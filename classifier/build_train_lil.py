import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os
# from insurance_fragments import insurance_data

# Modal imports
import modal
from modal import Image, App, Volume, gpu, method

# Modal configuration
app = App("insurance-fraud-detector")

#save_name
save_name = "state_dict_1B_200_1e-3.pth"

# Create Modal image with required dependencies
image = (
    Image.debian_slim()
    .pip_install([
        "torch",
        "transformers",
        "numpy",
        "tqdm",
        "accelerate",
        "bitsandbytes",
    ])
    .apt_install(["git"])
)

# Create volume for model persistence
volume = Volume.from_name("insurance-models", create_if_missing=True)

class InsuranceFraudDetector(nn.Module):
    def __init__(self, model_name: str = "google/gemma-3-1b-it", num_labels: int = 2, device: str = None):
        """
        Initialize the insurance fraud detector with a pre-trained Gemma model.
        
        Args:
            model_name: Name or path of the pre-trained Gemma model
            num_labels: Number of output labels (2 for binary classification)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        super().__init__()
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model and tokenizer
        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16 if 'cuda' in self.device else torch.float32,
            trust_remote_code=True
        )
        
        # Freeze the backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get hidden size of the model
        hidden_size = self.backbone.config.hidden_size
        # hidden_size = 3840
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Set classifier to same dtype as backbone model
        classifier_dtype = torch.bfloat16 if 'cuda' in self.device else torch.float32
        self.classifier = self.classifier.to(self.device, dtype=classifier_dtype)
        self.loss_fn = nn.BCELoss()
        
        # Verify only classifier parameters are trainable
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model loaded on {self.device}")
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
    def get_trainable_parameters(self):
        """Return only the trainable parameters (classification head)."""
        return filter(lambda p: p.requires_grad, self.parameters())
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Ground truth labels for training
            
        Returns:
            Dictionary containing logits and (if labels provided) loss
        """
        # Get hidden states from the backbone model
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Use the hidden states of the last token
        last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
        last_token_hidden = last_hidden_state[:, -1, :]  # (batch_size, hidden_size)
        
        # Ensure dtype consistency between backbone output and classifier
        classifier_dtype = next(self.classifier.parameters()).dtype
        if last_token_hidden.dtype != classifier_dtype:
            last_token_hidden = last_token_hidden.to(dtype=classifier_dtype)
        
        # Pass through classifier (includes sigmoid)
        probs = self.classifier(last_token_hidden)  # (batch_size, 1)
        probs = probs.squeeze(-1)  # (batch_size,) - remove last dimension
        
        # Convert probs to float32 for loss calculation if needed
        if probs.dtype != torch.float32:
            probs_for_loss = probs.float()
        else:
            probs_for_loss = probs
        
        output = {"logits": probs}  # Keep same key for compatibility
        
        # Calculate loss if labels are provided
        if labels is not None:
            labels_float = labels.float()  # BCELoss expects float labels
            loss = self.loss_fn(probs_for_loss, labels_float)
            output["loss"] = loss
            
        return output
    
    def predict(self, text: str, threshold: float = 0.5) -> Dict[str, float]:
        """
        Make a prediction on a single input text.
        
        Args:
            text: Input insurance policy text
            threshold: Threshold for positive class
            
        Returns:
            Dictionary with prediction probabilities and class
        """
        self.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=2048,  # Adjust based on GPU memory
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self(**inputs)
            prob_fraud = outputs["logits"].item()  # Already a probability from sigmoid
        
        return {
            "probability": prob_fraud,
            "prediction": 1 if prob_fraud >= threshold else 0,
            "class": "suspicious" if prob_fraud >= threshold else "normal"
        }


class InsuranceDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 2048):
        """
        Dataset for insurance fraud detection.
        
        Args:
            data: List of dictionaries with 'text' and 'label' keys
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(item["label"], dtype=torch.long)
        }


@app.function(
    image=image,
    gpu="H100",
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("modal")],
    timeout=3600,  # 1 hour timeout
)
def train_model(
    train_data: List[Dict],
    val_data: Optional[List[Dict]] = None,
    batch_size: int = 4,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    model_name: str = "google/gemma-3-1b-it",  # Using non-gated model for Modal
    model_save_path: str = "/models/" + save_name
):
    """
    Train the insurance fraud detection model on Modal.
    
    Args:
        train_data: Training data
        val_data: Optional validation data
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        model_name: Name of the pre-trained model to use
        model_save_path: Path to save the trained model
    """
    # Initialize model inside Modal function
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InsuranceFraudDetector(model_name=model_name, device=device)
    
    # Create datasets
    train_dataset = InsuranceDataset(train_data, model.tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if val_data:
        val_dataset = InsuranceDataset(val_data, model.tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Set up optimizer (only train the classifier parameters)
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=learning_rate
    )
    
    best_val_loss = float('inf')
    train_losses = []
    batch_train_losses = []
    val_losses = []
    batch_val_losses = []
    test_accuracies = []
    batch_val_accuracies = []
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Store initial parameters for comparison
        initial_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()
        
        # Training
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            # Move batch to device
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Backward pass and optimize
            loss = outputs["loss"]
            loss.backward()
            
            # Calculate gradient norm before optimizer step
            total_grad_norm = 0
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_grad_norm += param_norm.item() ** 2
            total_grad_norm = total_grad_norm ** (1. / 2)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Check if parameters actually changed
            param_changed = False
            max_param_change = 0
            for name, param in model.named_parameters():
                if param.requires_grad and name in initial_params:
                    param_change = (param.data - initial_params[name]).abs().max().item()
                    max_param_change = max(max_param_change, param_change)
                    if param_change > 1e-8:  # Small threshold for numerical precision
                        param_changed = True
                        print("param_change=", param_change)
                        print("param.data=", param.data.abs().max().item())
                        print("initial_params[name]=", initial_params[name].abs().max().item())
                        # Update initial params for next comparison
                        initial_params[name] = param.data.clone()
            
            total_loss += loss.item()
            batch_train_losses.append(loss.item())
            
            # Validate on validation data after each batch
            if val_data:
                model.eval()
                val_correct = 0
                val_total = 0
                val_loss_batch = 0
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input_ids = val_batch["input_ids"].to(model.device)
                        val_attention_mask = val_batch["attention_mask"].to(model.device)
                        val_labels = val_batch["labels"].to(model.device)
                        
                        val_outputs = model(
                            input_ids=val_input_ids,
                            attention_mask=val_attention_mask,
                            labels=val_labels
                        )
                        
                        # Track validation loss
                        val_loss_batch += val_outputs["loss"].item()
                        
                        # Calculate accuracy for binary classification
                        probs = val_outputs["logits"]
                        predicted = (probs > 0.5).float()
                        val_total += val_labels.size(0)
                        val_correct += (predicted == val_labels.float()).sum().item()
                        
                        # Debug: Print first batch info
                        if batch_idx == 0 and val_total <= val_labels.size(0):
                            tqdm.write(f"    Debug - Probs range: {probs.min():.3f} to {probs.max():.3f}")
                            tqdm.write(f"    Debug - Labels: {val_labels.float()[:5].tolist()}")
                            tqdm.write(f"    Debug - Predictions: {predicted[:5].tolist()}")
                
                # Calculate and store batch validation metrics
                batch_val_loss = val_loss_batch / len(val_loader)
                batch_val_accuracy = 100 * val_correct / val_total if val_total > 0 else 0
                
                batch_val_losses.append(batch_val_loss)
                batch_val_accuracies.append(batch_val_accuracy)
                test_accuracies.append(batch_val_accuracy)  # Keep for backward compatibility
                
                model.train()  # Switch back to training mode
            
            # Log batch loss, validation accuracy, and parameter update info
            if val_data:
                param_status = "✓" if param_changed else "✗"
                tqdm.write(f"  Batch {batch_idx}, Train Loss: {loss.item():.4f}, Val Loss: {batch_val_loss:.4f}, Val Acc: {batch_val_accuracy:.2f}%, Grad Norm: {total_grad_norm:.6f}, Params Updated: {param_status} (Max Change: {max_param_change:.2e})")
            else:
                param_status = "✓" if param_changed else "✗"
                tqdm.write(f"  Batch {batch_idx}, Train Loss: {loss.item():.4f}, Grad Norm: {total_grad_norm:.6f}, Params Updated: {param_status} (Max Change: {max_param_change:.2e})")
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_data:
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(model.device)
                    attention_mask = batch["attention_mask"].to(model.device)
                    labels = batch["labels"].to(model.device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs["loss"].item()
                    
                    # Calculate accuracy for binary classification
                    predicted = (outputs["logits"] > 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels.float()).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_accuracy = 100 * correct / total
            
            print(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save to volume first (only classifier weights to save space)
                classifier_state = {k: v.cpu() for k, v in model.state_dict().items() if 'classifier' in k}
                torch.save({
                    'classifier_state_dict': classifier_state,
                    'model_name': model_name,
                    'train_losses': train_losses,
                    'batch_train_losses': batch_train_losses,
                    'val_losses': val_losses,
                    'batch_val_losses': batch_val_losses,
                    'test_accuracies': test_accuracies,
                    'batch_val_accuracies': batch_val_accuracies,
                    'epoch': epoch + 1
                }, "/models/" + save_name)
                print(f"Model saved to volume")
    
    # Return both the trained model and the state dict for local saving
    # Move to CPU to avoid CUDA deserialization issues
    classifier_state = {k: v.cpu() for k, v in model.state_dict().items() if 'classifier' in k}
    return {
        'classifier_state_dict': classifier_state,  # Only save trainable classifier weights
        'model_name': model_name,  # Save model name to reload backbone later
        'train_losses': train_losses,
        'batch_train_losses': batch_train_losses,
        'val_losses': val_losses,
        'batch_val_losses': batch_val_losses,
        'test_accuracies': test_accuracies,
        'batch_val_accuracies': batch_val_accuracies,
        'final_epoch': num_epochs
    }


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("modal")],
)
def load_model(model_path: str = "/models/" + save_name, device: str = None) -> InsuranceFraudDetector:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model
        device: Device to load the model on
        
    Returns:
        Loaded InsuranceFraudDetector model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the saved model
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = InsuranceFraudDetector(device=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/models": volume},
    secrets=[modal.Secret.from_name("modal")],
)
def predict_fraud(text: str, model_path: str = "/models/" + save_name, threshold: float = 0.5) -> Dict[str, float]:
    """
    Make a fraud prediction on Modal.
    
    Args:
        text: Input insurance policy text
        model_path: Path to the trained model
        threshold: Threshold for positive class
        
    Returns:
        Dictionary with prediction probabilities and class
    """
    # Load the model
    model = load_model.local(model_path)
    
    # Make prediction
    return model.predict(text, threshold)


@app.local_entrypoint()
def main():
    """
    Main function to run training on Modal.
    """
    # Load data from insurance_fragments
    print("Loading training data...")
    from insurance_fragments import insurance_data
    train_data = insurance_data[:200]  # Use the data from insurance_fragments module
    
    # Split data into train/val (80/20 split)
    split_idx = int(0.8 * len(train_data))
    train_split = train_data[:split_idx]
    val_split = train_data[split_idx:]
    
    print(f"Training samples: {len(train_split)}")
    print(f"Validation samples: {len(val_split)}")
    
    # Run training on Modal
    print("Starting training on Modal...")
    model_data = train_model.remote(
        train_data=train_split,
        val_data=val_split,
        batch_size=2,  # Small batch size for large model
        num_epochs=1,
        learning_rate=1e-3,
        model_name="google/gemma-3-1b-it" 
    )
    
    # Save the trained model locally
    local_model_path = "./" + save_name
    torch.save(model_data, local_model_path)
    print(f"Model saved locally to: {local_model_path}")
    
    # Example prediction after training
    # test_text = "This policy requires payment of a large upfront fee before any coverage begins."
    # print(f"\nMaking prediction on Modal...")
    # prediction = predict_fraud.remote(test_text)
    # print(f"Text: {test_text}")
    # print(f"Prediction: {prediction}")


# For local testing (non-Modal execution)
if __name__ == "__main__":
    # This block runs when executed locally (not on Modal)
    print("Running locally - use 'modal run build_train.py' to run on Modal")
    
    # # Example data for local testing
    # example_data = [
    #     {"text": "This insurance policy has a 30-day waiting period before any claims can be made.", "label": 0},
    #     {"text": "All claims must be submitted within 24 hours of the incident.", "label": 1},
    # ]
    
    # Initialize model locally
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InsuranceFraudDetector(model_name="google/gemma-3-1b-it", device=device)
    
    # Example prediction
    test_text = "This policy requires payment of a large upfront fee before any coverage begins."
    prediction = model.predict(test_text)
    print(f"\nLocal prediction:")
    print(f"Text: {test_text}")
    print(f"Prediction: {prediction}")
