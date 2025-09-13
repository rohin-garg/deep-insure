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
from insurance_fragments.py import data

class InsuranceFraudDetector(nn.Module):
    def __init__(self, model_name: str = "google/gemma-3-12b-it", num_labels: int = 2, device: str = None):
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
            torch_dtype=torch.bfloat16 if 'cuda' in self.device else torch.float32,
            trust_remote_code=True
        )
        
        # Freeze the backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get hidden size of the model
        hidden_size = self.backbone.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels)
        )
        
        self.classifier = self.classifier.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        
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
        
        # Pass through classifier
        logits = self.classifier(last_token_hidden)  # (batch_size, num_labels)
        
        output = {"logits": logits}
        
        # Calculate loss if labels are provided
        if labels is not None:
            loss = self.loss_fn(logits, labels)
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
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            
        # Get probability of positive class (class 1)
        prob_fraud = probs[0, 1].item()
        
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


def train_model(
    model: InsuranceFraudDetector,
    train_data: List[Dict],
    val_data: Optional[List[Dict]] = None,
    batch_size: int = 4,
    num_epochs: int = 5,
    learning_rate: float = 2e-5,
    model_save_path: str = "insurance_fraud_detector.pth"
) -> None:
    """
    Train the insurance fraud detection model.
    
    Args:
        model: The model to train
        train_data: Training data
        val_data: Optional validation data
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        model_save_path: Path to save the trained model
    """
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
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
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
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
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
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs["logits"], 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            
            print(f"Epoch {epoch + 1} - Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'tokenizer': model.tokenizer,
                    'config': model.backbone.config
                }, model_save_path)
                print(f"Model saved to {model_save_path}")
    
    print("Training complete!")


def load_model(model_path: str, device: str = None) -> InsuranceFraudDetector:
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


# Example usage
if __name__ == "__main__":
    # Example data (replace with your actual data)
    example_data = [
        {"text": "This insurance policy has a 30-day waiting period before any claims can be made.", "label": 0},
        {"text": "All claims must be submitted within 24 hours of the incident.", "label": 1},
        # Add more training examples here
    ]
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InsuranceFraudDetector(device=device)
    
    # Split data into train/val (in a real scenario, use proper splitting)
    train_data = example_data[:int(0.8 * len(example_data))]
    val_data = example_data[int(0.8 * len(example_data)):]
    
    # Train the model (uncomment to train)
    # train_model(
    #     model=model,
    #     train_data=train_data,
    #     val_data=val_data,
    #     batch_size=2,  # Adjust based on GPU memory
    #     num_epochs=3,
    #     model_save_path="insurance_fraud_detector.pth"
    # )
    
    # Example prediction (after training)
    test_text = "This policy requires payment of a large upfront fee before any coverage begins."
    prediction = model.predict(test_text)
    print(f"\nPrediction for test text:")
    print(f"Text: {test_text}")
    print(f"Prediction: {prediction}")
