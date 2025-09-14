import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from build_train import InsuranceFraudDetector


# class InsuranceFraudDetector(nn.Module):
#     def __init__(self, model_name: str = "google/gemma-3-12b-it", num_labels: int = 2, device: str = None):
#         """
#         Initialize the insurance fraud detector with a pre-trained Gemma model.
        
#         Args:
#             model_name: Name or path of the pre-trained Gemma model
#             num_labels: Number of output labels (2 for binary classification)
#             device: Device to run the model on ('cuda' or 'cpu')
#         """
#         super().__init__()
        
#         # Set device
#         self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
#         # Load pre-trained model and tokenizer
#         print(f"Loading model {model_name}...")
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.backbone = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             device_map="auto",
#             dtype=torch.bfloat16 if 'cuda' in self.device else torch.float32,
#             trust_remote_code=True
#         )
        
#         # Freeze the backbone model
#         for param in self.backbone.parameters():
#             param.requires_grad = False
        
#         # Get hidden size of the model
#         hidden_size = 3840
        
#         # Classification head
#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_size, 512),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(512, 1),
#             nn.Sigmoid()
#         )
        
#         # Set classifier to same dtype as backbone model
#         classifier_dtype = torch.bfloat16 if 'cuda' in self.device else torch.float32
#         self.classifier = self.classifier.to(self.device, dtype=classifier_dtype)
#         self.loss_fn = nn.BCELoss()
        
#         # Verify only classifier parameters are trainable
#         trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
#         total_params = sum(p.numel() for p in self.parameters())
#         print(f"Model loaded on {self.device}")
#         print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
#     def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
#                 labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
#         """
#         Forward pass through the model.
        
#         Args:
#             input_ids: Input token IDs
#             attention_mask: Attention mask
#             labels: Ground truth labels for training
            
#         Returns:
#             Dictionary containing logits and (if labels provided) loss
#         """
#         # Get hidden states from the backbone model
#         outputs = self.backbone(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             output_hidden_states=True,
#             return_dict=True
#         )
        
#         # Use the hidden states of the last token
#         last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_size)
#         last_token_hidden = last_hidden_state[:, -1, :]  # (batch_size, hidden_size)
        
#         # Ensure dtype consistency between backbone output and classifier
#         classifier_dtype = next(self.classifier.parameters()).dtype
#         if last_token_hidden.dtype != classifier_dtype:
#             last_token_hidden = last_token_hidden.to(dtype=classifier_dtype)
        
#         # Pass through classifier (includes sigmoid)
#         probs = self.classifier(last_token_hidden)  # (batch_size, 1)
#         probs = probs.squeeze(-1)  # (batch_size,) - remove last dimension
        
#         # Convert probs to float32 for loss calculation if needed
#         if probs.dtype != torch.float32:
#             probs_for_loss = probs.float()
#         else:
#             probs_for_loss = probs
        
#         output = {"logits": probs}  # Keep same key for compatibility
        
#         # Calculate loss if labels are provided
#         if labels is not None:
#             labels_float = labels.float()  # BCELoss expects float labels
#             loss = self.loss_fn(probs_for_loss, labels_float)
#             output["loss"] = loss
            
#         return output
    
#     def predict(self, text: str, threshold: float = 0.5) -> Dict[str, float]:
#         """
#         Make a prediction on a single input text.
        
#         Args:
#             text: Input insurance policy text
#             threshold: Threshold for positive class
            
#         Returns:
#             Dictionary with prediction probabilities and class
#         """
#         self.eval()
        
#         # Tokenize input
#         inputs = self.tokenizer(
#             text,
#             padding=True,
#             truncation=True,
#             max_length=2048,  # Adjust based on GPU memory
#             return_tensors="pt"
#         ).to(self.device)
        
#         # Get predictions
#         with torch.no_grad():
#             outputs = self(**inputs)
#             prob_fraud = outputs["logits"].item()  # Already a probability from sigmoid
        
#         return {
#             "probability": prob_fraud,
#             "prediction": 1 if prob_fraud >= threshold else 0,
#             "class": "suspicious" if prob_fraud >= threshold else "normal"
#         }


def load_trained_model(model_path: str, model_name: str = "google/gemma-3-12b-it", device: str = None) -> InsuranceFraudDetector:
    """
    Load a trained insurance fraud detector model.
    
    Args:
        model_path: Path to the saved model (.pth file)
        model_name: Name of the pre-trained Gemma model to use as backbone
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Loaded InsuranceFraudDetector model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the checkpoint
    print(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model name from checkpoint if available
    saved_model_name = checkpoint.get('model_name', model_name)
    
    # Initialize the model with the Gemma backbone
    print(f"Initializing model with backbone: {saved_model_name}")
    model = InsuranceFraudDetector(model_name=saved_model_name, device=device)
    
    # Load the classifier state dict
    if 'classifier_state_dict' in checkpoint:
        print("Loading classifier weights...")
        classifier_state = checkpoint['classifier_state_dict']
        # Filter to only classifier parameters
        classifier_dict = {k.replace('classifier.', ''): v for k, v in classifier_state.items() if 'classifier' in k}
        model.classifier.load_state_dict(classifier_dict)
    else:
        print("Warning: No classifier_state_dict found in checkpoint")
    
    model.eval()
    print("Model loaded successfully!")
    return model


def load_training_stats(model_path: str) -> Dict:
    """
    Load the saved model state dict and extract training statistics.
    
    Args:
        model_path: Path to the saved model (.pth file)
        
    Returns:
        Dictionary containing training statistics
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    return {
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
        'test_accuracies': checkpoint.get('test_accuracies', []),
        'epoch': checkpoint.get('epoch', checkpoint.get('final_epoch', 0))
    }

def plot_loss_curves(train_losses: List[float], val_losses: Optional[List[float]] = None, 
                    save_path: Optional[str] = None, show_plot: bool = True):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses per epoch
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training loss
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, marker='o')
    
    # Plot validation loss if available
    if val_losses and len(val_losses) > 0:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, marker='s')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()

def visualize_batch_train_losses(model_path: str, save_path: Optional[str] = None, show_plot: bool = True):
    """
    Visualize the batch training losses from training data.
    
    Args:
        model_path: Path to the saved model (.pth file)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract batch training losses
    batch_train_losses = checkpoint.get('batch_train_losses', [])
    
    if not batch_train_losses:
        print("No batch training losses found in checkpoint")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create batch indices
    batches = range(1, len(batch_train_losses) + 1)
    
    # Plot batch training loss
    plt.plot(batches, batch_train_losses, 'b-', label='Batch Training Loss', linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Batch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some styling
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Batch training loss plot saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()

def visualize_batch_val_accuracies(model_path: str, save_path: Optional[str] = None, show_plot: bool = True):
    """
    Visualize the batch validation accuracies from training data.
    
    Args:
        model_path: Path to the saved model (.pth file)
        save_path: Optional path to save the plot
        show_plot: Whether to display the plot
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract batch validation accuracies
    batch_val_accuracies = checkpoint.get('batch_val_accuracies', [])
    
    if not batch_val_accuracies:
        print("No batch validation accuracies found in checkpoint")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Create batch indices
    batches = range(1, len(batch_val_accuracies) + 1)
    
    # Plot batch validation accuracy
    plt.plot(batches, batch_val_accuracies, 'g-', label='Batch Validation Accuracy', linewidth=1.5, alpha=0.8)
    
    plt.xlabel('Batch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy vs Batch')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)  # Accuracy is typically 0-100%
    
    # Add some styling
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Batch validation accuracy plot saved to: {save_path}")
    
    # Show plot
    if show_plot:
        plt.show()

def print_training_summary(stats: Dict):
    """
    Print a summary of training statistics.
    
    Args:
        stats: Dictionary containing training statistics
    """
    train_losses = stats['train_losses']
    val_losses = stats['val_losses']
    epoch = stats['epoch']
    
    print("=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Total Epochs: {epoch}")
    print(f"Training Loss - Initial: {train_losses[0]:.4f}, Final: {train_losses[-1]:.4f}")
    
    if val_losses:
        print(f"Validation Loss - Initial: {val_losses[0]:.4f}, Final: {val_losses[-1]:.4f}")
        print(f"Best Validation Loss: {min(val_losses):.4f} (Epoch {val_losses.index(min(val_losses)) + 1})")
    
    # Calculate improvement
    if len(train_losses) > 1:
        improvement = ((train_losses[0] - train_losses[-1]) / train_losses[0]) * 100
        print(f"Training Loss Improvement: {improvement:.2f}%")
    
    print("=" * 50)

def main():
    """
    Main function to visualize batch training losses and batch validation accuracies.
    """
    # Path to the saved model (adjust as needed)
    model_path = "./state_dict_1.pth"
    
    try:
        print("Loading model checkpoint...")
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            print("Make sure to run the training script first to generate the model file.")
            return
        
        # Load the checkpoint to analyze available data
        checkpoint = torch.load(model_path, map_location='cpu')
        
        print("\n" + "="*50)
        print("CHECKPOINT ANALYSIS")
        print("="*50)
        
        # Print available data in checkpoint
        print(f"Available keys in checkpoint: {list(checkpoint.keys())}")
        
        # Check for batch-level data
        batch_train_losses = checkpoint.get('batch_train_losses', [])
        batch_val_accuracies = checkpoint.get('batch_val_accuracies', [])
        
        print(f"\nBatch-level data available:")
        print(f"  Batch training losses: {len(batch_train_losses)} data points")
        print(f"  Batch validation accuracies: {len(batch_val_accuracies)} data points")
        
        print("\n" + "="*50)
        print("GENERATING BATCH-LEVEL VISUALIZATIONS")
        print("="*50)
        
        # Visualize batch training losses
        print("\n1. Generating batch training losses visualization...")
        try:
            visualize_batch_train_losses(
                model_path=model_path,
                save_path="batch_train_losses.png",
                show_plot=True
            )
        except Exception as e:
            print(f"Error generating batch training losses visualization: {e}")
        
        # Visualize batch validation accuracies
        print("\n2. Generating batch validation accuracies visualization...")
        try:
            visualize_batch_val_accuracies(
                model_path=model_path,
                save_path="batch_val_accuracies.png",
                show_plot=True
            )
        except Exception as e:
            print(f"Error generating batch validation accuracies visualization: {e}")
        
        print("\nBatch-level visualization complete!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run the training script first to generate the model file.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()