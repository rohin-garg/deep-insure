import modal
import torch
import torch.nn as nn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
import os
from pathlib import Path


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
            torch_dtype=torch.bfloat16,  # Use torch_dtype instead of dtype
            trust_remote_code=True,
            attn_implementation="flash_attention_2"  # Enable Flash Attention for speed
        )
        
        # Freeze the backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get hidden size of the model
        hidden_size = 3840
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
        # Set classifier to same dtype as backbone model
        self.classifier = self.classifier.to(self.device, dtype=torch.bfloat16)
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
    
    @torch.inference_mode()  # More efficient than torch.no_grad() for inference
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
            max_length=1024,  # Reduced from 2048 for faster inference
            return_tensors="pt"
        ).to(self.device)
        
        # Get predictions
        outputs = self(**inputs)
        prob_fraud = outputs["logits"].item()  # Already a probability from sigmoid
        
        return {
            "probability": prob_fraud,
            "prediction": 1 if prob_fraud >= threshold else 0,
            "class": "suspicious" if prob_fraud >= threshold else "normal"
        }


# -------------------
# Modal setup
# -------------------
app = modal.App("insurance-fraud-detector-gpu-2")

# Optimized image with all necessary packages
# image = modal.Image.debian_slim(python_version="3.11").pip_install([
#     "torch>=2.0.0",
#     "transformers>=4.35.0",
#     "fastapi",
#     "accelerate>=0.24.0",
#     "flash-attn>=2.3.0",  # For Flash Attention
#     "pydantic"
# ])
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(["torch", "transformers", "fastapi", "accelerate", "pydantic"])
    .pip_install("numpy")  # Add separately if needed
    .pip_install("tqdm")   # Add separately if needed
)

# Persistent volume where checkpoint resides
volume = modal.Volume.from_name("insurance-models")
MODEL_DIR = Path("/models")
CHECKPOINT_FILE = MODEL_DIR / "state_dict_100_2e-3.pth"

class PredictionRequest(BaseModel):
    features: str
    threshold: float = 0.5

# -------------------
# Model container class - FIXED
# -------------------
@app.cls(
    image=image,
    gpu="H100",
    volumes={MODEL_DIR: volume},
    timeout=1200,
    keep_warm=1,  # Keep 1 instance warm to avoid cold starts
    container_idle_timeout=300,  # Keep containers alive for 5 minutes
    allow_concurrent_inputs=10  # Allow multiple concurrent requests
)
class ModelContainer:
    @modal.enter()
    def load_model(self):
        """Initialize model when container starts"""
        print("Initializing model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model
        self.model = InsuranceFraudDetector(device=device)
        
        # Load checkpoint from volume
        print(f"Loading checkpoint from {CHECKPOINT_FILE}...")
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
        classifier_state = checkpoint['classifier_state_dict']
        classifier_dict = {k.replace('classifier.', ''): v for k, v in classifier_state.items() if 'classifier' in k}
        self.model.classifier.load_state_dict(classifier_dict)
        self.model.eval()
        
        # Warmup with a dummy prediction to compile/optimize
        print("Warming up model...")
        dummy_text = "This is a sample insurance claim for testing purposes."
        _ = self.model.predict(dummy_text)
        
        print("Model loaded and warmed up successfully!")

    @modal.method()
    def predict(self, text: str, threshold: float = 0.5):
        """Make prediction on input text"""
        return self.model.predict(text, threshold)

    @modal.method()
    def batch_predict(self, texts: List[str], threshold: float = 0.5):
        """Make predictions on multiple texts"""
        results = []
        for text in texts:
            results.append(self.model.predict(text, threshold))
        return results

# -------------------
# FastAPI app
# -------------------
web_app = FastAPI()

@web_app.post("/predict")
async def predict_endpoint(request: PredictionRequest):
    """Single prediction endpoint"""
    container = ModelContainer()
    result = container.predict.remote(request.features, request.threshold)
    return {"prob": result["probability"], "class": result["class"]}

@web_app.post("/batch_predict") 
async def batch_predict_endpoint(request: dict):
    """Batch prediction endpoint"""
    texts = request["features"]  # List of strings
    threshold = request.get("threshold", 0.5)
    
    container = ModelContainer()
    results = container.batch_predict.remote(texts, threshold)
    
    return {"predictions": [{"prob": r["probability"], "class": r["class"]} for r in results]}

@web_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

# -------------------
# Serve the app
# -------------------
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("modal")],
)
@modal.asgi_app()
def serve():
    return web_app