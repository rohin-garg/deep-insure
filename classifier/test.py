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
            dtype=torch.bfloat16 if 'cuda' in self.device else torch.float32,
            trust_remote_code=True
        )
        
        # Freeze the backbone model
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Get hidden size of the model
        # hidden_size = self.backbone.config.hidden_size
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

checkpoint_path = "/models/state_dict_100_2e-3.pth"
model = InsuranceFraudDetector(input_dim=768, device=device)
checkpoint = torch.load(checkpoint_path, map_location=device)
classifier_state = checkpoint['classifier_state_dict']
classifier_dict = {k.replace('classifier.', ''): v for k, v in classifier_state.items() if 'classifier' in k}
model.classifier.load_state_dict(classifier_dict)
print("Classifier weights loaded successfully!")
model.eval()

print(model.predict("This policy requires payment of a large upfront fee before any coverage begins."))


# # -------------------
# # Modal setup
# # -------------------
# app = modal.App("insurance-fraud-detector-gpu")

# # install dependencies in the container
# image = (
#     modal.Image.debian_slim()
#     .pip_install("torch", "fastapi", "uvicorn", "pydantic")
# )

# # -------------------
# # FastAPI app
# # -------------------
# web_app = FastAPI()

# # Request schema
# class InferenceRequest(BaseModel):
#     features: list[float]  # numeric features for the model

# if not torch.cuda.is_available():
#         device = 'cpu'
# else:
#     device = 'cuda'
# checkpoint_path = "/models/state_dict_100_2e-3.pth"
# # Load model only once per container
# with image.imports():
#     model = InsuranceFraudDetector(input_dim=768, device=device)
#     # model.load_state_dict(torch.load(model_path))
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     classifier_state = checkpoint['classifier_state_dict']
#     classifier_dict = {k.replace('classifier.', ''): v for k, v in classifier_state.items() if 'classifier' in k}
#     model.classifier.load_state_dict(classifier_dict)
#     print("Classifier weights loaded successfully!")
#     model.eval()


# @web_app.post("/predict")
# def predict(req: InferenceRequest):
#     with torch.no_grad():
#         x = torch.tensor(req.features, dtype=torch.float32).unsqueeze(0)
#         y = model(x).item()
#     return {"fraud_probability": y}


# # -------------------
# # Modal function to serve
# # -------------------
# @app.function(
#     image=image,
#     gpu="H100",     # or "A100", "H100"
#     timeout=1200     # optional, in seconds
# )
# @modal.asgi_app()
# def serve():
#     return web_app




