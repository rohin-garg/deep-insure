import modal
import torch
from transformers import AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import json

# Modal setup
app = modal.App("insurance-redflag-model")

# Install dependencies
image = modal.Image.debian_slim().pip_install(
    "torch", "transformers", "fastapi", "uvicorn", "accelerate"
)

# Create volume for model weights
volume = modal.Volume.from_name("insurance-models", create_if_missing=True)

@app.function(
    image=image, 
    min_containers=1, 
    secrets=[modal.Secret.from_name("modal")],
    volumes={"/models": volume},
    gpu="A100-40GB",
    timeout=1200
    keep_warm=1  # 15 minutes for model loading
)
def load_model():
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Define the model class within Modal
    class InsuranceFraudDetector(nn.Module):
        def __init__(self, model_name: str = "google/gemma-3-12b-it", num_labels: int = 2, device: str = None):
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
            
        def predict(self, text: str, threshold: float = 0.5):
            self.eval()
            
            # Tokenize input
            inputs = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.backbone(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True
                )
                
                # Use the hidden states of the last token
                last_hidden_state = outputs.hidden_states[-1]
                last_token_hidden = last_hidden_state[:, -1, :]
                
                # Ensure dtype consistency
                classifier_dtype = next(self.classifier.parameters()).dtype
                if last_token_hidden.dtype != classifier_dtype:
                    last_token_hidden = last_token_hidden.to(dtype=classifier_dtype)
                
                # Pass through classifier
                prob_fraud = self.classifier(last_token_hidden).squeeze(-1).item()
            
            return {
                "probability": prob_fraud,
                "prediction": 1 if prob_fraud >= threshold else 0,
                "class": "suspicious" if prob_fraud >= threshold else "normal"
            }
    
    # Load the checkpoint
    checkpoint_path = "/models/state_dict_100_2e-3.pth"
    if not torch.cuda.is_available():
        device = 'cpu'
    else:
        device = 'cuda'
    
    # Initialize model
    model = InsuranceFraudDetector(device=device)
    
    # Load classifier weights if checkpoint exists
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'classifier_state_dict' in checkpoint:
            classifier_state = checkpoint['classifier_state_dict']
            classifier_dict = {k.replace('classifier.', ''): v for k, v in classifier_state.items() if 'classifier' in k}
            model.classifier.load_state_dict(classifier_dict)
            print("Classifier weights loaded successfully!")
        else:
            print("Warning: No classifier_state_dict found in checkpoint")
    except FileNotFoundError:
        print("Warning: No checkpoint found, using untrained classifier")
    
    model.eval()
    return model

# API endpoint
class InputData(BaseModel):
    text: str

# Define a container class to hold the loaded model
class ModelContainer:
    def __init__(self):
        # This will be set when the container starts
        self._model = None
    
    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded. Make sure to call load_model() first.")
        return self._model
    
    @modal.enter()
    def load_model(self):
        # This runs once when the container starts
        print("Loading model in container...")
        self._model = load_model.remote()
        print("Model loaded successfully in container")
    
    @modal.method()
    def predict(self, text: str, threshold: float = 0.5):
        return self.model.predict(text, threshold)

# Create a container instance
model_container = ModelContainer()

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("modal")],
    volumes={"/models": volume},
    gpu="A100-40GB",
    timeout=60,  # Shorter timeout for inference
    container_idle_timeout=1200  # Keep container alive for 10 minutes of inactivity
)
@modal.fastapi_endpoint(method="POST")
def predict(request):
    data = json.loads(request.body.decode())
    text = data["text"]
    threshold = data.get("threshold", 0.5)
    
    # Use the pre-loaded model from the container
    prediction = model_container.predict.remote(text, threshold)
    return prediction

# Deploy
if __name__ == "__main__":
    app.deploy()