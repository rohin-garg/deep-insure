import modal
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------
# Modal setup
# -----------------------------
stub = modal.App("insurance-fraud-h100")

# Image with necessary packages
image = modal.Image.debian_slim().pip_install("torch", "transformers", "tqdm")

# Persistent volume for checkpoint
# volume = modal.SharedVolume().persist("insurance-model-vol")  # attach your model files here
volume = modal.NetworkFileSystem.persisted("insurance-model-vol")

# -----------------------------
# Model class
# -----------------------------
class InsuranceFraudDetector(nn.Module):
    def __init__(self, model_name: str = "google/gemma-3-12b-it", device: str = "cuda"):
        super().__init__()
        self.device = device
        print(f"Loading model {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            trust_remote_code=True
        )
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        hidden_size = 3840
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
            nn.Sigmoid()
        ).to(self.device, dtype=torch.bfloat16)

        self.loss_fn = nn.BCELoss()
        print("Model initialized.")

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        last_hidden_state = outputs.hidden_states[-1]
        last_token_hidden = last_hidden_state[:, -1, :]
        if last_token_hidden.dtype != next(self.classifier.parameters()).dtype:
            last_token_hidden = last_token_hidden.to(next(self.classifier.parameters()).dtype)
        probs = self.classifier(last_token_hidden).squeeze(-1)
        return probs

    def predict(self, text: str, threshold: float = 0.5):
        self.eval()
        inputs = self.tokenizer(
            text, padding=True, truncation=True, max_length=2048, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            prob_fraud = self(inputs["input_ids"], attention_mask=inputs["attention_mask"]).item()
        return {
            "probability": prob_fraud,
            "prediction": 1 if prob_fraud >= threshold else 0,
            "class": "suspicious" if prob_fraud >= threshold else "normal"
        }

# -----------------------------
# Load model once and keep in memory
# -----------------------------
class ModelWrapper:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = InsuranceFraudDetector(device="cuda")
            # Load checkpoint
            checkpoint_path = "/model/state_dict_100_2e-3.pth"
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            classifier_state = checkpoint['classifier_state_dict']
            classifier_dict = {k.replace('classifier.', ''): v for k, v in classifier_state.items() if 'classifier' in k}
            cls._model.classifier.load_state_dict(classifier_dict)
            cls._model.eval()
        return cls._model

# -----------------------------
# Modal function
# -----------------------------
@stub.function(
    image=image,
    gpu="H100",
    network_filesystems={"/model": volume},
    timeout=600
)
def run_inference(test_text: str):
    model = ModelWrapper.get_model()
    return model.predict(test_text)

# -----------------------------
# Local entrypoint for CLI testing
# -----------------------------
@stub.local_entrypoint()
def main(test_text: str = None):
    if test_text is None:
        test_text = "This policy requires payment of a large upfront fee before any coverage begins."
    print(run_inference.call(test_text))
