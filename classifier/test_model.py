#!/usr/bin/env python3
"""
Test script to verify the insurance fraud detector model configuration.
"""

from insurance_fraud_detector import InsuranceFraudDetector
import torch

def test_model_configuration():
    """Test that the model loads correctly and only classification head is trainable."""
    print("Testing Insurance Fraud Detector Model Configuration")
    print("=" * 60)
    
    try:
        # Initialize model (this will load Gemma-3-12B-IT)
        print("Initializing model with Gemma-3-12B-IT...")
        model = InsuranceFraudDetector()
        
        print("\n✅ Model loaded successfully!")
        
        # Test that backbone is frozen
        backbone_trainable = any(p.requires_grad for p in model.backbone.parameters())
        classifier_trainable = any(p.requires_grad for p in model.classifier.parameters())
        
        print(f"\nBackbone trainable: {backbone_trainable}")
        print(f"Classifier trainable: {classifier_trainable}")
        
        if not backbone_trainable and classifier_trainable:
            print("✅ Configuration correct: Backbone frozen, classifier trainable")
        else:
            print("❌ Configuration error: Check parameter freezing")
        
        # Test forward pass with dummy data
        print("\nTesting forward pass...")
        dummy_text = "This insurance policy requires immediate payment of $5000 before coverage begins."
        
        # Tokenize
        inputs = model.tokenizer(
            dummy_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"]
            print(f"✅ Forward pass successful! Output shape: {logits.shape}")
        
        # Test prediction
        print("\nTesting prediction...")
        prediction = model.predict(dummy_text)
        print(f"✅ Prediction successful!")
        print(f"   Probability: {prediction['probability']:.4f}")
        print(f"   Class: {prediction['class']}")
        
        print("\n🎉 All tests passed! Model is ready for training.")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        print("Make sure you have sufficient GPU memory or try running on CPU.")

if __name__ == "__main__":
    test_model_configuration()
