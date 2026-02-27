import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

class DiffusionDetector(nn.Module):
    """
    Phase 10: Strict Binary Diffusion Classifier
    
    Designed to detect Text-to-Image models (Midjourney, DALL-E, SDXL) by isolating
    over-smoothed microtextures, repetitive gradients, and frequency-space clustering.
    Specifically excludes "Edited" logic, leaving that to the photographic PRNU pipeline.
    
    Outputs probability distribution:
    Class 0: Real
    Class 1: Diffusion_AI
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.classes = ["Real", "Diffusion_AI"]

        # Using EfficientNet-B2 for diffusion micro-pattern detection
        self.model = models.efficientnet_b2(weights="IMAGENET1K_V1") 
        
        # Modify the classification head for binary classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)
        
        self.to(self.device)
        self.eval()

    def forward(self, x):
        """
        Expects a normalized 224x224 tensor.
        Returns raw logits.
        """
        return self.model(x)

    def predict(self, x):
        """
        Returns the parsed class label and the normalized probability.
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        pred_idx = np.argmax(probs)
        predicted_class = self.classes[pred_idx]
        
        return {
            "predicted_class": predicted_class,
            "probability_distribution": {
                "Real": float(probs[0]),
                "Diffusion_AI": float(probs[1])
            },
            "ai_probability": float(probs[1]) 
        }
