import torch
import torch.nn as nn
import torchvision.models as models

class ContentTypeClassifier(nn.Module):
    """
    Ultra-lightweight pipeline router (<20ms inference).
    Identifies base structural features to categorize an input as:
    0: Photograph -> Routes to forensic PRNU + Deepfake detectors
    1: Digital Illustration -> Routes to T2I Diffusion detectors
    2: 3D Render -> Routes to T2I Diffusion detectors
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        
        # Use MobileNetV3 Small for extreme low-latency classification with ImageNet weights
        self.model = models.mobilenet_v3_small(weights="DEFAULT") 
        
        # Modify final classifier for 3-class output
        in_features = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(in_features, 3)
        
        self.to(self.device)
        self.eval()

    def forward(self, x):
        """
        Expects a normalized 224x224 tensor.
        Returns 3-class softmax probabilities.
        """
        logits = self.model(x)
        return torch.softmax(logits, dim=1)

    def classify(self, x):
        probs = self.forward(x)
        pred_class = torch.argmax(probs, dim=1).item()
        
        classes = ["Photograph", "Digital Illustration", "3D Render"]
        return classes[pred_class]
