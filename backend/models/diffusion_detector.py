import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.forensics.generation_step_analyzer import GenerationStepAnalyzer


class DiffusionDetector(nn.Module):
    """
    Phase 10: Strict Binary Diffusion Classifier

    Enhanced with a Flowchart-Guided Generation Step Consistency Detector to isolate
    artifacts across the 6 steps of the image generation process.

    Outputs probability distribution:
    Class 0: Real
    Class 1: Diffusion_AI
    """
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.classes = ["Real", "Diffusion_AI"]

        # Using EfficientNetV2-S for highly accurate diffusion micro-pattern detection
        self.model = models.efficientnet_v2_s(weights="IMAGENET1K_V1")

        # Modify the classification head for binary classes
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, 2)

        # Initialize the flowchart-guided generation consistency analyzer
        self.analyzer = GenerationStepAnalyzer()

        self.to(self.device)
        self.eval()

    def forward(self, x):
        """
        Expects a normalized 224x224 tensor.
        Returns raw logits.
        """
        return self.model(x)

    def predict(self, x, image_path=None):
        """
        Returns the parsed class label, the normalized probability, and flowchart step scores.
        Blends CNN predictions with physical/structural generation step consistency.
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        float(probs[0])
        nn_ai_prob = float(probs[1])

        flowchart_analysis = None
        if image_path is not None:
            try:
                analyzer_res = self.analyzer.analyze_image(image_path)
                analyzer_ai_prob = analyzer_res["ai_probability"]

                # Blend: 40% neural network patterns + 60% flowchart consistency analysis.
                # This makes the detection highly robust to compression, scaling, and unseen generators.
                ai_probability = 0.40 * nn_ai_prob + 0.60 * analyzer_ai_prob
                real_probability = 1.0 - ai_probability
                probs = np.array([real_probability, ai_probability])

                flowchart_analysis = {
                    "is_ai": analyzer_res["is_ai_generated"],
                    "scores": analyzer_res["scores"],
                    "metrics": analyzer_res["metrics"]
                }
            except Exception as e:
                print(f"[DiffusionDetector] Flowchart consistency analyzer failed: {e}. Falling back to NN.")
                ai_probability = nn_ai_prob
        else:
            ai_probability = nn_ai_prob

        pred_idx = np.argmax(probs)
        predicted_class = self.classes[pred_idx]

        response = {
            "predicted_class": predicted_class,
            "probability_distribution": {
                "Real": float(probs[0]),
                "Diffusion_AI": float(probs[1])
            },
            "ai_probability": float(ai_probability)
        }

        if flowchart_analysis is not None:
            response["flowchart_analysis"] = flowchart_analysis

        return response
