import torch
import torch.nn as nn
import timm

class TemporalVideoDetector(nn.Module):
    """
    Temporal Video Detection Model (Option A)
    Uses EfficientNet-B0 to extract frame features, and an LSTM to analyze the sequence for:
    - Face morph flicker
    - Lighting inconsistency
    - Expression instability
    - Frame-to-frame texture drift
    """
    def __init__(self, sequence_length=8, hidden_dim=256, num_layers=2):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_extractor = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg')
        
        # EfficientNet-B0 feature dimension is typically 1280
        feature_dim = self.feature_extractor.num_features
        
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        x: (Batch, Sequence, Channels, Height, Width)
        Wait, for batch inference, we'll flatten Batch x Sequence, pass through EfficientNet, then reshape.
        """
        b, seq, c, h, w = x.shape
        
        # Flatten for feature extraction
        x_flat = x.view(b * seq, c, h, w)
        
        # Extract features
        features = self.feature_extractor(x_flat)  # (B*Seq, Feature_Dim)
        
        # Reshape for LSTM
        features = features.view(b, seq, -1)  # (B, Seq, Feature_Dim)
        
        # LSTM
        lstm_out, (hn, cn) = self.lstm(features)  # lstm_out: (B, Seq, Hidden_Dim)
        
        # Take the output of the last time step
        last_step_out = lstm_out[:, -1, :]  # (B, Hidden_Dim)
        
        # Classify
        prob = self.classifier(last_step_out)  # (B, 1)
        
        return prob
        
    def predict_video(self, frames_tensor):
        """
        Clean interface for predicting a single video's sequence of frames.
        frames_tensor: (Sequence, Channels, Height, Width)
        Returns: float probability (0 = Real, 1 = AI)
        """
        with torch.no_grad():
            x = frames_tensor.unsqueeze(0)  # Add batch dimension -> (1, Seq, C, H, W)
            x = x.to(next(self.parameters()).device)
            # Ensure proper casting if model is FP16
            if next(self.parameters()).dtype == torch.float16:
                x = x.half()
            
            prob = self.forward(x)
            return float(prob.cpu().item())
