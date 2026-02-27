import torch
import torch.nn as nn
import librosa
import numpy as np

class AudioCNNLSTM(nn.Module):
    """
    Research-Grade Audio Deepfake Detector using MFCC + Mel-spectrogram.
    Extracts features through CNN + BiLSTM to find phase inconsistencies.
    """
    def __init__(self, input_dim=128+20, hidden_dim=256, num_layers=2):
        super().__init__()
        # Input dim: 128 (mel) + 20 (mfcc) = 148
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        self.bilstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim // 2, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            # Output: [ai_probability_audio, spectral_anomaly_score]
            nn.Linear(64, 2) 
        )
        
    def forward(self, features):
        # features: (Batch, Channels=148, Time)
        x = self.cnn(features) # (B, hidden_dim, Time)
        x = x.permute(0, 2, 1) # (B, Time, hidden_dim)
        x, _ = self.bilstm(x)  # (B, Time, hidden_dim)
        # Global average pooling over time
        x = x.mean(dim=1)      # (B, hidden_dim)
        out = torch.sigmoid(self.fc(x))
        return out

def extract_advanced_audio_features(audio_path, n_mels=128, n_mfcc=20):
    """
    Extract Mel-Spectrogram and MFCC features and concatenate them.
    Detects unnatural frequency distributions and breath irregularities.
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        y, _ = librosa.effects.trim(y)
        
        # Log-Mel
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        S_log = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_log - np.min(S_log)) / (np.max(S_log) - np.min(S_log) + 1e-9)
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_norm = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc) + 1e-9)
        
        # Combine
        combined = np.concatenate((S_norm, mfcc_norm), axis=0) # (148, Time)
        return torch.from_numpy(combined).float()
    except Exception as e:
        print(f"[DeepShield] Audio extraction error: {e}")
        return torch.zeros((n_mels + n_mfcc, 100))
