"""
OpenSeek — Audio Model Training Script

Usage:
  python train/train_audio.py --data_dir ./data --epochs 20 --batch_size 16

data_dir must contain two subfolders:
  data/real/   ← real/authentic audio files (.wav, .mp3)
  data/fake/   ← deepfake/synthetic voice files

Outputs:
  weights/audio_model.pt  ← trained weights
"""
import os
import argparse
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ─── Simple 1D-CNN for MFCC ──────────────────────────────────────────────────

class AudioSpoofDetector(nn.Module):
    def __init__(self, n_mfcc=40):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mfcc, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ─── Dataset ─────────────────────────────────────────────────────────────────

class AudioDataset(Dataset):
    def __init__(self, data_dir, n_mfcc=40, duration=5.0):
        self.samples = []
        self.n_mfcc = n_mfcc
        self.sr = 16000
        self.length = int(self.sr * duration)
        
        real_dir = os.path.join(data_dir, "real")
        fake_dir = os.path.join(data_dir, "fake")
        
        for d, label in [(real_dir, 0.0), (fake_dir, 1.0)]:
            for f in os.listdir(d):
                if f.lower().endswith((".wav", ".mp3", ".flac")):
                    self.samples.append((os.path.join(d, f), label))
        print(f"[Dataset] Found {len(self.samples)} audio files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            y, _ = librosa.load(path, sr=self.sr, mono=True, duration=5.0)
            if len(y) < self.length:
                y = np.pad(y, (0, self.length - len(y)))
            else:
                y = y[:self.length]
            
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.n_mfcc)
            # Normalize
            mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)
            return torch.tensor(mfcc, dtype=torch.float32), torch.tensor([label], dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros((self.n_mfcc, 157)), torch.tensor([label], dtype=torch.float32)

# ─── Training ─────────────────────────────────────────────────────────────────

def train(data_dir, epochs=20, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Device: {device}")

    dataset = AudioDataset(data_dir)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioSpoofDetector().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    weights_path = os.path.join(os.path.dirname(__file__), "..", "weights", "audio_model.pt")
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += ((preds > 0.5).float() == y).sum().item()
        
        print(f"Epoch {epoch}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {correct/len(dataset)*100:.1f}%")
        torch.save(model.state_dict(), weights_path)

    print(f"\n[Done] weights saved to {weights_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(args.data_dir, args.epochs)
