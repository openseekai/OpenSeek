import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

from models.temporal_video_detector import TemporalVideoDetector
from utils.video_utils import VideoProcessor

# Dataset Requirements:
# - Minimum 10k real clips (YouTube, phone cameras, Instagram)
# - Minimum 10k AI clips (Runway, Pika, Sora-style, Deepfake faces)
# Directory structure expected:
# dataset/
# ├── real/
# │   ├── vid1.mp4
# │   └── ...
# └── ai/
#     ├── vid2.mp4
#     └── ...

class VideoTemporalDataset(Dataset):
    def __init__(self, root_dir, seq_length=12):
        self.root_dir = root_dir
        self.processor = VideoProcessor(frame_count=seq_length)
        self.samples = []
        
        # Load Real Videos (Label 0)
        real_dir = os.path.join(root_dir, "real")
        if os.path.exists(real_dir):
            for f in os.listdir(real_dir):
                if f.endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((os.path.join(real_dir, f), 0.0))
                    
        # Load AI Videos (Label 1)
        ai_dir = os.path.join(root_dir, "ai")
        if os.path.exists(ai_dir):
            for f in os.listdir(ai_dir):
                if f.endswith(('.mp4', '.avi', '.mov')):
                    self.samples.append((os.path.join(ai_dir, f), 1.0))
                    
        print(f"Loaded {len(self.samples)} video paths.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # Extract 12 frames
        frames_tensor = self.processor.extract_frames(video_path)
        
        return frames_tensor, torch.tensor([label], dtype=torch.float32)

def train_temporal_model(data_dir, epochs=10, batch_size=4, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}...")
    
    dataset = VideoTemporalDataset(data_dir)
    if len(dataset) == 0:
        print("Dataset empty. Please populate dataset/real and dataset/ai")
        return
        
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    model = TemporalVideoDetector(sequence_length=12).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (frames, labels) in enumerate(loader):
            frames, labels = frames.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            outputs = model(frames)
            loss = criterion(outputs, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 9:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(loader)}], Loss: {running_loss/10:.4f}")
                running_loss = 0.0
                
        print(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f}s")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"weights/temporal_video_epoch_{epoch+1}.pth")
        
    print("Training Complete. Models saved to weights/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to database containing 'real' and 'ai' subdirs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    os.makedirs("weights", exist_ok=True)
    train_temporal_model(args.data_dir, args.epochs, args.batch_size)
