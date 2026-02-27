import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from tqdm import tqdm

"""
Phase 11: Real Training Pipeline (Engine Fuel)
This file dictates how DeepShield model architectures are fundamentally trained on heavily augmented
diffusion generators to actually identify subtle structural frequency deviations.
"""

# 1. Dataset Structure & Transformations (Crucial Augmentations)
train_transforms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class DiffusionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Expects directory structure:
        root_dir/
            real/
            diffusion/
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 0: Real Photograph, 1: Diffusion
        real_dir = os.path.join(root_dir, 'real')
        diffusion_dir = os.path.join(root_dir, 'diffusion')
        
        if os.path.exists(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(real_dir, fname))
                    self.labels.append(0)
                    
        if os.path.exists(diffusion_dir):
            for fname in os.listdir(diffusion_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.image_paths.append(os.path.join(diffusion_dir, fname))
                    self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_balanced_sampler(dataset):
    """3. Evaluates dataset imbalance and forces sampler to pull 50/50 batches."""
    labels = dataset.labels
    class_counts = np.bincount(labels)
    
    if len(class_counts) < 2 or class_counts[0] == 0 or class_counts[1] == 0:
        return None
        
    class_weights = 1.0 / class_counts
    sample_weights = np.array([class_weights[t] for t in labels])
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

def train_diffusion_model(data_dir="../dataset", num_epochs=15, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Dataset directories not found. Please structure as:")
        print("  dataset/train/real")
        print("  dataset/train/diffusion")
        print("  dataset/val/real")
        print("  dataset/val/diffusion")
        return

    train_dataset = DiffusionDataset(train_dir, transform=train_transforms)
    val_dataset = DiffusionDataset(val_dir, transform=val_transforms)
    
    if len(train_dataset) == 0:
        print("No training images found!")
        return

    train_sampler = create_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler, 
        num_workers=4, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # 4. Model Setup
    model = models.efficientnet_b2(weights="IMAGENET1K_V1")
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model = model.to(device)
    
    # 5. Loss Function (Added safeguards for missing datasets so bincount works cleanly)
    class_counts = np.bincount(train_dataset.labels)
    weight_tensor = torch.tensor([1.0 / float(c) if c > 0 else 1.0 for c in class_counts], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    # 6. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    
    best_val_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # 7. Training Loop Requirements
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 20)
        
        # Train phase
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc="Training")
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_bar.set_postfix({'loss': loss.item()})
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)
                
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_loss = val_loss / len(val_dataset)
        
        # Validation Metrics
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = 0.0
            
        print(f"Val Loss: {val_loss:.4f}  |  Acc: {acc:.4f}  |  Prec: {prec:.4f}  |  Rec: {rec:.4f}  |  F1: {f1:.4f}  |  AUC: {auc:.4f}")
        
        scheduler.step(acc)
        
        if acc > best_val_acc:
            best_val_acc = acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), "deepshield_diffusion_b2.pth")
            print(">>> Saved new best model weights! <<<")
            
    # 8. Threshold Calibration
    print("\nTraining Complete! Calibrating Threshold...")
    model.load_state_dict(best_model_wts)
    model.eval()
    
    try:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        # Optimal threshold using Youden's J statistic
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        print(f"\n[!] Optimal ROC-AUC Decision Threshold: {optimal_threshold:.4f}")
        print(f"-> You MUST update models/diffusion_detector.py to use {optimal_threshold:.4f} instead of 0.5")
    except Exception as e:
        print(f"Could not calculate ROC boundary calibration: {e}")

if __name__ == "__main__":
    train_diffusion_model()
