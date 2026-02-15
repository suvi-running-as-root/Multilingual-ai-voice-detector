#!/usr/bin/env python3
"""
Training script for VoiceDetectorV2 classifier
Matches the detector_v2.py architecture exactly
"""

import os
import sys
import torch
import librosa
import numpy as np
import argparse
import json
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch.nn.functional as F

# ============= Configuration =============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "facebook/wav2vec2-xls-r-300m"  # Match detector_v2.py
MAX_DURATION = 6  # seconds (match detector)

print("="*60)
print("VoiceDetectorV2 Classifier Training")
print("="*60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")
print(f"Max audio duration: {MAX_DURATION}s")
print("="*60)

# ============= Dataset =============
class VoiceDataset(Dataset):
    def __init__(self, root_dir, feature_extractor, split='train'):
        self.samples = []
        self.feature_extractor = feature_extractor
        self.split = split
        
        print(f"\nLoading {split} dataset from {root_dir}...")
        
        # Load human samples
        human_dir = Path(root_dir) / "human"
        if human_dir.exists():
            human_files = list(human_dir.glob("*.wav"))
            for file in human_files:
                self.samples.append((str(file), 0))  # 0 = Human
            print(f"  Human samples: {len(human_files)}")
        
        # Load AI samples (including subdirectories)
        ai_dir = Path(root_dir) / "ai"
        if ai_dir.exists():
            ai_files = list(ai_dir.rglob("*.wav"))  # rglob = recursive
            for file in ai_files:
                self.samples.append((str(file), 1))  # 1 = AI
            print(f"  AI samples: {len(ai_files)}")
        
        print(f"  Total {split} samples: {len(self.samples)}")
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {root_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        
        try:
            y, sr = librosa.load(file_path, sr=16000, mono=True)
            
            # Limit duration
            max_samples = 16000 * MAX_DURATION
            if len(y) > max_samples:
                y = y[:max_samples]
            
            # Normalize
            max_val = np.abs(y).max()
            if max_val > 0:
                y = y / max_val
            
            # Extract features
            inputs = self.feature_extractor(
                y,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            
            return inputs.input_values.squeeze(0), torch.tensor(label)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            # Return random tensor and label to avoid crash
            return torch.randn(16000 * 2), torch.tensor(label)

# ============= Model =============
def create_model():
    """Create model matching detector_v2.py exactly"""
    print("\nLoading XLS-R backbone...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    backbone = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    
    # Freeze backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    hidden_size = backbone.config.hidden_size  # 1024 for XLS-R-300M
    
    # Classifier head (EXACTLY matching detector_v2.py)
    classifier = nn.Sequential(
        nn.Linear(hidden_size, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 2)
    )
    
    print(f"✓ Backbone loaded (hidden size: {hidden_size})")
    print(f"✓ Classifier created (trainable params: {sum(p.numel() for p in classifier.parameters())})")
    
    return feature_extractor, backbone, classifier

# ============= Training =============
def train_epoch(backbone, classifier, loader, optimizer, criterion, device):
    classifier.train()
    backbone.eval()  # Keep backbone frozen
    
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training")
    for input_values, labels in pbar:
        input_values = input_values.to(device)
        labels = labels.to(device)
        
        # Extract features (no gradients for backbone)
        with torch.no_grad():
            outputs = backbone(input_values)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1)  # Mean pooling
        
        # Classify
        logits = classifier(pooled)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update progress
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100*correct/total:.2f}%'
        })
    
    return total_loss / len(loader), 100 * correct / total

# ============= Validation =============
def validate(backbone, classifier, loader, criterion, device):
    classifier.eval()
    backbone.eval()
    
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for input_values, labels in pbar:
            input_values = input_values.to(device)
            labels = labels.to(device)
            
            # Extract features
            outputs = backbone(input_values)
            hidden_states = outputs.last_hidden_state
            pooled = hidden_states.mean(dim=1)
            
            # Classify
            logits = classifier(pooled)
            loss = criterion(logits, labels)
            
            # Get predictions
            probs = F.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=1)
            
            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # AI probability
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    # Calculate metrics
    accuracy = 100 * np.mean(np.array(all_predictions) == np.array(all_labels))
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
    
    return total_loss / len(loader), accuracy, auc, all_predictions, all_labels

# ============= Main =============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset/training_data')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    
    # Create model
    feature_extractor, backbone, classifier = create_model()
    backbone.to(DEVICE)
    classifier.to(DEVICE)
    
    # Load dataset
    full_dataset = VoiceDataset(args.data_dir, feature_extractor)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"\nDataset split:")
    print(f"  Train: {train_size} samples")
    print(f"  Val: {val_size} samples")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    history = []
    
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print("-"*60)
        
        # Train
        train_loss, train_acc = train_epoch(
            backbone, classifier, train_loader, optimizer, criterion, DEVICE
        )
        
        # Validate
        val_loss, val_acc, val_auc, val_preds, val_labels = validate(
            backbone, classifier, val_loader, criterion, DEVICE
        )
        
        # Print metrics
        print(f"\nClassification Report:")
        print(classification_report(
            val_labels, val_preds,
            target_names=['Human', 'AI'],
            digits=2
        ))
        
        print(f"Confusion Matrix:")
        print(confusion_matrix(val_labels, val_preds))
        
        print(f"\nROC AUC Score: {val_auc:.4f}")
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f}")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_auc': val_auc
        })
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            checkpoint = {
                'epoch': epoch + 1,
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc / 100,  # Convert to 0-1
                'val_auc': val_auc,
                'train_acc': train_acc / 100,
                'model_name': MODEL_NAME
            }
            
            torch.save(checkpoint, save_dir / 'best.pt')
            print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Save latest
        torch.save(checkpoint, save_dir / 'latest.pt')
        
        print(f"{'='*60}\n")
    
    # Save training history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
    