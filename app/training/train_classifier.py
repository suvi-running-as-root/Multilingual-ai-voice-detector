"""
Training script for Voice Detector V2 Classifier Head
PHASE 12: Train only the classifier head while keeping XLS-R backbone frozen
"""

import os
import sys
from pathlib import Path

# Fix imports - add project root to path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from tqdm import tqdm
import json

# Now import from app.models
from app.models.detector_v2 import VoiceDetectorV2


class AudioDataset(Dataset):
    """Dataset for AI vs Human audio classification"""
    
    def __init__(self, data_dir: str, max_duration: float = 2.0, target_sr: int = 16000):
        self.data_dir = Path(data_dir)
        self.max_duration = max_duration
        self.target_sr = target_sr
        self.max_samples = int(target_sr * max_duration)
        
        # Load file paths and labels
        self.samples = []
        
        # Human samples (label 0)
        human_dir = self.data_dir / "human"
        if human_dir.exists():
            for audio_file in human_dir.glob("*.wav"):
                self.samples.append((str(audio_file), 0))
        
        # AI samples (label 1)
        ai_dir = self.data_dir / "ai"
        if ai_dir.exists():
            for audio_file in ai_dir.glob("*.wav"):
                self.samples.append((str(audio_file), 1))
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"  - Human: {sum(1 for _, label in self.samples if label == 0)}")
        print(f"  - AI: {sum(1 for _, label in self.samples if label == 1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        audio_path, label = self.samples[idx]
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            # Trim silence
            y, _ = librosa.effects.trim(y, top_db=30)
            
            # Handle length
            if len(y) > self.max_samples:
                # Random crop during training
                start = np.random.randint(0, len(y) - self.max_samples)
                y = y[start:start + self.max_samples]
            elif len(y) < self.max_samples:
                # Pad with zeros
                y = np.pad(y, (0, self.max_samples - len(y)), mode='constant')
            
            # Normalize
            max_val = np.abs(y).max()
            if max_val > 0:
                y = y / max_val
            
            return torch.FloatTensor(y), label
            
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(self.max_samples), label


def train_epoch(model, classifier, feature_extractor, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    classifier.train()
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (audio, labels) in enumerate(pbar):
        audio = audio.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.no_grad():
            inputs = feature_extractor(
                audio.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            embeddings = hidden_states.mean(dim=1)
        
        logits = classifier(embeddings)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


def validate(model, classifier, feature_extractor, val_loader, criterion, device):
    """Validate the model"""
    classifier.eval()
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        
        for audio, labels in pbar:
            audio = audio.to(device)
            labels = labels.to(device)
            
            inputs = feature_extractor(
                audio.cpu().numpy(),
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            embeddings = hidden_states.mean(dim=1)
            
            logits = classifier(embeddings)
            probs = torch.softmax(logits, dim=-1)
            
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{total_loss / len(val_loader):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Human', 'AI']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nROC AUC Score: {auc:.4f}")
    
    return total_loss / len(val_loader), 100. * correct / total, auc


def main():
    parser = argparse.ArgumentParser(description='Train Voice Detector V2 Classifier')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Voice Detector V2 - Classifier Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("\nLoading dataset...")
    full_dataset = AudioDataset(args.data_dir)
    
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows
        pin_memory=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if args.device == 'cuda' else False
    )
    
    print("\nInitializing model...")
    detector = VoiceDetectorV2()
    
    model = detector.model
    classifier = detector.classifier
    feature_extractor = detector.feature_extractor
    
    device = torch.device(args.device)
    model = model.to(device)
    classifier = classifier.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    
    for param in classifier.parameters():
        param.requires_grad = True
    
    print(f"Trainable parameters: {sum(p.numel() for p in classifier.parameters() if p.requires_grad)}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(classifier.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': []
    }
    
    print("\nStarting training...\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(
            model, classifier, feature_extractor, train_loader,
            criterion, optimizer, device
        )
        
        val_loss, val_acc, val_auc = validate(
            model, classifier, feature_extractor, val_loader,
            criterion, device
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val AUC: {val_auc:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print()
        
        checkpoint = {
            'epoch': epoch,
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_acc': val_acc,
            'val_auc': val_auc,
            'history': history
        }
        
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest.pt'))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, os.path.join(args.save_dir, 'best.pt'))
            print(f"✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        print("=" * 60)
    
    print("\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nCheckpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    main()