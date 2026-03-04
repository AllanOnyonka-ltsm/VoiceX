#!/usr/bin/env python3
"""
Example: Training a Simple AST Model for TB Detection

This is a simplified example showing how to:
1. Load audio data
2. Extract mel spectrograms
3. Train a model
4. Export to ONNX for VoiceX

Note: Replace with your actual dataset and model architecture.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split


# =============================================================================
# Dataset
# =============================================================================

class CoughDataset(Dataset):
    """Dataset for cough audio classification."""
    
    def __init__(self, audio_paths, labels, sample_rate=16000, duration=3.0):
        self.audio_paths = audio_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = int(sample_rate * duration)
    
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        # Load audio
        y, sr = librosa.load(
            self.audio_paths[idx],
            sr=self.sample_rate,
            mono=True,
            duration=self.duration
        )
        
        # Pad or truncate to target length
        if len(y) < self.target_length:
            y = np.pad(y, (0, self.target_length - len(y)))
        else:
            y = y[:self.target_length]
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize to target shape (128, 224)
        from scipy.ndimage import zoom
        mel_spec_resized = zoom(mel_spec_db, (1, 224 / mel_spec_db.shape[1]), order=1)
        
        # Normalize
        mel_spec_norm = (mel_spec_resized - mel_spec_resized.mean()) / (mel_spec_resized.std() + 1e-8)
        
        # Convert to tensor (1, 128, 224)
        mel_tensor = torch.from_numpy(mel_spec_norm).float().unsqueeze(0)
        
        # Label: 1 for TB positive, 0 for negative
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return mel_tensor, label


# =============================================================================
# Model Architecture
# =============================================================================

class SimpleAST(nn.Module):
    """
    Simplified Audio Spectrogram Transformer for TB detection.
    
    In production, use the full AST architecture from the paper:
    "AST: Audio Spectrogram Transformer" (Gong et al., 2021)
    """
    
    def __init__(self):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 14))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Mel spectrogram tensor of shape (batch, 1, 128, 224)
        
        Returns:
            Logits for binary classification
        """
        features = self.cnn(x)
        output = self.classifier(features)
        return output.squeeze(-1)


# =============================================================================
# Training
# =============================================================================

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """Train the model."""
    
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Accuracy
            pred = (torch.sigmoid(output) > 0.5).float()
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                
                pred = (torch.sigmoid(output) > 0.5).float()
                val_correct += (pred == target).sum().item()
                val_total += target.size(0)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("  Saved best model")
    
    return model


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("VoiceX Model Training Example")
    print("="*60)
    
    # Configuration
    DATA_DIR = "data/coughs"  # Replace with your dataset path
    TB_POSITIVE_DIR = f"{DATA_DIR}/positive"
    TB_NEGATIVE_DIR = f"{DATA_DIR}/negative"
    
    # Check if data exists
    if not Path(DATA_DIR).exists():
        print(f"\nNote: Dataset not found at {DATA_DIR}")
        print("Creating dummy dataset for demonstration...")
        
        # Create dummy data
        import os
        os.makedirs(TB_POSITIVE_DIR, exist_ok=True)
        os.makedirs(TB_NEGATIVE_DIR, exist_ok=True)
        
        # Generate dummy audio files
        for i in range(10):
            # This would be your actual audio files
            pass
        
        print("\nPlease place your audio files in:")
        print(f"  - TB Positive: {TB_POSITIVE_DIR}")
        print(f"  - TB Negative: {TB_NEGATIVE_DIR}")
        print("\nThen run this script again.")
        return
    
    # Load data paths
    positive_files = list(Path(TB_POSITIVE_DIR).glob("*.wav"))
    negative_files = list(Path(TB_NEGATIVE_DIR).glob("*.wav"))
    
    all_files = positive_files + negative_files
    all_labels = [1] * len(positive_files) + [0] * len(negative_files)
    
    print(f"\nDataset:")
    print(f"  TB Positive: {len(positive_files)}")
    print(f"  TB Negative: {len(negative_files)}")
    print(f"  Total: {len(all_files)}")
    
    # Split train/val
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    
    # Create datasets
    train_dataset = CoughDataset(train_files, train_labels)
    val_dataset = CoughDataset(val_files, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Create model
    model = SimpleAST()
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = train_model(model, train_loader, val_loader, epochs=10, device=device)
    
    # Export to ONNX
    print("\n" + "="*60)
    print("Exporting to ONNX...")
    print("="*60)
    
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 128, 224)
    
    torch.onnx.export(
        model,
        dummy_input,
        "tb_classifier.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['mel_spectrogram'],
        output_names=['output'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print("Model exported to tb_classifier.onnx")
    
    # Test inference
    print("\nTesting inference...")
    import onnxruntime as ort
    
    session = ort.InferenceSession("tb_classifier.onnx")
    test_input = np.random.randn(1, 1, 128, 224).astype(np.float32)
    output = session.run(None, {'mel_spectrogram': test_input})
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output[0].shape}")
    print(f"Output value: {output[0][0]}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Start the API server:")
    print("   python api_server.py --model tb_classifier.onnx")
    print("\n2. Test the API:")
    print("   python test_api.py --generate-audio")


if __name__ == "__main__":
    main()
