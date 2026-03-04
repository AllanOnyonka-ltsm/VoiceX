#!/usr/bin/env python3
"""
Export PyTorch AST Model to ONNX for VoiceX Edge

Usage:
    python export_model.py --checkpoint model.pth --output tb_classifier.onnx
"""

import argparse
import torch
import torch.nn as nn
from pathlib import Path


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: tuple = (1, 1, 128, 224),
    opset_version: int = 11
):
    """Export PyTorch model to ONNX format."""
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['mel_spectrogram'],
        output_names=['risk_score', 'confidence'],
        dynamic_axes={
            'mel_spectrogram': {0: 'batch_size'},
            'risk_score': {0: 'batch_size'},
            'confidence': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")
    
    # Verify with onnxruntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        print(f"ONNX model verified successfully")
        print(f"Inputs: {[i.name for i in session.get_inputs()]}")
        print(f"Outputs: {[o.name for o in session.get_outputs()]}")
    except Exception as e:
        print(f"Warning: Could not verify model: {e}")


def export_torchscript(
    model: nn.Module,
    output_path: str,
    example_input: torch.Tensor
):
    """Export model to TorchScript for Python API."""
    
    model.eval()
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save
    traced_model.save(output_path)
    
    print(f"TorchScript model saved to {output_path}")


# Example AST Model Architecture
class TBASTModel(nn.Module):
    """
    Example AST (Audio Spectrogram Transformer) model for TB detection.
    
    Replace this with your actual model architecture.
    """
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        
        # Simple CNN feature extractor (replace with AST)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 7))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Mel spectrogram tensor of shape (batch, 1, 128, 224)
        
        Returns:
            Tensor of shape (batch, 2) with [risk_score, confidence]
        """
        x = self.features(x)
        x = self.classifier(x)
        
        # Apply sigmoid to get probabilities
        risk_score = torch.sigmoid(x[:, 0])
        confidence = torch.sigmoid(x[:, 1])
        
        return torch.stack([risk_score, confidence], dim=1)


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output", type=str, default="tb_classifier.onnx", help="Output ONNX path")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript", "both"], default="onnx")
    parser.add_argument("--input-shape", type=str, default="1,1,128,224", help="Input shape (batch,channels,height,width)")
    
    args = parser.parse_args()
    
    # Parse input shape
    input_shape = tuple(map(int, args.input_shape.split(',')))
    
    # Create model (replace with your model loading)
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}")
        model = TBASTModel()
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    else:
        print("No checkpoint provided, creating new model")
        model = TBASTModel()
    
    model.eval()
    
    # Export
    if args.format in ["onnx", "both"]:
        export_to_onnx(model, args.output, input_shape)
    
    if args.format in ["torchscript", "both"]:
        ts_output = args.output.replace('.onnx', '.pt')
        dummy_input = torch.randn(*input_shape)
        export_torchscript(model, ts_output, dummy_input)


if __name__ == "__main__":
    main()
