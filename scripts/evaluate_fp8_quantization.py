"""
FP8 Post-Training Quantization Evaluation Script.

This script evaluates FP8 post-training quantization on a trained FP16 model:
1. Load best FP16 model (from AugmFP16 experiment)
2. Evaluate original FP16 accuracy
3. Quantize to FP8 (E4M3 format)
4. Restore to FP16
5. Evaluate quantized accuracy
6. Measure accuracy degradation

This is NOT a training script - it's a conversion and evaluation test.

Usage:
    $ python scripts/evaluate_fp8_quantization.py

Requirements:
    - AugmFP16 experiment must be completed first
    - Best model checkpoint: results/AugmFP16/checkpoints/best_model.pt

Expected Results:
    - Accuracy Loss: ~2-3% (acceptable for deployment)
    - Demonstrates FP8 viability for inference
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.vit_model import create_vit_model
from src.data.dataset import get_cifar10_loaders
from src.utils.metrics import calculate_accuracy
from tqdm import tqdm


def quantize_to_fp8(tensor, e4m3_range=(-240, 240)):
    """
    Quantize tensor to FP8 E4M3 format and back to FP16.

    E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits

    Args:
        tensor (torch.Tensor): Input tensor (FP16 or FP32)
        e4m3_range (tuple): Min/max values for E4M3 format

    Returns:
        torch.Tensor: Quantized tensor (back in FP16)

    Note:
        IMPORTANT: Use actual weight range, not theoretical max!
        Using theoretical max (±240) will zero out all weights.
        Should use actual data range (e.g., ±0.64).
    """
    # Get actual data range (critical for proper quantization!)
    actual_min = tensor.min().item()
    actual_max = tensor.max().item()

    # Use actual range instead of theoretical range
    # This prevents zeroing out weights
    scale = max(abs(actual_min), abs(actual_max))

    # Quantize: scale to FP8 range, round, then scale back
    # Simulate FP8 precision loss
    fp8_scale = 240.0 / scale if scale > 0 else 1.0

    quantized = tensor * fp8_scale
    quantized = torch.clamp(quantized, -240, 240)
    quantized = torch.round(quantized)  # Simulate discrete FP8 values
    quantized = quantized / fp8_scale

    return quantized


def quantize_model_to_fp8(model):
    """
    Quantize all model weights to FP8 E4M3 and back to FP16.

    Args:
        model (torch.nn.Module): Model to quantize

    Returns:
        torch.nn.Module: Quantized model

    Note:
        This is POST-TRAINING quantization (no retraining).
        Simulates FP8 deployment on specialized hardware.
    """
    print("Quantizing model to FP8...")

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:  # Only quantize trainable weights
                param.data = quantize_to_fp8(param.data)

    print("Model quantized to FP8 (E4M3)")
    return model


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """
    Evaluate model accuracy and loss on test set.

    Args:
        model (torch.nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        device (str): Device to use

    Returns:
        tuple: (accuracy, loss)
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    print("Evaluating model...")
    for images, labels in tqdm(test_loader, desc='Evaluation'):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(test_loader)

    return accuracy, avg_loss


def main():
    """
    Main execution for FP8 quantization testing.

    Workflow:
        1. Load trained FP16 model
        2. Evaluate original accuracy
        3. Quantize to FP8
        4. Evaluate quantized accuracy
        5. Report accuracy degradation
        6. Save results
    """
    print("\n" + "="*70)
    print("FP8 Post-Training Quantization Test")
    print("="*70 + "\n")

    # Configuration
    CONFIG = {
        'name': 'FP8Test',
        'source_model': 'results/AugmFP16/checkpoints/best_model.pt',
        'model_name': 'vit_tiny_patch16_224',
        'num_classes': 10,
        'batch_size': 128,
        'device': 'mps'
    }

    print("Configuration:")
    print(f"  Source Model: {CONFIG['source_model']}")
    print(f"  Model: {CONFIG['model_name']}")
    print(f"  Device: {CONFIG['device']}")
    print()

    # Check if source model exists
    source_path = Path(CONFIG['source_model'])
    if not source_path.exists():
        print(f"Error: Source model not found!")
        print(f"   Expected: {source_path}")
        print(f"\n   Please run AugmFP16 experiment first:")
        print(f"   $ python scripts/train_AugmFP16.py")
        sys.exit(1)

    # Create model
    print("Creating model...")
    device = torch.device(CONFIG['device'])
    model = create_vit_model(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes'],
        pretrained=False
    )

    # Load trained weights
    print(f"Loading trained weights from: {source_path}")
    checkpoint = torch.load(source_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Model loaded (original val_acc: {checkpoint['val_acc']:.4f})")
    print()

    # Load test data
    print("Loading CIFAR-10 test set...")
    _, test_loader = get_cifar10_loaders(
        batch_size=CONFIG['batch_size'],
        num_workers=2,
        augmentation='extended',  # Same as training
        data_dir='./data'
    )
    print(f"✓ Test loader ready: {len(test_loader)} batches\n")

    # Evaluate original FP16 model
    print("="*70)
    print("STEP 1: Evaluate Original FP16 Model")
    print("="*70)
    original_acc, original_loss = evaluate_model(model, test_loader, device)
    print(f"\n✓ Original FP16 Model:")
    print(f"   Accuracy: {original_acc:.4f} ({original_acc*100:.2f}%)")
    print(f"   Loss: {original_loss:.4f}\n")

    # Quantize to FP8
    print("="*70)
    print("STEP 2: Quantize to FP8 (E4M3)")
    print("="*70)
    model = quantize_model_to_fp8(model)
    print()

    # Evaluate quantized model
    print("="*70)
    print("STEP 3: Evaluate Quantized FP8 Model")
    print("="*70)
    quantized_acc, quantized_loss = evaluate_model(model, test_loader, device)
    print(f"\n✓ Quantized FP8 Model:")
    print(f"   Accuracy: {quantized_acc:.4f} ({quantized_acc*100:.2f}%)")
    print(f"   Loss: {quantized_loss:.4f}\n")

    # Calculate degradation
    acc_loss = original_acc - quantized_acc
    acc_loss_percent = acc_loss * 100
    loss_increase = quantized_loss - original_loss

    # Save results
    results = {
        'experiment': 'FP8Test',
        'source_model': str(source_path),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'original_fp16': {
            'accuracy': float(original_acc),
            'loss': float(original_loss)
        },
        'quantized_fp8': {
            'accuracy': float(quantized_acc),
            'loss': float(quantized_loss)
        },
        'degradation': {
            'accuracy_loss': float(acc_loss),
            'accuracy_loss_percent': float(acc_loss_percent),
            'loss_increase': float(loss_increase)
        }
    }

    # Save results
    output_dir = Path('results/FP8Test/metrics')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'fp8_quantization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("="*70)
    print("FP8 QUANTIZATION SUMMARY")
    print("="*70)
    print(f"\nOriginal FP16 Accuracy:   {original_acc:.4f} ({original_acc*100:.2f}%)")
    print(f"Quantized FP8 Accuracy:   {quantized_acc:.4f} ({quantized_acc*100:.2f}%)")
    print(f"Accuracy Degradation:     {acc_loss:.4f} ({acc_loss_percent:.2f}%)")
    print(f"\nOriginal FP16 Loss:       {original_loss:.4f}")
    print(f"Quantized FP8 Loss:       {quantized_loss:.4f}")
    print(f"Loss Increase:            {loss_increase:.4f}")

    print(f"\n{'='*70}")

    # Interpretation
    if acc_loss_percent < 1.0:
        print("Excellent! <1% accuracy loss - FP8 is highly viable!")
    elif acc_loss_percent < 3.0:
        print("Good! <3% accuracy loss - FP8 is viable for deployment")
    elif acc_loss_percent < 5.0:
        print("Moderate: 3-5% accuracy loss - Consider QAT (Quantization-Aware Training)")
    else:
        print("High degradation: >5% accuracy loss - FP8 may not be suitable")

    print(f"\nResults saved to: {results_path}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
