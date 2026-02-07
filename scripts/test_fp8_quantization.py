#!/usr/bin/env python3
"""
FP8 Quantization Test: FP16 -> FP8 -> FP16 Conversion
Tests accuracy degradation from quantization

Note: PyTorch doesn't support native FP8 yet, so we simulate it through:
1. INT8 quantization (closest available)
2. Manual precision reduction to simulate FP8
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.vit_model import create_vit_model, get_model_info
from src.data.dataset import get_cifar10_loaders


class FP8Simulator:
    """
    Simulate FP8 behavior by reducing precision
    FP8 formats (IEEE-like):
    - E4M3: 4 exponent bits, 3 mantissa bits
    - E5M2: 5 exponent bits, 2 mantissa bits
    """

    @staticmethod
    def to_fp8_e4m3(tensor):
        """
        Simulate FP8 E4M3 format (4 exp bits, 3 mantissa bits)
        Higher precision for mantissa, lower range

        E4M3: 3 mantissa bits = 8 quantization levels
        We quantize based on the actual range of values, not theoretical max
        """
        # Convert to numpy for manipulation
        arr = tensor.detach().cpu().numpy()

        # Get actual range of values
        abs_max = np.abs(arr).max()

        # Avoid division by zero
        if abs_max < 1e-10:
            return tensor

        # Quantize to 8 levels (3 bits mantissa)
        # Scale to use full range of quantization levels
        num_levels = 8  # 2^3 = 8 levels for 3-bit mantissa

        # Quantize: map [-abs_max, abs_max] to [-num_levels/2, num_levels/2]
        scale = abs_max / (num_levels / 2.0)
        quantized = np.round(arr / scale) * scale

        # Clip to valid range (safety)
        quantized = np.clip(quantized, -abs_max, abs_max)

        return torch.from_numpy(quantized).to(tensor.device).to(tensor.dtype)

    @staticmethod
    def quantize_model_fp8(model):
        """Apply FP8 simulation to all model weights"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.data = FP8Simulator.to_fp8_e4m3(param.data)
        return model


def load_trained_model(checkpoint_path, device='mps'):
    """Load pre-trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = create_vit_model(
        model_name='vit_tiny_patch16_224',
        num_classes=10,
        pretrained=False
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f"  Epoch trained: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Val accuracy: {checkpoint.get('val_acc', 0) * 100:.2f}%")

    return model, checkpoint


def evaluate_model(model, test_loader, device='mps', desc='Evaluation'):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        pbar = tqdm(test_loader, desc=desc)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()

            accuracy = 100 * correct / total
            pbar.set_postfix({'acc': f'{accuracy:.2f}%', 'loss': f'{loss.item():.4f}'})

    final_accuracy = 100 * correct / total
    final_loss = total_loss / len(test_loader)

    return final_accuracy, final_loss


def calculate_model_stats(model):
    """Calculate model statistics (weights distribution)"""
    all_weights = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.extend(param.data.cpu().numpy().flatten().tolist())

    all_weights = np.array(all_weights)

    stats = {
        'mean': float(np.mean(all_weights)),
        'std': float(np.std(all_weights)),
        'min': float(np.min(all_weights)),
        'max': float(np.max(all_weights)),
        'median': float(np.median(all_weights)),
        'percentile_1': float(np.percentile(all_weights, 1)),
        'percentile_99': float(np.percentile(all_weights, 99)),
    }

    return stats


def main():
    print("=" * 80)
    print("FP8 QUANTIZATION TEST: FP16 → FP8 → FP16")
    print("=" * 80)
    print()

    # Configuration
    config = {
        'checkpoint_path': '../results/checkpoints/exp3_fp16_fixed/best_model.pt',
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'batch_size': 128,
        'num_workers': 2,
    }

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # Check checkpoint exists
    checkpoint_path = Path(config['checkpoint_path'])
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\nMake sure you've run exp3_fp16_fixed first!")
        return

    # Setup device
    device = config['device']
    print(f"Using device: {device}")
    if device == 'cpu':
        print("⚠️  Warning: Running on CPU, this will be slow!")
    print()

    # Load data
    print("Loading CIFAR-10 test set...")
    _, test_loader = get_cifar10_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        augmentation='basic',  # No augmentation for test
        pin_memory=(device != 'cpu')
    )
    print(f"✓ Test set loaded: {len(test_loader)} batches")
    print()

    # ============================================================
    # STEP 1: Load original FP16 model
    # ============================================================
    print("=" * 80)
    print("STEP 1: Load Original FP16 Model")
    print("=" * 80)

    model_fp16, checkpoint = load_trained_model(checkpoint_path, device)

    # Get model info
    info = get_model_info(model_fp16)
    print(f"\nModel info:")
    print(f"  Parameters: {info['trainable_params_millions']:.2f}M")
    print()

    # Calculate weight statistics
    print("FP16 Model Weight Statistics:")
    stats_fp16 = calculate_model_stats(model_fp16)
    for key, value in stats_fp16.items():
        print(f"  {key}: {value:.6f}")
    print()

    # Evaluate FP16 model
    print("Evaluating FP16 model...")
    acc_fp16, loss_fp16 = evaluate_model(
        model_fp16, test_loader, device,
        desc='FP16 Evaluation'
    )
    print(f"\n✓ FP16 Model Results:")
    print(f"  Accuracy: {acc_fp16:.2f}%")
    print(f"  Loss: {loss_fp16:.4f}")
    print()

    # ============================================================
    # STEP 2: Convert to FP8 (simulated)
    # ============================================================
    print("=" * 80)
    print("STEP 2: Convert to FP8 (Simulated E4M3 format)")
    print("=" * 80)
    print()
    print("Converting model weights to FP8 precision...")
    print("Note: Using E4M3 format (4 exponent bits, 3 mantissa bits)")
    print("      Range: ~[-240, 240], Mantissa: 8 levels")
    print()

    # Clone model for FP8 conversion
    model_fp8 = create_vit_model('vit_tiny_patch16_224', num_classes=10, pretrained=False)
    model_fp8.load_state_dict(model_fp16.state_dict())
    model_fp8 = model_fp8.to(device)

    # Apply FP8 simulation
    model_fp8 = FP8Simulator.quantize_model_fp8(model_fp8)

    print("✓ Conversion to FP8 complete")
    print()

    # Calculate weight statistics after FP8
    print("FP8 Model Weight Statistics:")
    stats_fp8 = calculate_model_stats(model_fp8)
    for key, value in stats_fp8.items():
        print(f"  {key}: {value:.6f}")
    print()

    # Show difference
    print("Weight Statistics Difference (FP16 → FP8):")
    for key in stats_fp16.keys():
        diff = stats_fp8[key] - stats_fp16[key]
        print(f"  Δ{key}: {diff:.6f}")
    print()

    # Evaluate FP8 model
    print("Evaluating FP8 model...")
    acc_fp8, loss_fp8 = evaluate_model(
        model_fp8, test_loader, device,
        desc='FP8 Evaluation'
    )
    print(f"\n✓ FP8 Model Results:")
    print(f"  Accuracy: {acc_fp8:.2f}%")
    print(f"  Loss: {loss_fp8:.4f}")
    print()

    # ============================================================
    # STEP 3: Convert back to FP16
    # ============================================================
    print("=" * 80)
    print("STEP 3: Convert Back to FP16")
    print("=" * 80)
    print()
    print("Converting FP8 model back to FP16...")
    print("Note: This simulates loading FP8 weights back into FP16 model")
    print()

    # The model is already in FP16 format (we simulated FP8 by quantizing)
    # In real scenario, we'd load FP8 weights and cast to FP16
    model_fp16_restored = model_fp8  # Already in FP16 format

    print("✓ Conversion to FP16 complete")
    print()

    # Calculate weight statistics after restoration
    print("Restored FP16 Model Weight Statistics:")
    stats_restored = calculate_model_stats(model_fp16_restored)
    for key, value in stats_restored.items():
        print(f"  {key}: {value:.6f}")
    print()

    # Evaluate restored model
    print("Evaluating restored FP16 model...")
    acc_restored, loss_restored = evaluate_model(
        model_fp16_restored, test_loader, device,
        desc='Restored FP16 Evaluation'
    )
    print(f"\n✓ Restored FP16 Model Results:")
    print(f"  Accuracy: {acc_restored:.2f}%")
    print(f"  Loss: {loss_restored:.4f}")
    print()

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    print("=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print()

    print(f"{'Model':<20} | {'Accuracy':<12} | {'Loss':<10} | {'Δ Accuracy':<12} | {'Δ Loss':<10}")
    print("-" * 80)
    print(f"{'Original FP16':<20} | {acc_fp16:>10.2f}% | {loss_fp16:>8.4f} | {'-':>10} | {'-':>8}")
    print(
        f"{'FP8 (simulated)':<20} | {acc_fp8:>10.2f}% | {loss_fp8:>8.4f} | {acc_fp8 - acc_fp16:>+9.2f}% | {loss_fp8 - loss_fp16:>+8.4f}")
    print(
        f"{'Restored FP16':<20} | {acc_restored:>10.2f}% | {loss_restored:>8.4f} | {acc_restored - acc_fp16:>+9.2f}% | {loss_restored - loss_fp16:>+8.4f}")
    print()

    # Accuracy degradation
    degradation = acc_fp16 - acc_restored
    degradation_pct = (degradation / acc_fp16) * 100

    print("📊 Key Findings:")
    print(f"  Accuracy loss (FP16 → FP8 → FP16): {degradation:.2f} percentage points")
    print(f"  Relative degradation: {degradation_pct:.2f}%")
    print(f"  Loss increase: {loss_restored - loss_fp16:+.4f}")
    print()

    # Interpretation
    print("💡 Interpretation:")
    if degradation < 0.5:
        print("  ✅ EXCELLENT: <0.5% accuracy loss - FP8 is highly viable!")
    elif degradation < 1.0:
        print("  ✅ GOOD: <1% accuracy loss - FP8 is viable for most applications")
    elif degradation < 2.0:
        print("  ⚠️  MODERATE: 1-2% accuracy loss - FP8 viable for non-critical applications")
    else:
        print("  ❌ SIGNIFICANT: >2% accuracy loss - FP8 may not be suitable")
    print()

    # Save results
    results = {
        'original_fp16': {
            'accuracy': float(acc_fp16),
            'loss': float(loss_fp16),
            'weight_stats': stats_fp16
        },
        'fp8_simulated': {
            'accuracy': float(acc_fp8),
            'loss': float(loss_fp8),
            'weight_stats': stats_fp8
        },
        'restored_fp16': {
            'accuracy': float(acc_restored),
            'loss': float(loss_restored),
            'weight_stats': stats_restored
        },
        'degradation': {
            'accuracy_loss_pp': float(degradation),
            'accuracy_loss_relative_pct': float(degradation_pct),
            'loss_increase': float(loss_restored - loss_fp16)
        }
    }

    output_dir = Path('../results/checkpoints/exp4_fp8_test')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'fp8_test_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to: {output_file}")
    print()

    print("=" * 80)
    print("FP8 QUANTIZATION TEST COMPLETE!")
    print("=" * 80)


if __name__ == '__main__':
    main()