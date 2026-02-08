"""
General Training Script for Vision Transformer Experiments.

This script provides a flexible training pipeline that reads configuration from YAML files.
Supports all experiment types: FP32, FP16, different augmentation strategies, etc.

Usage:
    $ python scripts/train.py --config configs/BaseFP32.yaml
    $ python scripts/train.py --config configs/AugmFP16.yaml --device cuda

Features:
    - Automatic hardware monitoring (CPU, memory, thermal, GPU on Apple Silicon)
    - Mixed precision training (AMP) for FP16/FP32
    - Wall-clock timing tracking with detailed reports
    - Comprehensive metrics and checkpointing
    - GPU monitoring (automatically requests sudo on Apple Silicon)

Output:
    - Checkpoints: results/{ExperimentName}/checkpoints/
    - Metrics: results/{ExperimentName}/metrics/final_metrics.json
    - Timing: results/{ExperimentName}/metrics/timing_report.json
    - Hardware: results/{ExperimentName}/metrics/hardware_stats.json
    - GPU stats: results/{ExperimentName}/metrics/gpu_stats.csv (if sudo granted)
"""

import sys
import json
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models.vit_model import create_vit_model, get_model_info
from src.data.dataset import get_cifar10_loaders
from src.training.trainer import ViTTrainer


def load_config(config_path):
    """
    Load experiment configuration from YAML file.

    Args:
        config_path (str): Path to YAML config file

    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_timing_report(timing_data, output_path):
    """
    Save detailed timing report to JSON file.

    Args:
        timing_data (dict): Timing statistics
        output_path (Path): Where to save the report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(timing_data, f, indent=2)

    print(f"✓ Timing report saved to: {output_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Train Vision Transformer on CIFAR-10')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file (e.g., configs/BaseFP32.yaml)')
    parser.add_argument('--device', type=str, default=None,
                        help='Override device from config (mps, cuda, cpu)')
    args = parser.parse_args()

    # Load configuration
    print(f"\n{'='*70}")
    print(f"Loading configuration from: {args.config}")
    print(f"{'='*70}\n")

    config = load_config(args.config)

    # Override device if specified
    if args.device:
        config['hardware']['device'] = args.device

    # Print configuration
    print(f"Experiment: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"\nConfiguration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Precision: {'FP16 (AMP)' if config['training']['use_amp'] else 'FP32'}")
    print(f"  Augmentation: {config['data']['augmentation']}")
    print(f"  Batch Size: {config['data']['batch_size']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")
    print(f"  Weight Decay: {config['training']['weight_decay']}")
    print(f"  Label Smoothing: {config['training']['label_smoothing']}")
    print(f"  Gradient Clipping: {config['training']['gradient_clip']}")
    print(f"  Warmup Epochs: {config['training']['warmup_epochs']}")
    print(f"  Device: {config['hardware']['device']}")
    print()

    # Create model
    print("📦 Creating model...")
    model = create_vit_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    )
    model_info = get_model_info(model)
    print(f"✓ Model created: {model_info['trainable_params_millions']:.2f}M parameters\n")

    # Create data loaders
    print("📊 Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        augmentation=config['data']['augmentation'],
        data_dir=config['paths']['data_dir']
    )
    print(f"✓ Data loaders ready: {len(train_loader)} train batches, {len(test_loader)} test batches\n")

    # Initialize trainer
    print("🚀 Initializing trainer...")
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=config['hardware']['device'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        save_dir=config['paths']['save_dir'],
        label_smoothing=config['training']['label_smoothing'],
        gradient_clip=config['training']['gradient_clip'],
        warmup_epochs=config['training']['warmup_epochs'],
        use_amp=config['training']['use_amp']
    )
    print("✓ Trainer initialized\n")

    # Record start time
    start_time = time.time()
    start_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"⏱️  Training started at: {start_timestamp}\n")

    # Train
    try:
        trainer.train(
            num_epochs=config['training']['num_epochs'],
            save_every=config['training']['save_every']
        )

        # Record end time
        end_time = time.time()
        end_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_duration = end_time - start_time

        # Calculate timing statistics
        timing_data = {
            'experiment_name': config['name'],
            'start_time': start_timestamp,
            'end_time': end_timestamp,
            'total_duration_seconds': total_duration,
            'total_duration_minutes': total_duration / 60,
            'total_duration_hours': total_duration / 3600,
            'num_epochs': config['training']['num_epochs'],
            'avg_epoch_time_seconds': total_duration / config['training']['num_epochs'],
            'best_val_accuracy': trainer.metrics.get_best_acc(),
            'precision': config['training']['precision'],
            'augmentation': config['data']['augmentation'],
            'config_file': args.config
        }

        # Save timing report
        timing_report_path = Path(config['paths']['save_dir']).parent / 'metrics' / 'timing_report.json'
        save_timing_report(timing_data, timing_report_path)

        # Print summary
        print("\n" + "="*70)
        print("🎉 Training Completed Successfully!")
        print("="*70)
        print(f"\nTiming Summary:")
        print(f"  Started: {start_timestamp}")
        print(f"  Ended: {end_timestamp}")
        print(f"  Total Duration: {timing_data['total_duration_minutes']:.1f} minutes ({timing_data['total_duration_hours']:.2f} hours)")
        print(f"  Average Time per Epoch: {timing_data['avg_epoch_time_seconds']:.1f} seconds")
        print(f"\nResults:")
        print(f"  Best Validation Accuracy: {timing_data['best_val_accuracy']:.4f}")
        print(f"  Checkpoints: {trainer.checkpoint_dir}")
        print(f"  Metrics: {trainer.metrics_dir}")
        print(f"  Timing Report: {timing_report_path}")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"\n\n⚠️  Training interrupted after {total_duration/60:.1f} minutes")
        print(f"Partial results saved to: {trainer.metrics_dir}")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
