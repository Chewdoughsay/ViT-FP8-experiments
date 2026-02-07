#!/usr/bin/env python3
"""
Generic training script for ViT experiments
Reads configuration from YAML files with proper path handling

Usage:
    python scripts/train.py --config configs/exp3_fp16_fixed.yaml
"""
import argparse
import yaml
import sys
from pathlib import Path

# Find project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.vit_model import create_vit_model, get_model_info
from src.data.dataset import get_cifar10_loaders
from src.training.trainer import ViTTrainer


def load_config(config_path):
    """Load YAML configuration file and resolve paths"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve all paths relative to project root
    if 'paths' in config:
        for key in ['data_dir', 'save_dir']:
            if key in config['paths']:
                path_value = config['paths'][key]

                # If relative path, make it relative to project root
                if not Path(path_value).is_absolute():
                    config['paths'][key] = str(project_root / path_value)

    return config


def print_config(config):
    """Pretty print configuration"""
    print("=" * 70)
    print(f"EXPERIMENT: {config['name']}")
    print("=" * 70)
    print(f"Description: {config['description']}")
    print()

    print("Configuration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Dataset: {config['data']['dataset']}")
    print(f"  Batch size: {config['data']['batch_size']}")
    print(f"  Augmentation: {config['data']['augmentation']}")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Weight decay: {config['training']['weight_decay']}")

    if config['training'].get('label_smoothing'):
        print(f"  Label smoothing: {config['training']['label_smoothing']}")
    if config['training'].get('gradient_clip'):
        print(f"  Gradient clip: {config['training']['gradient_clip']}")
    if config['training'].get('warmup_epochs'):
        print(f"  Warmup epochs: {config['training']['warmup_epochs']}")
    if config['training'].get('use_amp'):
        print(f"  Mixed precision: {config['training']['precision']}")

    print()
    print("Paths (resolved):")
    print(f"  Data dir: {config['paths']['data_dir']}")
    print(f"  Save dir: {config['paths']['save_dir']}")
    print()


def setup_device(config):
    """Setup compute device"""
    device = config['hardware']['device']

    if device == 'mps':
        if torch.backends.mps.is_available():
            print(f"✓ Using Apple Silicon GPU (MPS)")
            return 'mps'
        else:
            print(f"⚠ MPS not available, falling back to CPU")
            return 'cpu'
    elif device == 'cuda':
        if torch.cuda.is_available():
            print(f"✓ Using NVIDIA GPU (CUDA)")
            return 'cuda'
        else:
            print(f"⚠ CUDA not available, falling back to CPU")
            return 'cpu'
    else:
        print(f"Using CPU")
        return 'cpu'


def ensure_directories(config):
    """Ensure all necessary directories exist"""
    # Data directory
    Path(config['paths']['data_dir']).mkdir(parents=True, exist_ok=True)

    # Save directory and subdirectories
    save_dir = Path(config['paths']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create results subdirectories
    results_root = project_root / 'results'
    (results_root / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (results_root / 'logs').mkdir(parents=True, exist_ok=True)
    (results_root / 'plots').mkdir(parents=True, exist_ok=True)
    (results_root / 'metrics').mkdir(parents=True, exist_ok=True)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train ViT model with YAML configuration'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file (e.g., configs/exp3_fp16_fixed.yaml)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    print_config(config)

    # Setup device
    device = setup_device(config)

    # Ensure directories exist
    ensure_directories(config)

    # Create data loaders
    print("Loading dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        augmentation=config['data']['augmentation'],
        data_dir=config['paths']['data_dir'],
        pin_memory=(device != 'cpu')
    )
    print(f"✓ Data loaded: {len(train_loader)} train batches, {len(test_loader)} test batches")
    print()

    # Create model
    print("Creating model...")
    model = create_vit_model(
        model_name=config['model']['name'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    )

    # Print model info
    info = get_model_info(model)
    print(f"✓ Model created: {config['model']['name']}")
    print(f"  Parameters: {info['trainable_params_millions']:.2f}M")
    print()

    # Create trainer
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        save_dir=config['paths']['save_dir'],
        label_smoothing=config['training'].get('label_smoothing', 0.0),
        gradient_clip=config['training'].get('gradient_clip', None),
        warmup_epochs=config['training'].get('warmup_epochs', 0),
        use_amp=config['training'].get('use_amp', False)
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # TODO: Add resume functionality
        pass

    # Start training
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print()

    trainer.train(
        num_epochs=config['training']['num_epochs'],
        save_every=config['training'].get('save_every', 10)
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {config['paths']['save_dir']}")


if __name__ == '__main__':
    main()

#python scripts/train.py --config configs/exp3_fp16_fixed.yaml