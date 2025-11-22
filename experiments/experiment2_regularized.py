"""
Experiment 2: ViT-Tiny FP32 cu regularizare îmbunătățită
- 50 epoci (full training)
- Data augmentation extins
- Label smoothing
- Gradient clipping
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from src.models.vit_model import create_vit_model, get_model_info
from src.data.dataset import get_cifar10_loaders_v2  # Vom crea versiunea îmbunătățită
from src.training.trainer import ViTTrainer


def main():
    # Configurație experiment
    config = {
        'model_name': 'vit_tiny_patch16_224',
        'num_classes': 10,
        'batch_size': 64,
        'learning_rate': 1e-3,
        'weight_decay': 0.1,  # Crescut de la 0.05
        'num_epochs': 50,  # Full training
        'num_workers': 2,
        'warmup_epochs': 5,  # Learning rate warmup
        'label_smoothing': 0.1,  # Nou
        'gradient_clip': 1.0,  # Nou
    }

    print("=" * 70)
    print("EXPERIMENT 2: ViT-Tiny FP32 cu Regularizare Îmbunătățită")
    print("=" * 70)
    print(f"\nConfigurations:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    print("🔥 ÎMBUNĂTĂȚIRI față de Experiment #1:")
    print("  ✓ 50 epoci (vs 5)")
    print("  ✓ Weight decay crescut: 0.1 (vs 0.05)")
    print("  ✓ Label smoothing: 0.1")
    print("  ✓ Gradient clipping: 1.0")
    print("  ✓ Data augmentation extins (Mixup, RandomErasing, ColorJitter)")
    print("  ✓ Learning rate warmup: 5 epoci")
    print()

    # Check device
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"✓ Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        print(f"⚠ MPS not available, using CPU")
    print()

    # Load data cu augmentation îmbunătățit
    print("Loading CIFAR-10 dataset with enhanced augmentation...")
    train_loader, test_loader = get_cifar10_loaders_v2(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    print(f"✓ Data loaded: {len(train_loader)} train batches, {len(test_loader)} test batches\n")

    # Create model
    print("Creating ViT-Tiny model...")
    model = create_vit_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        pretrained=False  # From scratch, cum ai cerut
    )

    # Print model info
    info = get_model_info(model)
    print(f"✓ Model created:")
    print(f"  Total parameters: {info['trainable_params_millions']:.2f}M")
    print()

    # Create trainer cu îmbunătățiri
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        save_dir='results/checkpoints/experiment2_regularized',
        label_smoothing=config['label_smoothing'],  # Nou
        gradient_clip=config['gradient_clip'],  # Nou
        warmup_epochs=config['warmup_epochs']  # Nou
    )

    # Start training
    print(f"\n🚀 Starting training for {config['num_epochs']} epochs...")
    print(f"⏱️  Estimated time: ~{config['num_epochs'] * 9:.0f} minutes ({config['num_epochs'] * 9 / 60:.1f} hours)")
    print()

    trainer.train(
        num_epochs=config['num_epochs'],
        save_every=10
    )

    print("\n" + "=" * 70)
    print("Experiment 2 completed! Check results/ directory for outputs.")
    print("=" * 70)


if __name__ == '__main__':
    main()
