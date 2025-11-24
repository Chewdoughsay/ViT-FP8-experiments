"""
Experiment 3: ViT-Tiny în Mixed Precision (FP16) pe Apple Silicon (M4)
Scop:
1. Reducerea timpului de antrenare (prin batch size mai mare).
2. Verificarea stabilității numerice (dacă pierdem acuratețe față de FP32).
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.vit_model import create_vit_model, get_model_info
from src.data.dataset import get_cifar10_loaders
from src.training.trainer import ViTTrainer


def main():
    # Configurație (Identică cu Exp 2, dar cu AMP și Batch Size dublu)
    config = {
        'model_name': 'vit_tiny_patch16_224',
        'num_classes': 10,
        'batch_size': 128,  # <--- DUBLAT (era 64) datorită FP16
        'learning_rate': 1e-3,
        'weight_decay': 0.1,  # Păstrat de la Exp 2
        'num_epochs': 50,
        'warmup_epochs': 5,
        'label_smoothing': 0.1,
        'gradient_clip': 1.0,
        'use_amp': True,  # <--- ACTIVAT Mixed Precision
        'num_workers': 4  # Putem crește puțin și workerii pe M4
    }

    print("=" * 70)
    print(f"EXPERIMENT 3: ViT-Tiny FP16 (Mixed Precision) pe {torch.backends.mps.is_available() and 'MPS' or 'CPU'}")
    print("=" * 70)

    # Verificări sistem
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'mps':
        print("✓ Apple Silicon acceleration active")
        print("✓ Mixed Precision mode: ENABLED")

    # Load data
    print(f"\nLoading CIFAR-10 dataset (Batch Size: {config['batch_size']})...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )

    # Create model
    print("Creating ViT-Tiny model...")
    model = create_vit_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        pretrained=False
    )

    # Trainer
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        save_dir='results/checkpoints/experiment3_fp16',  # Folder nou
        label_smoothing=config['label_smoothing'],
        gradient_clip=config['gradient_clip'],
        warmup_epochs=config['warmup_epochs'],
        use_amp=config['use_amp']  # Trimitem flag-ul
    )

    # Start training
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == '__main__':
    main()