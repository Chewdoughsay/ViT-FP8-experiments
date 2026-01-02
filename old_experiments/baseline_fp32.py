"""
Experiment 1: Baseline ViT-Tiny în FP32 pe CIFAR-10
"""

import os, certifi

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = os.environ["SSL_CERT_FILE"]

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from src.models.vit_model import create_vit_model, get_model_info
from src.data.dataset import get_cifar10_loaders
from src.training.trainer import ViTTrainer


def main():
    # Configurație experiment
    config = {
        'model_name': 'vit_tiny_patch16_224',
        'num_classes': 10,
        'batch_size': 128,  # Ajustează după RAM disponibil
        'learning_rate': 1e-3,
        'weight_decay': 0.05,
        'num_epochs': 50,  # Pentru început, apoi poți crește
        'num_workers': 2,  # Pentru MacBook
    }
    
    print("="*70)
    print("EXPERIMENT 1: ViT-Tiny FP32 pe CIFAR-10")
    print("="*70)
    print(f"\nConfigurations:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Check device (MPS pentru Apple Silicon)
    if torch.backends.mps.is_available():
        device = 'mps'
        print(f"✓ Using Apple Silicon GPU (MPS)")
    else:
        device = 'cpu'
        print(f"⚠ MPS not available, using CPU")
    print()
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    print(f"✓ Data loaded: {len(train_loader)} train batches, {len(test_loader)} test batches\n")
    
    # Create model
    print("Creating ViT-Tiny model...")
    model = create_vit_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        pretrained=False
    )
    
    # Print model info
    info = get_model_info(model)
    print(f"✓ Model created:")
    print(f"  Total parameters: {info['trainable_params_millions']:.2f}M")
    print()
    
    # Create trainer
    trainer = ViTTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        save_dir='../experiments/results/checkpoints/baseline_fp32'
    )
    
    # Start training
    trainer.train(
        num_epochs=config['num_epochs'],
        save_every=10
    )
    
    print("\n" + "="*70)
    print("Experiment completed! Check results/ directory for outputs.")
    print("="*70)


if __name__ == '__main__':
    main()