"""
CIFAR-10 Dataset Loader
Single loader with configurable augmentation level
Paths are always relative to project root
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


# Find project root (from src/data/dataset.py)
def get_project_root():
    """Get project root directory"""
    current = Path(__file__).resolve().parent
    # From src/data/dataset.py, go up 2 levels to root
    return current.parent.parent


# Default paths relative to project root
PROJECT_ROOT = get_project_root()
DEFAULT_DATA_DIR = PROJECT_ROOT / 'data'


def get_cifar10_loaders(
    batch_size=128,
    num_workers=2,
    augmentation='basic',
    data_dir=None,  # ← Changed from './data' to None
    pin_memory=False
):
    """
    Create CIFAR-10 data loaders with configurable augmentation

    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of worker processes for data loading
        augmentation: 'basic' or 'extended'
            - basic: RandomCrop + RandomHorizontalFlip
            - extended: basic + ColorJitter + RandomRotation + RandomErasing
        data_dir: Directory to store/load dataset (default: PROJECT_ROOT/data)
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        train_loader, test_loader
    """

    # Use default data dir if not specified
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR

    # Ensure data_dir exists
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Test transforms (no augmentation)
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 stats
            std=[0.2470, 0.2435, 0.2616]
        ),
    ])

    # Train transforms - configurable augmentation
    if augmentation == 'extended':
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(p=0.5),

            # Extended augmentation
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomRotation(15),

            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),

            # RandomErasing (simulated occlusion)
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value='random'
            ),
        ])
    else:  # basic
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=28),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616]
            ),
        ])

    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=str(data_dir),  # Convert Path to str
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform_test
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, test_loader


def get_dataset_info():
    """Return CIFAR-10 dataset information"""
    return {
        'name': 'CIFAR-10',
        'num_classes': 10,
        'train_size': 50000,
        'test_size': 10000,
        'image_size': (32, 32),
        'num_channels': 3,
        'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    }


if __name__ == '__main__':
    # Test the loader
    print("Testing CIFAR-10 loaders...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DEFAULT_DATA_DIR}")

    # Test basic augmentation
    print("\n1. Basic augmentation:")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=32,
        augmentation='basic'
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Test extended augmentation
    print("\n2. Extended augmentation:")
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=32,
        augmentation='extended'
    )
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Test batch
    images, labels = next(iter(train_loader))
    print(f"\n3. Batch shape: {images.shape}")
    print(f"   Labels shape: {labels.shape}")

    # Dataset info
    print("\n4. Dataset info:")
    info = get_dataset_info()
    for key, value in info.items():
        print(f"   {key}: {value}")