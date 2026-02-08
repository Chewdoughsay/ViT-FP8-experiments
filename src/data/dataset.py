"""
CIFAR-10 dataset loader with configurable augmentation.

This module provides a unified interface for loading CIFAR-10 data with
different augmentation strategies. Features:
- Automatic path resolution relative to project root
- Configurable augmentation levels (basic vs extended)
- Normalized using CIFAR-10 statistics
- Resized to 224×224 for Vision Transformers
- Persistent workers for efficient data loading

The loader automatically handles:
- Dataset download (if not already present)
- Train/test split (50,000 train / 10,000 test)
- Image preprocessing and normalization
- Batch loading with multi-worker support

Example:
    >>> from src.data.dataset import get_cifar10_loaders
    >>>
    >>> # Basic augmentation (default)
    >>> train_loader, test_loader = get_cifar10_loaders(
    ...     batch_size=128,
    ...     augmentation='basic'
    ... )
    >>>
    >>> # Extended augmentation for better generalization
    >>> train_loader, test_loader = get_cifar10_loaders(
    ...     batch_size=128,
    ...     augmentation='extended'
    ... )
    >>>
    >>> # Iterate through batches
    >>> for images, labels in train_loader:
    ...     # images: [128, 3, 224, 224]
    ...     # labels: [128]
    ...     outputs = model(images)
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path


# Find project root (from src/data/dataset.py)
def get_project_root():
    """
    Get the project root directory.

    Returns:
        Path: Absolute path to project root (parent of src/ directory)

    Example:
        >>> root = get_project_root()
        >>> print(root)
        /Users/username/PycharmProjects/ViT-FP8-experiments
    """
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
    Create CIFAR-10 train and test data loaders with configurable augmentation.

    Loads CIFAR-10 dataset with preprocessing for Vision Transformers:
    - Images resized from 32×32 to 224×224 (ViT standard input size)
    - Normalized using CIFAR-10 mean and std
    - Training set augmented (validation set is not augmented)
    - Automatic download if dataset not present

    Args:
        batch_size (int): Number of samples per batch. Default: 128
        num_workers (int): Number of subprocesses for data loading. Default: 2
            - 0 = single-process loading (slower but simpler)
            - 2-4 = good for most systems
            - Higher values may not improve speed (depends on CPU/disk)
        augmentation (str): Data augmentation level. Default: 'basic'
            - 'basic': RandomCrop with padding + RandomHorizontalFlip
            - 'extended': basic + ColorJitter + RandomRotation + RandomErasing
        data_dir (str, optional): Directory to store/load CIFAR-10 data.
            Default: None (uses PROJECT_ROOT/data)
        pin_memory (bool): Pin memory in RAM for faster GPU transfer. Default: False
            - Set to True when using CUDA (speeds up data transfer to GPU)
            - Keep False for CPU or MPS

    Returns:
        tuple: (train_loader, test_loader)
            - train_loader (DataLoader): Training data with augmentation
            - test_loader (DataLoader): Test data without augmentation

    Augmentation Details:
        Basic augmentation (standard for CIFAR-10):
            - Resize to 224×224
            - RandomCrop 224×224 with padding=28 (adds some variation)
            - RandomHorizontalFlip with p=0.5
            - Normalization

        Extended augmentation (stronger regularization):
            - All basic augmentations, plus:
            - ColorJitter: brightness, contrast, saturation, hue variations
            - RandomRotation: ±15 degrees
            - RandomErasing: randomly occludes patches (p=0.5, simulates missing data)

    Data Statistics:
        - Training samples: 50,000
        - Test samples: 10,000
        - Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
        - Original size: 32×32×3
        - Resized to: 224×224×3 (for Vision Transformers)
        - Normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]

    Example:
        >>> # Basic usage
        >>> train_loader, test_loader = get_cifar10_loaders(batch_size=128)
        >>> print(f"Train batches: {len(train_loader)}")
        Train batches: 391
        >>> print(f"Test batches: {len(test_loader)}")
        Test batches: 79
        >>>
        >>> # Extended augmentation for better generalization
        >>> train_loader, test_loader = get_cifar10_loaders(
        ...     batch_size=64,
        ...     num_workers=4,
        ...     augmentation='extended',
        ...     pin_memory=True  # For CUDA
        ... )
        >>>
        >>> # Check batch shape
        >>> images, labels = next(iter(train_loader))
        >>> print(f"Image batch: {images.shape}")  # [128, 3, 224, 224]
        >>> print(f"Label batch: {labels.shape}")  # [128]

    Notes:
        - First run will download CIFAR-10 (~170MB) to data_dir
        - Persistent workers (num_workers > 0) speeds up data loading
        - Extended augmentation improves generalization but slows training slightly
        - Test set is never augmented (only resized and normalized)
        - Images are returned as tensors in range [~-2, ~2] after normalization
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
    """
    Return metadata about the CIFAR-10 dataset.

    Returns:
        dict: Dataset information with keys:
            - name (str): Dataset name
            - num_classes (int): Number of classes
            - train_size (int): Number of training samples
            - test_size (int): Number of test samples
            - image_size (tuple): Original image dimensions (height, width)
            - num_channels (int): Number of color channels
            - classes (list): Class names in order

    Example:
        >>> info = get_dataset_info()
        >>> print(f"Dataset: {info['name']}")
        Dataset: CIFAR-10
        >>> print(f"Classes: {info['classes']}")
        Classes: ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
        >>> print(f"Training samples: {info['train_size']}")
        Training samples: 50000
    """
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