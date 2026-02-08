"""
Vision Transformer model creation using the timm library.

This module provides utilities for creating and analyzing Vision Transformer (ViT)
models using the timm (PyTorch Image Models) library. Includes:
- Model creation with configurable architecture and classes
- Parameter counting utilities
- Predefined model configurations (Tiny, Small, Base)

The Vision Transformer architecture:
- Splits images into patches (e.g., 16×16)
- Projects patches into embeddings
- Processes with transformer encoder blocks
- Classification head predicts class probabilities

Example:
    >>> from src.models.vit_model import create_vit_model, get_model_info
    >>>
    >>> # Create ViT-Tiny for CIFAR-10
    >>> model = create_vit_model('vit_tiny_patch16_224', num_classes=10)
    >>>
    >>> # Get model statistics
    >>> info = get_model_info(model)
    >>> print(f"Parameters: {info['trainable_params_millions']:.2f}M")
    >>>
    >>> # Test forward pass
    >>> x = torch.randn(2, 3, 224, 224)
    >>> y = model(x)  # Output: [2, 10]
"""
import timm
import torch
import torch.nn as nn


def create_vit_model(model_name='vit_tiny_patch16_224', num_classes=10, pretrained=False):
    """
    Create a Vision Transformer model from timm library.

    Instantiates a ViT model with the specified architecture and number of
    output classes. Optionally loads pretrained ImageNet weights.

    Args:
        model_name (str): Model architecture name from timm. Default: 'vit_tiny_patch16_224'
            Common options:
                - 'vit_tiny_patch16_224': ~5.7M params (fastest)
                - 'vit_small_patch16_224': ~22M params (balanced)
                - 'vit_base_patch16_224': ~86M params (most accurate)
            See MODEL_CONFIGS for details or timm docs for full list.
        num_classes (int): Number of output classes. Default: 10 (for CIFAR-10)
            - 10 for CIFAR-10
            - 1000 for ImageNet
            - Custom value for other datasets
        pretrained (bool): Load pretrained ImageNet weights. Default: False
            - True: Use ImageNet-1K weights (requires fine-tuning for CIFAR-10)
            - False: Random initialization (train from scratch)

    Returns:
        torch.nn.Module: Vision Transformer model ready for training or inference

    Example:
        >>> # Create ViT-Tiny from scratch for CIFAR-10
        >>> model = create_vit_model('vit_tiny_patch16_224', num_classes=10)
        >>>
        >>> # Create ViT-Base with ImageNet pretrained weights
        >>> model = create_vit_model(
        ...     'vit_base_patch16_224',
        ...     num_classes=10,
        ...     pretrained=True
        ... )
        >>>
        >>> # Test forward pass
        >>> x = torch.randn(4, 3, 224, 224)
        >>> outputs = model(x)
        >>> print(outputs.shape)  # [4, 10]

    Notes:
        - Input images must be 224×224×3 (or model-specific size)
        - Pretrained weights are from ImageNet-1K (1000 classes)
        - Classification head is automatically adjusted for num_classes
        - Models expect normalized images (use appropriate transforms)
        - See timm documentation for full list of available models
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    
    return model


def count_parameters(model):
    """
    Count total and trainable parameters in a model.

    Iterates through all model parameters and counts the total number of
    elements (parameters) as well as those that require gradients (trainable).

    Args:
        model (torch.nn.Module): PyTorch model to analyze

    Returns:
        tuple: (total_params, trainable_params)
            - total_params (int): Total number of parameters in the model
            - trainable_params (int): Number of parameters with requires_grad=True

    Example:
        >>> model = create_vit_model('vit_tiny_patch16_224', num_classes=10)
        >>> total, trainable = count_parameters(model)
        >>> print(f"Total: {total:,} | Trainable: {trainable:,}")
        Total: 5,717,416 | Trainable: 5,717,416
        >>>
        >>> # Freeze some layers
        >>> for param in model.parameters():
        ...     param.requires_grad = False
        >>> total, trainable = count_parameters(model)
        >>> print(f"After freezing - Trainable: {trainable}")
        After freezing - Trainable: 0

    Notes:
        - Counts all parameters (weights, biases, embeddings, etc.)
        - Trainable count reflects current requires_grad state
        - For frozen models, trainable < total
        - Parameter count determines memory usage and compute requirements
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_info(model):
    """
    Get comprehensive information about a model.

    Analyzes the model and returns a dictionary with parameter counts
    in both absolute values and millions for readability.

    Args:
        model (torch.nn.Module): PyTorch model to analyze

    Returns:
        dict: Model information with keys:
            - total_parameters (int): Total parameter count
            - trainable_parameters (int): Trainable parameter count
            - total_params_millions (float): Total parameters in millions
            - trainable_params_millions (float): Trainable parameters in millions

    Example:
        >>> model = create_vit_model('vit_small_patch16_224', num_classes=10)
        >>> info = get_model_info(model)
        >>> print(f"Model size: {info['trainable_params_millions']:.2f}M parameters")
        Model size: 21.67M parameters
        >>>
        >>> # Compare model sizes
        >>> models = ['vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224']
        >>> for name in models:
        ...     model = create_vit_model(name, num_classes=10)
        ...     info = get_model_info(model)
        ...     print(f"{name}: {info['trainable_params_millions']:.1f}M params")
        vit_tiny_patch16_224: 5.7M params
        vit_small_patch16_224: 21.7M params
        vit_base_patch16_224: 86.6M params

    Notes:
        - Useful for comparing model sizes before training
        - Helps estimate memory requirements (larger models need more VRAM)
        - Parameter count correlates with computational cost
        - Millions format (M) is standard in deep learning papers
    """
    total_params, trainable_params = count_parameters(model)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_params_millions': total_params / 1e6,
        'trainable_params_millions': trainable_params / 1e6,
    }
    
    return info


# Predefined model configurations for easy reference
MODEL_CONFIGS = {
    'vit_tiny': {
        'name': 'vit_tiny_patch16_224',
        'params_approx': '5.7M',
        'description': 'ViT-Tiny - smallest model, good for fast testing'
    },
    'vit_small': {
        'name': 'vit_small_patch16_224',
        'params_approx': '22M',
        'description': 'ViT-Small - medium model, good compromise'
    },
    'vit_base': {
        'name': 'vit_base_patch16_224',
        'params_approx': '86M',
        'description': 'ViT-Base - standard model, larger size'
    },
}
"""
Predefined Vision Transformer Configurations:

vit_tiny (5.7M params):
    - Fastest training and inference
    - Good for prototyping and small datasets
    - May underfit on complex tasks

vit_small (22M params):
    - Balanced speed and accuracy
    - Recommended for CIFAR-10 experiments
    - Good compromise for resource-constrained environments

vit_base (86M params):
    - Highest accuracy potential
    - Requires more compute and memory
    - Best for final production models or large datasets

Usage:
    >>> config = MODEL_CONFIGS['vit_tiny']
    >>> model = create_vit_model(config['name'], num_classes=10)
"""


if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Test models
    print("=== Testing ViT Models ===\n")
    
    for key, config in MODEL_CONFIGS.items():
        print(f"{key.upper()}:")
        model = create_vit_model(config['name'], num_classes=10, pretrained=False)
        info = get_model_info(model)
        print(f"  Parameters: {info['trainable_params_millions']:.2f}M")
        print(f"  Description: {config['description']}")
        
        # Test forward pass
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        print(f"  Output shape: {output.shape}")  # Should be [2, 10]
        print()