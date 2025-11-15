"""
Vision Transformer models folosind timm library
"""
import timm
import torch
import torch.nn as nn


def create_vit_model(model_name='vit_tiny_patch16_224', num_classes=10, pretrained=False):
    """
    Creează un model ViT
    
    Args:
        model_name: Numele modelului din timm
            Opțiuni: 'vit_tiny_patch16_224', 'vit_small_patch16_224', 
                     'vit_base_patch16_224', etc.
        num_classes: Număr de clase (10 pentru CIFAR-10)
        pretrained: Folosește greutăți pre-antrenate pe ImageNet
        
    Returns:
        model: PyTorch model
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    
    return model


def count_parameters(model):
    """Numără parametrii trainable ai modelului"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_model_info(model):
    """Returnează informații despre model"""
    total_params, trainable_params = count_parameters(model)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'total_params_millions': total_params / 1e6,
        'trainable_params_millions': trainable_params / 1e6,
    }
    
    return info


# Configurații predefinite
MODEL_CONFIGS = {
    'vit_tiny': {
        'name': 'vit_tiny_patch16_224',
        'params_approx': '5.7M',
        'description': 'ViT-Tiny - cel mai mic model, bun pentru testare rapidă'
    },
    'vit_small': {
        'name': 'vit_small_patch16_224',
        'params_approx': '22M',
        'description': 'ViT-Small - model mediu, bun compromis'
    },
    'vit_base': {
        'name': 'vit_base_patch16_224',
        'params_approx': '86M',
        'description': 'ViT-Base - model standard, mai mare'
    },
}


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