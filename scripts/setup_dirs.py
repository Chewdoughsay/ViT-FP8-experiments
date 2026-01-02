#!/usr/bin/env python3
"""
Setup Project Directories
Creates all necessary directories for the ViT experiments project
"""
from pathlib import Path


def setup_directories():
    """Create all necessary project directories"""
    # Find project root
    project_root = Path(__file__).resolve().parent.parent
    
    print(f"Setting up directories in: {project_root}")
    print()
    
    # Define directory structure
    directories = [
        # Data
        'data',
        
        # Results
        'results',
        'results/checkpoints',
        'results/logs',
        'results/plots',
        'results/metrics',
        
        # Source code (should already exist, but just in case)
        'src',
        'src/data',
        'src/models',
        'src/training',
        'src/utils',
        
        # Scripts (should already exist)
        'scripts',
        
        # Configs (should already exist)
        'configs',
        
        # Notebooks (optional)
        'notebooks',
        
        # Old experiments backup
        'old_experiments',
    ]
    
    # Create each directory
    created = []
    already_exists = []
    
    for dir_path in directories:
        full_path = project_root / dir_path
        
        if full_path.exists():
            already_exists.append(dir_path)
        else:
            full_path.mkdir(parents=True, exist_ok=True)
            created.append(dir_path)
    
    # Report
    if created:
        print("✅ Created directories:")
        for dir_path in created:
            print(f"   {dir_path}/")
    
    if already_exists:
        print()
        print("📁 Already exist:")
        for dir_path in already_exists:
            print(f"   {dir_path}/")
    
    print()
    print("🎉 Project directories ready!")
    print()
    print("Next steps:")
    print("  1. Check that configs/ has your YAML files")
    print("  2. Run: python scripts/train.py --config configs/exp3_fp16_fixed.yaml")


if __name__ == '__main__':
    setup_directories()
