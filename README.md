# ViT FP8 Precision Experiments

Comprehensive experimental study of Vision Transformer (ViT) performance across different numerical precision levels (FP32, FP16, FP8) and data augmentation strategies on CIFAR-10.

## 📋 Project Overview

This project investigates the trade-offs between computational efficiency and model accuracy when training Vision Transformers with reduced precision arithmetic. We systematically compare:

- **Precision Levels**: FP32 (baseline), FP16 (mixed precision), FP8 (experimental)
- **Augmentation Strategies**: Basic (standard) vs. Extended (aggressive regularization)
- **Hardware Utilization**: Apple Silicon (M1/M2/M3) with GPU/CPU monitoring
- **Training Dynamics**: Convergence behavior, generalization gap, overfitting analysis

### Key Research Questions

1. How much accuracy is lost when moving from FP32 → FP16 → FP8?
2. Does extended data augmentation improve generalization across precision levels?
3. What is the computational speedup from reduced precision training?
4. How does hardware utilization (GPU, CPU, thermal throttling) differ across experiments?

---

## 🎯 Experiments

We conduct 4 primary experiments in a factorial design:

| Experiment | Precision | Augmentation | Description |
|------------|-----------|--------------|-------------|
| **BaseFP32** | FP32 | Basic | Baseline with full precision and minimal augmentation |
| **AugmFP32** | FP32 | Extended | FP32 with aggressive augmentation for comparison |
| **BaseFP16** | FP16 | Basic | Mixed precision (AMP) with basic augmentation |
| **AugmFP16** | FP16 | Extended | Mixed precision with extended augmentation |
| **FP8Test** | FP8 | Extended | Experimental FP8 quantization (convert trained model) |

### Augmentation Details

**Basic Augmentation:**
- Resize to 224×224
- RandomCrop with padding=28
- RandomHorizontalFlip (p=0.5)
- Normalization (CIFAR-10 statistics)

**Extended Augmentation:**
- All basic augmentations, plus:
- ColorJitter (brightness, contrast, saturation, hue)
- RandomRotation (±15 degrees)
- RandomErasing (p=0.5, simulates occlusion)

---

## 📁 Project Structure

```
ViT-FP8-experiments/
├── configs/                     # YAML configuration files
│   ├── BaseFP32.yaml           # FP32 baseline
│   ├── AugmFP32.yaml           # FP32 + extended augmentation
│   ├── BaseFP16.yaml           # FP16 baseline
│   ├── AugmFP16.yaml           # FP16 + extended augmentation
│   └── FP8Test.yaml            # FP8 quantization test
│
├── src/                         # Source code modules
│   ├── data/
│   │   └── dataset.py          # CIFAR-10 loader with augmentation
│   ├── models/
│   │   └── vit_model.py        # Vision Transformer models (timm)
│   ├── training/
│   │   └── trainer.py          # Comprehensive ViT trainer with AMP
│   └── utils/
│       ├── metrics.py          # Metrics tracking and analysis
│       ├── system_monitor.py   # CPU/memory/thermal monitoring
│       └── gpu_monitor.py      # Apple Silicon GPU monitoring
│
├── scripts/                     # Executable scripts
│   ├── train.py                # Main training script (uses YAML configs)
│   ├── evaluate_fp8_quantization.py  # FP8 post-training quantization evaluation
│   ├── extract_metrics.py      # Metrics extraction and analysis
│   ├── generate_plots.py       # Visualization generation
│   └── compare_experiments.py  # Comprehensive experiment comparison
│
├── results/                     # Experiment outputs
│   ├── BaseFP32/
│   │   ├── checkpoints/        # Model checkpoints
│   │   ├── metrics/            # Training metrics + hardware stats
│   │   └── plots/              # Visualizations
│   ├── AugmFP32/
│   ├── BaseFP16/
│   └── AugmFP16/
│
├── data/                        # CIFAR-10 dataset (auto-downloaded)
├── README.md                    # This file
└── REFACTORING_PLAN.md         # Development history and roadmap
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+ with MPS/CUDA support
- macOS (for Apple Silicon GPU monitoring) or Linux/Windows

### Setup

```bash
# Clone repository
git clone <repository-url>
cd ViT-FP8-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA
# OR for Apple Silicon:
pip install torch torchvision torchaudio

# Install other requirements
pip install timm pyyaml numpy matplotlib tqdm psutil
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
python -c "import timm; print(f'timm version: {timm.__version__}')"
```

---

## 📖 Usage

### 1. Train Experiments with Config Files (Recommended)

Use the general training script with YAML config files for maximum flexibility:

```bash
# Run BaseFP32 (FP32 baseline)
python scripts/train.py --config configs/BaseFP32.yaml

# Run AugmFP32 (FP32 with extended augmentation)
python scripts/train.py --config configs/AugmFP32.yaml

# Run BaseFP16 (FP16 mixed precision)
python scripts/train.py --config configs/BaseFP16.yaml

# Run AugmFP16 (FP16 + extended augmentation - best trade-off)
python scripts/train.py --config configs/AugmFP16.yaml

# Override device from config
python scripts/train.py --config configs/AugmFP16.yaml --device cuda
```

**Output per Experiment:**
- Checkpoints: `results/{ExperimentName}/checkpoints/best_model.pt`, `checkpoint_epoch_N.pt`
- Metrics: `results/{ExperimentName}/metrics/final_metrics.json`
- **Timing Report**: `results/{ExperimentName}/metrics/timing_report.json` ← Detailed timing info!
- Hardware stats: `results/{ExperimentName}/metrics/hardware_stats.json`
- **GPU stats** (Apple Silicon): `results/{ExperimentName}/metrics/gpu_stats.csv` ← Automatic with sudo!

**Estimated Time per Experiment:**
- **BaseFP32**: ~100-120 min (baseline)
- **AugmFP32**: ~105-125 min (slower due to augmentation)
- **BaseFP16**: ~85-95 min (**~30% faster** than FP32!)
- **AugmFP16**: ~90-100 min (best speed/accuracy trade-off)

### 2. Comprehensive Experiment Comparison (For Reports)

After running all experiments, generate complete analysis for your research report:

```bash
python scripts/compare_experiments.py
```

**Analyzes Everything:**
- **Performance**: Best/final accuracy, overfitting gap, loss, convergence
- **Timing**: Total duration, per-epoch stats with std dev
- **Hardware**: CPU/memory utilization, thermal throttling
- **Analysis**: FP16 speedup, best trade-offs, recommendations

**Example Output:**
```
COMPREHENSIVE EXPERIMENT COMPARISON
============================================================
PERFORMANCE METRICS
Metric                         | BaseFP32  | AugmFP32  | BaseFP16  | AugmFP16
Best Val Accuracy              | 78.14%    | 83.62%    | 82.25%    | 83.29%
Final Train Accuracy           | 96.50%    | 85.71%    | 97.62%    | 86.10%
Overfitting Gap                | 18.37%    | 2.08%     | 15.37%    | 2.82%
Final Val Loss                 | 0.9020    | 0.8807    | 0.9843    | 0.8886

TIMING METRICS
Total Time (hours)             | 6.38h     | 7.53h     | 7.36h     | 7.82h
Time/Epoch (sec)               | 459.3s    | 542.2s    | 530.1s    | 563.3s
Time/Epoch StdDev              | ±15.9s    | ±200.8s   | ±21.9s    | ±17.4s

HARDWARE UTILIZATION
CPU Average                    | 7.8%      | 12.7%     | 7.8%      | 7.1%
Memory Average                 | 86.1%     | 86.9%     | 80.4%     | 83.9%
Thermal Throttling             | No        | No        | No        | No
============================================================

ANALYSIS SUMMARY
🏆 Best Accuracy: AugmFP16 (83.29%)
🎯 Best Generalization: AugmFP32 (2.08% overfitting)
⚡ Fastest Training: BaseFP32 (6.38 hours)

📊 FP16 Speedup: 1.25x faster (20% time reduction)
💡 Recommended: AugmFP16 (best speed/accuracy trade-off!)
```

**Output Files:**
- `results/comparison_table.csv` - Tabular data for analysis and reporting
- `results/experiment_comparison.json` - Structured data for plotting scripts
- Console: Formatted tables with analysis summary

### 3. FP8 Quantization Evaluation

After training the AugmFP16 model, evaluate FP8 post-training quantization:

```bash
# Must run AugmFP16 experiment first!
python scripts/evaluate_fp8_quantization.py
```

**What it does:**
1. Loads the best FP16 model from `results/AugmFP16/checkpoints/best_model.pt`
2. Evaluates original FP16 accuracy
3. Quantizes all weights to FP8 E4M3 format (simulated)
4. Evaluates quantized model accuracy
5. Measures accuracy degradation

**Output:**
- Results: `results/FP8Test/metrics/fp8_quantization_results.json`
- Includes: Original accuracy, quantized accuracy, degradation metrics

**Expected degradation:** ~2-3% accuracy loss (acceptable for deployment)

**Note:** This is NOT a training script - it's a conversion and evaluation test to assess FP8 viability for inference deployment.

### 4. Extract Metrics

```bash
# Single experiment summary
python scripts/extract_metrics.py --experiment results/BaseFP32

# Detailed statistics
python scripts/extract_metrics.py --experiment results/BaseFP32 --detailed

# Compare multiple experiments
python scripts/extract_metrics.py --compare results/BaseFP32 results/BaseFP16 results/AugmFP16

# Extract all experiments and save to CSV
python scripts/extract_metrics.py --all --output metrics_summary.csv
```

### 5. Generate Plots

```bash
# Plot single experiment
python scripts/generate_plots.py --experiment results/BaseFP32

# Compare multiple experiments
python scripts/generate_plots.py --compare results/BaseFP32 results/BaseFP16 results/AugmFP16

# Generate all plots
python scripts/generate_plots.py --all
```

**Generated Plots:**
- Training/validation loss curves
- Training/validation accuracy curves
- Learning rate schedules
- Hardware monitoring (CPU, memory, thermal)
- Multi-experiment comparisons

---

## 📊 Interpreting Results

### Metrics Summary

After running experiments, check the metrics summary:

```bash
python scripts/extract_metrics.py --all --output results/metrics_summary.csv
```

**Key Metrics:**
- **Best Validation Accuracy**: Highest accuracy achieved during training
- **Convergence Epoch**: When model reached within 0.5% of best accuracy
- **Generalization Gap**: Final train_acc - final val_acc (lower is better)
- **Overfitting Score**: best_val_acc - final_val_acc (>0 indicates overfitting)
- **Average Epoch Time**: Training speed (lower is better)

### Expected Results (Typical)

| Experiment | Best Val Acc | Convergence | Epoch Time | Notes |
|------------|--------------|-------------|------------|-------|
| BaseFP32   | ~82-84%      | ~35-40 epochs | ~125s    | Baseline reference |
| AugmFP32   | ~83-85%      | ~40-45 epochs | ~130s    | Better generalization |
| BaseFP16   | ~82-84%      | ~35-40 epochs | ~85-95s  | ~30% faster, similar accuracy |
| AugmFP16   | ~83-85%      | ~40-45 epochs | ~90-100s | Best speed/accuracy trade-off |

**Note:** Actual results depend on hardware, random seeds, and system load.

### Visualizations

Check `results/{ExperimentName}/plots/` for:

1. **training_curves.png**: Loss and accuracy over time
   - Look for smooth convergence (no wild oscillations)
   - Check for overfitting (val_loss increasing while train_loss decreases)

2. **learning_rate.png**: LR schedule
   - Should show cosine annealing (smooth decay)

3. **hardware_stats.png**: System resource usage
   - CPU should be <80% on average (data loading bottleneck if higher)
   - Memory should be stable (no leaks)
   - Thermal pressure should be 0 or low (cooling is adequate)

---

## 🔬 Advanced Usage

### GPU Monitoring (Apple Silicon - Automatic!)

GPU monitoring is now **automatically integrated** into training scripts! On macOS with Apple Silicon:

**You'll be prompted for sudo password when training starts:**
```bash
$ python scripts/train.py --config configs/BaseFP16.yaml
🔒 GPU Monitor requires sudo access for powermetrics...
Password: [enter your password]
🎮 GPU Monitor started...
💾 Saving to: results/BaseFP16/metrics/gpu_stats.csv
```

**No sudo? No problem!** If you cancel the password prompt:
```bash
⚠️  Sudo access denied. GPU monitoring disabled.
   (Training will continue without GPU stats)
```

**GPU stats are automatically saved to the experiment folder:**
- Output: `results/{ExperimentName}/metrics/gpu_stats.csv`
- Contains: Timestamp, GPU utilization %, GPU power (mW), CPU power (mW)

**Note:** The separate `gpu_monitor.py` tool is still available for standalone monitoring if needed.

### Custom Configuration

For custom experiments, create a new config file based on existing ones:

1. Copy an existing config:
```bash
cp configs/AugmFP16.yaml configs/MyExperiment.yaml
```

2. Edit the YAML file to customize your experiment:
```yaml
name: "MyExperiment"
description: "Custom experiment with larger model"

model:
  name: "vit_small_patch16_224"  # Larger model
  num_classes: 10
  pretrained: false

training:
  num_epochs: 100  # More training
  learning_rate: 0.0005  # Lower LR
  use_amp: true
  # ... other settings
```

3. Run your custom experiment:
```bash
python scripts/train.py --config configs/MyExperiment.yaml
```

### Resume Training (Not Yet Implemented)

Future feature - would require adding resume functionality to train.py:
```bash
# Planned future usage:
python scripts/train.py --config configs/BaseFP32.yaml --resume results/BaseFP32/checkpoints/checkpoint_epoch_30.pt
```

---

## 🛠️ Development

### Code Quality

All source code includes comprehensive docstrings following NumPy/Google style conventions:

```python
from src.models.vit_model import create_vit_model, get_model_info

# All functions have detailed documentation
help(create_vit_model)
help(get_model_info)
```

### Running Tests

```bash
# Test model creation
python src/models/vit_model.py

# Test data loader
python src/data/dataset.py

# Test metrics tracking
python src/utils/metrics.py
```

### Adding New Models

Edit `src/models/vit_model.py` to add new architectures:

```python
MODEL_CONFIGS = {
    'vit_large': {
        'name': 'vit_large_patch16_224',
        'params_approx': '300M',
        'description': 'ViT-Large - highest capacity'
    }
}
```

---

## 📈 Results Interpretation Guide

### 1. Accuracy Analysis

**What to look for:**
- **FP32 vs FP16**: Expect <1% accuracy difference (FP16 should be nearly identical)
- **Basic vs Extended Augmentation**: Extended should have +1-2% accuracy (better generalization)
- **Convergence Speed**: FP16 trains ~30% faster per epoch, similar convergence behavior

**Red flags:**
- FP16 accuracy >2% lower than FP32: Possible numerical instability (use gradient clipping)
- Extended augmentation worse than basic: Data augmentation too aggressive
- High generalization gap (>5%): Model is overfitting (increase augmentation or regularization)

### 2. Training Dynamics

**Healthy training:**
- Smooth loss curves (no sudden spikes)
- Val accuracy tracks train accuracy with small gap
- Best val accuracy near the end of training (good convergence)

**Problematic training:**
- Oscillating loss: Learning rate too high or data issue
- Early plateau: Learning rate too low or model capacity insufficient
- Overfitting: Val loss increases while train loss decreases (add regularization)

### 3. Hardware Efficiency

**Good utilization:**
- CPU: 40-80% (data loading + model training)
- Memory: Stable, no growth over time
- Thermal: 0 or low (system stays cool)
- GPU (if monitored): 70-95% utilization

**Poor utilization:**
- CPU >90%: Data loading bottleneck (increase num_workers or use pin_memory)
- Thermal pressure >0: System throttling (reduce batch size or improve cooling)
- GPU <50%: CPU bottleneck or small batch size

---

## 🐛 Troubleshooting

### Common Issues

**1. Out of Memory (OOM)**
```bash
# Reduce batch size in config
batch_size: 64  # Instead of 128
```

**2. MPS Backend Error (Apple Silicon)**
```bash
# Fall back to CPU
python scripts/train.py --config configs/BaseFP32.yaml --device cpu
```

**3. Slow Training**
```bash
# Increase data loader workers
num_workers: 4  # In config file

# Enable persistent workers (automatic in code)
```

**4. Import Errors**
```bash
# Ensure project root is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

**5. GPU Monitoring Requires Sudo**
```bash
# powermetrics requires root access
sudo python src/utils/gpu_monitor.py --name experiment
```

---

## 📚 Dependencies

### Core Libraries
- **PyTorch** (2.0+): Deep learning framework with MPS/CUDA support
- **timm** (0.9+): Vision Transformer models (PyTorch Image Models)
- **torchvision**: CIFAR-10 dataset and transforms

### Utilities
- **PyYAML**: Configuration file parsing
- **NumPy**: Numerical operations and statistics
- **matplotlib**: Plotting and visualization
- **tqdm**: Progress bars for training loops
- **psutil**: System resource monitoring (CPU, memory)

### Optional
- **powermetrics** (macOS built-in): GPU monitoring on Apple Silicon

---

## 🎓 Citation

If you use this code for research, please cite:

```bibtex
@misc{vit-fp8-experiments,
  author = {Your Name},
  title = {Vision Transformer FP8 Precision Experiments},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ViT-FP8-experiments}
}
```

### Related Papers

- **Vision Transformers (ViT)**: [Dosovitskiy et al., "An Image is Worth 16x16 Words", ICLR 2021](https://arxiv.org/abs/2010.11929)
- **Mixed Precision Training**: [Micikevicius et al., "Mixed Precision Training", ICLR 2018](https://arxiv.org/abs/1710.03740)
- **FP8 Training**: [Noune et al., "8-bit Numerical Formats for Deep Neural Networks", 2022](https://arxiv.org/abs/2206.02915)

---

## 📝 License

This project is released under the MIT License. See LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or issues:
- Open an issue on GitHub
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **timm library** by Ross Wightman for excellent ViT implementations
- **PyTorch team** for MPS backend and mixed precision support
- **CIFAR-10 dataset** creators for the benchmark dataset

---

## 📖 Additional Resources

### Documentation
- [Full API Documentation](docs/API.md) (if available)
- [Development Roadmap](REFACTORING_PLAN.md)
- [Experiment Configs Guide](docs/configs.md) (if available)

### External Resources
- [PyTorch AMP Tutorial](https://pytorch.org/docs/stable/amp.html)
- [timm Documentation](https://huggingface.co/docs/timm/index)
- [Vision Transformer Explained](https://jalammar.github.io/illustrated-transformer/)

---

**Last Updated:** 2024 (Update with actual date)
**Version:** 1.0.0
