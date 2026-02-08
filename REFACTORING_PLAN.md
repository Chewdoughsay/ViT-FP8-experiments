# ViT-FP8 Experiments: Refactoring & Scientific Report Preparation

## Context

This refactoring addresses a messy research codebase that needs to be cleaned up and re-executed for a scientific report. The project compares Vision Transformer training across different precision levels (FP32/FP16/FP8) with varying regularization strategies.

**Current Issues:**
- Inconsistent experiment naming (exp1_baseline, experiment2_regularized, exp3_fp16_fixed, etc.)
- Redundant scripts (plot_results.py superseded by generate_plots.py, broken run_all.py)
- Missing documentation for scientific reporting
- Old experiment data mixed with new data
- Need fresh runs with cleaned codebase

**Goals:**
1. Rename 4 experiments to clear, consistent names: BaseFP32, AugmFP32, BaseFP16, AugmFP16
2. Remove redundant code (~200 lines)
3. Document all functions for scientific report
4. Preserve all existing data
5. Rerun all experiments with clean codebase

**Approach:** Phased execution (rename → document → rerun) for easier review between stages.

---

## Phase 1: Preserve & Rename (Execute First)

### 1.1 Archive Current State

**IMPORTANT:** The results/ folder (~20GB) is in .gitignore and won't be committed to git. You MUST back it up to external storage.

**Backup to external hard drive:**
```bash
# Mount your external drive first, then:
# Replace /Volumes/ExternalDrive with your actual mount point
cd /Users/alextudose/PycharmProjects/ViT-FP8-experiments
timestamp=$(date +%Y%m%d_%H%M%S)
cp -r results/ /Volumes/ExternalDrive/ViT_results_backup_${timestamp}/

# Verify backup completed
ls -lh /Volumes/ExternalDrive/ViT_results_backup_${timestamp}/
```

**Git commit code snapshot (results/ excluded by .gitignore):**
```bash
git add -A
git commit -m "Pre-refactoring snapshot: preserve code state before renaming

Note: results/ (~20GB) backed up to external drive separately

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### 1.2 Rename & Update Experiment Names

**Naming Convention:**
```
OLD                          → NEW
exp1_baseline               → BaseFP32  (FP32 baseline, minimal augmentation)
experiment2_regularized     → AugmFP32  (FP32 augmented, full regularization)
experiment3_fp16 (archive)  → BaseFP16_OLD_ARCHIVE
exp3_fp16_fixed             → AugmFP16  (FP16 augmented, full regularization)
exp4_fp8_test               → FP8Test   (FP8 quantization test)
```

**Configuration Files (5 files):**

1. Rename & update [configs/exp1_baseline.yaml](configs/exp1_baseline.yaml):
   ```bash
   mv configs/exp1_baseline.yaml configs/BaseFP32.yaml
   ```
   - Line 1: `name: "exp1_baseline"` → `name: "BaseFP32"`
   - Line 39: `save_dir: "results/checkpoints/exp1_baseline"` → `save_dir: "results/checkpoints/BaseFP32"`

2. Rename & update [configs/exp2_regularized.yaml](configs/exp2_regularized.yaml):
   ```bash
   mv configs/exp2_regularized.yaml configs/AugmFP32.yaml
   ```
   - Line 1: `name: "exp2_regularized"` → `name: "AugmFP32"`
   - Line 39: Update `save_dir` to `"results/checkpoints/AugmFP32"`

3. Rename & update [configs/exp3_fp16_fixed.yaml](configs/exp3_fp16_fixed.yaml):
   ```bash
   mv configs/exp3_fp16_fixed.yaml configs/AugmFP16.yaml
   ```
   - Line 1: `name: "exp3_fp16_fixed"` → `name: "AugmFP16"`
   - Line 39: Update `save_dir` to `"results/checkpoints/AugmFP16"`

4. Rename [configs/exp4_fp8_test.yaml](configs/exp4_fp8_test.yaml):
   ```bash
   mv configs/exp4_fp8_test.yaml configs/FP8Test.yaml
   ```
   - Line 1: `name: "exp4_fp8_test"` → `name: "FP8Test"`
   - Line 8: `load_checkpoint: "results/checkpoints/exp3_fp16_fixed/best_model.pt"` → `"results/checkpoints/AugmFP16/best_model.pt"`
   - Update `save_dir` to `"results/checkpoints/FP8Test"`

5. **Create new [configs/BaseFP16.yaml](configs/BaseFP16.yaml)** (based on exp1_baseline.yaml):
   - Copy BaseFP32.yaml structure
   - Set `use_amp: true`, `precision: "fp16"`
   - Set `name: "BaseFP16"`
   - Set `save_dir: "results/checkpoints/BaseFP16"`
   - Keep basic augmentation, minimal regularization

**Results Directories:**
```bash
cd results/checkpoints/
mv baseline_fp32 BaseFP32
mv experiment2_regularized AugmFP32
mv exp3_fp16_fixed AugmFP16
mv experiment3_fp16 experiment3_fp16_OLD_ARCHIVE  # Archive old baseline FP16
mv exp4_fp8_test FP8Test
```

**Python Scripts (3 critical files):**

1. [scripts/extract_metrics.py](scripts/extract_metrics.py) - Line 296-301:
   ```python
   # OLD:
   experiment_names = ['baseline_fp32', 'experiment2_regularized',
                       'experiment3_fp16', 'exp3_fp16_fixed']

   # NEW:
   experiment_names = ['BaseFP32', 'AugmFP32', 'BaseFP16', 'AugmFP16']
   ```
   Also update table headers and LaTeX output labels throughout file.

2. [scripts/generate_plots.py](scripts/generate_plots.py) - Lines 38-57:
   ```python
   # Lines 38-49: LABEL_MAPPING
   LABEL_MAPPING = {
       'BaseFP32': ('Baseline FP32', 'Base FP32', 1),
       'AugmFP32': ('Augmented FP32', 'Augm FP32', 2),
       'BaseFP16': ('Baseline FP16', 'Base FP16', 3),
       'AugmFP16': ('Augmented FP16', 'Augm FP16', 4),
   }

   # Lines 52-57: COLOR_MAPPING
   COLOR_MAPPING = {
       'BaseFP32': '#e74c3c',   # Red
       'AugmFP32': '#2ecc71',   # Green
       'BaseFP16': '#e67e22',   # Orange
       'AugmFP16': '#3498db',   # Blue
   }
   ```

3. Delete [configs/temp_verification_config.yaml](configs/temp_verification_config.yaml) (temporary file, no longer needed)

### 1.3 Clean Up Redundant Code

**Delete deprecated scripts:**
```bash
cd scripts/
git rm plot_results.py        # 140 lines - superseded by generate_plots.py
git rm run_all.py              # 55 lines - old experiment structure, broken
git commit -m "Remove deprecated scripts (plot_results.py, run_all.py)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

**Archive verification scripts:**
```bash
mkdir -p results/verification_run_fp32/verification_scripts/
mv scripts/verify_fp32_augmented.py results/verification_run_fp32/verification_scripts/
mv scripts/compare_verification.py results/verification_run_fp32/verification_scripts/
echo "Archived one-off verification scripts (2024-12-28 verification run)" > results/verification_run_fp32/README.txt

git add results/verification_run_fp32/
git commit -m "Archive verification scripts to results directory

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### 1.4 Verify Renaming

**Test that renamed configs are valid:**
```bash
# Test config loading (dry run)
python scripts/train.py --config configs/BaseFP32.yaml --help
python scripts/train.py --config configs/AugmFP32.yaml --help
python scripts/train.py --config configs/BaseFP16.yaml --help
python scripts/train.py --config configs/AugmFP16.yaml --help

# Test metrics extraction with new names
python scripts/extract_metrics.py

# Test plot generation (scan mode)
python scripts/generate_plots.py --scan
```

**Commit Phase 1:**
```bash
git add -A
git commit -m "Phase 1 complete: Rename experiments to BaseFP32/AugmFP32/BaseFP16/AugmFP16

Changes:
- Renamed 5 config files with updated name/save_dir fields
- Created new BaseFP16.yaml for baseline FP16 experiment
- Renamed 5 results directories
- Updated experiment names in extract_metrics.py and generate_plots.py
- Deleted plot_results.py and run_all.py (redundant)
- Archived verification scripts
- Removed temp_verification_config.yaml

All files use consistent naming: {Base/Augm}{FP32/FP16}

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 2: Documentation (Execute Second)

### 2.1 Document Core Training Module

**File:** [src/training/trainer.py](src/training/trainer.py) (260 lines)

Add comprehensive docstrings to:

1. **`ViTTrainer` class** (class-level docstring):
   - Explain purpose: trainer for Vision Transformers with monitoring
   - List capabilities: mixed precision, hardware monitoring, checkpointing
   - Document all Args: model, loaders, device, learning_rate, etc.
   - Include usage example

2. **`train_epoch()` method** (lines ~93-125):
   - Explain: executes one training epoch
   - Detail: forward pass, loss computation, backprop, gradient clipping
   - Returns: (average_loss, average_accuracy)
   - Note: handles AMP automatically

3. **`validate()` method** (lines ~127-151):
   - Explain: evaluates model on validation set
   - Detail: no gradient computation, evaluation mode
   - Returns: (val_loss, val_accuracy)

4. **`train()` method** (lines ~153-240):
   - Explain: main training loop orchestrator
   - Detail: warmup, LR scheduling, checkpointing, hardware monitoring
   - Args: num_epochs, save_every
   - Note: handles Ctrl+C gracefully

5. **`save_checkpoint()` method** (lines ~242-260):
   - Explain: saves model checkpoint with training state
   - Detail: saves model, optimizer, scheduler state dicts
   - Args: epoch, val_acc, is_best

### 2.2 Document Hardware Monitoring

**File:** [src/utils/system_monitor.py](src/utils/system_monitor.py)

Add docstrings to:

1. **`SystemMonitor` class** (class-level):
   - Explain: background hardware resource monitor
   - List metrics: CPU, memory, thermal pressure (macOS)
   - Args: interval (sampling rate)
   - Include usage example

2. **`__init__()` method**:
   - Document interval parameter

3. **`_get_thermal_pressure()` method**:
   - Explain: reads thermal throttling level from macOS sysctl
   - Returns: int (thermal level, 0 = no throttling)

4. **`_monitor_loop()` method**:
   - Explain: background monitoring loop (runs in thread)
   - Detail: samples CPU/memory/thermal at intervals

5. **`start()` and `stop()` methods**:
   - start(): begins background monitoring thread
   - stop(): returns (summary, full_stats) with aggregated data

**File:** [src/utils/gpu_monitor.py](src/utils/gpu_monitor.py)

Add:

1. **Module docstring**:
   - Explain: GPU power monitoring for Apple Silicon using powermetrics
   - Requirements: macOS, sudo access
   - Metrics: GPU utilization, GPU power, CPU power

2. **`monitor_stream()` function**:
   - Args: output_file (CSV path), interval (milliseconds)
   - Explain: continuously monitors and logs to CSV
   - Detail: parses powermetrics output, handles Ctrl+C gracefully

### 2.3 Create Project README

**File:** [README.md](README.md) (NEW - create at project root)

**Structure:**
```markdown
# ViT FP8 Precision Experiments

## Project Overview
- Brief description of project goals
- 4 experiments comparing FP32/FP16 precision with baseline/augmented regularization

## Experiments
| Name | Precision | Augmentation | Purpose |
|------|-----------|--------------|---------|
| BaseFP32 | FP32 | Basic | Baseline without regularization |
| AugmFP32 | FP32 | Extended | Best model with full regularization |
| BaseFP16 | FP16 | Basic | Mixed precision baseline |
| AugmFP16 | FP16 | Extended | Test regularization in FP16 |

## Project Structure
- configs/ - YAML configuration files
- src/ - Source code (models, data, training, utils)
- scripts/ - Execution scripts (train.py, extract_metrics.py, generate_plots.py)
- results/ - Outputs (checkpoints, metrics, plots)

## Installation
- Requirements: Python 3.11+, PyTorch 2.0+, timm, torchvision

## Usage
- Training: `python scripts/train.py --config configs/AugmFP32.yaml`
- Generate plots: `python scripts/generate_plots.py`
- Extract metrics: `python scripts/extract_metrics.py`

## Results
- Summary table of best accuracies
- Links to generated plots in results/plots_v2/

## Reproducibility
- All experiments use fixed random seeds
- Configurations saved in configs/
- Hardware stats logged
```

### 2.4 Commit Phase 2

```bash
git add src/training/trainer.py
git add src/utils/system_monitor.py
git add src/utils/gpu_monitor.py
git add README.md
git commit -m "Phase 2 complete: Add comprehensive documentation

Changes:
- Documented ViTTrainer class and all methods (train, validate, train_epoch, save_checkpoint)
- Documented SystemMonitor for hardware monitoring
- Documented gpu_monitor.py for Apple Silicon GPU tracking
- Created comprehensive README.md with project overview, usage, structure

All core functions now have scientific-quality docstrings.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Phase 3: Rerun Experiments (Execute Third)

### 3.1 Create Batch Rerun Script

**File:** [scripts/rerun_all_experiments.py](scripts/rerun_all_experiments.py) (NEW)

Create script that:
- Runs all 4 experiments sequentially: BaseFP32, AugmFP32, BaseFP16, AugmFP16
- Includes 2-minute thermal cooldown between experiments
- Handles KeyboardInterrupt gracefully
- Prints summary at end
- Usage: `python scripts/rerun_all_experiments.py`
- Supports `--skip` flag to skip specific experiments

### 3.2 Pre-Rerun Checklist

Before running experiments:
- [ ] All Phase 1 & 2 changes committed
- [ ] Old results backed up to external drive (from Step 1.1)
- [ ] All 4 config files ready (BaseFP32.yaml, AugmFP32.yaml, BaseFP16.yaml, AugmFP16.yaml)
- [ ] Sufficient disk space (>10GB)
- [ ] Laptop plugged in with good cooling

### 3.3 Execute Reruns

**Option A - Batch script (recommended):**
```bash
python scripts/rerun_all_experiments.py
```

**Option B - Individual runs:**
```bash
# Run each experiment with cooldown between
python scripts/train.py --config configs/BaseFP32.yaml
sleep 120  # 2-minute cooldown

python scripts/train.py --config configs/AugmFP32.yaml
sleep 120

python scripts/train.py --config configs/BaseFP16.yaml
sleep 120

python scripts/train.py --config configs/AugmFP16.yaml
```

**Optional: GPU monitoring (separate terminal):**
```bash
sudo python src/utils/gpu_monitor.py --name rerun_all_4experiments
```

**Expected duration:** ~12-16 hours total (3-4 hours per experiment)

### 3.4 Post-Rerun Analysis

**Generate all analysis outputs:**
```bash
# Extract metrics and create comparison tables
python scripts/extract_metrics.py
# Generates: results/metrics/all_experiments_summary.json
#           results/metrics/metrics_comparison_4exp.tex

# Generate all plots
python scripts/generate_plots.py
# Generates: results/plots_v2/*.png (8 comprehensive plots)

# Verify outputs
ls -lh results/metrics/
ls -lh results/plots_v2/
```

### 3.5 Backup Results to External Drive

**IMPORTANT:** Since results/ is in .gitignore, backup to external storage:

```bash
# Backup fresh experiment results to external drive
timestamp=$(date +%Y%m%d_%H%M%S)
cp -r results/ /Volumes/ExternalDrive/ViT_results_final_${timestamp}/

# Create summary file for reference
cat > results/RESULTS_SUMMARY.txt << EOF
Fresh runs completed: ${timestamp}

BaseFP32: [accuracy]% validation accuracy, [time]h training
AugmFP32: [accuracy]% validation accuracy, [time]h training
BaseFP16: [accuracy]% validation accuracy, [time]h training
AugmFP16: [accuracy]% validation accuracy, [time]h training

Full results backed up to external drive.
Metrics: results/metrics/all_experiments_summary.json
Plots: results/plots_v2/
EOF

# Commit code changes only (results/ excluded by .gitignore)
git add scripts/ configs/ README.md src/
git commit -m "Phase 3 complete: Add rerun script and execute experiments

Fresh runs completed for all 4 experiments.
Results (~20GB) backed up to external drive separately.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Critical Files to Modify

### Phase 1 (Renaming):
1. **configs/exp1_baseline.yaml** → rename + update lines 1, 39
2. **configs/exp2_regularized.yaml** → rename + update lines 1, 39
3. **configs/exp3_fp16_fixed.yaml** → rename + update lines 1, 39
4. **configs/exp4_fp8_test.yaml** → rename + update lines 1, 8
5. **configs/BaseFP16.yaml** → create new (copy from BaseFP32, modify precision)
6. **scripts/extract_metrics.py** → update line 296 experiment names list
7. **scripts/generate_plots.py** → update lines 38-57 (LABEL_MAPPING, COLOR_MAPPING)

### Phase 2 (Documentation):
1. **src/training/trainer.py** → add 5 comprehensive docstrings
2. **src/utils/system_monitor.py** → add class + 5 method docstrings
3. **src/utils/gpu_monitor.py** → add module + function docstring
4. **README.md** → create new (~200 lines)

### Phase 3 (Reruns):
1. **scripts/rerun_all_experiments.py** → create new script
2. All 4 configs → used for fresh training runs

---

## Verification & Testing

### After Phase 1:
```bash
# Verify configs load correctly
python scripts/train.py --config configs/BaseFP32.yaml --help
python scripts/train.py --config configs/AugmFP32.yaml --help
python scripts/train.py --config configs/BaseFP16.yaml --help
python scripts/train.py --config configs/AugmFP16.yaml --help

# Verify metrics extraction works
python scripts/extract_metrics.py

# Verify plot generation (scan mode, doesn't generate plots)
python scripts/generate_plots.py --scan
```

### After Phase 2:
```bash
# Check README renders properly
cat README.md

# Verify docstrings are present (spot check)
python -c "from src.training.trainer import ViTTrainer; help(ViTTrainer.train)"
python -c "from src.utils.system_monitor import SystemMonitor; help(SystemMonitor)"
```

### After Phase 3:
```bash
# Verify all experiments completed
ls -lh results/checkpoints/BaseFP32/final_metrics.json
ls -lh results/checkpoints/AugmFP32/final_metrics.json
ls -lh results/checkpoints/BaseFP16/final_metrics.json
ls -lh results/checkpoints/AugmFP16/final_metrics.json

# Verify metrics extracted
cat results/metrics/all_experiments_summary.json

# Verify plots generated with new names
ls results/plots_v2/*.png
```

---

## Expected Outcomes

### After Phase 1 (Renaming):
- ✅ 4 experiments renamed: BaseFP32, AugmFP32, BaseFP16, AugmFP16
- ✅ All results directories renamed
- ✅ 14 files updated with new names
- ✅ Redundant code removed (195 lines)
- ✅ Old experiment3_fp16 archived as experiment3_fp16_OLD_ARCHIVE
- ✅ Verification scripts moved to results/verification_run_fp32/

### After Phase 2 (Documentation):
- ✅ ViTTrainer fully documented (class + 5 methods)
- ✅ SystemMonitor fully documented (class + 5 methods)
- ✅ gpu_monitor.py documented (module + function)
- ✅ README.md created with project overview
- ✅ All core functions have scientific-quality docstrings

### After Phase 3 (Reruns):
- ✅ Fresh training runs for all 4 experiments
- ✅ Consistent results with latest codebase
- ✅ Metrics comparison table (LaTeX + JSON)
- ✅ 8 comprehensive plots generated
- ✅ Ready for scientific report/paper

### Final State:
- Clean, consistent naming across entire project
- Zero code redundancy
- Comprehensive documentation
- Fresh experimental results
- Publication-ready metrics and visualizations

---

## Rollback Plan

If issues occur during any phase:

**Phase 1 rollback (code):**
```bash
git reset --hard HEAD~1  # Undo last commit
```

**Phase 1 rollback (results):**
```bash
# Restore from external drive backup:
rm -rf results/
cp -r /Volumes/ExternalDrive/ViT_results_backup_TIMESTAMP/ results/
```

**Phase 2 rollback:**
```bash
git reset --hard HEAD~1  # Remove documentation commit
# Results unaffected - still on external drive
```

**Phase 3 rollback:**
```bash
# Restore old results from external drive backup if needed
# New results also backed up separately
# No code rollback needed - experiments are additive
```

---

## Timeline Estimates

| Phase | Tasks | Estimated Time |
|-------|-------|----------------|
| **Phase 1** | Archive, rename configs/dirs, update scripts, delete redundant code | 1.5-2 hours |
| **Phase 2** | Add docstrings (3 files), create README | 2-3 hours |
| **Phase 3** | Create rerun script, execute 4 experiments, generate analysis | 14-18 hours (mostly unattended) |
| **Total** | End-to-end | ~18-23 hours |

**Note:** Phase 3 experiments can run overnight. Active work is ~4-5 hours.

---

## Success Criteria

- [ ] All experiment names follow BaseFP32/AugmFP32/BaseFP16/AugmFP16 convention
- [ ] No references to old names (exp1, exp2, exp3, experiment2, etc.) in active code
- [ ] plot_results.py and run_all.py deleted
- [ ] Verification scripts archived in results/
- [ ] All ViTTrainer, SystemMonitor, gpu_monitor functions documented
- [ ] README.md created and comprehensive
- [ ] All 4 experiments run successfully with fresh code
- [ ] Metrics comparison table generated (LaTeX + JSON)
- [ ] 8 plots generated in results/plots_v2/ with new naming
- [ ] Git history shows clear phase commits for code changes
- [ ] Old results preserved on external drive
- [ ] New results backed up to external drive
