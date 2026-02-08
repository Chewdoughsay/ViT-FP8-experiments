"""
Trainer class for Vision Transformer models
Includes: Mixed Precision (AMP), Hardware Monitoring, Metrics Tracking
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import time

from src.utils.metrics import MetricsTracker, calculate_accuracy, Timer

# Importam monitorul nou (cu fallback daca nu exista psutil)
try:
    from src.utils.system_monitor import SystemMonitor
    HAS_MONITOR = True
except ImportError:
    print("⚠️ SystemMonitor not found. Install psutil: pip install psutil")
    HAS_MONITOR = False


class ViTTrainer:
    """
    Comprehensive trainer for Vision Transformer models with advanced features.

    This trainer provides a complete training pipeline for Vision Transformers with:
    - Mixed Precision Training (AMP) for FP16/FP32
    - Hardware monitoring (CPU, memory, thermal throttling)
    - Automatic metrics tracking and checkpointing
    - Learning rate scheduling with optional warmup
    - Gradient clipping and label smoothing
    - Organized output directories (separate checkpoints and metrics)

    The trainer automatically organizes outputs into experiment-specific directories:
    - Checkpoints: save_dir (e.g., results/BaseFP32/checkpoints/)
    - Metrics: save_dir/../metrics/ (e.g., results/BaseFP32/metrics/)
    - Hardware stats: metrics directory

    Args:
        model (torch.nn.Module): Vision Transformer model to train
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Validation/test data loader
        device (str): Device for training ('mps', 'cuda', or 'cpu'). Default: 'mps'
        learning_rate (float): Initial learning rate for AdamW optimizer. Default: 1e-3
        weight_decay (float): L2 regularization strength. Default: 0.05
        save_dir (str): Directory for saving checkpoints. Metrics will be saved to
            save_dir/../metrics/. Default: 'results/checkpoints'
        label_smoothing (float): Label smoothing factor (0.0 = no smoothing, 0.1 = 10%).
            Default: 0.0
        gradient_clip (float, optional): Maximum gradient norm for clipping. None = no clipping.
            Default: None
        warmup_epochs (int): Number of epochs for linear learning rate warmup. Default: 0
        use_amp (bool): Enable Automatic Mixed Precision (AMP) training. Uses FP16 on
            MPS/CUDA, BF16 on CPU. Default: False

    Attributes:
        checkpoint_dir (Path): Directory where model checkpoints are saved
        metrics_dir (Path): Directory where metrics and hardware stats are saved
        metrics (MetricsTracker): Tracks training/validation metrics per epoch
        monitor (SystemMonitor): Monitors CPU, memory, and thermal throttling
        criterion (CrossEntropyLoss): Loss function with optional label smoothing
        optimizer (AdamW): AdamW optimizer with weight decay
        scheduler (CosineAnnealingLR): Cosine annealing learning rate scheduler
        scaler (GradScaler, optional): Gradient scaler for mixed precision training

    Example:
        >>> from src.models.vit_model import create_vit_model
        >>> from src.data.dataset import get_cifar10_loaders
        >>>
        >>> # Create model and data loaders
        >>> model = create_vit_model('vit_tiny_patch16_224', num_classes=10)
        >>> train_loader, test_loader = get_cifar10_loaders(batch_size=128)
        >>>
        >>> # Initialize trainer with mixed precision
        >>> trainer = ViTTrainer(
        ...     model=model,
        ...     train_loader=train_loader,
        ...     test_loader=test_loader,
        ...     device='mps',
        ...     learning_rate=1e-3,
        ...     save_dir='results/experiment1/checkpoints',
        ...     use_amp=True,
        ...     warmup_epochs=5
        ... )
        >>>
        >>> # Train for 50 epochs
        >>> trainer.train(num_epochs=50, save_every=10)

    Notes:
        - The trainer uses AdamW optimizer with cosine annealing LR scheduling
        - Checkpoints include model, optimizer, scheduler states and all metrics
        - Best model is automatically saved when validation accuracy improves
        - Hardware monitoring requires psutil (install: pip install psutil)
        - Mixed precision (AMP) is recommended for faster training on modern hardware
    """

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device='mps',
        learning_rate=1e-3,
        weight_decay=0.05,
        save_dir='results/checkpoints',
        label_smoothing=0.0,
        gradient_clip=None,
        warmup_epochs=0,
        use_amp=False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.gradient_clip = gradient_clip
        self.warmup_epochs = warmup_epochs
        self.use_amp = use_amp

        # 1. Setup directories: checkpoints and metrics
        self.checkpoint_dir = Path(save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Derive metrics directory from checkpoint directory
        # e.g., results/BaseFP32/checkpoints -> results/BaseFP32/metrics
        experiment_root = self.checkpoint_dir.parent
        self.metrics_dir = experiment_root / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # 2. Metrics tracker (saves to metrics directory)
        self.metrics = MetricsTracker(save_dir=self.metrics_dir)

        # 3. Hardware monitor
        self.monitor = SystemMonitor(interval=2.0) if HAS_MONITOR else None

        # 4. Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50
        )

        # 5. Mixed precision scaler
        self.scaler = None
        if self.use_amp:
            print(f"⚡ Mixed Precision (AMP) enabled for device: {device}")
            self.scaler = torch.amp.GradScaler(device)

        print(f"Trainer initialized on device: {device}")
        print(f"Checkpoints: {self.checkpoint_dir}")
        print(f"Metrics: {self.metrics_dir}")

    def train_epoch(self):
        """
        Execute one training epoch with gradient updates.

        Performs forward pass, loss computation, backward pass, and optimizer step
        for all batches in the training set. Supports both standard and mixed
        precision training with optional gradient clipping.

        Returns:
            tuple: (average_loss, average_accuracy)
                - average_loss (float): Mean loss across all training batches
                - average_accuracy (float): Mean accuracy across all training batches

        Notes:
            - Uses tqdm progress bar to display real-time loss and accuracy
            - With AMP enabled: Uses autocast context and GradScaler
            - With gradient clipping: Clips gradients before optimizer step
            - Model is set to training mode (enables dropout, batch norm updates)
        """
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # --- AMP LOGIC ---
            if self.use_amp:
                device_type = 'cuda' if 'cuda' in self.device else ('mps' if 'mps' in self.device else 'cpu')
                dtype = torch.float16 if device_type != 'cpu' else torch.bfloat16

                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()

                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                self.optimizer.step()

            acc = calculate_accuracy(outputs.detach(), labels)
            total_loss += loss.item()
            total_acc += acc

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

        return total_loss / num_batches, total_acc / num_batches

    @torch.no_grad()
    def validate(self):
        """
        Evaluate model performance on validation/test set.

        Runs inference on the test set without computing gradients. Used to
        monitor generalization and select the best model checkpoint.

        Returns:
            tuple: (average_loss, average_accuracy)
                - average_loss (float): Mean loss across all validation batches
                - average_accuracy (float): Mean accuracy across all validation batches

        Notes:
            - @torch.no_grad() decorator disables gradient computation for efficiency
            - Model is set to evaluation mode (disables dropout, uses running stats for batch norm)
            - Uses tqdm progress bar to display real-time validation metrics
            - No gradient updates or optimizer steps are performed
        """
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = len(self.test_loader)

        pbar = tqdm(self.test_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            total_loss += loss.item()
            total_acc += acc
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})

        return total_loss / num_batches, total_acc / num_batches

    def train(self, num_epochs, save_every=10):
        """
        Execute complete training loop for specified number of epochs.

        Runs the full training pipeline including:
        - Hardware monitoring (CPU, memory, thermal pressure)
        - Optional learning rate warmup phase
        - Training and validation per epoch
        - Metrics tracking (loss, accuracy, learning rate, epoch time)
        - Automatic checkpoint saving (best model + periodic checkpoints)
        - Graceful handling of keyboard interrupts
        - Final summary with hardware statistics

        Args:
            num_epochs (int): Total number of training epochs to run
            save_every (int): Save checkpoint every N epochs. Default: 10

        Behavior:
            - Saves best model when validation accuracy improves
            - Saves periodic checkpoints every `save_every` epochs
            - If interrupted (Ctrl+C), saves interrupted_metrics.json
            - Always saves final_metrics.json and hardware_stats.json at the end
            - Displays real-time progress with epoch summaries

        Output Files:
            Checkpoints directory (self.checkpoint_dir):
                - best_model.pt: Model with highest validation accuracy
                - checkpoint_epoch_N.pt: Periodic checkpoints

            Metrics directory (self.metrics_dir):
                - final_metrics.json: Training/validation metrics per epoch
                - interrupted_metrics.json: Metrics if training was interrupted
                - hardware_stats.json: CPU, memory, thermal monitoring data

        Example:
            >>> trainer = ViTTrainer(model, train_loader, test_loader, device='mps')
            >>> trainer.train(num_epochs=50, save_every=10)
            Starting training for 50 epochs
            ⚡ Optimized with Mixed Precision (AMP)
            ============================================================
            🖥️  System Monitor started...

            Epoch 1/50
            Training: 100%|██████████| 391/391 [01:23<00:00, 4.69it/s]
            Validation: 100%|██████████| 79/79 [00:12<00:00, 6.21it/s]
            Epoch 1 Summary: [Time: 95.2s]
              Train: Loss 2.1234 | Acc 0.2456
              Val:   Loss 1.9876 | Acc 0.2891
              🎉 New Best Acc: 0.2891
            ...

        Notes:
            - Training can be safely interrupted with Ctrl+C without losing progress
            - Learning rate warmup (if enabled) is applied for the first N epochs
            - After warmup, cosine annealing scheduler reduces LR until the end
            - Hardware monitoring is optional (requires psutil)
            - All metrics are automatically saved for post-training analysis
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        if self.use_amp:
            print("🚀 Optimized with Mixed Precision (AMP)")
        print(f"{'='*60}\n")

        # Start Monitor
        if self.monitor:
            self.monitor.start()

        # Warmup
        warmup_sched = None
        if self.warmup_epochs > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, total_iters=self.warmup_epochs
            )

        best_val_acc = 0.0

        try:
            for epoch in range(1, num_epochs + 1):
                print(f"\nEpoch {epoch}/{num_epochs}")

                # Measure epoch time
                epoch_start_time = time.time()

                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.validate()

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time

                # Update learning rate scheduler
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                    warmup_sched.step()
                elif self.scheduler is not None:
                    self.scheduler.step()

                # Save metrics (loss, accuracy, time, learning rate)
                self.metrics.update(
                    train_loss=train_loss, train_acc=train_acc,
                    val_loss=val_loss, val_acc=val_acc,
                    learning_rates=current_lr,
                    epoch_time=epoch_duration
                )

                print(f"Epoch {epoch} Summary: [Time: {epoch_duration:.1f}s]")
                print(f"  Train: Loss {train_loss:.4f} | Acc {train_acc:.4f}")
                print(f"  Val:   Loss {val_loss:.4f} | Acc {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"  🎉 New Best Acc: {val_acc:.4f}")
                    self.save_checkpoint(epoch, val_acc, is_best=True)

                if epoch % save_every == 0:
                    self.save_checkpoint(epoch, val_acc, is_best=False)

        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user!")
            # Save metrics even if interrupted
            self.metrics.save('interrupted_metrics.json')

        finally:
            # Stop monitor and save hardware stats
            if self.monitor:
                summary, full_stats = self.monitor.stop()
                print("\nHardware Summary:")
                print(f"  Avg CPU: {summary['avg_cpu']:.1f}%")
                print(f"  Avg RAM: {summary['avg_mem']:.1f}%")
                if summary['throttled']:
                    print(f"  🔥 WARNING: Thermal Throttling detected! (Level {summary['max_thermal']})")
                else:
                    print(f"  ✅ Thermals OK (No throttling)")

                # Save hardware stats to metrics directory
                import json
                with open(self.metrics_dir / 'hardware_stats.json', 'w') as f:
                    json.dump(full_stats, f)

            # Save final metrics
            self.metrics.save('final_metrics.json')
            print(f"\n{'='*60}")
            print(f"Training completed (or stopped)!")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            print(f"{'='*60}\n")

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """
        Save model checkpoint with full training state.

        Saves a complete checkpoint containing model weights, optimizer state,
        scheduler state, and all training metrics. This allows resuming training
        or loading the model for inference/evaluation.

        Args:
            epoch (int): Current epoch number
            val_acc (float): Validation accuracy at this epoch
            is_best (bool): If True, saves as 'best_model.pt'. If False, saves as
                'checkpoint_epoch_N.pt'. Default: False

        Checkpoint Contents:
            - epoch: Current epoch number
            - model_state_dict: Model weights and biases
            - optimizer_state_dict: Optimizer state (momentum, adaptive LR, etc.)
            - scheduler_state_dict: LR scheduler state (if scheduler exists)
            - val_acc: Validation accuracy at this checkpoint
            - metrics: Full training history (all epochs up to this point)

        Saved Files:
            - best_model.pt: Overwritten when validation accuracy improves
            - checkpoint_epoch_N.pt: Unique file for each periodic checkpoint

        Example:
            >>> # Load checkpoint for inference
            >>> checkpoint = torch.load('results/BaseFP32/checkpoints/best_model.pt')
            >>> model.load_state_dict(checkpoint['model_state_dict'])
            >>> print(f"Best accuracy: {checkpoint['val_acc']:.4f}")
            >>>
            >>> # Resume training from checkpoint
            >>> checkpoint = torch.load('results/BaseFP32/checkpoints/checkpoint_epoch_30.pt')
            >>> model.load_state_dict(checkpoint['model_state_dict'])
            >>> optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            >>> scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            >>> start_epoch = checkpoint['epoch'] + 1

        Notes:
            - Best model is saved whenever validation accuracy improves
            - Periodic checkpoints allow resuming if training is interrupted
            - All checkpoints include complete training state for reproducibility
            - Checkpoints can be large (size depends on model architecture)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'metrics': self.metrics.metrics
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            save_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, save_path)
            print(f"  💾 Best model saved: {save_path}")
        else:
            save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved: {save_path}")


if __name__ == '__main__':
    print("Trainer module ready.")