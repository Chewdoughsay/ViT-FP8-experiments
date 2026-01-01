"""
Trainer class pentru antrenarea modelelor ViT
Include: Mixed Precision (AMP), Hardware Monitoring, Metrics Tracking
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
    """Trainer pentru modele Vision Transformer"""

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

        # 1. Setup Folder
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 2. Metrics
        self.metrics = MetricsTracker(save_dir=self.save_dir)

        # 3. Hardware Monitor
        self.monitor = SystemMonitor(interval=2.0) if HAS_MONITOR else None

        # 4. Optimizer & Loss
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

        # 5. AMP Scaler
        self.scaler = None
        if self.use_amp:
            print(f"⚡ Mixed Precision (AMP) enabled for device: {device}")
            self.scaler = torch.amp.GradScaler(device)

        print(f"Trainer initialized on device: {device}")
        print(f"Saving to: {self.save_dir}")

    def train_epoch(self):
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

                # Masurare timp per epoca
                start_time = time.time()

                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc = self.validate()

                end_time = time.time()
                epoch_duration = end_time - start_time

                # Scheduler step
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                    warmup_sched.step()
                elif self.scheduler is not None:
                    self.scheduler.step()

                # Save Metrics (Loss, Acc, Time)
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
            # La final (sau daca dam stop), oprim monitorul si salvam datele hardware
            if self.monitor:
                summary, full_stats = self.monitor.stop()
                print("\nHardware Summary:")
                print(f"  Avg CPU: {summary['avg_cpu']:.1f}%")
                print(f"  Avg RAM: {summary['avg_mem']:.1f}%")
                if summary['throttled']:
                    print(f"  🔥 WARNING: Thermal Throttling detected! (Level {summary['max_thermal']})")
                else:
                    print(f"  ✅ Thermals OK (No throttling)")

                # Salvam stats hardware in folder
                import json
                with open(self.save_dir / 'hardware_stats.json', 'w') as f:
                    json.dump(full_stats, f)

            # Save final metrics standard
            self.metrics.save('final_metrics.json')
            print(f"\n{'='*60}")
            print(f"Training completed (or stopped)!")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            print(f"{'='*60}\n")

    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Salvează checkpoint-ul modelului"""
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
            save_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, save_path)
            print(f"  💾 Best model saved: {save_path}")
        else:
            save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved: {save_path}")


if __name__ == '__main__':
    print("Trainer module ready.")