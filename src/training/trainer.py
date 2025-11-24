"""
Trainer class pentru antrenarea modelelor ViT
Supports: Mixed Precision (AMP), Label Smoothing, Gradient Clipping, Metrics Tracking
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

from src.utils.metrics import MetricsTracker, calculate_accuracy, Timer


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

        # 1. Setup Folder Salvare (CRITIC: Trebuie definit înainte de MetricsTracker)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 2. Metrics Tracking (Acum știe exact unde să salveze)
        self.metrics = MetricsTracker(save_dir=self.save_dir)

        # 3. Loss & Optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 4. Scheduler (Cosine Annealing)
        # Nota: Poți ajusta T_max dacă schimbi numărul de epoci
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50
        )

        # 5. Mixed Precision Scaler (AMP)
        self.scaler = None
        if self.use_amp:
            print(f"⚡ Mixed Precision (AMP) enabled for device: {device}")
            self.scaler = torch.amp.GradScaler(device)

        print(f"Trainer initialized on device: {device}")
        print(f"Checkpoints/Metrics will be saved to: {self.save_dir}")

    def train_epoch(self):
        """Antrenează modelul pentru o epocă"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Reset gradients
            self.optimizer.zero_grad()

            # --- LOGICA MIXED PRECISION (AMP) ---
            if self.use_amp:
                # Select device type for autocast context
                device_type = 'cuda' if 'cuda' in self.device else ('mps' if 'mps' in self.device else 'cpu')
                dtype = torch.float16 if device_type != 'cpu' else torch.bfloat16

                # Forward pass in FP16
                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                # Backward pass cu scalare
                self.scaler.scale(loss).backward()

                # Gradient Clipping (trebuie făcut pe gradienții unscaled)
                if self.gradient_clip:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # --- LOGICA STANDARD (FP32) ---
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()

                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

                self.optimizer.step()
            # -----------------------------------

            # Calculează metrici (pe detach pentru viteză)
            acc = calculate_accuracy(outputs.detach(), labels)
            total_loss += loss.item()
            total_acc += acc

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        return avg_loss, avg_acc

    @torch.no_grad()
    def validate(self):
        """Validează modelul pe test set"""
        self.model.eval()
        total_loss = 0
        total_acc = 0
        num_batches = len(self.test_loader)

        pbar = tqdm(self.test_loader, desc='Validation')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Inference standard
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            acc = calculate_accuracy(outputs, labels)

            total_loss += loss.item()
            total_acc += acc

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        return avg_loss, avg_acc

    def train(self, num_epochs, save_every=10):
        """
        Antrenează modelul pentru mai multe epoci
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        if self.use_amp:
            print("🚀 Optimized with Mixed Precision (AMP)")
        print(f"{'='*60}\n")

        best_val_acc = 0.0

        # Warmup scheduler logic
        warmup_sched = None
        if self.warmup_epochs > 0:
            warmup_sched = optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, total_iters=self.warmup_epochs
            )

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"{'-'*60}")

            with Timer(f"Epoch {epoch}"):
                # Training
                train_loss, train_acc = self.train_epoch()

                # Validation
                val_loss, val_acc = self.validate()

                # Update scheduler
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                    warmup_sched.step()
                elif self.scheduler is not None:
                    self.scheduler.step()

                # Save metrics
                self.metrics.update(
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                    learning_rates=current_lr
                )

                # Print summary
                print(f"\nEpoch {epoch} Summary:")
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
                print(f"  LR: {current_lr:.6f}")

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"  🎉 New best validation accuracy: {val_acc:.4f}")
                    self.save_checkpoint(epoch, val_acc, is_best=True)

                # Save checkpoint periodically
                if epoch % save_every == 0:
                    self.save_checkpoint(epoch, val_acc, is_best=False)

        # Save final metrics (Va salva în folderul experimentului)
        self.metrics.save('final_metrics.json')
        print(f"\n{'='*60}")
        print(f"Training completed!")
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