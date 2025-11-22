"""
Trainer class pentru antrenarea modelelor ViT
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
        label_smoothing=0.0,      # Nou
        gradient_clip=None,        # Nou
        warmup_epochs=0,           # Nou
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.gradient_clip = gradient_clip
        self.warmup_epochs = warmup_epochs
        self.base_lr = learning_rate

        # Loss cu label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Scheduler (va fi ajustat după warmup)
        self.scheduler = None
        self.warmup_scheduler = None

        # Metrics tracking
        self.metrics = MetricsTracker()

        # Save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        print(f"Trainer initialized on device: {device}")
        if label_smoothing > 0:
            print(f"  Using label smoothing: {label_smoothing}")
        if gradient_clip is not None:
            print(f"  Using gradient clipping: {gradient_clip}")
        if warmup_epochs > 0:
            print(f"  Using warmup: {warmup_epochs} epochs")

    def _setup_schedulers(self, num_epochs):
        """Setup learning rate schedulers cu warmup"""
        if self.warmup_epochs > 0:
            # Warmup scheduler
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=self.warmup_epochs
            )

            # Main scheduler (după warmup)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - self.warmup_epochs,
                eta_min=1e-6
            )
        else:
            # Doar cosine annealing
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
                eta_min=1e-6
            )

    def train_epoch(self, epoch):
        """Antrenează modelul pentru o epocă"""
        self.model.train()
        total_loss = 0
        total_acc = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (dacă e activat)
            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )

            self.optimizer.step()

            # Calculează metrici
            acc = calculate_accuracy(outputs, labels)
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

        Args:
            num_epochs: Numărul de epoci
            save_every: Salvează checkpoint-uri la fiecare X epoci
        """
        print(f"\n{'='*60}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*60}\n")

        # Setup schedulers
        self._setup_schedulers(num_epochs)

        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"{'-'*60}")

            with Timer(f"Epoch {epoch}"):
                # Training
                train_loss, train_acc = self.train_epoch(epoch)

                # Validation
                val_loss, val_acc = self.validate()

                # Update scheduler
                if self.warmup_epochs > 0 and epoch <= self.warmup_epochs:
                    self.warmup_scheduler.step()
                else:
                    if self.scheduler is not None:
                        self.scheduler.step()

                current_lr = self.optimizer.param_groups[0]['lr']

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

                # Track best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    print(f"  🎉 New best validation accuracy: {val_acc:.4f}")
                    self.save_checkpoint(epoch, val_acc, is_best=True)

                # Save checkpoint periodically
                if epoch % save_every == 0:
                    self.save_checkpoint(epoch, val_acc, is_best=False)

        # Save final metrics
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
            print(f"  💾 Checkpoint saved: {save_path}")


if __name__ == '__main__':
    print("Trainer module - import and use in experiments")