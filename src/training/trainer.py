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
        device='mps',  # 'mps' pentru Apple Silicon
        learning_rate=1e-3,
        weight_decay=0.05,
        save_dir='results/checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        
        # Loss și optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100  # Ajustează după numărul de epoci
        )
        
        # Metrics tracking
        self.metrics = MetricsTracker()
        
        # Save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Trainer initialized on device: {device}")
    
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
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
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
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"{'-'*60}")
            
            with Timer(f"Epoch {epoch}"):
                # Training
                train_loss, train_acc = self.train_epoch()
                
                # Validation
                val_loss, val_acc = self.validate()
                
                # Update scheduler
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
                
                # Save checkpoint
                if epoch % save_every == 0:
                    self.save_checkpoint(epoch, val_acc)
        
        # Save final metrics
        self.metrics.save('final_metrics.json')
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best validation accuracy: {self.metrics.get_best_acc():.4f}")
        print(f"{'='*60}\n")
    
    def save_checkpoint(self, epoch, val_acc):
        """Salvează checkpoint-ul modelului"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'metrics': self.metrics.metrics
        }
        
        save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")


if __name__ == '__main__':
    print("Trainer module - import and use in experiments")