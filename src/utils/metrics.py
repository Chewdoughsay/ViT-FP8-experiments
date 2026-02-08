"""
Utilities for calculating and logging training metrics.

This module provides tools for tracking, saving, and analyzing metrics during
Vision Transformer training. Includes:
- MetricsTracker: Tracks loss, accuracy, learning rate, and epoch time
- calculate_accuracy: Computes classification accuracy from model outputs
- Timer: Context manager for measuring execution time

Example:
    >>> from src.utils.metrics import MetricsTracker, Timer
    >>>
    >>> # Track metrics during training
    >>> tracker = MetricsTracker(save_dir='results/metrics')
    >>> tracker.update(train_loss=0.5, train_acc=0.85, val_loss=0.6, val_acc=0.82)
    >>> tracker.save('final_metrics.json')
    >>>
    >>> # Time an operation
    >>> with Timer("Data loading"):
    ...     data = load_dataset()
    Data loading took 2.35 seconds
"""
import time
import json
from pathlib import Path
import torch


class MetricsTracker:
    """
    Tracks and persists training metrics across epochs.

    Maintains time-series data for train/validation loss, accuracy, learning rates,
    and epoch timing. Automatically saves to JSON for post-training analysis and plotting.

    Args:
        save_dir (str): Directory for saving metrics JSON files. Default: 'results/metrics'

    Attributes:
        save_dir (Path): Directory where metrics are saved
        metrics (dict): Dictionary containing metric lists:
            - train_loss: Training loss per epoch
            - train_acc: Training accuracy per epoch
            - val_loss: Validation loss per epoch
            - val_acc: Validation accuracy per epoch
            - epoch_time: Duration of each epoch in seconds
            - learning_rates: Learning rate per epoch
        current_epoch (int): Current epoch number (0-indexed)

    Example:
        >>> tracker = MetricsTracker(save_dir='results/BaseFP32/metrics')
        >>>
        >>> # Update after each epoch
        >>> tracker.update(
        ...     train_loss=0.52, train_acc=0.84,
        ...     val_loss=0.61, val_acc=0.81,
        ...     epoch_time=125.3, learning_rates=0.001
        ... )
        >>>
        >>> # Save to disk
        >>> tracker.save('final_metrics.json')
        Metrics saved to results/BaseFP32/metrics/final_metrics.json
        >>>
        >>> # Get best accuracy
        >>> best_acc = tracker.get_best_acc()
        >>> print(f"Best validation accuracy: {best_acc:.4f}")

    Notes:
        - Metrics are stored in memory until save() is called
        - JSON format allows easy loading in Python or plotting tools
        - All metric lists are aligned by epoch (same length)
        - Use reset() to clear metrics for a new training run
    """

    def __init__(self, save_dir='results/metrics'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.reset()

    def reset(self):
        """
        Reset all metrics to empty lists.

        Clears all stored metrics and resets epoch counter. Use this when
        starting a new training run with the same tracker instance.

        Example:
            >>> tracker.reset()
            >>> # Now ready for a fresh training run
        """
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_time': [],
            'learning_rates': [],
        }
        self.current_epoch = 0

    def update(self, **kwargs):
        """
        Add metrics for the current epoch.

        Appends values to the corresponding metric lists. Only keys that match
        existing metrics are stored (unknown keys are silently ignored).

        Args:
            **kwargs: Metric name-value pairs. Valid keys:
                - train_loss (float): Training loss for this epoch
                - train_acc (float): Training accuracy for this epoch
                - val_loss (float): Validation loss for this epoch
                - val_acc (float): Validation accuracy for this epoch
                - epoch_time (float): Epoch duration in seconds
                - learning_rates (float): Learning rate for this epoch

        Example:
            >>> tracker.update(
            ...     train_loss=0.52, train_acc=0.84,
            ...     val_loss=0.61, val_acc=0.81,
            ...     epoch_time=125.3, learning_rates=0.001
            ... )
        """
        for key, value in kwargs.items():
            if key in self.metrics:
                self.metrics[key].append(value)

    def get_best_acc(self):
        """
        Return the best (maximum) validation accuracy achieved.

        Returns:
            float: Best validation accuracy, or 0.0 if no validation has been run.

        Example:
            >>> best_acc = tracker.get_best_acc()
            >>> print(f"Best accuracy: {best_acc:.4f}")
            Best accuracy: 0.8456
        """
        if self.metrics['val_acc']:
            return max(self.metrics['val_acc'])
        return 0.0

    def save(self, filename='metrics.json'):
        """
        Save all metrics to a JSON file.

        Writes the complete metrics dictionary to disk in JSON format.
        The file can be loaded later for analysis or plotting.

        Args:
            filename (str): Name of the JSON file. Default: 'metrics.json'

        Example:
            >>> tracker.save('final_metrics.json')
            Metrics saved to results/BaseFP32/metrics/final_metrics.json

        JSON Structure:
            {
              "train_loss": [1.23, 0.98, 0.76, ...],
              "train_acc": [0.56, 0.67, 0.74, ...],
              "val_loss": [1.45, 1.12, 0.89, ...],
              "val_acc": [0.52, 0.63, 0.71, ...],
              "epoch_time": [120.5, 118.3, 119.8, ...],
              "learning_rates": [0.001, 0.001, 0.0009, ...]
            }
        """
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {save_path}")

    def load(self, filename='metrics.json'):
        """
        Load metrics from a JSON file.

        Reads a previously saved metrics file and replaces the current metrics.
        Useful for resuming training or analyzing past experiments.

        Args:
            filename (str): Name of the JSON file to load. Default: 'metrics.json'

        Example:
            >>> tracker.load('final_metrics.json')
            Metrics loaded from results/BaseFP32/metrics/final_metrics.json
            >>> print(len(tracker.metrics['train_loss']))
            50  # Number of epochs in the loaded data
        """
        load_path = self.save_dir / filename
        with open(load_path, 'r') as f:
            self.metrics = json.load(f)
        print(f"Metrics loaded from {load_path}")


def calculate_accuracy(outputs, targets):
    """
    Calculate classification accuracy from model outputs.

    Computes the fraction of correct predictions by comparing the predicted
    class (argmax of outputs) with the ground truth labels.

    Args:
        outputs (torch.Tensor): Model predictions with shape [batch_size, num_classes].
            Raw logits or probabilities (both work since argmax is used).
        targets (torch.Tensor): Ground truth labels with shape [batch_size].
            Integer class indices (0 to num_classes-1).

    Returns:
        float: Accuracy as a value between 0.0 (no correct predictions) and 1.0
            (all predictions correct).

    Example:
        >>> outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
        >>> targets = torch.tensor([1, 0, 1])
        >>> acc = calculate_accuracy(outputs, targets)
        >>> print(f"Accuracy: {acc:.4f}")
        Accuracy: 1.0000

        >>> # All predictions wrong
        >>> targets_wrong = torch.tensor([0, 1, 0])
        >>> acc = calculate_accuracy(outputs, targets_wrong)
        >>> print(f"Accuracy: {acc:.4f}")
        Accuracy: 0.0000

    Notes:
        - Works with any number of classes (outputs.shape[1])
        - Expects integer labels in targets (not one-hot encoded)
        - Returns accuracy as a float (not percentage)
        - Use .detach() on outputs if they require gradients
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy


class Timer:
    """
    Context manager for measuring and printing execution time.

    Simple utility for timing code blocks. Automatically prints the elapsed
    time when exiting the context.

    Args:
        name (str): Description of the operation being timed. Default: "Operation"

    Attributes:
        name (str): Operation name for the printed message
        start_time (float): Timestamp when entering the context
        end_time (float): Timestamp when exiting the context
        elapsed (float): Total elapsed time in seconds

    Example:
        >>> with Timer("Model forward pass"):
        ...     outputs = model(inputs)
        Model forward pass took 0.15 seconds

        >>> with Timer("Full training epoch"):
        ...     train_one_epoch(model, loader, optimizer)
        Full training epoch took 125.43 seconds

    Notes:
        - Automatically prints elapsed time (no need to manually print)
        - Time is measured in seconds with 2 decimal places
        - Can be nested (inner timers will print inside outer ones)
        - Access elapsed time via the context variable if needed:
            >>> with Timer("Operation") as t:
            ...     do_something()
            >>> print(f"Exact time: {t.elapsed:.6f}s")
    """

    def __init__(self, name="Operation"):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        print(f"{self.name} took {self.elapsed:.2f} seconds")


if __name__ == '__main__':
    # Test
    tracker = MetricsTracker(save_dir='test_metrics')
    tracker.update(
        train_loss=0.5,
        train_acc=0.85,
        val_loss=0.6,
        val_acc=0.82,
        epoch_time=120.5
    )
    tracker.save('test.json')
    print(f"Best accuracy: {tracker.get_best_acc()}")