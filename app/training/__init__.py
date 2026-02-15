"""
Training utilities for Voice Detector V2.

This package contains:
- train_classifier: Main training script for the V2 classifier head
- utils: Checkpoint loading and testing utilities
"""

from .train_classifier import (
    AudioDataset,
    train_epoch,
    validate,
    main as train_classifier_main
)

from .utils import (
    load_trained_classifier,
    test_on_audio_file,
    batch_test
)

__all__ = [
    # Training components
    'AudioDataset',
    'train_epoch',
    'validate',
    'train_classifier_main',
    
    # Utilities
    'load_trained_classifier',
    'test_on_audio_file',
    'batch_test',
]

__version__ = '2.0.0'

# Convenience function
def train_classifier(data_dir, **kwargs):
    """
    Convenience wrapper for training the V2 classifier.
    
    Args:
        data_dir: Path to training data directory with 'human/' and 'ai/' subdirectories
        **kwargs: Additional arguments for training (epochs, batch_size, lr, etc.)
    
    Example:
        >>> from app.training import train_classifier
        >>> train_classifier('dataset/training_data', epochs=5, batch_size=16)
    """
    import argparse
    
    # Build arguments
    args = argparse.Namespace(
        data_dir=data_dir,
        val_split=kwargs.get('val_split', 0.2),
        epochs=kwargs.get('epochs', 5),
        batch_size=kwargs.get('batch_size', 16),
        lr=kwargs.get('lr', 1e-3),
        save_dir=kwargs.get('save_dir', './checkpoints'),
        device=kwargs.get('device', 'cuda' if __import__('torch').cuda.is_available() else 'cpu')
    )
    
    # Run training
    import sys
    sys.argv = ['train_classifier.py']  # Reset argv to avoid conflicts
    
    # Import and run
    from .train_classifier import main
    return main(args)