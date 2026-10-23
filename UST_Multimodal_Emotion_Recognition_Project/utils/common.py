"""
Common utilities for IEMOCAP MemoCMT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import yaml
import json
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.utils.rnn import pad_sequence


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance
    
    Args:
        alpha (float or Tensor): Weighting factor for rare class (default: 1.0)
        gamma (float): Focusing parameter to reduce the relative loss for well-classified examples (default: 2.0)
        num_classes (int): Number of classes
        reduction (str): Reduction method ('none', 'mean', 'sum')
    """
    
    def __init__(self, alpha=1.0, gamma=2.0, num_classes=6, reduction='mean'):
        super(FocalLoss, self).__init__()
        
        if isinstance(alpha, (float, int)):
            self.alpha = torch.ones(num_classes) * alpha
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
            
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Args:
            inputs (Tensor): Predictions of shape (N, C) where N is batch size and C is number of classes
            targets (Tensor): Ground truth labels of shape (N,)
        
        Returns:
            Tensor: Focal loss
        """
        # Compute softmax probabilities
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Move alpha to same device as inputs
        if self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)
        
        # Get alpha for each sample
        at = self.alpha.gather(0, targets)
        
        # Compute focal loss
        focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def setup_logging(log_dir: str, experiment_name: str) -> logging.Logger:
    """Setup logging configuration"""
    
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(
        log_path / f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Configuration saved to {save_path}")


def create_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create necessary directories from config"""
    
    paths = {}
    
    if 'paths' in config:
        for path_name, path_value in config['paths'].items():
            path = Path(path_value)
            path.mkdir(parents=True, exist_ok=True)
            paths[path_name] = path
    
    # Additional standard directories
    standard_dirs = ['checkpoints', 'logs', 'results', 'plots']
    for dir_name in standard_dirs:
        if dir_name not in paths:
            path = Path(f"./{dir_name}")
            path.mkdir(parents=True, exist_ok=True)
            paths[dir_name] = path
    
    return paths


def get_device() -> torch.device:
    """Get the best available device (CUDA if available, else CPU)"""
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def get_device_info() -> Dict[str, Any]:
    """Get device information"""
    
    device_info = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'cuda_available': torch.cuda.is_available(),
        'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        try:
            device_info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory / 1e9,
                'gpu_memory_allocated': torch.cuda.memory_allocated(0) / 1e9,
                'gpu_memory_reserved': torch.cuda.memory_reserved(0) / 1e9
            })
        except Exception as e:
            print(f"Warning: Could not get CUDA device info: {e}")
            device_info.update({
                'gpu_name': 'CUDA Device (info unavailable)',
                'gpu_memory_total': 0.0,
                'gpu_memory_allocated': 0.0,
                'gpu_memory_reserved': 0.0
            })
    
    return device_info


def activation_check():
    """Check if virtual environment is activated and dependencies are available"""
    
    print("Environment Activation Check:")
    print("=" * 40)
    
    # Check virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Virtual environment activated")
        print(f"   Python executable: {sys.executable}")
        print(f"   Virtual env path: {sys.prefix}")
    else:
        print("Virtual environment may not be activated")
        print(f"   Python executable: {sys.executable}")
    
    # Check key dependencies
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers', 
        'datasets': 'Datasets',
        'librosa': 'Librosa',
        'sklearn': 'Scikit-learn',
        'yaml': 'YAML',
        'pytorch_lightning': 'PyTorch Lightning'
    }
    
    missing = []
    for dep, name in dependencies.items():
        try:
            if dep == 'sklearn':
                import sklearn
            elif dep == 'yaml':
                import yaml  
            elif dep == 'pytorch_lightning':
                import pytorch_lightning as pl
            else:
                __import__(dep)
            print(f"   {name}")
        except ImportError:
            print(f"   {name} - Missing")
            missing.append(dep)
    
    # Check device
    device_info = get_device_info()
    try:
        if device_info['cuda_available'] and device_info['num_gpus'] > 0:
            print(f"   CUDA available: {device_info['num_gpus']} GPU(s)")
            if 'gpu_name' in device_info:
                print(f"   GPU: {device_info['gpu_name']}")
        else:
            print("   CUDA not available - using CPU")
    except Exception as e:
        print(f"   CUDA check failed: {e} - using CPU")
    
    print("=" * 40)
    
    if missing:
        print(f"Missing dependencies: {missing}")
        return False
    else:
        print("All dependencies available")
        return True


def print_model_info(model: torch.nn.Module):
    """Print model architecture and parameter information"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print("=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    print(f"Parameter Count:")
    print(f"   Total parameters:     {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Frozen parameters:    {frozen_params:,}")
    print(f"   Trainable ratio:      {trainable_params/total_params:.1%}")
    
    print(f"\nModel Structure:")
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"   {name:20s}: {num_params:>8,} params ({trainable:>8,} trainable)")
    
    print("=" * 60)


def create_attention_heatmap(
    attention_weights: torch.Tensor,
    tokens: List[str],
    save_path: Optional[str] = None,
    title: str = "Attention Heatmap"
) -> plt.Figure:
    """Create attention heatmap visualization"""
    
    # Convert to numpy and get mean across heads if needed
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    if attention_weights.ndim == 4:  # (batch, heads, seq, seq)
        attention_weights = attention_weights[0].mean(0)  # Average across heads
    elif attention_weights.ndim == 3:  # (heads, seq, seq)
        attention_weights = attention_weights.mean(0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        cbar=True,
        square=True,
        ax=ax
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Key Tokens')
    ax.set_ylabel('Query Tokens')
    
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention heatmap saved to {save_path}")
    
    return fig


def create_feature_importance_plot(
    importance_scores: Union[torch.Tensor, np.ndarray],
    feature_names: List[str],
    top_k: int = 20,
    save_path: Optional[str] = None,
    title: str = "Feature Importance"
) -> plt.Figure:
    """Create feature importance bar plot"""
    
    if isinstance(importance_scores, torch.Tensor):
        importance_scores = importance_scores.detach().cpu().numpy()
    
    # Get top k features
    if len(importance_scores) > top_k:
        top_indices = np.argsort(np.abs(importance_scores))[-top_k:]
        importance_scores = importance_scores[top_indices]
        feature_names = [feature_names[i] for i in top_indices]
    
    # Sort for plotting
    sorted_indices = np.argsort(importance_scores)
    importance_scores = importance_scores[sorted_indices]
    feature_names = [feature_names[i] for i in sorted_indices]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
    
    colors = ['red' if x < 0 else 'green' for x in importance_scores]
    bars = ax.barh(range(len(importance_scores)), importance_scores, color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Importance Score')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, importance_scores)):
        ax.text(score + (0.01 if score > 0 else -0.01), i, f'{score:.3f}',
                ha='left' if score > 0 else 'right', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    return fig


def create_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Dict[str, List[float]] = None,
    val_metrics: Dict[str, List[float]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """Create training curves plot"""
    
    epochs = range(1, len(train_losses) + 1)
    
    # Determine number of subplots
    n_plots = 1  # Loss plot
    if train_metrics and val_metrics:
        n_plots += len(train_metrics)
    
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Metrics plots
    if train_metrics and val_metrics:
        for i, metric_name in enumerate(train_metrics.keys(), 1):
            if i < len(axes):
                axes[i].plot(epochs, train_metrics[metric_name], 'b-', 
                           label=f'Train {metric_name}', linewidth=2)
                axes[i].plot(epochs, val_metrics[metric_name], 'r-', 
                           label=f'Val {metric_name}', linewidth=2)
                axes[i].set_title(f'{metric_name.capitalize()} Score')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric_name.capitalize())
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig


def format_time(seconds: float) -> str:
    """Format time duration"""
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


def save_predictions(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    metadata: Optional[Dict] = None,
    save_path: str = "predictions.json"
):
    """Save model predictions to file"""
    
    results = {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist(),
        'labels': labels.tolist(),
        'metadata': metadata or {}
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Predictions saved to {save_path}")


def create_mask_from_lengths(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create attention mask from sequence lengths"""
    
    if max_len is None:
        max_len = lengths.max().item()
    
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    
    return mask.float()


def pad_and_mask_sequences(sequences: List[torch.Tensor], padding_value: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences and create attention masks"""
    
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value)
    
    # Create attention masks
    lengths = torch.tensor([len(seq) for seq in sequences])
    attention_mask = create_mask_from_lengths(lengths, padded_sequences.size(1))
    
    return padded_sequences, attention_mask


def compute_class_weights(labels: np.ndarray, method: str = 'balanced') -> torch.Tensor:
    """Compute class weights for imbalanced datasets"""
    
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(labels)
    
    if method == 'balanced':
        weights = compute_class_weight('balanced', classes=classes, y=labels)
    else:
        # Manual inverse frequency
        class_counts = np.bincount(labels)
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * len(classes)  # Normalize
    
    return torch.FloatTensor(weights)


def print_gpu_memory_usage():
    """Print current GPU memory usage"""
    
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"GPU Memory Usage:")
        print(f"   Allocated: {allocated:.2f} GB")
        print(f"   Reserved:  {reserved:.2f} GB") 
        print(f"   Total:     {total:.2f} GB")
        print(f"   Usage:     {allocated/total:.1%}")
    else:
        print("CUDA not available")


def test_common_utils():
    """Test common utilities"""
    
    print("🧪 Testing common utilities...")
    
    # Test device info
    device_info = get_device_info()
    print(f"Device info: {device_info}")
    
    # Test seed setting
    set_seed(42)
    
    # Test GPU memory
    print_gpu_memory_usage()
    
    # Test time formatting
    print(f"Time format test: {format_time(3725)}")
    
    # Test mask creation
    lengths = torch.tensor([5, 3, 7])
    mask = create_mask_from_lengths(lengths, max_len=8)
    print(f"Mask shape: {mask.shape}")
    print(f"Mask:\n{mask}")
    
    # Test sequence padding
    seqs = [torch.randn(3, 4), torch.randn(5, 4), torch.randn(2, 4)]
    padded, mask = pad_and_mask_sequences(seqs)
    print(f"Padded shape: {padded.shape}, Mask shape: {mask.shape}")
    
    print("Common utilities test completed!")


if __name__ == "__main__":
    test_common_utils()
