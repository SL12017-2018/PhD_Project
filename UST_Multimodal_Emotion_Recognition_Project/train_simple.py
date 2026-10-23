"""
Simple Training Script for IEMOCAP MemoCMT (Without PyTorch Lightning)
Compatible with standard PyTorch training loop
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from transformers import AutoTokenizer, Wav2Vec2Processor
import numpy as np
from tqdm import tqdm

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our modules
from data.iemocap import create_dataloaders
from models.model import IEMOCAPMemoCMTModel, load_config
from utils.common import set_seed, setup_logging, create_directories, print_model_info, format_time
from utils.metrics import IEMOCAPMetrics

warnings.filterwarnings('ignore')


class SimpleTrainer:
    
    def __init__(
        self,
        model: IEMOCAPMemoCMTModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Training configuration
        training_config = config.get('training', {})
        self.num_epochs = training_config.get('num_epochs', 30)
        self.learning_rate = training_config.get('learning_rate', 1e-4)
        self.encoder_lr = training_config.get('encoder_lr', 5e-6)
        self.weight_decay = training_config.get('weight_decay', 0.01)
        self.gradient_clip_norm = training_config.get('gradient_clip_norm', 1.0)
        self.gradient_accumulation = training_config.get('gradient_accumulation', 1)
        
        # Mixed precision training
        self.use_amp = training_config.get('use_amp', True)
        if self.use_amp:
            self.scaler = GradScaler()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss function
        self.criterion = self._create_loss_function()
        
        # Metrics
        emotion_labels = config.get('dataset', {}).get('emotion_labels', {})
        self.metrics_calc = IEMOCAPMetrics(emotion_labels)
        
        # Training tracking
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.patience = training_config.get('patience', 7)
        
        # Storage for training history
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = {'accuracy': [], 'weighted_f1': []}
        self.val_metrics = {'accuracy': [], 'weighted_f1': []}
        
        print(f"Simple trainer initialized on {device}")
        print(f"   Epochs: {self.num_epochs}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Batch size: {len(train_loader.dataset) // len(train_loader)}")
        print(f"   Mixed precision: {self.use_amp}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with different learning rates for encoders"""
        
        # Separate parameters for encoders and rest of the model
        encoder_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'text_encoder' in name or 'audio_encoder' in name:
                    encoder_params.append(param)
                else:
                    other_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {'params': other_params, 'lr': self.learning_rate, 'weight_decay': self.weight_decay}
        ]
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params, 
                'lr': self.encoder_lr, 
                'weight_decay': self.weight_decay * 0.1
            })
        
        optimizer = AdamW(param_groups, eps=1e-8)
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        
        total_steps = len(self.train_loader) * self.num_epochs // self.gradient_accumulation
        
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[group['lr'] for group in self.optimizer.param_groups],
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=1000.0
        )
        
        return scheduler
    
    def _create_loss_function(self):
        """Create loss function"""
        
        training_config = self.config.get('training', {})
        loss_type = training_config.get('loss_type', 'focal')
        
        if loss_type == 'focal':
            return FocalLoss(
                alpha=training_config.get('focal_alpha', 0.25),
                gamma=training_config.get('focal_gamma', 2.0)
            )
        elif loss_type == 'label_smoothing':
            return nn.CrossEntropyLoss(
                label_smoothing=training_config.get('label_smoothing', 0.1)
            )
        else:
            return nn.CrossEntropyLoss()
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        
        self.model.train()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            with autocast(enabled=self.use_amp):
                outputs = self.model(batch, return_explanations=False, return_uncertainty=False)
                loss = self.criterion(outputs['logits'], batch['labels'])
                loss = loss / self.gradient_accumulation
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
            
            # Track metrics
            running_loss += loss.item() * self.gradient_accumulation
            predictions = outputs['predictions'].cpu().numpy()
            labels = batch['labels'].cpu().numpy()
            
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item() * self.gradient_accumulation:.4f}",
                'LR': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # Calculate F1 scores
        from sklearn.metrics import f1_score
        weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1
        }
        
        return epoch_loss, metrics
    
    def validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        
        self.model.eval()
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validating", leave=False)
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                with autocast(enabled=self.use_amp):
                    outputs = self.model(batch, return_explanations=False, return_uncertainty=False)
                    loss = self.criterion(outputs['logits'], batch['labels'])
                
                # Track metrics
                running_loss += loss.item()
                predictions = outputs['predictions'].cpu().numpy()
                probabilities = outputs['probabilities'].cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_probabilities.append(probabilities)
                all_labels.extend(labels)
                
                # Update progress bar
                progress_bar.set_postfix({'Val Loss': f"{loss.item():.4f}"})
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.val_loader)
        
        all_predictions = np.array(all_predictions)
        all_probabilities = np.vstack(all_probabilities)
        all_labels = np.array(all_labels)
        
        # Comprehensive metrics
        metrics = self.metrics_calc.compute_all_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        return epoch_loss, metrics
    
    def train(self, save_dir: str = "./checkpoints"):
        """Main training loop"""
        
        print(f"Starting training for {self.num_epochs} epochs")
        print("=" * 60)
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        start_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 40)
            
            # Training phase
            train_loss, train_metrics = self.train_epoch()
            
            # Validation phase
            val_loss, val_metrics = self.validate_epoch()
            
            # Store history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            for metric_name in self.train_metrics.keys():
                if metric_name in train_metrics:
                    self.train_metrics[metric_name].append(train_metrics[metric_name])
                if metric_name in val_metrics:
                    self.val_metrics[metric_name].append(val_metrics[metric_name])
            
            # Print epoch results
            epoch_time = time.time() - epoch_start
            print(f"Epoch time: {format_time(epoch_time)}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"Train F1: {train_metrics['weighted_f1']:.4f} | Val F1: {val_metrics['weighted_f1']:.4f}")
            
            # Check for best model
            current_val_f1 = val_metrics['weighted_f1']
            if current_val_f1 > self.best_val_f1:
                self.best_val_f1 = current_val_f1
                self.patience_counter = 0
                
                # Save best model
                best_model_path = save_path / "best_model.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_f1': self.best_val_f1,
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                    'config': self.config
                }, best_model_path)
                
                print(f"New best model saved! F1: {current_val_f1:.4f}")
            else:
                self.patience_counter += 1
                print(f"Patience: {self.patience_counter}/{self.patience}")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = save_path / f"checkpoint_epoch_{epoch + 1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'train_losses': self.train_losses,
                    'val_losses': self.val_losses,
                }, checkpoint_path)
        
        # Training completed
        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        
        return save_path / "best_model.pth"
    
    def test(self, model_path: Optional[str] = None) -> Dict[str, float]:
        """Test the model"""
        
        print("\nTesting model...")
        
        # Load best model if path provided
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.test_loader, desc="Testing")
            
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(batch, return_explanations=False, return_uncertainty=False)
                
                predictions = outputs['predictions'].cpu().numpy()
                probabilities = outputs['probabilities'].cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                
                all_predictions.extend(predictions)
                all_probabilities.append(probabilities)
                all_labels.extend(labels)
        
        # Calculate comprehensive metrics
        all_predictions = np.array(all_predictions)
        all_probabilities = np.vstack(all_probabilities)
        all_labels = np.array(all_labels)
        
        test_metrics = self.metrics_calc.compute_all_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        # Print test results
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        self.metrics_calc.print_metrics_summary(test_metrics)
        
        return test_metrics


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def train_model_simple(config_path: str, args: argparse.Namespace):
    """Main training function using simple trainer"""
    
    print("Starting IEMOCAP MemoCMT Training (Simple Mode)")
    print("=" * 60)
    
    # Load configuration
    config = load_config(config_path)
    
    # Override config with command line arguments
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    paths = create_directories(config)
    experiment_name = f"iemocap_memocmt_{args.experiment_name}"
    
    # Initialize tokenizer and processor
    text_config = config.get('model', {}).get('text', {})
    audio_config = config.get('model', {}).get('audio', {})
    
    print("Loading pretrained models...")
    tokenizer = AutoTokenizer.from_pretrained(
        text_config.get('model_name', 'j-hartmann/emotion-english-distilroberta-base')
    )
    
    processor = Wav2Vec2Processor.from_pretrained(
        audio_config.get('model_name', 'superb/wav2vec2-base-superb-er')
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        root_path=config.get('dataset', {}).get('root_path', '../iemocap_dataset'),
        tokenizer=tokenizer,
        processor=processor,
        batch_size=config['training']['batch_size'],
        num_workers=config.get('dataloader', {}).get('num_workers', 4),
        sample_rate=config.get('dataset', {}).get('sample_rate', 16000)
    )
    
    # Create model
    print("Creating model...")
    model = IEMOCAPMemoCMTModel(config, tokenizer, processor)
    print_model_info(model)
    
    # Create trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = SimpleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Train model
    checkpoint_dir = paths.get('checkpoints', './checkpoints') / experiment_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = trainer.train(str(checkpoint_dir))
    
    # Unfreeze encoders and continue training if requested
    if args.finetune_encoders:
        print("\nPhase 2: Fine-tuning with unfrozen encoders")
        
        # Unfreeze encoders
        model.unfreeze_encoders()
        
        # Create new trainer with lower learning rates
        finetune_config = config.copy()
        finetune_config['training']['learning_rate'] *= 0.1
        finetune_config['training']['encoder_lr'] *= 0.1
        finetune_config['training']['num_epochs'] = 5
        
        finetune_trainer = SimpleTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=finetune_config,
            device=device
        )
        
        # Continue training
        best_model_path = finetune_trainer.train(str(checkpoint_dir))
    
    # Test final model
    test_metrics = trainer.test(str(best_model_path))
    
    print(f"\nTraining completed!")
    print(f"Best model: {best_model_path}")
    print(f"Checkpoints: {checkpoint_dir}")
    
    return model, trainer, best_model_path


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Train IEMOCAP MemoCMT Model (Simple)')
    
    parser.add_argument('--config', type=str, default='configs/iemocap_memocmt.yaml')
    parser.add_argument('--experiment_name', type=str, default='baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--finetune_encoders', action='store_true')
    
    args = parser.parse_args()
    
    # Check config file
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Start training
    try:
        model, trainer, best_model_path = train_model_simple(str(config_path), args)
        print(f"\nTraining successful!")
        
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
