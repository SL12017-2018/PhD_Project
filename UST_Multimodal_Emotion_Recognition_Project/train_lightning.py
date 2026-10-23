
"""
PyTorch Lightning training script for IEMOCAP MemoCMT model
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our modules
from data.iemocap import IEMOCAPDataset, collate_dialogue_batch
from models.model import IEMOCAPMemoCMTModel
from utils.common import set_seed, get_device, activation_check
from utils.metrics import compute_emotion_metrics

from transformers import AutoTokenizer, Wav2Vec2FeatureExtractor


class IEMOCAPLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for IEMOCAP MemoCMT model"""
    
    def __init__(self, config, tokenizer, processor):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        
        # Create model
        self.model = IEMOCAPMemoCMTModel(config, tokenizer, processor)
        
        # Enable CUDA optimizations if available
        if torch.cuda.is_available():
            # Enable mixed precision training
            self.automatic_optimization = True
            # Set memory format for optimal CUDA performance
            if hasattr(self.model, 'to_memory_format'):
                self.model = self.model.to(memory_format=torch.channels_last)
        
        # Loss function
        if config['training']['loss']['type'] == 'focal':
            from utils.common import FocalLoss
            self.criterion = FocalLoss(
                alpha=config['training']['loss']['focal_alpha'],
                gamma=config['training']['loss']['focal_gamma'],
                num_classes=config['dataset']['num_classes']  # Fixed: should be dataset not model
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Metrics storage
        self.val_predictions = []
        self.val_labels = []
        self.test_predictions = []
        self.test_labels = []
    
    def forward(self, batch):
        return self.model(batch)
    
    def training_step(self, batch, batch_idx):
        # CUDA memory management
        if torch.cuda.is_available() and batch_idx % 100 == 0:
            torch.cuda.empty_cache()  # Clear cache periodically
        
        outputs = self(batch)
        loss = self.criterion(outputs['logits'], batch['labels'])
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log learning rates
        for i, optimizer in enumerate(self.optimizers()):
            for j, param_group in enumerate(optimizer.param_groups):
                self.log(f'lr_optimizer_{i}_group_{j}', param_group['lr'], on_step=True)
        
        # Monitor GPU memory usage for RTX 4080 (every 50 steps)
        if batch_idx % 50 == 0 and torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1e9  # GB
            gpu_reserved = torch.cuda.memory_reserved() / 1e9  # GB
            gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()) / 1e9
            
            self.log('gpu_memory_gb', gpu_memory, on_step=True)
            self.log('gpu_reserved_gb', gpu_reserved, on_step=True)
            self.log('gpu_free_gb', gpu_free, on_step=True)
            
            # Log GPU utilization percentage
            gpu_util = (gpu_reserved / 16.0) * 100  # RTX 4080 has 16GB
            self.log('gpu_utilization_%', gpu_util, on_step=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs['logits'], batch['labels'])
        
        # Store predictions for epoch-end metrics
        self.val_predictions.extend(outputs['predictions'].cpu().numpy())
        self.val_labels.extend(batch['labels'].cpu().numpy())
        
        # Log loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_predictions) > 0:
            # Compute metrics
            metrics = compute_emotion_metrics(
                self.val_labels, 
                self.val_predictions,
                self.config['dataset']['emotion_labels']
            )
            
            # Log metrics
            for metric_name, value in metrics.items():
                self.log(f'val_{metric_name}', value, on_epoch=True, prog_bar=True)
            
            # Clear for next epoch
            self.val_predictions = []
            self.val_labels = []
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs['logits'], batch['labels'])
        
        # Store predictions for epoch-end metrics
        self.test_predictions.extend(outputs['predictions'].cpu().numpy())
        self.test_labels.extend(batch['labels'].cpu().numpy())
        
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        return loss
    
    def on_test_epoch_end(self):
        if len(self.test_predictions) > 0:
            # Compute metrics
            metrics = compute_emotion_metrics(
                self.test_labels,
                self.test_predictions, 
                self.config['dataset']['emotion_labels']
            )
            
            # Log metrics
            for metric_name, value in metrics.items():
                self.log(f'test_{metric_name}', value, on_epoch=True)
            
            # Print final results
            print(f"\n{'='*50}")
            print("Final Test Results:")
            print("=" * 50)
            for metric_name, value in metrics.items():
                print(f"   {metric_name}: {value:.4f}")
    
    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        
        # Separate encoder and other parameters
        encoder_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'text_encoder' in name or 'audio_encoder' in name:
                encoder_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        if len(encoder_params) > 0:
            param_groups.append({
                'params': encoder_params,
                'lr': self.config['training']['optimizer']['encoder_lr'],
                'weight_decay': self.config['training']['optimizer']['weight_decay']
            })
        
        if len(other_params) > 0:
            param_groups.append({
                'params': other_params,
                'lr': self.config['training']['optimizer']['lr'],
                'weight_decay': self.config['training']['optimizer']['weight_decay']
            })
        
        # Optimizer
        if self.config['training']['optimizer']['type'] == 'adamw':
            optimizer = torch.optim.AdamW(param_groups)
        elif self.config['training']['optimizer']['type'] == 'adam':
            optimizer = torch.optim.Adam(param_groups)
        else:
            raise ValueError(f"Unknown optimizer: {self.config['training']['optimizer']['type']}")
        
        # Scheduler
        scheduler_config = None
        if self.config['training']['scheduler']['type'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['max_epochs'],
                eta_min=self.config['training']['scheduler']['min_lr']
            )
            scheduler_config = {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        elif self.config['training']['scheduler']['type'] == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.config['training']['scheduler']['factor'],
                patience=self.config['training']['scheduler']['patience'],
                min_lr=self.config['training']['scheduler']['min_lr']
            )
            scheduler_config = {
                'scheduler': scheduler,
                'monitor': 'val_weighted_f1',
                'interval': 'epoch',
                'frequency': 1
            }
        
        if scheduler_config:
            return [optimizer], [scheduler_config]
        else:
            return optimizer


class IEMOCAPDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for IEMOCAP"""
    
    def __init__(self, config, tokenizer, processor):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.data_path = config['dataset']['data_path']
        
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = IEMOCAPDataset(
                root_path=self.data_path,
                split='train',
                max_length=self.config['dataset']['max_length'],
                max_audio_length=self.config['dataset']['max_audio_length']
            )
            
            # Use part of train for validation if no separate val split
            # For now, we'll create a simple train/val split
            train_size = int(0.8 * len(self.train_dataset))
            val_size = len(self.train_dataset) - train_size
            
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(
                self.train_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
        if stage == 'test' or stage is None:
            # For testing, we can use a held-out portion or the validation set
            self.test_dataset = self.val_dataset if hasattr(self, 'val_dataset') else None
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True,
            collate_fn=lambda batch: collate_dialogue_batch(batch, self.tokenizer, self.processor)
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True,
            collate_fn=lambda batch: collate_dialogue_batch(batch, self.tokenizer, self.processor)
        )
    
    def test_dataloader(self):
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['dataset']['num_workers'],
            pin_memory=True,
            collate_fn=lambda batch: collate_dialogue_batch(batch, self.tokenizer, self.processor)
        )


def main():
    parser = argparse.ArgumentParser(description='Train IEMOCAP MemoCMT with PyTorch Lightning')
    parser.add_argument('--config', type=str, default='configs/iemocap_memocmt.yaml',
                        help='Path to config file')
    parser.add_argument('--gpus', type=int, default=-1,
                        help='Number of GPUs to use (-1 for all available)')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        choices=['32', '16-mixed', 'bf16-mixed'],
                        help='Training precision')
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator type')
    parser.add_argument('--log_name', type=str, default='iemocap_memocmt',
                        help='Experiment name for logging')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    args = parser.parse_args()
    
    print("IEMOCAP MemoCMT Training with PyTorch Lightning")
    print("=" * 60)
    
    # Check environment
    activation_check()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Configuration loaded from {args.config}")
    
    # Set seed
    set_seed(config['training']['seed'])
    
    # Create tokenizers
    print("Loading tokenizers and processors...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['text']['model_name'])
    processor = Wav2Vec2FeatureExtractor.from_pretrained(config['model']['audio']['model_name'])
    
    # Create data module
    print("Creating data module...")
    data_module = IEMOCAPDataModule(config, tokenizer, processor)
    
    # Create model
    print("Creating model...")
    model = IEMOCAPLightningModule(config, tokenizer, processor)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['training']['checkpoint_dir'],
        filename='{epoch:02d}-{val_weighted_f1:.4f}',
        monitor='val_weighted_f1',
        mode='max',
        save_top_k=3,
        save_last=True,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if config['training']['early_stopping']['enabled']:
        early_stop_callback = EarlyStopping(
            monitor='val_weighted_f1',
            patience=config['training']['early_stopping']['patience'],
            mode='max',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    # Setup loggers
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config['training']['log_dir'],
        name=args.log_name,
        version=None
    )
    loggers.append(tb_logger)
    
    # Weights & Biases logger (optional)
    if args.use_wandb:
        try:
            wandb_logger = WandbLogger(
                project='iemocap-memocmt',
                name=args.log_name,
                config=config
            )
            loggers.append(wandb_logger)
            print("Weights & Biases logging enabled")
        except Exception as e:
            print(f"Could not initialize W&B: {e}")
    
    # Setup trainer - Optimized for RTX 4080 16GB
    # Force CPU first to avoid CUDA initialization issues
    import os
    os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '1')
    
    # Check if CUDA is truly available without triggering initialization errors
    use_gpu = False
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache
            _ = torch.tensor([1.0], device='cuda')  # Test CUDA
            use_gpu = True
            print("CUDA initialized successfully")
    except Exception as e:
        print(f"CUDA error: {e}. Using CPU training.")
        use_gpu = False
    
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if use_gpu else 'cpu',
        devices=1 if use_gpu else 'auto',
        precision='16-mixed' if use_gpu else 32,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=config['training']['gradient_clip_val'],
        accumulate_grad_batches=config['training']['gradient_accumulation_steps'],
        val_check_interval=config['training']['val_check_interval'],
        log_every_n_steps=config['training']['log_every_n_steps'],
        deterministic=False,  # Set to False for better GPU performance
        enable_progress_bar=True,
        enable_model_summary=True,
        strategy='auto',
        # GPU optimizations for RTX 4080
        sync_batchnorm=False,
        detect_anomaly=False,
        benchmark=use_gpu,  # Enable CUDNN benchmark for consistent input shapes
    )
    
    print("Starting training...")
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Test the model
    if data_module.test_dataloader() is not None:
        print("Running final test...")
        trainer.test(model, data_module, ckpt_path='best')
    
    print("Training completed!")
    print(f"Checkpoints saved in: {config['training']['checkpoint_dir']}")
    print(f"Logs saved in: {config['training']['log_dir']}")


if __name__ == '__main__':
    main()
