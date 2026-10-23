
"""
Full training script for IEMOCAP emotion recognition
Uses the complete dataset with progressive encoder unfreezing
"""

import os
import sys
import time
import gc
import yaml
import torch
import torch.nn as nn
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, Any, Optional

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.iemocap import IEMOCAPDataset
from models.model import IEMOCAPMemoCMTModel
from utils.collate import iemocap_collate_fn
from utils.results_saver import ResultsSaver

def get_system_info():
    """Get current system resource information"""
    # GPU info
    gpus = GPUtil.getGPUs()
    gpu_info = {
        'available': torch.cuda.is_available(),
        'count': len(gpus),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None',
        'memory_total': f"{gpus[0].memoryTotal}MB" if gpus else 'N/A',
        'memory_free': f"{gpus[0].memoryFree}MB" if gpus else 'N/A'
    }

    # CPU/RAM info
    cpu_info = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_total': f"{psutil.virtual_memory().total // (1024**3)}GB",
        'memory_available': f"{psutil.virtual_memory().available // (1024**3)}GB",
        'memory_percent': psutil.virtual_memory().percent
    }

    return gpu_info, cpu_info

def monitor_resources():
    """Display current resource usage"""
    gpu_info, cpu_info = get_system_info()

    print(f"GPU memory: {gpu_info['memory_free']} / {gpu_info['memory_total']}")
    print(f"CPU usage: {cpu_info['cpu_percent']:.1f}% | RAM usage: {cpu_info['memory_percent']:.1f}%")

def cleanup_memory():
    """Clean up GPU and system memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

class ResultsLoggingCallback(L.Callback):
    """Custom callback to save training metrics"""

    def __init__(self, results_saver: ResultsSaver):
        super().__init__()
        self.results_saver = results_saver

    def on_train_epoch_end(self, trainer, pl_module):
        """Save metrics at the end of each training epoch"""
        epoch_metrics = {}

        # Extract metrics from trainer
        if hasattr(trainer, 'logged_metrics') and trainer.logged_metrics:
            for key, value in trainer.logged_metrics.items():
                try:
                    if isinstance(value, torch.Tensor):
                        epoch_metrics[key] = value.item()
                    elif isinstance(value, (int, float)):
                        epoch_metrics[key] = value
                    else:
                        epoch_metrics[key] = float(value) if value is not None else None
                except (ValueError, TypeError):
                    print(f"Could not convert metric {key}: {value}")
                    continue

        # Add learning rate
        if trainer.optimizers:
            try:
                lr = trainer.optimizers[0].param_groups[0]['lr']
                epoch_metrics['learning_rate'] = lr
            except (IndexError, KeyError):
                pass

        # Add epoch number for reference
        epoch_metrics['epoch'] = trainer.current_epoch

        # Log to results saver
        try:
            self.results_saver.log_epoch_metrics(trainer.current_epoch, epoch_metrics)
        except Exception as e:
            print(f"Warning: Could not log epoch metrics: {e}")
class ProgressiveUnfreezingCallback(L.Callback):
    """Gradually unfreeze encoder layers during training"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.text_schedule = config.get('model', {}).get('text', {}).get('unfreeze_schedule', {})
        self.audio_schedule = config.get('model', {}).get('audio', {}).get('unfreeze_schedule', {})

    def on_train_epoch_start(self, trainer, pl_module):
        """Check if we need to unfreeze layers at the start of each epoch"""
        current_epoch = trainer.current_epoch

        # Check text encoder unfreezing
        for milestone, layers in self.text_schedule.items():
            epoch_num = int(milestone.split('_')[1])
            if current_epoch == epoch_num:
                print(f"\nUnfreezing top {layers} layers of text encoder at epoch {current_epoch}")
                pl_module.text_encoder.unfreeze_top_layers(layers)

        # Check audio encoder unfreezing
        for milestone, layers in self.audio_schedule.items():
            epoch_num = int(milestone.split('_')[1])
            if current_epoch == epoch_num:
                print(f"Unfreezing top {layers} layers of audio encoder at epoch {current_epoch}")
                pl_module.audio_encoder.unfreeze_top_layers(layers)

def create_data_split(dataset, train_ratio=0.8):
    """Create train/val split from the dataset"""
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = total_size - train_size
    
    # Create deterministic split
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset

def main():
    """Main training function"""

    print("Starting full IEMOCAP training with R-AIME")
    print("=" * 60)

    # Record start time
    start_time = time.time()

    # Initial resource status
    print("Checking system resources before starting:")
    monitor_resources()

    # Load configuration
    config_path = "configs/conservative_test.yaml"
    print(f"Loading configuration from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize results saver
    experiment_name = config['logging']['experiment_name']
    results_saver = ResultsSaver(
        results_dir=config['evaluation']['results_dir'],
        experiment_name=experiment_name
    )

    # Extract key parameters
    batch_size = config['training']['batch_size']
    max_epochs = config['training']['max_epochs']
    max_audio_len_s = config['dataset']['max_audio_len_s']
    max_length = config['dataset']['max_length']

    print(f"Batch size: {batch_size}")
    print(f"Max epochs: {max_epochs}")
    print(f"Max audio length: {max_audio_len_s}s")
    print(f"Max text length: {max_length}")

    # Set random seed
    L.seed_everything(config['training']['seed'])
    print(f"Random seed set to {config['training']['seed']}")

    # GPU setup
    if torch.cuda.is_available():
        print(f"CUDA is available: True")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        # Set tensor core precision for better performance
        torch.set_float32_matmul_precision('medium')

    try:
        print("\nCreating full dataset...")

        # Create full dataset
        full_dataset = IEMOCAPDataset(
            root_path=config['dataset']['data_path'],
            split="train",
            sample_rate=config['dataset']['sample_rate'],
            max_audio_len_s=max_audio_len_s,
            max_length=max_length,
            context_window=config['dataset']['context_window']
        )

        print(f"Full dataset: {len(full_dataset)} samples")

        # Create train/val split
        train_dataset, val_dataset = create_data_split(full_dataset, train_ratio=0.8)
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")

        print("\nResource Status after Dataset Creation:")
        monitor_resources()

        # Create data loaders
        print("\nCreating data loaders...")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config['dataloader']['num_workers'],
            pin_memory=config['dataloader']['pin_memory'],
            collate_fn=iemocap_collate_fn,
            persistent_workers=config['dataloader']['persistent_workers'],
            prefetch_factor=config['dataloader']['prefetch_factor']
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config['dataloader']['num_workers'],
            pin_memory=config['dataloader']['pin_memory'],
            collate_fn=iemocap_collate_fn,
            persistent_workers=config['dataloader']['persistent_workers'],
            prefetch_factor=config['dataloader']['prefetch_factor']
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Test single batch
        print("\nTesting single batch...")
        batch = next(iter(train_loader))
        print(f"Batch keys: {list(batch.keys())}")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} {value.dtype}")

        print("\nResource Status after Data Loading:")
        monitor_resources()

        # Create model
        print("\nCreating model...")

        model = IEMOCAPMemoCMTModel(
            config=config
        )

        print(f"Model created successfully")

        # Verify R-AIME components are enabled
        print("\nR-AIME Architecture Verification:")
        print(f"Fusion Agent: {'Enabled' if hasattr(model.raime_head, 'fusion_agent') else 'Disabled'}")
        print(f"Decision Agent: {'Enabled' if hasattr(model.raime_head, 'decision_agent') else 'Disabled'}")
        print(f"Temperature Scaling: {'Enabled' if model.raime_head.decision_agent.use_temperature else 'Disabled'}")
        print(f"Explainability Agent: {'Enabled' if hasattr(model.raime_head, 'explainability_agent') else 'Disabled'}")
        print(f"MemoCMT: {'Enabled' if hasattr(model, 'memocmt') else 'Disabled'}")

        if hasattr(model.raime_head, 'explainability_agent'):
            print("Full R-AIME architecture is ACTIVE!")
        else:
            print("Partial R-AIME architecture")

        # Test R-AIME functionality with a sample batch
        print("\nTesting R-AIME components...")
        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            test_outputs = model(sample_batch, return_explanations=True, return_uncertainty=True)

        raime_components = []
        if 'feature_attribution' in test_outputs:
            raime_components.append("Feature Attribution")
        if 'concept_activations' in test_outputs:
            raime_components.append("Concept Activations")
        if 'uncertainty' in test_outputs:
            raime_components.append("Uncertainty Estimation")
        if 'confidence' in test_outputs:
            raime_components.append("Confidence Estimation")
        if 'temperature' in test_outputs and test_outputs['temperature'] is not None:
            raime_components.append("Temperature Scaling")

        print(f"Active R-AIME components: {', '.join(raime_components)}")
        print(f"Output keys: {list(test_outputs.keys())}")


        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        print("\nResource Status after Model Creation:")
        monitor_resources()

        # Setup callbacks
        callbacks = []

        # Results logging callback
        results_callback = ResultsLoggingCallback(results_saver)
        callbacks.append(results_callback)

        # Progressive unfreezing
        callbacks.append(ProgressiveUnfreezingCallback(config))
        
        # Model checkpoint
        checkpoint_callback = ModelCheckpoint(
            dirpath=config['training']['checkpoint_dir'],
            filename='best-{epoch:02d}-{val_weighted_f1:.3f}',
            monitor=config['checkpoints']['monitor'],
            mode=config['checkpoints']['mode'],
            save_top_k=config['checkpoints']['save_top_k'],
            save_last=config['checkpoints']['save_last']
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        if config['training']['early_stopping']['enabled']:
            early_stop_callback = EarlyStopping(
                monitor=config['training']['early_stopping']['monitor'],
                mode=config['training']['early_stopping']['mode'],
                patience=config['training']['early_stopping']['patience'],
                min_delta=config['training']['early_stopping']['min_delta']
            )
            callbacks.append(early_stop_callback)
        
        # Learning rate monitor
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)
        
        # Setup logger
        logger = TensorBoardLogger(
            save_dir=config['training']['log_dir'],
            name=config['logging']['experiment_name'],
            version=f"run_{int(time.time())}"
        )
        
        # Create trainer
        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            precision=config['hardware']['precision'],
            gradient_clip_val=config['training']['gradient_clip_val'],
            accumulate_grad_batches=config['training']['gradient_accumulation_steps'],
            val_check_interval=config['training']['val_check_interval'],
            log_every_n_steps=config['training']['log_every_n_steps'],
            callbacks=callbacks,
            logger=logger,
            enable_checkpointing=config['training']['enable_checkpointing'],
            deterministic=True
        )

        print(f"\nStarting training for {max_epochs} epochs...")
        print("Final Resource Status before Training:")
        monitor_resources()

        # Start training
        trainer.fit(model, train_loader, val_loader)

        print("\nTraining completed successfully!")
        print("Final Resource Status:")
        monitor_resources()

        # Calculate training time
        end_time = time.time()
        training_time = end_time - start_time
        print(f"Total training time: {training_time/60:.2f} minutes")

        # Collect model information
        model_info = {
            'architecture': 'MemoCMT + R-AIME',
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'text_encoder': config['model']['text']['model_name'],
            'audio_encoder': config['model']['audio']['model_name'],
            'memocmt_layers': config['model']['memocmt']['num_layers'],
            'projection_dim': config['model']['memocmt']['proj_dim'],
            'raime_hidden_dim': config['model']['raime']['hidden_dim'],
            'num_classes': config['dataset']['num_classes'],
            'batch_size': config['training']['batch_size'],
            'max_epochs': config['training']['max_epochs'],
            'learning_rate': config['training']['optimizer']['lr']
        }

        # Get final validation metrics
        if trainer.callback_metrics:
            val_metrics = {}
            for key, value in trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    val_metrics[key] = value.item()
                else:
                    val_metrics[key] = value
        else:
            val_metrics = {}

        # Save comprehensive results
        results_saver.save_final_results(
            model_info=model_info,
            train_metrics={},  # Training metrics are logged per epoch
            val_metrics=val_metrics,
            config=config,
            training_time=training_time
        )

        # Save best model path
        if checkpoint_callback.best_model_path:
            print(f"Best model saved at: {checkpoint_callback.best_model_path}")
            print(f"Best {config['checkpoints']['monitor']}: {checkpoint_callback.best_model_score:.4f}")

            # Save model info to results
            with open(results_saver.experiment_dir / "best_model_info.txt", 'w') as f:
                f.write(f"Best Model Information\n")
                f.write(f"=====================\n")
                f.write(f"Model Path: {checkpoint_callback.best_model_path}\n")
                f.write(f"Best Score: {checkpoint_callback.best_model_score:.4f}\n")
                f.write(f"Monitor Metric: {config['checkpoints']['monitor']}\n")

        print(f"\nAll results saved to: {results_saver.experiment_dir}")
        print("Files generated:")
        print("   - results.json (complete results)")
        print("   - detailed_results.txt (human-readable report)")
        print("   - metrics_summary.csv (key metrics)")
        print("   - training_history.csv (epoch-by-epoch metrics)")
        print("   - per_class_metrics.csv (per-emotion class performance)")
        print("   - confusion_matrix_validation.png (confusion matrix)")
        print("   - training_curves.png (training progress plots)")

        return 0

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()

        print("\nEmergency Resource Status:")
        monitor_resources()

        return 1

    finally:
        print("\nCleaning up memory...")
        cleanup_memory()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
