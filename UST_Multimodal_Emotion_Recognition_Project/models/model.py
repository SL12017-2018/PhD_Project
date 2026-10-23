"""
Complete End-to-End IEMOCAP MemoCMT Model
Integrates all components: Encoders + MemoCMT + R-AIME Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Dict, List, Tuple, Optional, Any
import yaml
from pathlib import Path

from .text_encoder import TextEmotionEncoder
from .audio_encoder import AudioEmotionEncoder
from .memocmt import MemoCMT
from .raime_head import RAIMEHead


class IEMOCAPMemoCMTModel(L.LightningModule):
    """
    Complete IEMOCAP MemoCMT Model using PyTorch Lightning
    
    Architecture:
    Text/Audio Inputs → Pretrained Encoders → MemoCMT → R-AIME Head → Emotion Prediction
    """
    
    def __init__(
        self,
        config: Dict,
        tokenizer = None,
        processor = None
    ):
        super().__init__()
        
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        
        # Model configuration
        model_config = config.get('model', {})
        text_config = model_config.get('text', {})
        audio_config = model_config.get('audio', {})
        memocmt_config = model_config.get('memocmt', {})
        raime_config = model_config.get('raime', {})
        
        # Dataset configuration
        dataset_config = config.get('dataset', {})
        self.num_classes = dataset_config.get('num_classes', 6)
        self.emotion_labels = dataset_config.get('emotion_labels', {})
        
        # Training configuration
        training_config = config.get('training', {})
        self.learning_rate = training_config.get('learning_rate', 1e-4)
        self.encoder_lr = training_config.get('encoder_lr', 5e-6)
        self.weight_decay = training_config.get('weight_decay', 0.01)
        self.warmup_ratio = training_config.get('warmup_ratio', 0.1)
        
        # Loss configuration
        self.loss_type = training_config.get('loss_type', 'focal')
        self.focal_alpha = training_config.get('focal_alpha', 0.25)
        self.focal_gamma = training_config.get('focal_gamma', 2.0)
        self.label_smoothing = training_config.get('label_smoothing', 0.1)
        
        # Initialize components
        self._init_encoders(text_config, audio_config)
        self._init_memocmt(memocmt_config, text_config, audio_config)
        self._init_raime_head(raime_config, memocmt_config)
        
        # Loss functions
        self._init_loss_functions()
        
        # Metrics tracking
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        # Save hyperparameters
        self.save_hyperparameters(config)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.text_encoder.model, 'gradient_checkpointing_enable'):
            self.text_encoder.model.gradient_checkpointing_enable()
        if hasattr(self.audio_encoder.model, 'gradient_checkpointing_enable'):
            self.audio_encoder.model.gradient_checkpointing_enable()
        
        print(f"✅ IEMOCAPMemoCMTModel initialized:")
        print(f"   Text encoder: {config['model']['text']['model_name']}")
        print(f"   Audio encoder: {config['model']['audio']['model_name']}")
        print(f"   MemoCMT projection dim: {config['model']['memocmt']['proj_dim']}")
        print(f"   Number of emotion classes: {config['dataset']['num_classes']}")
        print(f"   Gradient checkpointing: Enabled")
    
    def _init_encoders(self, text_config: Dict, audio_config: Dict):
        """Initialize text and audio encoders"""
        
        # Text encoder
        self.text_encoder = TextEmotionEncoder(
            model_name=text_config.get('model_name', 'j-hartmann/emotion-english-distilroberta-base'),
            pool=text_config.get('pool', 'cls'),
            freeze=text_config.get('freeze', True),
            unfreeze_layers=text_config.get('unfreeze_layers', 2),
            output_dim=text_config.get('output_dim', 768)
        )
        
        # Audio encoder
        self.audio_encoder = AudioEmotionEncoder(
            model_name=audio_config.get('model_name', 'superb/wav2vec2-base-superb-er'),
            freeze=audio_config.get('freeze', True),
            unfreeze_layers=audio_config.get('unfreeze_layers', 2),
            output_dim=audio_config.get('output_dim', 768)
        )
    
    def _init_memocmt(self, memocmt_config: Dict, text_config: Dict, audio_config: Dict):
        """Initialize MemoCMT module"""
        
        self.memocmt = MemoCMT(
            text_dim=text_config.get('output_dim', 768),
            audio_dim=audio_config.get('output_dim', 768),
            proj_dim=memocmt_config.get('proj_dim', 256),
            num_layers=memocmt_config.get('num_layers', 3),
            num_heads=memocmt_config.get('num_heads', 8),
            ff_dim=memocmt_config.get('ff_dim', 512),
            dropout=memocmt_config.get('dropout', 0.2),
            speaker_embed_dim=memocmt_config.get('speaker_embed_dim', 32),
            global_mem_dim=memocmt_config.get('global_mem_dim', 256),
            use_residual=memocmt_config.get('use_residual', True),
            use_layer_norm=memocmt_config.get('use_layer_norm', True)
        )
    
    def _init_raime_head(self, raime_config: Dict, memocmt_config: Dict):
        """Initialize R-AIME head"""
        
        self.raime_head = RAIMEHead(
            input_dim=memocmt_config.get('proj_dim', 256),
            num_classes=self.num_classes,
            hidden_dim=raime_config.get('hidden_dim', 512),
            dropout=raime_config.get('dropout', 0.2),
            use_temperature=raime_config.get('use_temperature_scaling', True),
            explainability=raime_config.get('explainability', True)
        )
    
    def _init_loss_functions(self):
        """Initialize loss functions"""
        
        if self.loss_type == 'focal':
            self.criterion = FocalLoss(
                alpha=self.focal_alpha,
                gamma=self.focal_gamma,
                num_classes=self.num_classes
            )
        elif self.loss_type == 'label_smoothing':
            self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def unfreeze_encoders(self):
        """Unfreeze top layers of encoders for fine-tuning"""
        print("Unfreezing encoder layers...")
        self.text_encoder.unfreeze_top_layers()
        self.audio_encoder.unfreeze_top_layers()
    
    def forward(
        self, 
        batch: Dict[str, torch.Tensor],
        return_explanations: bool = False,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the complete model
        
        Args:
            batch: Batch dict containing text_inputs, audio_inputs, speaker_ids, etc.
            return_explanations: Whether to return explainability outputs
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            Model outputs including predictions, features, attention maps, etc.
        """
        
        # Reset memory for new batch
        batch_size = batch['text_inputs']['input_ids'].size(0)
        self.memocmt.reset_memory(batch_size)
        
        # 1. Encode text and audio
        text_outputs = self.text_encoder(batch['text_inputs'])
        audio_outputs = self.audio_encoder(batch['audio_inputs'])
        
        text_features = text_outputs['turn_pooled']  # (B, text_dim)
        audio_features = audio_outputs['turn_pooled'] # (B, audio_dim)
        
        # 2. Cross-modal fusion with MemoCMT
        memocmt_outputs = self.memocmt(
            text_features=text_features,
            audio_features=audio_features,
            speaker_ids=batch['speaker_ids'],
            update_memory=True
        )
        
        fused_features = memocmt_outputs['fused_features']  # (B, proj_dim)
        
        # 3. Final prediction with R-AIME head
        raime_outputs = self.raime_head(
            fused_features=fused_features,
            return_explanations=return_explanations,
            return_uncertainty=return_uncertainty
        )
        
        # Combine all outputs
        outputs = {
            # Core predictions
            'logits': raime_outputs['logits'],
            'probabilities': raime_outputs['probabilities'],
            'predictions': raime_outputs['predictions'],
            'confidence': raime_outputs['confidence'],
            
            # Features
            'text_features': text_features,
            'audio_features': audio_features,
            'fused_features': fused_features,
            'processed_features': raime_outputs['processed_features'],
            
            # MemoCMT outputs
            'memocmt_text_features': memocmt_outputs['text_features'],
            'memocmt_audio_features': memocmt_outputs['audio_features'],
            'attention_maps': memocmt_outputs['attention_maps'],
            'memory_states': memocmt_outputs['memory_states'],
            
            # R-AIME outputs
            'importance_scores': raime_outputs['importance_scores'],
            'calibrated_logits': raime_outputs['calibrated_logits'],
        }
        
        # Add optional outputs
        if return_uncertainty and 'uncertainty' in raime_outputs:
            outputs['uncertainty'] = raime_outputs['uncertainty']
        
        if return_explanations:
            for key in ['feature_attribution', 'concept_activations', 'attention_importance']:
                if key in raime_outputs:
                    outputs[key] = raime_outputs[key]
        
        return outputs
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step"""
        
        outputs = self(batch, return_explanations=False, return_uncertainty=False)
        
        # Compute loss
        loss = self.criterion(outputs['logits'], batch['labels'])
        
        # Compute metrics
        predictions = outputs['predictions']
        labels = batch['labels']
        
        accuracy = (predictions == labels).float().mean()
        confidence = outputs['confidence'].mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_confidence', confidence, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step"""
        
        outputs = self(batch, return_explanations=True, return_uncertainty=True)
        
        # Compute loss
        loss = self.criterion(outputs['logits'], batch['labels'])
        
        # Store outputs for epoch-level metrics
        step_output = {
            'val_loss': loss,
            'predictions': outputs['predictions'],
            'probabilities': outputs['probabilities'],
            'labels': batch['labels'],
            'confidence': outputs['confidence'],
        }
        
        if 'uncertainty' in outputs:
            step_output['uncertainty'] = outputs['uncertainty']
        
        self.validation_step_outputs.append(step_output)
        
        return step_output
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step"""
        
        outputs = self(batch, return_explanations=True, return_uncertainty=True)
        
        # Compute loss
        loss = self.criterion(outputs['logits'], batch['labels'])
        
        # Store outputs for epoch-level metrics
        step_output = {
            'test_loss': loss,
            'predictions': outputs['predictions'],
            'probabilities': outputs['probabilities'],
            'labels': batch['labels'],
            'confidence': outputs['confidence'],
            'attention_maps': outputs.get('attention_maps', None),
            'feature_attribution': outputs.get('feature_attribution', None)
        }
        
        if 'uncertainty' in outputs:
            step_output['uncertainty'] = outputs['uncertainty']
        
        self.test_step_outputs.append(step_output)
        
        return step_output
    
    def on_validation_epoch_end(self):
        """Compute validation metrics at epoch end with per-class breakdown"""
        
        if not self.validation_step_outputs:
            return
        
        # Aggregate outputs
        all_predictions = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_probabilities = torch.cat([x['probabilities'] for x in self.validation_step_outputs])
        avg_loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean()
        
        # Compute metrics with per-class breakdown
        metrics = self.compute_metrics(
            all_predictions, all_labels, all_probabilities, return_per_class=True
        )
        
        # Log overall metrics
        self.log('val_loss_epoch', avg_loss, prog_bar=True)
        self.log('val_accuracy', metrics['accuracy'], prog_bar=True)
        self.log('val_weighted_f1', metrics['weighted_f1'], prog_bar=True)
        self.log('val_macro_f1', metrics['macro_f1'], prog_bar=True)
        self.log('val_weighted_precision', metrics['weighted_precision'], prog_bar=False)
        self.log('val_weighted_recall', metrics['weighted_recall'], prog_bar=False)
        self.log('val_macro_precision', metrics['macro_precision'], prog_bar=False)
        self.log('val_macro_recall', metrics['macro_recall'], prog_bar=False)
        
        # Log per-class metrics
        emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
        for emotion in emotion_labels:
            if f'{emotion}_f1' in metrics:
                self.log(f'val_{emotion}_precision', metrics[f'{emotion}_precision'], prog_bar=False)
                self.log(f'val_{emotion}_recall', metrics[f'{emotion}_recall'], prog_bar=False)
                self.log(f'val_{emotion}_f1', metrics[f'{emotion}_f1'], prog_bar=False)
        
        # Print detailed per-class metrics to console
        if hasattr(self.trainer, 'global_rank') and self.trainer.global_rank == 0:
            print(f"\n📊 Validation Results - Epoch {self.current_epoch}:")
            print(f"🎯 Overall Metrics:")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Weighted F1: {metrics['weighted_f1']:.4f}")
            print(f"   Macro F1: {metrics['macro_f1']:.4f}")
            print(f"   Weighted Precision: {metrics['weighted_precision']:.4f}")
            print(f"   Weighted Recall: {metrics['weighted_recall']:.4f}")
            
            print(f"\n📋 Per-Class Metrics:")
            print(f"{'Emotion':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
            print("-" * 50)
            for emotion in emotion_labels:
                if f'{emotion}_f1' in metrics:
                    precision = metrics[f'{emotion}_precision']
                    recall = metrics[f'{emotion}_recall']
                    f1 = metrics[f'{emotion}_f1']
                    print(f"{emotion:<12} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
            
            # Print classification report
            if 'classification_report' in metrics:
                print(f"\n📈 Detailed Classification Report:")
                print(metrics['classification_report'])
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def on_test_epoch_end(self):
        """Compute test metrics at epoch end"""
        
        if not self.test_step_outputs:
            return
        
        # Aggregate outputs
        all_predictions = torch.cat([x['predictions'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        all_probabilities = torch.cat([x['probabilities'] for x in self.test_step_outputs])
        avg_loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean()
        
        # Compute metrics
        metrics = self.compute_metrics(all_predictions, all_labels, all_probabilities)
        
        # Log metrics
        self.log('test_loss_epoch', avg_loss)
        for metric_name, metric_value in metrics.items():
            self.log(f'test_{metric_name}', metric_value)
        
        # Clear outputs
        self.test_step_outputs.clear()
    
    def compute_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor, 
        probabilities: torch.Tensor,
        return_per_class: bool = False
    ) -> Dict[str, float]:
        """Compute evaluation metrics with optional per-class breakdown"""
        
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score, 
            classification_report, confusion_matrix
        )
        
        # Convert to numpy
        pred_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        probs_np = probabilities.cpu().numpy()
        
        # Compute overall metrics
        accuracy = accuracy_score(labels_np, pred_np)
        weighted_f1 = f1_score(labels_np, pred_np, average='weighted', zero_division=0)
        macro_f1 = f1_score(labels_np, pred_np, average='macro', zero_division=0)
        weighted_precision = precision_score(labels_np, pred_np, average='weighted', zero_division=0)
        weighted_recall = recall_score(labels_np, pred_np, average='weighted', zero_division=0)
        macro_precision = precision_score(labels_np, pred_np, average='macro', zero_division=0)
        macro_recall = recall_score(labels_np, pred_np, average='macro', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall
        }
        
        # Add per-class metrics if requested
        if return_per_class:
            # Get emotion labels
            emotion_labels = ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
            
            # Per-class precision, recall, f1-score
            per_class_precision = precision_score(labels_np, pred_np, average=None, zero_division=0)
            per_class_recall = recall_score(labels_np, pred_np, average=None, zero_division=0)
            per_class_f1 = f1_score(labels_np, pred_np, average=None, zero_division=0)
            
            # Add per-class metrics to the dict
            for i, emotion in enumerate(emotion_labels):
                if i < len(per_class_precision):
                    metrics[f'{emotion}_precision'] = per_class_precision[i]
                    metrics[f'{emotion}_recall'] = per_class_recall[i]
                    metrics[f'{emotion}_f1'] = per_class_f1[i]
            
            # Generate classification report
            metrics['classification_report'] = classification_report(
                labels_np, pred_np, target_names=emotion_labels, zero_division=0
            )
            
            # Generate confusion matrix
            metrics['confusion_matrix'] = confusion_matrix(labels_np, pred_np)
        
        return metrics
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers"""
        
        # Separate parameters for encoders and rest of the model
        encoder_params = []
        other_params = []
        
        for name, param in self.named_parameters():
            if 'text_encoder' in name or 'audio_encoder' in name:
                if param.requires_grad:  # Only include unfrozen parameters
                    encoder_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.learning_rate,
                'weight_decay': self.weight_decay
            })
        
        if encoder_params:
            param_groups.append({
                'params': encoder_params,
                'lr': self.encoder_lr,
                'weight_decay': self.weight_decay * 0.1  # Lower weight decay for pre-trained params
            })
        
        optimizer = torch.optim.AdamW(param_groups)
        
        # Learning rate scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs,
                eta_min=1e-6
            ),
            'interval': 'epoch',
            'frequency': 1,
            'monitor': 'val_weighted_f1',
            'strict': True,
        }
        
        return [optimizer], [scheduler]


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, num_classes: int = 6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config_path: str, tokenizer=None, processor=None) -> IEMOCAPMemoCMTModel:
    """Create model from configuration file"""
    config = load_config(config_path)
    model = IEMOCAPMemoCMTModel(config, tokenizer, processor)
    return model, config


def test_complete_model():
    """Test the complete model"""
    
    # Create dummy config
    config = {
        'dataset': {
            'num_classes': 6,
            'emotion_labels': {'angry': 0, 'happy': 1, 'sad': 2, 'neutral': 3, 'excited': 4, 'frustrated': 5}
        },
        'model': {
            'text': {
                'model_name': 'j-hartmann/emotion-english-distilroberta-base',
                'output_dim': 768,
                'freeze': True
            },
            'audio': {
                'model_name': 'superb/wav2vec2-base-superb-er',
                'output_dim': 768,
                'freeze': True
            },
            'memocmt': {
                'proj_dim': 256,
                'num_layers': 2,
                'num_heads': 4
            },
            'raime': {
                'hidden_dim': 512,
                'use_temperature_scaling': True,
                'explainability': True
            }
        },
        'training': {
            'learning_rate': 1e-4,
            'loss_type': 'focal'
        }
    }
    
    # Create model
    model = IEMOCAPMemoCMTModel(config)
    
    # Create dummy batch
    batch_size = 4
    seq_len = 128
    audio_len = 16000 * 2  # 2 seconds
    
    batch = {
        'text_inputs': {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len)
        },
        'audio_inputs': {
            'input_values': torch.randn(batch_size, audio_len),
            'attention_mask': torch.ones(batch_size, audio_len // 320)  # Approximate for wav2vec2
        },
        'speaker_ids': torch.randint(0, 2, (batch_size,)),
        'labels': torch.randint(0, 6, (batch_size,))
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(batch, return_explanations=True, return_uncertainty=True)
    
    print("Complete Model Test:")
    print(f"Batch size: {batch_size}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Predictions shape: {outputs['predictions'].shape}")
    print(f"Confidence shape: {outputs['confidence'].shape}")
    print(f"Text features shape: {outputs['text_features'].shape}")
    print(f"Audio features shape: {outputs['audio_features'].shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    print(f"Attention maps layers: {len(outputs['attention_maps'])}")
    print(f"Memory states layers: {len(outputs['memory_states'])}")
    print("Complete model working correctly!")


if __name__ == "__main__":
    test_complete_model()
