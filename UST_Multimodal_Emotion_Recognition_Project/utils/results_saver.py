
"""
Results Saver Module
Comprehensive evaluation metrics and results saving for research documentation
"""

import os
import csv
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class ResultsSaver:
    """Save comprehensive training and evaluation results"""
    
    def __init__(self, results_dir: str = "./results", experiment_name: str = "experiment"):
        self.results_dir = Path(results_dir)
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment-specific directory
        self.experiment_dir = self.results_dir / f"{experiment_name}_{self.timestamp}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Results will be saved to: {self.experiment_dir}")
        
        # Initialize metrics tracking
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_weighted_f1': [],
            'val_macro_f1': [],
            'learning_rate': []
        }
        
        self.final_results = {}
        
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float]):
        """Log metrics for each epoch"""
        self.training_history['epoch'].append(epoch)
        
        # Ensure all metric lists have the same length by padding with None
        current_length = len(self.training_history['epoch'])
        
        # First, pad all existing lists to current length
        for key in self.training_history:
            if key != 'epoch':
                while len(self.training_history[key]) < current_length - 1:
                    self.training_history[key].append(None)
        
        # Add new metrics or None if not present
        for key in self.training_history:
            if key != 'epoch':
                if key in metrics:
                    self.training_history[key].append(metrics[key])
                else:
                    self.training_history[key].append(None)
        
        # Save training history after each epoch
        self.save_training_history()
    
    def compute_comprehensive_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute comprehensive evaluation metrics"""
        
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(len(np.unique(y_true)))]
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(class_names):
            metrics[f'{class_name}_precision'] = per_class_precision[i] if i < len(per_class_precision) else 0.0
            metrics[f'{class_name}_recall'] = per_class_recall[i] if i < len(per_class_recall) else 0.0
            metrics[f'{class_name}_f1'] = per_class_f1[i] if i < len(per_class_f1) else 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Classification report
        class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        metrics['classification_report'] = class_report
        
        # ROC AUC if probabilities are provided
        if y_prob is not None and len(np.unique(y_true)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
        elif y_prob is not None:
            try:
                metrics['roc_auc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['roc_auc_weighted'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            except ValueError:
                pass
        
        return metrics
    
    def save_final_results(
        self,
        model_info: Dict[str, Any],
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        test_metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        training_time: Optional[float] = None
    ):
        """Save final comprehensive results"""
        
        self.final_results = {
            'experiment_info': {
                'name': self.experiment_name,
                'timestamp': self.timestamp,
                'training_time_minutes': training_time / 60 if training_time else None
            },
            'model_info': model_info,
            'configuration': config,
            'training_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'training_history': self.training_history
        }
        
        # Save to JSON
        json_file = self.experiment_dir / "results.json"
        with open(json_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._convert_numpy_to_list(self.final_results.copy())
            json.dump(json_results, f, indent=2, default=str)
        print(f"Results saved to JSON: {json_file}")
        
        # Save to detailed TXT file
        self.save_results_txt()
        
        # Save metrics to CSV
        self.save_results_csv()
        
        # Save confusion matrices as images
        if 'confusion_matrix' in val_metrics:
            self.save_confusion_matrix(val_metrics['confusion_matrix'], "validation")
        if test_metrics and 'confusion_matrix' in test_metrics:
            self.save_confusion_matrix(test_metrics['confusion_matrix'], "test")
        
        # Save training curves
        self.save_training_curves()
        
    def save_results_txt(self):
        """Save detailed results to TXT file"""
        txt_file = self.experiment_dir / "detailed_results.txt"
        
        with open(txt_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"IEMOCAP MULTIMODAL EMOTION RECOGNITION RESULTS\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write("=" * 80 + "\n\n")
            
            # Model information
            if 'model_info' in self.final_results:
                f.write("MODEL INFORMATION:\n")
                f.write("-" * 40 + "\n")
                for key, value in self.final_results['model_info'].items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Configuration
            if 'configuration' in self.final_results and self.final_results['configuration']:
                f.write("CONFIGURATION:\n")
                f.write("-" * 40 + "\n")
                self._write_dict_to_file(f, self.final_results['configuration'])
                f.write("\n")
            
            # Training metrics
            if 'training_metrics' in self.final_results:
                f.write("TRAINING METRICS:\n")
                f.write("-" * 40 + "\n")
                self._write_metrics_to_file(f, self.final_results['training_metrics'])
                f.write("\n")
            
            # Validation metrics
            if 'validation_metrics' in self.final_results:
                f.write("VALIDATION METRICS:\n")
                f.write("-" * 40 + "\n")
                self._write_metrics_to_file(f, self.final_results['validation_metrics'])
                f.write("\n")
            
            # Test metrics
            if 'test_metrics' in self.final_results and self.final_results['test_metrics']:
                f.write("TEST METRICS:\n")
                f.write("-" * 40 + "\n")
                self._write_metrics_to_file(f, self.final_results['test_metrics'])
                f.write("\n")
            
            # Classification reports
            for split in ['training_metrics', 'validation_metrics', 'test_metrics']:
                if split in self.final_results and self.final_results[split]:
                    if 'classification_report' in self.final_results[split]:
                        f.write(f"\n{split.upper().replace('_', ' ')} CLASSIFICATION REPORT:\n")
                        f.write("-" * 50 + "\n")
                        report = self.final_results[split]['classification_report']
                        if isinstance(report, dict):
                            self._write_classification_report(f, report)
                        else:
                            f.write(str(report))
                        f.write("\n")
        
        print(f"Detailed results saved to TXT: {txt_file}")
    
    def save_results_csv(self):
        """Save key metrics to CSV file"""
        csv_file = self.experiment_dir / "metrics_summary.csv"
        
        # Prepare metrics data
        metrics_data = []
        
        # Add summary metrics
        for split in ['training_metrics', 'validation_metrics', 'test_metrics']:
            if split in self.final_results and self.final_results[split]:
                split_name = split.replace('_metrics', '')
                metrics = self.final_results[split]
                
                row = {
                    'split': split_name,
                    'accuracy': metrics.get('accuracy', ''),
                    'weighted_f1': metrics.get('weighted_f1', ''),
                    'macro_f1': metrics.get('macro_f1', ''),
                    'weighted_precision': metrics.get('weighted_precision', ''),
                    'weighted_recall': metrics.get('weighted_recall', ''),
                    'macro_precision': metrics.get('macro_precision', ''),
                    'macro_recall': metrics.get('macro_recall', ''),
                    'roc_auc_macro': metrics.get('roc_auc_macro', ''),
                    'roc_auc_weighted': metrics.get('roc_auc_weighted', '')
                }
                metrics_data.append(row)
        
        # Save summary CSV
        if metrics_data:
            df_summary = pd.DataFrame(metrics_data)
            df_summary.to_csv(csv_file, index=False)
            print(f"Metrics summary saved to CSV: {csv_file}")
        
        # Save training history CSV
        if self.training_history['epoch']:
            history_csv = self.experiment_dir / "training_history.csv"
            df_history = pd.DataFrame(self.training_history)
            df_history.to_csv(history_csv, index=False)
            print(f"Training history saved to CSV: {history_csv}")
        
        # Save per-class metrics CSV
        self.save_per_class_metrics_csv()
    
    def save_per_class_metrics_csv(self):
        """Save per-class metrics to CSV"""
        per_class_file = self.experiment_dir / "per_class_metrics.csv"
        
        # Emotion class names (update based on your dataset)
        class_names = ['angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated']
        
        per_class_data = []
        
        for split in ['training_metrics', 'validation_metrics', 'test_metrics']:
            if split in self.final_results and self.final_results[split]:
                split_name = split.replace('_metrics', '')
                metrics = self.final_results[split]
                
                for class_name in class_names:
                    row = {
                        'split': split_name,
                        'emotion_class': class_name,
                        'precision': metrics.get(f'{class_name}_precision', ''),
                        'recall': metrics.get(f'{class_name}_recall', ''),
                        'f1_score': metrics.get(f'{class_name}_f1', '')
                    }
                    per_class_data.append(row)
        
        if per_class_data:
            df_per_class = pd.DataFrame(per_class_data)
            df_per_class.to_csv(per_class_file, index=False)
            print(f"Per-class metrics saved to CSV: {per_class_file}")
    
    def save_training_history(self):
        """Save training history to CSV"""
        if not self.training_history['epoch']:
            return
        
        # Ensure all lists have the same length
        max_length = max(len(v) for v in self.training_history.values())
        for key, values in self.training_history.items():
            while len(values) < max_length:
                values.append(None)
        
        history_file = self.experiment_dir / "training_history.csv"
        
        try:
            df = pd.DataFrame(self.training_history)
            df.to_csv(history_file, index=False)
        except Exception as e:
            print(f"Warning: Could not save training history: {e}")
            # Try to save with only epoch and available metrics
            safe_history = {'epoch': self.training_history['epoch']}
            for key, values in self.training_history.items():
                if key != 'epoch' and len(values) == len(safe_history['epoch']):
                    safe_history[key] = values
            
            try:
                df = pd.DataFrame(safe_history)
                df.to_csv(history_file, index=False)
            except Exception as e2:
                print(f"Warning: Could not save even safe training history: {e2}")
    
    def save_confusion_matrix(self, cm: np.ndarray, split_name: str):
        """Save confusion matrix as image"""
        plt.figure(figsize=(10, 8))
        
        class_names = ['Angry', 'Happy', 'Sad', 'Neutral', 'Excited', 'Frustrated']
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names[:cm.shape[1]],
            yticklabels=class_names[:cm.shape[0]]
        )
        
        plt.title(f'Confusion Matrix - {split_name.title()}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        cm_file = self.experiment_dir / f"confusion_matrix_{split_name}.png"
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved: {cm_file}")
    
    def save_training_curves(self):
        """Save training curves as images"""
        if not self.training_history['epoch']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.training_history['epoch']
        
        # Loss curves
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(epochs, self.training_history['train_accuracy'], 'b-', label='Training Accuracy')
        axes[0, 1].plot(epochs, self.training_history['val_accuracy'], 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 scores
        axes[1, 0].plot(epochs, self.training_history['val_weighted_f1'], 'g-', label='Weighted F1')
        axes[1, 0].plot(epochs, self.training_history['val_macro_f1'], 'orange', label='Macro F1')
        axes[1, 0].set_title('Validation F1 Scores')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(epochs, self.training_history['learning_rate'], 'm-', label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        curves_file = self.experiment_dir / "training_curves.png"
        plt.savefig(curves_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved: {curves_file}")
    
    def _write_dict_to_file(self, f, d, indent=0):
        """Recursively write dictionary to file"""
        for key, value in d.items():
            if isinstance(value, dict):
                f.write("  " * indent + f"{key}:\n")
                self._write_dict_to_file(f, value, indent + 1)
            else:
                f.write("  " * indent + f"{key}: {value}\n")
    
    def _write_metrics_to_file(self, f, metrics):
        """Write metrics to file, skipping complex objects"""
        for key, value in metrics.items():
            if key in ['confusion_matrix', 'classification_report']:
                continue
            if isinstance(value, (int, float)):
                f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
    
    def _write_classification_report(self, f, report):
        """Write classification report to file"""
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"\n{class_name}:\n")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric}: {value:.4f}\n")
    
    def _convert_numpy_to_list(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_list(item) for item in obj]
        else:
            return obj


def test_results_saver():
    """Test the results saver functionality"""
    saver = ResultsSaver("./test_results", "test_experiment")
    
    # Simulate training history
    for epoch in range(5):
        metrics = {
            'train_loss': 1.0 - epoch * 0.1,
            'train_accuracy': 0.5 + epoch * 0.1,
            'val_loss': 1.2 - epoch * 0.12,
            'val_accuracy': 0.4 + epoch * 0.12,
            'val_weighted_f1': 0.3 + epoch * 0.15,
            'val_macro_f1': 0.25 + epoch * 0.13,
            'learning_rate': 0.001 * (0.9 ** epoch)
        }
        saver.log_epoch_metrics(epoch, metrics)
    
    # Simulate final results
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 2, 1])
    y_prob = np.random.random((10, 3))
    
    val_metrics = saver.compute_comprehensive_metrics(y_true, y_pred, y_prob)
    
    model_info = {
        'total_parameters': 178406471,
        'trainable_parameters': 2506951,
        'architecture': 'MemoCMT + R-AIME'
    }
    
    config = {'model': {'text': {'model_name': 'test'}}}
    
    saver.save_final_results(
        model_info=model_info,
        train_metrics=val_metrics,
        val_metrics=val_metrics,
        config=config,
        training_time=3600
    )
    
    print("Results saver test completed!")


if __name__ == "__main__":
    test_results_saver()
