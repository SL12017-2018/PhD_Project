"""
Metrics and Evaluation Utilities for IEMOCAP MemoCMT
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class IEMOCAPMetrics:
    """Comprehensive metrics for IEMOCAP emotion recognition evaluation"""
    
    def __init__(self, emotion_labels: Dict[str, int], class_names: List[str] = None):
        self.emotion_labels = emotion_labels
        self.num_classes = len(emotion_labels)
        
        if class_names is None:
            # Create class names from emotion labels
            self.class_names = [None] * self.num_classes
            for emotion, idx in emotion_labels.items():
                self.class_names[idx] = emotion.capitalize()
        else:
            self.class_names = class_names
    
    def compute_all_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics
        
        Args:
            y_true: True labels (N,)
            y_pred: Predicted labels (N,)
            y_prob: Prediction probabilities (N, num_classes)
            
        Returns:
            Dictionary of all computed metrics
        """
        
        metrics = {}
        
        # Basic accuracy metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
        
        # Precision and Recall
        metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
        per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics['per_class_f1'] = dict(zip(self.class_names, per_class_f1))
        metrics['per_class_precision'] = dict(zip(self.class_names, per_class_precision))
        metrics['per_class_recall'] = dict(zip(self.class_names, per_class_recall))
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            output_dict=True,
            zero_division=0
        )
        
        # AUC metrics if probabilities available
        if y_prob is not None:
            try:
                # Multi-class AUC
                y_true_binarized = label_binarize(y_true, classes=list(range(self.num_classes)))
                if self.num_classes == 2:
                    auc = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['auc'] = auc
                else:
                    auc_macro = roc_auc_score(y_true_binarized, y_prob, average='macro', multi_class='ovr')
                    auc_weighted = roc_auc_score(y_true_binarized, y_prob, average='weighted', multi_class='ovr')
                    metrics['auc_macro'] = auc_macro
                    metrics['auc_weighted'] = auc_weighted
                
                # Per-class AUC
                per_class_auc = []
                for i in range(self.num_classes):
                    if len(np.unique(y_true_binarized[:, i])) > 1:  # Check if class exists
                        auc_class = roc_auc_score(y_true_binarized[:, i], y_prob[:, i])
                        per_class_auc.append(auc_class)
                    else:
                        per_class_auc.append(0.0)
                
                metrics['per_class_auc'] = dict(zip(self.class_names, per_class_auc))
                
            except Exception as e:
                print(f"Warning: Could not compute AUC metrics: {e}")
                metrics['auc_macro'] = 0.0
                metrics['auc_weighted'] = 0.0
        
        return metrics
    
    def print_metrics_summary(self, metrics: Dict[str, Any]):
        """Print a formatted summary of metrics"""
        
        print("=" * 60)
        print("IEMOCAP EMOTION RECOGNITION EVALUATION RESULTS")
        print("=" * 60)
        
        print(f"OVERALL PERFORMANCE:")
        print(f"   Accuracy:           {metrics['accuracy']:.4f}")
        print(f"   Weighted F1:        {metrics['weighted_f1']:.4f}")
        print(f"   Macro F1:           {metrics['macro_f1']:.4f}")
        print(f"   Micro F1:           {metrics['micro_f1']:.4f}")
        
        if 'auc_weighted' in metrics:
            print(f"   Weighted AUC:       {metrics['auc_weighted']:.4f}")
            print(f"   Macro AUC:          {metrics['auc_macro']:.4f}")
        
        print(f"\nPRECISION & RECALL:")
        print(f"   Weighted Precision: {metrics['weighted_precision']:.4f}")
        print(f"   Macro Precision:    {metrics['macro_precision']:.4f}")
        print(f"   Weighted Recall:    {metrics['weighted_recall']:.4f}")
        print(f"   Macro Recall:       {metrics['macro_recall']:.4f}")
        
        print(f"\nPER-CLASS PERFORMANCE:")
        for emotion, f1_score in metrics['per_class_f1'].items():
            precision = metrics['per_class_precision'][emotion]
            recall = metrics['per_class_recall'][emotion]
            print(f"   {emotion:12s}: F1={f1_score:.3f}, P={precision:.3f}, R={recall:.3f}")
        
        print("=" * 60)
    
    def plot_confusion_matrix(
        self, 
        confusion_matrix: np.ndarray, 
        save_path: Optional[str] = None,
        normalize: bool = True
    ):
        """Plot confusion matrix"""
        
        plt.figure(figsize=(10, 8))
        
        if normalize:
            cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            cm_normalized = confusion_matrix
            fmt = 'd'
            title = 'Confusion Matrix'
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'}
        )
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
        return plt.gcf()
    
    def plot_per_class_metrics(
        self, 
        metrics: Dict[str, Any], 
        save_path: Optional[str] = None
    ):
        """Plot per-class performance metrics"""
        
        emotions = list(metrics['per_class_f1'].keys())
        f1_scores = list(metrics['per_class_f1'].values())
        precision_scores = list(metrics['per_class_precision'].values())
        recall_scores = list(metrics['per_class_recall'].values())
        
        x = np.arange(len(emotions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, f1_scores, width, label='F1-Score', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x, precision_scores, width, label='Precision', color='lightcoral', alpha=0.8)
        bars3 = ax.bar(x + width, recall_scores, width, label='Recall', color='lightgreen', alpha=0.8)
        
        ax.set_xlabel('Emotions', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        add_value_labels(bars1)
        add_value_labels(bars2)
        add_value_labels(bars3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Per-class metrics plot saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def create_results_dataframe(self, metrics: Dict[str, Any]) -> pd.DataFrame:
        """Create a pandas DataFrame with results"""
        
        # Overall metrics
        overall_data = {
            'Metric': ['Accuracy', 'Weighted F1', 'Macro F1', 'Micro F1', 
                      'Weighted Precision', 'Macro Precision', 'Weighted Recall', 'Macro Recall'],
            'Score': [
                metrics['accuracy'], metrics['weighted_f1'], metrics['macro_f1'], metrics['micro_f1'],
                metrics['weighted_precision'], metrics['macro_precision'], 
                metrics['weighted_recall'], metrics['macro_recall']
            ]
        }
        
        if 'auc_weighted' in metrics:
            overall_data['Metric'].extend(['Weighted AUC', 'Macro AUC'])
            overall_data['Score'].extend([metrics['auc_weighted'], metrics['auc_macro']])
        
        overall_df = pd.DataFrame(overall_data)
        
        # Per-class metrics
        per_class_data = {
            'Emotion': list(metrics['per_class_f1'].keys()),
            'F1': list(metrics['per_class_f1'].values()),
            'Precision': list(metrics['per_class_precision'].values()),
            'Recall': list(metrics['per_class_recall'].values())
        }
        
        if 'per_class_auc' in metrics:
            per_class_data['AUC'] = list(metrics['per_class_auc'].values())
        
        per_class_df = pd.DataFrame(per_class_data)
        
        return overall_df, per_class_df
    
    def save_results(
        self, 
        metrics: Dict[str, Any], 
        save_dir: str,
        experiment_name: str = "iemocap_memocmt"
    ):
        """Save all results to files"""
        
        import os
        from pathlib import Path
        import json
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics as JSON
        metrics_copy = metrics.copy()
        
        # Convert numpy arrays to lists for JSON serialization
        if 'confusion_matrix' in metrics_copy:
            metrics_copy['confusion_matrix'] = metrics_copy['confusion_matrix'].tolist()
        
        with open(save_path / f"{experiment_name}_metrics.json", 'w') as f:
            json.dump(metrics_copy, f, indent=2, default=str)
        
        # Save DataFrames as CSV
        overall_df, per_class_df = self.create_results_dataframe(metrics)
        overall_df.to_csv(save_path / f"{experiment_name}_overall_metrics.csv", index=False)
        per_class_df.to_csv(save_path / f"{experiment_name}_per_class_metrics.csv", index=False)
        
        # Save plots
        self.plot_confusion_matrix(
            metrics['confusion_matrix'], 
            save_path / f"{experiment_name}_confusion_matrix.png"
        )
        plt.close()
        
        self.plot_per_class_metrics(
            metrics, 
            save_path / f"{experiment_name}_per_class_metrics.png"
        )
        plt.close()
        
        print(f"All results saved to {save_path}")
        
        return save_path


def compute_emotion_metrics(y_true: np.ndarray, y_pred: np.ndarray, emotion_labels: Dict[str, int]) -> Dict[str, float]:
    """
    Compute emotion recognition metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        emotion_labels: Mapping of emotion names to indices
        
    Returns:
        Dictionary of computed metrics
    """
    
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
    metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
    
    # Precision and Recall
    metrics['weighted_precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['macro_precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['weighted_recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['macro_recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    return metrics


def compute_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Compute calibration metrics (ECE - Expected Calibration Error)
    """
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return {'ece': ece, 'avg_confidence': confidences.mean(), 'avg_accuracy': accuracies.mean()}


def test_metrics():
    """Test metrics functionality"""
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 6
    
    emotion_labels = {
        'angry': 0, 'happy': 1, 'sad': 2, 'neutral': 3, 'excited': 4, 'frustrated': 5
    }
    
    # Generate dummy predictions and labels
    y_true = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)
    y_pred = np.argmax(y_prob, axis=1)
    
    # Create metrics calculator
    metrics_calc = IEMOCAPMetrics(emotion_labels)
    
    # Compute metrics
    metrics = metrics_calc.compute_all_metrics(y_true, y_pred, y_prob)
    
    # Print summary
    metrics_calc.print_metrics_summary(metrics)
    
    # Test calibration metrics
    cal_metrics = compute_calibration_metrics(y_true, y_prob)
    print(f"\nCALIBRATION METRICS:")
    print(f"   ECE: {cal_metrics['ece']:.4f}")
    print(f"   Avg Confidence: {cal_metrics['avg_confidence']:.4f}")
    print(f"   Avg Accuracy: {cal_metrics['avg_accuracy']:.4f}")
    
    # Create DataFrames
    overall_df, per_class_df = metrics_calc.create_results_dataframe(metrics)
    print(f"\nOVERALL METRICS DATAFRAME:")
    print(overall_df)
    
    print(f"\nPER-CLASS METRICS DATAFRAME:")
    print(per_class_df)
    
    print("Metrics test completed!")


if __name__ == "__main__":
    test_metrics()
