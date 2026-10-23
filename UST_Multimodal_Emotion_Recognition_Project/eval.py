"""
Evaluation Script for IEMOCAP MemoCMT
Comprehensive testing with explainability and visualization
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, Wav2Vec2Processor

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import our modules
from data.iemocap import create_dataloaders
from models.model import IEMOCAPMemoCMTModel, load_config
from utils.common import set_seed, load_config, create_directories
from utils.metrics import IEMOCAPMetrics, compute_calibration_metrics

warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Comprehensive model evaluator with explainability"""
    
    def __init__(
        self,
        model: IEMOCAPMemoCMTModel,
        tokenizer,
        processor,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        # Initialize metrics
        emotion_labels = config.get('dataset', {}).get('emotion_labels', {})
        self.metrics_calc = IEMOCAPMetrics(emotion_labels)
        
        print(f"Model evaluator initialized on {device}")
    
    def evaluate_model(
        self, 
        test_loader,
        save_predictions: bool = True,
        save_explanations: bool = True,
        results_dir: str = './results'
    ) -> Dict[str, any]:
        """
        Comprehensive model evaluation
        
        Args:
            test_loader: Test data loader
            save_predictions: Whether to save detailed predictions
            save_explanations: Whether to save explainability results
            results_dir: Directory to save results
            
        Returns:
            Dictionary with all evaluation results
        """
        
        print("🧪 Starting comprehensive model evaluation...")
        
        # Storage for results
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_confidences = []
        all_uncertainties = []
        
        # Explainability storage
        attention_maps = []
        feature_attributions = []
        importance_scores = []
        
        # Additional analysis
        per_emotion_correct = {emotion: [] for emotion in self.metrics_calc.class_names}
        per_speaker_results = {'F': {'preds': [], 'labels': []}, 'M': {'preds': [], 'labels': []}}
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with explanations
                outputs = self.model(
                    batch, 
                    return_explanations=save_explanations,
                    return_uncertainty=True
                )
                
                # Extract results
                predictions = outputs['predictions'].cpu().numpy()
                probabilities = outputs['probabilities'].cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                confidences = outputs['confidence'].cpu().numpy().flatten()
                
                # Store results
                all_predictions.extend(predictions)
                all_probabilities.append(probabilities)
                all_labels.extend(labels)
                all_confidences.extend(confidences)
                
                # Store uncertainties if available
                if 'uncertainty' in outputs:
                    uncertainties = outputs['uncertainty'].cpu().numpy().flatten()
                    all_uncertainties.extend(uncertainties)
                
                # Store explainability results
                if save_explanations:
                    if 'attention_maps' in outputs:
                        attention_maps.append(outputs['attention_maps'])
                    
                    if 'feature_attribution' in outputs:
                        feature_attributions.append(outputs['feature_attribution'].cpu().numpy())
                    
                    if 'importance_scores' in outputs:
                        importance_scores.append(outputs['importance_scores'].cpu().numpy())
                
                # Per-emotion analysis
                for pred, label in zip(predictions, labels):
                    emotion_name = self.metrics_calc.class_names[label]
                    per_emotion_correct[emotion_name].append(pred == label)
                
                # Per-speaker analysis (if speaker info available)
                if 'raw_speakers' in batch:
                    speakers = batch['raw_speakers']
                    for speaker, pred, label in zip(speakers, predictions, labels):
                        if speaker in per_speaker_results:
                            per_speaker_results[speaker]['preds'].append(pred)
                            per_speaker_results[speaker]['labels'].append(label)
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.vstack(all_probabilities)
        all_labels = np.array(all_labels)
        all_confidences = np.array(all_confidences)
        
        if all_uncertainties:
            all_uncertainties = np.array(all_uncertainties)
        
        print(f"Evaluation completed: {len(all_predictions)} samples")
        
        # Compute comprehensive metrics
        print("📈 Computing metrics...")
        metrics = self.metrics_calc.compute_all_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        # Compute calibration metrics
        calibration_metrics = compute_calibration_metrics(all_labels, all_probabilities)
        metrics.update(calibration_metrics)
        
        # Per-emotion analysis
        per_emotion_accuracy = {}
        for emotion, correct_list in per_emotion_correct.items():
            if correct_list:
                per_emotion_accuracy[emotion] = np.mean(correct_list)
            else:
                per_emotion_accuracy[emotion] = 0.0
        
        metrics['per_emotion_accuracy'] = per_emotion_accuracy
        
        # Per-speaker analysis
        per_speaker_metrics = {}
        for speaker, results in per_speaker_results.items():
            if results['preds']:
                speaker_acc = np.mean(np.array(results['preds']) == np.array(results['labels']))
                per_speaker_metrics[f'{speaker}_accuracy'] = speaker_acc
            else:
                per_speaker_metrics[f'{speaker}_accuracy'] = 0.0
        
        metrics.update(per_speaker_metrics)
        
        # Print results summary
        self.metrics_calc.print_metrics_summary(metrics)
        
        # Additional analysis
        if all_uncertainties:
            print(f"\n UNCERTAINTY ANALYSIS:")
            print(f"   Mean uncertainty: {np.mean(all_uncertainties):.4f}")
            print(f"   Std uncertainty:  {np.std(all_uncertainties):.4f}")
        
        print(f"\nPER-EMOTION ACCURACY:")
        for emotion, acc in per_emotion_accuracy.items():
            print(f"   {emotion:12s}: {acc:.3f}")
        
        print(f"\n PER-SPEAKER ACCURACY:")
        for speaker_metric, acc in per_speaker_metrics.items():
            print(f"   {speaker_metric:12s}: {acc:.3f}")
        
        # Save results
        if save_predictions or save_explanations:
            self._save_evaluation_results(
                metrics=metrics,
                predictions=all_predictions,
                probabilities=all_probabilities,
                labels=all_labels,
                confidences=all_confidences,
                uncertainties=all_uncertainties if all_uncertainties else None,
                attention_maps=attention_maps if attention_maps else None,
                feature_attributions=feature_attributions if feature_attributions else None,
                importance_scores=importance_scores if importance_scores else None,
                results_dir=results_dir
            )
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'labels': all_labels,
            'confidences': all_confidences,
            'uncertainties': all_uncertainties if all_uncertainties else None
        }
    
    def _save_evaluation_results(
        self,
        metrics: Dict,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
        attention_maps: Optional[List] = None,
        feature_attributions: Optional[List] = None,
        importance_scores: Optional[List] = None,
        results_dir: str = './results'
    ):
        """Save comprehensive evaluation results"""
        
        results_path = Path(results_dir)
        results_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving evaluation results to {results_path}")
        
        # Save metrics and plots
        self.metrics_calc.save_results(
            metrics=metrics,
            save_dir=str(results_path),
            experiment_name="iemocap_memocmt_evaluation"
        )
        
        # Save detailed predictions
        predictions_df = pd.DataFrame({
            'true_label': labels,
            'predicted_label': predictions,
            'confidence': confidences,
            'correct': predictions == labels
        })
        
        # Add class names
        predictions_df['true_emotion'] = [self.metrics_calc.class_names[i] for i in labels]
        predictions_df['predicted_emotion'] = [self.metrics_calc.class_names[i] for i in predictions]
        
        # Add probabilities for each class
        for i, emotion in enumerate(self.metrics_calc.class_names):
            predictions_df[f'prob_{emotion}'] = probabilities[:, i]
        
        # Add uncertainties if available
        if uncertainties is not None:
            predictions_df['uncertainty'] = uncertainties
        
        predictions_df.to_csv(results_path / 'detailed_predictions.csv', index=False)
        
        # Create additional visualizations
        self._create_additional_visualizations(
            metrics, predictions, probabilities, labels, confidences,
            uncertainties, results_path
        )
        
        # Save explainability results if available
        if attention_maps and feature_attributions:
            self._save_explainability_results(
                attention_maps, feature_attributions, importance_scores, results_path
            )
        
        print(f"All results saved to {results_path}")
    
    def _create_additional_visualizations(
        self,
        metrics: Dict,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: np.ndarray,
        confidences: np.ndarray,
        uncertainties: Optional[np.ndarray],
        results_path: Path
    ):
        """Create additional visualization plots"""
        
        # Confidence distribution plot
        plt.figure(figsize=(10, 6))
        correct_mask = predictions == labels
        
        plt.hist(confidences[correct_mask], bins=30, alpha=0.7, label='Correct', color='green')
        plt.hist(confidences[~correct_mask], bins=30, alpha=0.7, label='Incorrect', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(results_path / 'confidence_distribution.png', dpi=300)
        plt.close()
        
        # Uncertainty vs Confidence plot (if uncertainty available)
        if uncertainties is not None:
            plt.figure(figsize=(10, 6))
            colors = ['green' if c else 'red' for c in correct_mask]
            plt.scatter(confidences, uncertainties, c=colors, alpha=0.6)
            plt.xlabel('Confidence')
            plt.ylabel('Uncertainty')
            plt.title('Confidence vs Uncertainty')
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(confidences, uncertainties, 1)
            p = np.poly1d(z)
            plt.plot(confidences, p(confidences), "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.savefig(results_path / 'confidence_vs_uncertainty.png', dpi=300)
            plt.close()
        
        # Per-class probability distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, emotion in enumerate(self.metrics_calc.class_names):
            if i < len(axes):
                # Get probabilities for this class
                class_probs = probabilities[:, i]
                true_class_mask = labels == i
                
                axes[i].hist(class_probs[true_class_mask], bins=20, alpha=0.7, 
                           label=f'True {emotion}', color='blue')
                axes[i].hist(class_probs[~true_class_mask], bins=20, alpha=0.7, 
                           label=f'Not {emotion}', color='orange')
                axes[i].set_title(f'{emotion.capitalize()} Probability Distribution')
                axes[i].set_xlabel('Probability')
                axes[i].set_ylabel('Frequency')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(results_path / 'probability_distributions.png', dpi=300)
        plt.close()
        
        print(f"Additional visualizations saved")
    
    def _save_explainability_results(
        self,
        attention_maps: List,
        feature_attributions: List,
        importance_scores: List,
        results_path: Path
    ):
        """Save explainability results"""
        
        print("Saving explainability results...")
        
        # Create explainability directory
        explain_dir = results_path / 'explainability'
        explain_dir.mkdir(exist_ok=True)
        
        # Save attention maps (sample)
        if attention_maps:
            # Take first few samples for visualization
            for i, attn_map in enumerate(attention_maps[:5]):
                if isinstance(attn_map, list) and len(attn_map) > 0:
                    # Save attention weights from different layers
                    for layer_idx, layer_attn in enumerate(attn_map):
                        if 'text_cross' in layer_attn:
                            # Create attention heatmap
                            attention_weights = layer_attn['text_cross']
                            if isinstance(attention_weights, torch.Tensor):
                                attention_weights = attention_weights.detach().cpu().numpy()
                            
                            # Plot attention heatmap
                            plt.figure(figsize=(8, 6))
                            if attention_weights.ndim >= 3:
                                # Average across heads and batch
                                attn_viz = attention_weights[0].mean(0) if attention_weights.shape[0] > 0 else attention_weights.mean(0)
                            else:
                                attn_viz = attention_weights
                            
                            sns.heatmap(attn_viz, cmap='Blues', cbar=True)
                            plt.title(f'Cross-Modal Attention - Sample {i+1}, Layer {layer_idx+1}')
                            plt.xlabel('Audio Features')
                            plt.ylabel('Text Features')
                            plt.tight_layout()
                            plt.savefig(explain_dir / f'attention_sample_{i+1}_layer_{layer_idx+1}.png', dpi=300)
                            plt.close()
        
        # Save feature attributions summary
        if feature_attributions:
            # Aggregate feature attributions
            all_attributions = np.concatenate(feature_attributions, axis=0)
            mean_attribution = np.mean(np.abs(all_attributions), axis=0)
            
            # Create feature importance plot
            plt.figure(figsize=(12, 8))
            feature_indices = np.arange(len(mean_attribution))
            plt.bar(feature_indices, mean_attribution)
            plt.title('Mean Feature Attribution Importance')
            plt.xlabel('Feature Index')
            plt.ylabel('Attribution Magnitude')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(explain_dir / 'feature_attributions.png', dpi=300)
            plt.close()
        
        # Save importance scores summary
        if importance_scores:
            all_importance = np.concatenate(importance_scores, axis=0)
            
            plt.figure(figsize=(10, 6))
            plt.hist(all_importance.flatten(), bins=50, alpha=0.7, color='purple')
            plt.title('Feature Importance Scores Distribution')
            plt.xlabel('Importance Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(explain_dir / 'importance_distribution.png', dpi=300)
            plt.close()
        
        print(f"Explainability results saved to {explain_dir}")


def load_model_from_checkpoint(checkpoint_path: str, config: Dict, tokenizer, processor) -> IEMOCAPMemoCMTModel:
    """Load model from checkpoint"""
    
    print(f" Loading model from checkpoint: {checkpoint_path}")
    
    # Create model
    model = IEMOCAPMemoCMTModel(config, tokenizer, processor)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load state dict (handle Lightning checkpoint format)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    
    print("Model loaded successfully")
    return model


def main():
    """Main evaluation function"""
    
    parser = argparse.ArgumentParser(description='Evaluate IEMOCAP MemoCMT Model')
    
    # Model and config
    parser.add_argument(
        '--config',
        type=str,
        default='configs/iemocap_memocmt.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    
    # Evaluation settings
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./results/evaluation',
        help='Directory to save evaluation results'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    parser.add_argument(
        '--no_explanations',
        action='store_true',
        help='Skip explainability analysis'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Override batch size
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    
    # Initialize tokenizer and processor
    text_config = config.get('model', {}).get('text', {})
    audio_config = config.get('model', {}).get('audio', {})
    
    tokenizer = AutoTokenizer.from_pretrained(
        text_config.get('model_name', 'j-hartmann/emotion-english-distilroberta-base')
    )
    
    processor = Wav2Vec2Processor.from_pretrained(
        audio_config.get('model_name', 'superb/wav2vec2-base-superb-er')
    )
    
    # Load model from checkpoint
    model = load_model_from_checkpoint(args.checkpoint, config, tokenizer, processor)
    
    # Create test dataloader
    dataset_config = config.get('dataset', {})
    root_path = dataset_config.get('root_path', '../iemocap_dataset')
    
    _, _, test_loader = create_dataloaders(
        root_path=root_path,
        tokenizer=tokenizer,
        processor=processor,
        batch_size=args.batch_size,
        num_workers=4,
        sample_rate=dataset_config.get('sample_rate', 16000)
    )
    
    # Create evaluator
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = ModelEvaluator(model, tokenizer, processor, config, device)
    
    # Run evaluation
    results = evaluator.evaluate_model(
        test_loader=test_loader,
        save_predictions=True,
        save_explanations=not args.no_explanations,
        results_dir=args.results_dir
    )
    
    print(f"\n Evaluation completed!")
    print(f"Final Results:")
    print(f"   Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"   Weighted F1: {results['metrics']['weighted_f1']:.4f}")
    print(f"   Macro F1: {results['metrics']['macro_f1']:.4f}")
    
    if 'auc_weighted' in results['metrics']:
        print(f"   Weighted AUC: {results['metrics']['auc_weighted']:.4f}")
    
    print(f"Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
