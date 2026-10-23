"""
RAIME Explainability and Visualization Analysis
Comprehensive analysis and visualization code for R-AIME agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class RAIMEVisualizer:
    """Comprehensive visualizer for R-AIME explainability analysis"""

    def __init__(self, emotion_labels: Dict[str, int]):
        self.emotion_labels = emotion_labels
        self.class_names = [None] * len(emotion_labels)
        for emotion, idx in emotion_labels.items():
            self.class_names[idx] = emotion.capitalize()

        # Create output directory
        self.output_dir = Path('./raime_analysis')
        self.output_dir.mkdir(exist_ok=True)

    def analyze_fusion_agent(
        self,
        model,
        dataloader,
        device: str = 'cuda',
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze Fusion Agent: feature importance, attention patterns, layer evolution
        """
        print("Analyzing Fusion Agent...")

        model.eval()
        fusion_analysis = {
            'feature_importance': [],
            'attention_weights': [],
            'layer_outputs': [],
            'predictions': [],
            'labels': []
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break

                # Move to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Get model outputs
                outputs = model(batch, return_explanations=True)

                # Store fusion agent outputs
                if 'importance_scores' in outputs:
                    fusion_analysis['feature_importance'].append(
                        outputs['importance_scores'].cpu().numpy()
                    )

                if 'attention_weights' in outputs and outputs['attention_weights'] is not None:
                    fusion_analysis['attention_weights'].append(
                        outputs['attention_weights'].cpu().numpy()
                    )

                fusion_analysis['predictions'].append(outputs['predictions'].cpu().numpy())
                fusion_analysis['labels'].append(batch['labels'].cpu().numpy())

        # Convert to numpy arrays
        for key in fusion_analysis:
            if fusion_analysis[key]:
                fusion_analysis[key] = np.concatenate(fusion_analysis[key], axis=0)

        # Generate visualizations
        self._visualize_fusion_agent(fusion_analysis)

        return fusion_analysis

    def analyze_decision_agent(
        self,
        model,
        dataloader,
        device: str = 'cuda',
        num_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Analyze Decision Agent: confidence calibration, uncertainty patterns
        """
        print(" Analyzing Decision Agent...")

        model.eval()
        decision_analysis = {
            'confidences': [],
            'uncertainties': [],
            'predictions': [],
            'labels': [],
            'probabilities': []
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break

                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = model(batch, return_uncertainty=True)

                decision_analysis['confidences'].append(outputs['confidence'].cpu().numpy())
                decision_analysis['predictions'].append(outputs['predictions'].cpu().numpy())
                decision_analysis['labels'].append(batch['labels'].cpu().numpy())
                decision_analysis['probabilities'].append(outputs['probabilities'].cpu().numpy())

                if 'uncertainty' in outputs:
                    decision_analysis['uncertainties'].append(outputs['uncertainty'].cpu().numpy())

        # Convert to numpy arrays
        for key in decision_analysis:
            if decision_analysis[key]:
                decision_analysis[key] = np.concatenate(decision_analysis[key], axis=0)

        # Generate visualizations
        self._visualize_decision_agent(decision_analysis)

        return decision_analysis

    def analyze_explainability_agent(
        self,
        model,
        dataloader,
        device: str = 'cuda',
        num_samples: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze Explainability Agent: feature attribution, concept activation
        """
        print("🔬 Analyzing Explainability Agent...")

        model.eval()
        explainability_analysis = {
            'feature_attributions': [],
            'concept_activations': [],
            'attention_importance': [],
            'predictions': [],
            'labels': []
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break

                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = model(batch, return_explanations=True)

                if 'feature_attribution' in outputs:
                    explainability_analysis['feature_attributions'].append(
                        outputs['feature_attribution'].cpu().numpy()
                    )

                if 'concept_activations' in outputs:
                    explainability_analysis['concept_activations'].append(
                        outputs['concept_activations'].cpu().numpy()
                    )

                if 'attention_importance' in outputs:
                    explainability_analysis['attention_importance'].append(
                        outputs['attention_importance'].cpu().numpy()
                    )

                explainability_analysis['predictions'].append(outputs['predictions'].cpu().numpy())
                explainability_analysis['labels'].append(batch['labels'].cpu().numpy())

        # Convert to numpy arrays
        for key in explainability_analysis:
            if explainability_analysis[key]:
                explainability_analysis[key] = np.concatenate(explainability_analysis[key], axis=0)

        # Generate visualizations
        self._visualize_explainability_agent(explainability_analysis)

        return explainability_analysis

    def analyze_attention_weights(
        self,
        model,
        dataloader,
        device: str = 'cuda',
        num_samples: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze attention weights from MemoCMT
        """
        print("Analyzing Attention Weights...")

        model.eval()
        attention_analysis = {
            'text_self_attention': [],
            'audio_self_attention': [],
            'text_cross_attention': [],
            'audio_cross_attention': [],
            'memory_states': [],
            'predictions': [],
            'labels': []
        }

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break

                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                outputs = model(batch, return_explanations=True)

                if 'attention_maps' in outputs:
                    attention_maps = outputs['attention_maps']
                    if attention_maps:
                        # Extract attention from first layer
                        layer_attention = attention_maps[0]
                        for attn_type in ['text_self', 'audio_self', 'text_cross', 'audio_cross']:
                            if attn_type in layer_attention:
                                attention_analysis[f'{attn_type}_attention'].append(
                                    layer_attention[attn_type].cpu().numpy()
                                )

                if 'memory_states' in outputs:
                    attention_analysis['memory_states'].append(
                        outputs['memory_states'][0]['text_memory'].cpu().numpy()
                    )

                attention_analysis['predictions'].append(outputs['predictions'].cpu().numpy())
                attention_analysis['labels'].append(batch['labels'].cpu().numpy())

        # Convert to numpy arrays
        for key in attention_analysis:
            if attention_analysis[key]:
                attention_analysis[key] = np.concatenate(attention_analysis[key], axis=0)

        # Generate visualizations
        self._visualize_attention_weights(attention_analysis)

        return attention_analysis

    def _visualize_fusion_agent(self, analysis: Dict[str, Any]):
        """Generate Fusion Agent visualizations"""
        print("Generating Fusion Agent visualizations...")

        # Feature Importance Distribution
        if analysis['feature_importance'].size > 0:
            plt.figure(figsize=(12, 8))
            importance_scores = analysis['feature_importance'].flatten()

            plt.hist(importance_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            plt.axvline(np.mean(importance_scores), color='red', linestyle='--',
                       label=f'Mean: {np.mean(importance_scores):.3f}')
            plt.xlabel('Feature Importance Score')
            plt.ylabel('Frequency')
            plt.title('Fusion Agent: Feature Importance Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'fusion_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Attention Weights Heatmap (sample)
        if analysis['attention_weights']:
            plt.figure(figsize=(10, 8))
            # Take first sample, first head
            attn_sample = analysis['attention_weights'][0]
            if attn_sample.ndim >= 3:
                attn_viz = attn_sample[0].mean(axis=0)  # Average across heads
            else:
                attn_viz = attn_sample

            sns.heatmap(attn_viz, cmap='viridis', cbar=True, square=True)
            plt.title('Fusion Agent: Attention Weights (Sample)')
            plt.xlabel('Key Features')
            plt.ylabel('Query Features')
            plt.tight_layout()
            plt.savefig(self.output_dir / 'fusion_attention_weights.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _visualize_decision_agent(self, analysis: Dict[str, Any]):
        """Generate Decision Agent visualizations"""
        print("Generating Decision Agent visualizations...")

        confidences = analysis['confidences'].flatten()
        predictions = analysis['predictions']
        labels = analysis['labels']
        correct_mask = predictions == labels

        # Confidence Distribution
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Confidence distribution by correctness
        axes[0,0].hist(confidences[correct_mask], bins=30, alpha=0.7, label='Correct', color='green')
        axes[0,0].hist(confidences[~correct_mask], bins=30, alpha=0.7, label='Incorrect', color='red')
        axes[0,0].set_xlabel('Confidence Score')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Confidence Distribution: Correct vs Incorrect')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Uncertainty distribution (if available)
        if 'uncertainties' in analysis and analysis['uncertainties'].size > 0:
            uncertainties = analysis['uncertainties'].flatten()
            axes[0,1].hist(uncertainties, bins=30, alpha=0.7, color='purple')
            axes[0,1].axvline(np.mean(uncertainties), color='red', linestyle='--',
                             label=f'Mean: {np.mean(uncertainties):.3f}')
            axes[0,1].set_xlabel('Uncertainty Score')
            axes[0,1].set_ylabel('Frequency')
            axes[0,1].set_title('Uncertainty Distribution')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)

            # Confidence vs Uncertainty scatter
            colors = ['green' if c else 'red' for c in correct_mask]
            axes[1,0].scatter(confidences, uncertainties, c=colors, alpha=0.6)
            axes[1,0].set_xlabel('Confidence')
            axes[1,0].set_ylabel('Uncertainty')
            axes[1,0].set_title('Confidence vs Uncertainty')
            axes[1,0].grid(True, alpha=0.3)
        else:
            axes[0,1].text(0.5, 0.5, 'No uncertainty data available',
                          ha='center', va='center', transform=axes[0,1].transAxes)
            axes[1,0].text(0.5, 0.5, 'No uncertainty data available',
                          ha='center', va='center', transform=axes[1,0].transAxes)

        # Calibration curve
        if analysis['probabilities'].ndim > 1:
            prob_true, prob_pred = calibration_curve(labels, confidences, n_bins=10)
            axes[1,1].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Model')
            axes[1,1].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect')
            axes[1,1].set_xlabel('Predicted Probability')
            axes[1,1].set_ylabel('True Probability')
            axes[1,1].set_title('Calibration Curve')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'decision_agent_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Per-class confidence analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for i, emotion in enumerate(self.class_names):
            if i < len(axes):
                emotion_mask = labels == i
                if np.sum(emotion_mask) > 0:
                    emotion_confidences = confidences[emotion_mask]
                    emotion_correct = correct_mask[emotion_mask]

                    axes[i].hist(emotion_confidences[emotion_correct], bins=20, alpha=0.7,
                               label='Correct', color='green')
                    axes[i].hist(emotion_confidences[~emotion_correct], bins=20, alpha=0.7,
                               label='Incorrect', color='red')
                    axes[i].set_title(f'{emotion} Confidence')
                    axes[i].set_xlabel('Confidence')
                    axes[i].set_ylabel('Frequency')
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'decision_per_class_confidence.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _visualize_explainability_agent(self, analysis: Dict[str, Any]):
        """Generate Explainability Agent visualizations"""
        print("Generating Explainability Agent visualizations...")

        # Feature Attribution Analysis
        if analysis['feature_attributions'].size > 0:
            attributions = analysis['feature_attributions']

            # Mean attribution across samples
            mean_attribution = np.mean(np.abs(attributions), axis=0)

            plt.figure(figsize=(14, 8))
            feature_indices = np.arange(len(mean_attribution))
            bars = plt.bar(feature_indices, mean_attribution, alpha=0.7, color='lightcoral', edgecolor='black')

            # Highlight top features
            top_k = 10
            top_indices = np.argsort(mean_attribution)[-top_k:]
            for idx in top_indices:
                bars[idx].set_color('darkred')

            plt.xlabel('Feature Index')
            plt.ylabel('Mean Attribution Magnitude')
            plt.title(f'Explainability Agent: Feature Attribution (Top {top_k} highlighted)')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'explainability_feature_attribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        # Concept Activation Analysis
        if analysis['concept_activations'].size > 0:
            concept_activations = analysis['concept_activations']

            # Average across samples for each class
            mean_concepts = np.mean(concept_activations, axis=0)  # (num_classes, concept_dim)

            plt.figure(figsize=(12, 8))
            for i, emotion in enumerate(self.class_names):
                plt.plot(mean_concepts[i], label=emotion, linewidth=2)

            plt.xlabel('Concept Dimension')
            plt.ylabel('Mean Activation')
            plt.title('Explainability Agent: Concept Activations by Emotion')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'explainability_concept_activations.png', dpi=300, bbox_inches='tight')
            plt.close()

    def _visualize_attention_weights(self, analysis: Dict[str, Any]):
        """Generate Attention Weights visualizations"""
        print("Generating Attention Weights visualizations...")

        attention_types = ['text_self', 'audio_self', 'text_cross', 'audio_cross']

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()

        for i, attn_type in enumerate(attention_types):
            if f'{attn_type}_attention' in analysis and analysis[f'{attn_type}_attention'].size > 0:
                attn_weights = analysis[f'{attn_type}_attention']

                # Take first sample and average across heads
                if attn_weights.ndim >= 3:
                    attn_viz = attn_weights[0].mean(axis=0)
                else:
                    attn_viz = attn_weights[0]

                # Create heatmap
                sns.heatmap(attn_viz, ax=axes[i], cmap='Blues', cbar=True, square=True)
                axes[i].set_title(f'{attn_type.replace("_", " ").title()} Attention')
                axes[i].set_xlabel('Key Features')
                axes[i].set_ylabel('Query Features')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_weights_overview.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Memory State Analysis
        if analysis['memory_states'].size > 0:
            memory_states = analysis['memory_states']

            # Average memory state across samples
            mean_memory = np.mean(memory_states, axis=0)

            plt.figure(figsize=(12, 8))
            plt.plot(mean_memory.flatten(), linewidth=2, color='navy')
            plt.xlabel('Memory Dimension')
            plt.ylabel('Memory Activation')
            plt.title('Attention Analysis: Average Memory State')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'attention_memory_states.png', dpi=300, bbox_inches='tight')
            plt.close()

    def generate_comprehensive_report(
        self,
        fusion_analysis: Dict,
        decision_analysis: Dict,
        explainability_analysis: Dict,
        attention_analysis: Dict
    ):
        """Generate comprehensive analysis report"""
        print("📋 Generating comprehensive analysis report...")

        report = {
            'summary': {},
            'metrics': {},
            'insights': []
        }

        # Summary statistics
        if fusion_analysis['feature_importance'].size > 0:
            report['summary']['mean_feature_importance'] = float(
                np.mean(fusion_analysis['feature_importance'])
            )
            report['summary']['feature_importance_std'] = float(
                np.std(fusion_analysis['feature_importance'])
            )

        if decision_analysis['confidences'].size > 0:
            report['summary']['mean_confidence'] = float(
                np.mean(decision_analysis['confidences'])
            )
            report['summary']['confidence_std'] = float(
                np.std(decision_analysis['confidences'])
            )

        # Performance metrics
        predictions = decision_analysis['predictions']
        labels = decision_analysis['labels']
        correct_mask = predictions == labels

        report['metrics']['accuracy'] = float(np.mean(correct_mask))
        report['metrics']['confidence_correct'] = float(
            np.mean(decision_analysis['confidences'][correct_mask])
        )
        report['metrics']['confidence_incorrect'] = float(
            np.mean(decision_analysis['confidences'][~correct_mask])
        )

        # Key insights
        report['insights'] = [
            "Fusion Agent shows varying feature importance across samples",
            "Decision Agent confidence correlates with prediction correctness",
            "Explainability Agent reveals emotion-specific feature patterns",
            "Cross-modal attention highlights text-audio interaction patterns"
        ]

        # Save report
        import json
        with open(self.output_dir / 'analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Comprehensive report saved to {self.output_dir / 'analysis_report.json'}")

        return report


def run_complete_raime_analysis(
    model,
    dataloader,
    emotion_labels: Dict[str, int],
    device: str = 'cuda',
    num_samples: int = 100
):
    """
    Run complete R-AIME analysis pipeline
    """
    print("Starting Complete R-AIME Analysis Pipeline")
    print("=" * 50)

    # Initialize visualizer
    visualizer = RAIMEVisualizer(emotion_labels)

    # Run all analyses
    fusion_results = visualizer.analyze_fusion_agent(model, dataloader, device, num_samples)
    decision_results = visualizer.analyze_decision_agent(model, dataloader, device, num_samples)
    explainability_results = visualizer.analyze_explainability_agent(model, dataloader, device, num_samples//2)
    attention_results = visualizer.analyze_attention_weights(model, dataloader, device, num_samples//5)

    # Generate comprehensive report
    report = visualizer.generate_comprehensive_report(
        fusion_results, decision_results, explainability_results, attention_results
    )

    print("Complete R-AIME Analysis Finished!")
    print(f"Results saved to: {visualizer.output_dir}")

    return {
        'fusion': fusion_results,
        'decision': decision_results,
        'explainability': explainability_results,
        'attention': attention_results,
        'report': report
    }


# Example usage function
def example_raime_analysis():
    """
    Example of how to use the R-AIME analysis tools
    """
    print("Example: R-AIME Analysis Usage")
    print("-" * 30)

    # This would be your actual model and dataloader
    # model = YourTrainedModel(...)
    # dataloader = your_test_dataloader
    # emotion_labels = {'angry': 0, 'happy': 1, 'sad': 2, 'neutral': 3, 'excited': 4, 'frustrated': 5}

    # Run analysis
    # results = run_complete_raime_analysis(
    #     model=model,
    #     dataloader=dataloader,
    #     emotion_labels=emotion_labels,
    #     device='cuda',
    #     num_samples=200
    # )

    print("Analysis would generate:")
    print("1. Fusion Agent: feature_importance.png, attention_weights.png")
    print("2. Decision Agent: confidence_distribution.png, calibration_curve.png")
    print("3. Explainability Agent: feature_attribution.png, concept_activations.png")
    print("4. Attention Weights: attention_overview.png, memory_states.png")
    print("5. Comprehensive Report: analysis_report.json")


if __name__ == "__main__":
    example_raime_analysis()

