"""
R-AIME Head: Reinforcement-Augmented Interpretable Multimodal Emotion
Fusion Agent + Decision Agent + Explainability Agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from torch.nn.utils import spectral_norm


class TemperatureScaling(nn.Module):
    """Temperature scaling for model calibration"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits"""
        return logits / self.temperature
    
    def set_temperature(self, valid_loader, model, device):
        """
        Tune the temperature using validation set
        """
        nll_criterion = nn.CrossEntropyLoss()
        
        # First collect all logits and labels from validation set
        logits_list = []
        labels_list = []
        
        model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                # Get model outputs (this would need to be adapted to your model)
                outputs = model(batch)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                labels = batch['labels']
                
                logits_list.append(logits.cpu())
                labels_list.append(labels.cpu())
        
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        
        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        print(f'Before temperature - NLL: {before_temperature_nll:.3f}')
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)
        
        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.forward(logits), labels).item()
        print(f'After temperature - NLL: {after_temperature_nll:.3f}')
        print(f'Optimal temperature: {self.temperature.item():.3f}')


class FusionAgent(nn.Module):
    """Fusion Agent: combines multimodal features into joint representation"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_classes: int = 6,
        dropout: float = 0.15,
        num_layers: int = 3,
        attention_heads: int = 8,
        use_attention: bool = True,
        use_residual: bool = True,
        use_skip_connections: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_skip_connections = use_skip_connections
        self.num_layers = num_layers
        
        # Feature transformation layers
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Enhanced multi-layer fusion network
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Linear(hidden_dim, hidden_dim)),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Skip connections for each layer
        if use_skip_connections and num_layers > 1:
            self.skip_projections = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)
            ])
        
        # Enhanced attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Feature importance computation
        self.importance_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fused_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through fusion agent
        
        Args:
            fused_features: (B, input_dim)
            
        Returns:
            Dict with logits, features, and importance scores
        """
        
        # Initial projection
        x = self.input_proj(fused_features)  # (B, hidden_dim)
        residual = x
        
        # Enhanced multi-layer fusion with skip connections
        layer_outputs = [x]  # Store outputs for skip connections
        
        for i, layer in enumerate(self.fusion_layers):
            x_new = layer(x)
            
            # Apply skip connections from previous layers
            if self.use_skip_connections and i > 0:
                # Skip connection from input
                if i == 1:
                    x_new = x_new + self.skip_projections[i-1](residual)
                # Skip connection from previous layer
                else:
                    x_new = x_new + self.skip_projections[i-1](layer_outputs[-1])
            
            if self.use_residual:
                x = x + x_new  # Residual connection
            else:
                x = x_new
                
            layer_outputs.append(x)
        
        # Self-attention for feature refinement
        if self.use_attention:
            x_seq = x.unsqueeze(1)  # (B, 1, hidden_dim)
            attended, attention_weights = self.attention(x_seq, x_seq, x_seq)
            x_attended = attended.squeeze(1)  # (B, hidden_dim)
            x = self.attention_norm(x + x_attended)
        else:
            attention_weights = None
        
        # Compute feature importance
        importance_scores = self.importance_head(x)  # (B, 1)
        
        # Apply importance weighting
        weighted_features = x * importance_scores  # (B, hidden_dim)
        
        # Final classification
        logits = self.classifier(weighted_features)  # (B, num_classes)
        
        return {
            'logits': logits,
            'features': weighted_features,
            'raw_features': x,
            'importance_scores': importance_scores,
            'attention_weights': attention_weights
        }


class DecisionAgent(nn.Module):
    """Decision Agent: calibrates predictions and provides confidence estimates"""
    
    def __init__(
        self,
        num_classes: int = 6,
        use_temperature: bool = True,
        use_confidence: bool = True,
        uncertainty_method: str = "monte_carlo",
        mc_samples: int = 10
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.use_temperature = use_temperature
        self.use_confidence = use_confidence
        self.uncertainty_method = uncertainty_method
        self.mc_samples = mc_samples
        
        # Temperature scaling for calibration
        if use_temperature:
            self.temperature_scaling = TemperatureScaling()
        
        # Enhanced confidence estimation network
        if use_confidence:
            self.confidence_head = nn.Sequential(
                nn.Linear(num_classes, num_classes),
                nn.ReLU(),
                nn.Linear(num_classes, num_classes // 2),
                nn.ReLU(),
                nn.Linear(num_classes // 2, 1),
                nn.Sigmoid()
            )
        
        # Enhanced uncertainty estimation based on method
        if uncertainty_method == "monte_carlo":
            self.uncertainty_head = nn.Sequential(
                nn.Dropout(0.2),  # MC dropout
                nn.Linear(num_classes, num_classes // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(num_classes // 2, 1),
                nn.Softplus()  # Ensure positive output
            )
        else:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(num_classes, 1),
                nn.Softplus()  # Ensure positive output
            )
    
    def forward(
        self, 
        logits: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through decision agent
        
        Args:
            logits: (B, num_classes) - raw logits from fusion agent
            return_uncertainty: Whether to compute uncertainty estimates
            
        Returns:
            Dict with calibrated probabilities, confidence, and uncertainty
        """
        
        # Apply temperature scaling
        if self.use_temperature:
            calibrated_logits = self.temperature_scaling(logits)
        else:
            calibrated_logits = logits
        
        # Convert to probabilities
        probabilities = F.softmax(calibrated_logits, dim=-1)  # (B, num_classes)
        
        # Compute confidence
        if self.use_confidence:
            confidence = self.confidence_head(probabilities)  # (B, 1)
        else:
            # Use max probability as confidence
            confidence = torch.max(probabilities, dim=-1, keepdim=True)[0]
        
        # Compute uncertainty if requested
        uncertainty = None
        if return_uncertainty:
            # Predictive uncertainty (aleatoric + epistemic approximation)
            uncertainty = self.uncertainty_head(logits)  # (B, 1)
        
        # Get predictions
        predictions = torch.argmax(probabilities, dim=-1)  # (B,)
        
        return {
            'probabilities': probabilities,
            'predictions': predictions,
            'confidence': confidence,
            'uncertainty': uncertainty,
            'calibrated_logits': calibrated_logits,
            'temperature': self.temperature_scaling.temperature if self.use_temperature else None
        }


class ExplainabilityAgent(nn.Module):
    """Explainability Agent: provides interpretability through attention and gradients"""
    
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 6,
        attribution_method: str = "integrated_gradients",
        concept_bottleneck: bool = True
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.attribution_method = attribution_method
        self.concept_bottleneck = concept_bottleneck
        
        # Enhanced feature attribution network
        self.attribution_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Tanh()  # Attribution scores in [-1, 1]
        )
        
        # Enhanced concept activation layers
        if concept_bottleneck:
            # Concept bottleneck layers for interpretability
            self.concept_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.Sigmoid()  # Concept activations in [0, 1]
                ) for _ in range(num_classes)
            ])
        else:
            self.concept_layers = nn.ModuleList([
                nn.Linear(feature_dim, 64) for _ in range(num_classes)
            ])
    
    def compute_feature_attribution(
        self,
        features: torch.Tensor,
        target_class: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute feature attribution scores"""
        
        attribution_scores = self.attribution_head(features)  # (B, feature_dim)
        
        return attribution_scores
    
    def compute_integrated_gradients(
        self,
        model: nn.Module,
        inputs: torch.Tensor,
        target_class: torch.Tensor,
        steps: int = 50
    ) -> torch.Tensor:
        """
        Compute Integrated Gradients for input attribution
        
        Args:
            model: The model to compute gradients for
            inputs: Input features (B, feature_dim)
            target_class: Target class indices (B,)
            steps: Number of integration steps
            
        Returns:
            Attribution scores (B, feature_dim)
        """
        
        if not self.attribution_method == "integrated_gradients":
            return torch.zeros_like(inputs)
        
        # Create baseline (zeros)
        baseline = torch.zeros_like(inputs)
        
        # Generate alpha values
        alphas = torch.linspace(0, 1, steps, device=inputs.device)
        
        integrated_grads = torch.zeros_like(inputs)
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            outputs = model(interpolated)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
            
            # Compute gradients w.r.t. target class
            target_logits = logits.gather(1, target_class.unsqueeze(1)).squeeze(1)
            gradients = torch.autograd.grad(
                outputs=target_logits.sum(),
                inputs=interpolated,
                create_graph=False,
                retain_graph=False
            )[0]
            
            integrated_grads += gradients
        
        # Average and scale by (input - baseline)
        integrated_grads = integrated_grads / steps
        integrated_grads = integrated_grads * (inputs - baseline)
        
        return integrated_grads
    
    def compute_concept_activations(
        self,
        features: torch.Tensor,
        class_idx: Optional[int] = None
    ) -> torch.Tensor:
        """Compute concept activation vectors"""
        
        if class_idx is not None:
            # Activation for specific class
            activations = self.concept_layers[class_idx](features)
        else:
            # Activations for all classes
            all_activations = []
            for layer in self.concept_layers:
                activations = layer(features)
                all_activations.append(activations)
            activations = torch.stack(all_activations, dim=1)  # (B, num_classes, 64)
        
        return activations
    
    def forward(
        self,
        features: torch.Tensor,
        probabilities: torch.Tensor,
        predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through explainability agent
        
        Args:
            features: (B, feature_dim) - features from fusion agent
            probabilities: (B, num_classes) - class probabilities
            predictions: (B,) - predicted classes
            
        Returns:
            Dict with various explainability metrics
        """
        
        # Feature attribution
        feature_attribution = self.compute_feature_attribution(features, predictions)
        
        # Concept activations
        concept_activations = self.compute_concept_activations(features)
        
        # Attention-based importance (using probability distribution)
        attention_importance = probabilities.unsqueeze(-1) * features.unsqueeze(1)  # (B, num_classes, feature_dim)
        attention_importance = attention_importance.sum(dim=1)  # (B, feature_dim)
        
        # Confidence-based explanations
        confidence = torch.max(probabilities, dim=-1)[0]  # (B,)
        confidence_weighted_features = features * confidence.unsqueeze(-1)
        
        return {
            'feature_attribution': feature_attribution,
            'concept_activations': concept_activations,
            'attention_importance': attention_importance,
            'confidence_weighted_features': confidence_weighted_features
        }


class RAIMEHead(nn.Module):
    """
    R-AIME Head: Complete multimodal emotion recognition head
    Combines Fusion Agent + Decision Agent + Explainability Agent
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        hidden_dim: int = 512,
        dropout: float = 0.15,
        use_temperature: bool = True,
        explainability: bool = True,
        num_fusion_layers: int = 3,
        attention_heads: int = 8,
        use_skip_connections: bool = True,
        uncertainty_method: str = "monte_carlo",
        mc_samples: int = 10,
        feature_attribution_method: str = "integrated_gradients",
        concept_bottleneck: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.explainability = explainability
        self.num_fusion_layers = num_fusion_layers
        self.use_skip_connections = use_skip_connections
        self.uncertainty_method = uncertainty_method
        self.mc_samples = mc_samples
        
        # Enhanced Fusion Agent
        self.fusion_agent = FusionAgent(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            num_layers=num_fusion_layers,
            attention_heads=attention_heads,
            use_skip_connections=use_skip_connections
        )
        
        # Enhanced Decision Agent
        self.decision_agent = DecisionAgent(
            num_classes=num_classes,
            use_temperature=use_temperature,
            uncertainty_method=uncertainty_method,
            mc_samples=mc_samples
        )
        
        if explainability:
            self.explainability_agent = ExplainabilityAgent(
                feature_dim=hidden_dim,
                num_classes=num_classes,
                attribution_method=feature_attribution_method,
                concept_bottleneck=concept_bottleneck
            )
        
        print(f"RAIMEHead initialized:")
        print(f"   Input dim: {input_dim}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Num classes: {num_classes}")
        print(f"   Temperature scaling: {use_temperature}")
        print(f"   Explainability: {explainability}")
    
    def forward(
        self,
        fused_features: torch.Tensor,
        return_explanations: bool = True,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through R-AIME head
        
        Args:
            fused_features: (B, input_dim) - fused multimodal features
            return_explanations: Whether to compute explainability
            return_uncertainty: Whether to compute uncertainty estimates
            
        Returns:
            Comprehensive output dict with predictions, confidence, and explanations
        """
        
        # 1. Fusion Agent
        fusion_outputs = self.fusion_agent(fused_features)
        
        # 2. Decision Agent
        decision_outputs = self.decision_agent(
            fusion_outputs['logits'],
            return_uncertainty=return_uncertainty
        )
        
        # 3. Explainability Agent (if enabled)
        explanations = {}
        if self.explainability and return_explanations:
            explanations = self.explainability_agent(
                features=fusion_outputs['features'],
                probabilities=decision_outputs['probabilities'],
                predictions=decision_outputs['predictions']
            )
        
        # Combine all outputs
        outputs = {
            # Core predictions
            'logits': fusion_outputs['logits'],
            'probabilities': decision_outputs['probabilities'],
            'predictions': decision_outputs['predictions'],
            'confidence': decision_outputs['confidence'],
            
            # Features
            'fused_features': fused_features,
            'processed_features': fusion_outputs['features'],
            
            # Attention and importance
            'importance_scores': fusion_outputs['importance_scores'],
            'attention_weights': fusion_outputs['attention_weights'],
            
            # Calibration
            'calibrated_logits': decision_outputs['calibrated_logits'],
            'temperature': decision_outputs['temperature'],
        }
        
        # Add uncertainty if requested
        if return_uncertainty:
            outputs['uncertainty'] = decision_outputs['uncertainty']
        
        # Add explanations if available
        outputs.update(explanations)
        
        return outputs
    
    def calibrate(self, valid_loader, model, device):
        """Calibrate the decision agent using validation data"""
        if hasattr(self.decision_agent, 'temperature_scaling'):
            print("Calibrating decision agent...")
            self.decision_agent.temperature_scaling.set_temperature(valid_loader, model, device)
            print("Decision agent calibrated!")


def test_raime_head():
    """Test R-AIME head functionality"""
    
    # Test parameters
    batch_size = 4
    input_dim = 256
    num_classes = 6
    hidden_dim = 512
    
    # Create R-AIME head
    raime_head = RAIMEHead(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        use_temperature=True,
        explainability=True
    )
    
    # Create dummy input
    fused_features = torch.randn(batch_size, input_dim)
    
    # Forward pass
    outputs = raime_head(
        fused_features=fused_features,
        return_explanations=True,
        return_uncertainty=True
    )
    
    print("R-AIME Head test:")
    print(f"Input shape: {fused_features.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Probabilities shape: {outputs['probabilities'].shape}")
    print(f"Predictions shape: {outputs['predictions'].shape}")
    print(f"Confidence shape: {outputs['confidence'].shape}")
    print(f"Uncertainty shape: {outputs['uncertainty'].shape}")
    print(f"Feature attribution shape: {outputs['feature_attribution'].shape}")
    print(f"Concept activations shape: {outputs['concept_activations'].shape}")
    print(f"Temperature: {outputs['temperature']}")
    print("R-AIME head working correctly!")
    
    # Test individual components
    print("\nTesting individual agents:")
    
    # Test fusion agent
    fusion_outputs = raime_head.fusion_agent(fused_features)
    print(f"Fusion agent - Logits: {fusion_outputs['logits'].shape}")
    print(f"Fusion agent - Features: {fusion_outputs['features'].shape}")
    print(f"Fusion agent - Importance: {fusion_outputs['importance_scores'].shape}")
    
    # Test decision agent
    decision_outputs = raime_head.decision_agent(fusion_outputs['logits'], return_uncertainty=True)
    print(f"Decision agent - Probabilities: {decision_outputs['probabilities'].shape}")
    print(f"Decision agent - Confidence: {decision_outputs['confidence'].shape}")
    print(f"Decision agent - Uncertainty: {decision_outputs['uncertainty'].shape}")
    
    # Test explainability agent
    if raime_head.explainability:
        explain_outputs = raime_head.explainability_agent(
            features=fusion_outputs['features'],
            probabilities=decision_outputs['probabilities'],
            predictions=decision_outputs['predictions']
        )
        print(f"Explainability agent - Attribution: {explain_outputs['feature_attribution'].shape}")
        print(f"Explainability agent - Concepts: {explain_outputs['concept_activations'].shape}")
    
    print("All agents working correctly!")


if __name__ == "__main__":
    test_raime_head()
