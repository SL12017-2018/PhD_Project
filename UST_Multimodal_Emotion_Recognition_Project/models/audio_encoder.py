"""
Audio Encoder using Pretrained Wav2Vec2 for Speech Emotion Recognition
Wrapper around superb/wav2vec2-base-superb-er
"""

import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor, AutoConfig
from typing import Dict, Optional, Tuple
import torch.nn.functional as F


class AudioEmotionEncoder(nn.Module):
    """Audio encoder using pretrained wav2vec2 for speech emotion recognition"""
    
    def __init__(
        self,
        model_name: str = "superb/wav2vec2-base-superb-er",
        freeze: bool = True,
        unfreeze_layers: int = 2,
        output_dim: int = 768,
        sample_rate: int = 16000,
        pool: str = "mean",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.freeze = freeze
        self.unfreeze_layers = unfreeze_layers
        self.output_dim = output_dim
        self.sample_rate = sample_rate
        self.pool = pool
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Get the hidden size from config
        if hasattr(self.config, 'hidden_size'):
            hidden_size = self.config.hidden_size
        elif hasattr(self.config, 'd_model'):
            hidden_size = self.config.d_model
        else:
            hidden_size = 768  # Default for wav2vec2-base
        
        # Freeze parameters if specified
        if freeze:
            self.freeze_encoder()
        
        # Projection layer to standardize output dimension
        self.projection = nn.Linear(hidden_size, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Audio preprocessing layers (for robustness)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        print(f"AudioEmotionEncoder initialized:")
        print(f"   Model: {model_name}")
        print(f"   Hidden size: {hidden_size}")
        print(f"   Output dim: {output_dim}")
        print(f"   Sample rate: {sample_rate}")
        print(f"   Pool method: {pool}")
        print(f"   Frozen: {freeze}")
        if freeze:
            print(f"   Unfreezing top {unfreeze_layers} layers later")
    
    def freeze_encoder(self):
        """Freeze all encoder parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
        print("Audio encoder frozen")
    
    def unfreeze_top_layers(self, num_layers: Optional[int] = None):
        """Unfreeze top N transformer layers"""
        if num_layers is None:
            num_layers = self.unfreeze_layers
        
        # Get transformer layers (wav2vec2 structure)
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
            layers = self.model.encoder.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'layers'):
            layers = self.model.transformer.layers
        else:
            # Fallback: try to find any layer-like structures
            layers = []
            for name, module in self.model.named_modules():
                if 'layer' in name.lower() and isinstance(module, nn.TransformerEncoderLayer):
                    layers.append(module)
        
        # Unfreeze top layers
        total_layers = len(layers)
        layers_to_unfreeze = min(num_layers, total_layers)
        
        if layers_to_unfreeze > 0:
            for i in range(total_layers - layers_to_unfreeze, total_layers):
                for param in layers[i].parameters():
                    param.requires_grad = True
        
        # Unfreeze final layer norm and projection if exists
        if hasattr(self.model, 'layer_norm') and self.model.layer_norm is not None:
            for param in self.model.layer_norm.parameters():
                param.requires_grad = True
        
        if hasattr(self.model, 'projector') and self.model.projector is not None:
            for param in self.model.projector.parameters():
                param.requires_grad = True
        
        print(f"Unfroze top {layers_to_unfreeze} layers of audio encoder")
    
    def unfreeze_layers(self, num_layers: int):
        """Alias for unfreeze_top_layers for compatibility"""
        self.unfreeze_top_layers(num_layers)
    
    def pool_features(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool frame representations to utterance representation"""
        
        if attention_mask is None:
            # Create attention mask based on non-zero frames
            attention_mask = (hidden_states.abs().sum(dim=-1) > 1e-6).float()
        
        if self.pool == "mean":
            # Mean pooling with attention mask
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(hidden_states, dim=1)
        
        elif self.pool == "max":
            # Max pooling
            if attention_mask is not None:
                hidden_states = hidden_states.masked_fill(
                    attention_mask.unsqueeze(-1) == 0, -1e9
                )
            return torch.max(hidden_states, dim=1)[0]
        
        elif self.pool == "last":
            # Use last non-padded frame
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1).long() - 1  # -1 for 0-indexing
                lengths = torch.clamp(lengths, 0, hidden_states.size(1) - 1)
                return hidden_states[torch.arange(hidden_states.size(0)), lengths]
            else:
                return hidden_states[:, -1, :]
        
        elif self.pool == "first":
            # Use first frame
            return hidden_states[:, 0, :]
        
        elif self.pool == "attention":
            # Learnable attention pooling
            if not hasattr(self, 'attention_weights'):
                self.attention_weights = nn.Parameter(torch.randn(hidden_states.size(-1), 1))
            
            # Compute attention scores
            attn_scores = torch.matmul(hidden_states, self.attention_weights).squeeze(-1)
            
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
            
            attn_weights = F.softmax(attn_scores, dim=1)
            return torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pool}")
    
    def forward(self, audio_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through audio encoder
        
        Args:
            audio_inputs: Dict with 'input_values' and optionally 'attention_mask'
            
        Returns:
            Dict with:
                - 'turn_pooled': (B, output_dim) - pooled representations
                - 'hidden_states': (B, seq_len, hidden_dim) - all frame representations
                - 'attention_mask': (B, seq_len) - attention mask
        """
        
        input_values = audio_inputs['input_values']
        attention_mask = audio_inputs.get('attention_mask', None)
        
        # Forward through wav2vec2
        outputs = self.model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states (last layer)
        hidden_states = outputs.last_hidden_state  # (B, seq_len, hidden_dim)
        
        # Apply layer norm for stability
        hidden_states = self.layer_norm(hidden_states)
        
        # Pool to utterance representation
        pooled = self.pool_features(hidden_states, attention_mask)  # (B, hidden_dim)
        
        # Apply dropout and projection
        pooled = self.dropout(pooled)
        turn_pooled = self.projection(pooled)  # (B, output_dim)
        
        return {
            'turn_pooled': turn_pooled,
            'hidden_states': hidden_states,
            'attention_mask': attention_mask,
            'pooled_raw': pooled
        }
    
    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract features directly from waveform (for debugging)"""
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension
        
        # Create input dict
        audio_inputs = {'input_values': waveform}
        
        with torch.no_grad():
            outputs = self.forward(audio_inputs)
        
        return outputs['turn_pooled']


class SpectralAugmentedAudioEncoder(AudioEmotionEncoder):
    """Audio encoder with spectral augmentation for robustness"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Spectral augmentation parameters
        self.freq_mask_param = 30
        self.time_mask_param = 40
        self.augment_prob = 0.3
    
    def apply_spec_augment(self, hidden_states: torch.Tensor, training: bool = True) -> torch.Tensor:
        """Apply SpecAugment to hidden states during training"""
        
        if not training or torch.rand(1).item() > self.augment_prob:
            return hidden_states
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Time masking
        if seq_len > self.time_mask_param:
            time_mask_len = torch.randint(0, min(self.time_mask_param, seq_len // 4), (1,)).item()
            time_mask_start = torch.randint(0, seq_len - time_mask_len, (1,)).item()
            
            hidden_states = hidden_states.clone()
            hidden_states[:, time_mask_start:time_mask_start + time_mask_len, :] = 0
        
        return hidden_states
    
    def forward(self, audio_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward with spectral augmentation"""
        
        # Get base outputs
        outputs = super().forward(audio_inputs)
        
        # Apply spectral augmentation to hidden states
        if self.training:
            outputs['hidden_states'] = self.apply_spec_augment(
                outputs['hidden_states'], 
                training=self.training
            )
            
            # Re-pool with augmented features
            pooled = self.pool_features(
                outputs['hidden_states'], 
                outputs['attention_mask']
            )
            pooled = self.dropout(pooled)
            outputs['turn_pooled'] = self.projection(pooled)
        
        return outputs


def test_audio_encoder():
    """Test the audio encoder"""
    
    from transformers import Wav2Vec2Processor
    import numpy as np
    
    # Initialize processor and encoder
    processor = Wav2Vec2Processor.from_pretrained("superb/wav2vec2-base-superb-er")
    encoder = AudioEmotionEncoder(freeze=True)
    
    # Create dummy audio (3 samples, 2 seconds each at 16kHz)
    sample_rate = 16000
    duration = 2.0
    audio_length = int(sample_rate * duration)
    
    # Generate dummy audio signals
    dummy_audio = []
    for i in range(3):
        # Create different frequency patterns for variety
        t = np.linspace(0, duration, audio_length)
        freq = 440 + i * 110  # Different frequencies
        audio = np.sin(2 * np.pi * freq * t) * 0.3
        dummy_audio.append(audio)
    
    # Process audio
    inputs = processor(
        dummy_audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = encoder(inputs)
    
    print("Audio encoder test:")
    print(f"Input audio samples: {len(dummy_audio)}")
    print(f"Input values shape: {inputs['input_values'].shape}")
    print(f"Turn pooled shape: {outputs['turn_pooled'].shape}")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    print("Audio encoder working correctly!")
    
    # Test unfreezing
    encoder.unfreeze_top_layers(2)
    print("Unfreezing test completed!")
    
    # Test spectral augmented version
    aug_encoder = SpectralAugmentedAudioEncoder(freeze=True)
    aug_encoder.train()  # Enable training mode for augmentation
    
    with torch.no_grad():
        aug_outputs = aug_encoder(inputs)
    
    print(f"Augmented encoder output shape: {aug_outputs['turn_pooled'].shape}")
    print("Spectral augmented encoder working correctly!")


if __name__ == "__main__":
    test_audio_encoder()
