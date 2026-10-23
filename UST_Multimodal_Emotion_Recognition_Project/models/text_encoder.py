"""
Text Encoder using Pretrained DistilRoBERTa for Emotion Recognition
Wrapper around j-hartmann/emotion-english-distilroberta-base
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Optional


class TextEmotionEncoder(nn.Module):
    """Text encoder using pretrained emotion-tuned DistilRoBERTa"""
    
    def __init__(
        self,
        model_name: str = "j-hartmann/emotion-english-distilroberta-base",
        pool: str = "cls",
        freeze: bool = True,
        unfreeze_layers: int = 2,
        output_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.model_name = model_name
        self.pool = pool
        self.freeze = freeze
        self.unfreeze_layers = unfreeze_layers
        self.output_dim = output_dim
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            config=self.config,
            add_pooling_layer=False  # We'll do our own pooling
        )
        
        # Freeze parameters if specified
        if freeze:
            self.freeze_encoder()
        
        # Projection layer to standardize output dimension
        self.projection = nn.Linear(self.config.hidden_size, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        print(f"TextEmotionEncoder initialized:")
        print(f"   Model: {model_name}")
        print(f"   Hidden size: {self.config.hidden_size}")
        print(f"   Output dim: {output_dim}")
        print(f"   Pool method: {pool}")
        print(f"   Frozen: {freeze}")
        if freeze:
            print(f"   Unfreezing top {unfreeze_layers} layers later")
    
    def freeze_encoder(self):
        """Freeze all encoder parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
        print("Text encoder frozen")
    
    def unfreeze_top_layers(self, num_layers: Optional[int] = None):
        """Unfreeze top N transformer layers"""
        if num_layers is None:
            num_layers = self.unfreeze_layers
        
        # Get transformer layers
        if hasattr(self.model, 'transformer'):
            layers = self.model.transformer.layer
        elif hasattr(self.model, 'encoder'):
            layers = self.model.encoder.layer
        else:
            layers = []
        
        # Unfreeze top layers
        total_layers = len(layers)
        layers_to_unfreeze = min(num_layers, total_layers)
        
        for i in range(total_layers - layers_to_unfreeze, total_layers):
            for param in layers[i].parameters():
                param.requires_grad = True
        
        # Also unfreeze layer norm and pooler if exists
        if hasattr(self.model, 'LayerNorm'):
            for param in self.model.LayerNorm.parameters():
                param.requires_grad = True
        
        if hasattr(self.model, 'pooler') and self.model.pooler is not None:
            for param in self.model.pooler.parameters():
                param.requires_grad = True
        
        print(f"Unfroze top {layers_to_unfreeze} layers of text encoder")
    
    def unfreeze_layers(self, num_layers: int):
        """Alias for unfreeze_top_layers for compatibility"""
        self.unfreeze_top_layers(num_layers)
    
    def pool_features(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token representations to sentence representation"""
        
        if self.pool == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]
        
        elif self.pool == "mean":
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pool == "max":
            # Max pooling
            hidden_states = hidden_states.masked_fill(
                attention_mask.unsqueeze(-1) == 0, -1e9
            )
            return torch.max(hidden_states, dim=1)[0]
        
        elif self.pool == "mean_max":
            # Concatenate mean and max pooling
            mean_pool = self.pool_features(hidden_states, attention_mask, "mean")
            max_pool = self.pool_features(hidden_states, attention_mask, "max")
            return torch.cat([mean_pool, max_pool], dim=-1)
        
        else:
            raise ValueError(f"Unknown pooling method: {self.pool}")
    
    def forward(self, tokenized_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through text encoder
        
        Args:
            tokenized_inputs: Dict with 'input_ids', 'attention_mask', etc.
            
        Returns:
            Dict with:
                - 'turn_pooled': (B, output_dim) - pooled representations
                - 'hidden_states': (B, seq_len, hidden_dim) - all token representations
                - 'attention_mask': (B, seq_len) - attention mask
        """
        
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        
        # Forward through transformer
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get hidden states (last layer)
        hidden_states = outputs.last_hidden_state  # (B, seq_len, hidden_dim)
        
        # Pool to sentence representation
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
    
    def forward_context(self, context_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for context encoding"""
        
        outputs = self.forward(context_inputs)
        return outputs['turn_pooled']  # (B, output_dim)
    
    def get_attention_weights(self, tokenized_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Get attention weights for explainability"""
        
        input_ids = tokenized_inputs['input_ids']
        attention_mask = tokenized_inputs['attention_mask']
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True
        )
        
        # Return attention weights from last layer
        return outputs.attentions[-1]  # (B, num_heads, seq_len, seq_len)


class ContextualTextEncoder(nn.Module):
    """Text encoder with dialogue context integration"""
    
    def __init__(
        self,
        base_encoder: TextEmotionEncoder,
        context_dim: int = 256,
        context_layers: int = 2
    ):
        super().__init__()
        
        self.base_encoder = base_encoder
        self.context_dim = context_dim
        
        # Context integration network
        self.context_proj = nn.Linear(base_encoder.output_dim, context_dim)
        
        # Simple context aggregation (can be replaced with more sophisticated methods)
        self.context_attention = nn.MultiheadAttention(
            embed_dim=context_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        self.context_norm = nn.LayerNorm(context_dim)
        self.output_proj = nn.Linear(context_dim, base_encoder.output_dim)
        
    def forward(
        self,
        current_inputs: Dict[str, torch.Tensor],
        context_inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward with context"""
        
        # Encode current utterance
        current_outputs = self.base_encoder(current_inputs)
        current_repr = current_outputs['turn_pooled']  # (B, output_dim)
        
        if context_inputs is None:
            return current_outputs
        
        # Encode context
        context_repr = self.base_encoder.forward_context(context_inputs)  # (B, output_dim)
        
        # Project to context dimension
        current_ctx = self.context_proj(current_repr).unsqueeze(1)  # (B, 1, context_dim)
        context_ctx = self.context_proj(context_repr).unsqueeze(1)  # (B, 1, context_dim)
        
        # Concatenate current + context
        combined = torch.cat([current_ctx, context_ctx], dim=1)  # (B, 2, context_dim)
        
        # Apply self-attention for context integration
        attended, _ = self.context_attention(combined, combined, combined)
        
        # Take the current utterance's attended representation
        attended_current = attended[:, 0, :]  # (B, context_dim)
        attended_current = self.context_norm(attended_current)
        
        # Project back to output dimension
        context_enhanced = self.output_proj(attended_current)  # (B, output_dim)
        
        return {
            'turn_pooled': context_enhanced,
            'hidden_states': current_outputs['hidden_states'],
            'attention_mask': current_outputs['attention_mask'],
            'context_repr': context_repr
        }


def test_text_encoder():
    """Test the text encoder"""
    
    from transformers import AutoTokenizer
    
    # Initialize tokenizer and encoder
    tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    encoder = TextEmotionEncoder(freeze=True)
    
    # Test input
    texts = [
        "I'm really angry about this situation!",
        "This makes me so happy and excited!",
        "I feel quite sad and disappointed."
    ]
    
    # Tokenize
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    # Forward pass
    with torch.no_grad():
        outputs = encoder(inputs)
    
    print("Text encoder test:")
    print(f"Input texts: {len(texts)}")
    print(f"Turn pooled shape: {outputs['turn_pooled'].shape}")
    print(f"Hidden states shape: {outputs['hidden_states'].shape}")
    print("Text encoder working correctly!")
    
    # Test unfreezing
    encoder.unfreeze_top_layers(2)
    print("Unfreezing test completed!")


if __name__ == "__main__":
    test_text_encoder()
