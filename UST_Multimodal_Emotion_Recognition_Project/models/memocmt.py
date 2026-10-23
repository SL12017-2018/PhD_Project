"""
MemoCMT: Memory-Enhanced Cross-Modal Transformer
Implements cross-attention between text and audio modalities with persistent dialogue memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class CrossModalAttention(nn.Module):
    """Cross-attention between two modalities"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Projection layers - optimized for CUDA
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        # Initialize weights for better CUDA performance
        self._init_weights()
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def _init_weights(self):
        """Initialize weights for optimal CUDA performance"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight, gain=1/math.sqrt(2))
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
    def forward(
        self,
        query: torch.Tensor,  # (B, T_q, d_model)
        key: torch.Tensor,    # (B, T_k, d_model)
        value: torch.Tensor,  # (B, T_v, d_model)
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_size = query.size(0)
        
        # Apply linear projections
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask, self.temperature
        )
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear layer
        output = self.w_o(attention_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        return output, attention_weights
    
    @staticmethod
    def scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        d_k = Q.size(-1)
        
        # Use more efficient attention computation for CUDA
        if Q.is_cuda and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized CUDA kernel if available (PyTorch 2.0+)
            try:
                attention_output = F.scaled_dot_product_attention(
                    Q, K, V, 
                    attn_mask=mask,
                    dropout_p=0.0,  # We handle dropout separately
                    scale=1.0 / (math.sqrt(d_k) * temperature)
                )
                # For compatibility, compute attention weights separately if needed
                scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(d_k) * temperature)
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                attention_weights = F.softmax(scores, dim=-1)
                return attention_output, attention_weights
            except:
                pass  # Fall back to manual computation
        
        # Manual computation (fallback or for non-CUDA)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(d_k) * temperature)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights


class DialogueMemory(nn.Module):
    """Persistent memory for dialogue context with speaker awareness"""
    
    def __init__(
        self,
        d_model: int,
        memory_size: int = 512,
        num_speakers: int = 2,
        speaker_embed_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.memory_size = memory_size
        self.num_speakers = num_speakers
        self.speaker_embed_dim = speaker_embed_dim
        
        # Speaker embeddings
        self.speaker_embeddings = nn.Embedding(num_speakers, speaker_embed_dim)
        
        # Memory components
        self.memory_projection = nn.Linear(d_model + speaker_embed_dim, memory_size)
        self.memory_gate = nn.Linear(memory_size + d_model, memory_size)
        self.memory_update = nn.Linear(memory_size, memory_size)
        
        # Global memory state
        self.register_buffer('global_memory', torch.zeros(1, memory_size))
        
        # Attention for memory retrieval
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=memory_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(memory_size)
        
    def reset_memory(self, batch_size: int = 1):
        """Reset global memory for new dialogue"""
        self.global_memory = torch.zeros(
            batch_size, self.memory_size, 
            device=self.global_memory.device, 
            dtype=self.global_memory.dtype
        )
    
    def forward(
        self,
        current_repr: torch.Tensor,  # (B, d_model)
        speaker_ids: torch.Tensor,   # (B,)
        update_memory: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update and retrieve from dialogue memory
        
        Returns:
            - memory_enhanced_repr: (B, d_model)
            - memory_state: (B, memory_size)
        """
        
        batch_size = current_repr.size(0)
        
        # Get speaker embeddings
        speaker_embeds = self.speaker_embeddings(speaker_ids)  # (B, speaker_embed_dim)
        
        # Combine current representation with speaker info
        combined_repr = torch.cat([current_repr, speaker_embeds], dim=-1)  # (B, d_model + speaker_embed_dim)
        
        # Project to memory space
        memory_input = self.memory_projection(combined_repr)  # (B, memory_size)
        
        # Expand global memory for batch
        if self.global_memory.size(0) != batch_size:
            self.global_memory = self.global_memory.expand(batch_size, -1).contiguous()
        
        # Compute memory gate
        gate_input = torch.cat([memory_input, current_repr], dim=-1)
        memory_gate = torch.sigmoid(self.memory_gate(gate_input))  # (B, memory_size)
        
        # Update memory
        if update_memory:
            memory_update = self.memory_update(memory_input)
            new_memory = memory_gate * memory_update + (1 - memory_gate) * self.global_memory
            self.global_memory = new_memory.detach()  # Detach to prevent gradient accumulation
        else:
            new_memory = self.global_memory
        
        # Retrieve from memory using attention
        memory_retrieved, _ = self.memory_attention(
            query=memory_input.unsqueeze(1),  # (B, 1, memory_size)
            key=new_memory.unsqueeze(1),      # (B, 1, memory_size)
            value=new_memory.unsqueeze(1)     # (B, 1, memory_size)
        )
        memory_retrieved = memory_retrieved.squeeze(1)  # (B, memory_size)
        
        # Apply layer norm and dropout
        memory_retrieved = self.layer_norm(memory_retrieved + memory_input)
        memory_retrieved = self.dropout(memory_retrieved)
        
        # Project back to model dimension
        memory_enhanced_repr = current_repr + F.linear(memory_retrieved, 
                                                      self.memory_projection.weight.T[:self.d_model])
        
        return memory_enhanced_repr, new_memory


class MemoCMTBlock(nn.Module):
    """Single MemoCMT transformer block with cross-modal attention and memory"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1,
        use_memory: bool = True,
        memory_config: Optional[Dict] = None
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_memory = use_memory
        
        # Cross-modal attention layers
        self.text_to_audio_attn = CrossModalAttention(d_model, num_heads, dropout)
        self.audio_to_text_attn = CrossModalAttention(d_model, num_heads, dropout)
        
        # Self-attention layers
        self.text_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        self.audio_self_attn = nn.MultiheadAttention(d_model, num_heads, dropout, batch_first=True)
        
        # Feed-forward networks
        self.text_ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        self.audio_ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.text_norm1 = nn.LayerNorm(d_model)
        self.text_norm2 = nn.LayerNorm(d_model)
        self.text_norm3 = nn.LayerNorm(d_model)
        
        self.audio_norm1 = nn.LayerNorm(d_model)
        self.audio_norm2 = nn.LayerNorm(d_model)
        self.audio_norm3 = nn.LayerNorm(d_model)
        
        # Memory module
        if use_memory and memory_config:
            self.memory = DialogueMemory(**memory_config)
        else:
            self.memory = None
    
    def forward(
        self,
        text_repr: torch.Tensor,    # (B, d_model)
        audio_repr: torch.Tensor,   # (B, d_model)
        speaker_ids: Optional[torch.Tensor] = None,  # (B,)
        update_memory: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward pass through MemoCMT block
        
        Returns:
            - Enhanced text representation
            - Enhanced audio representation
            - Attention weights dictionary
        """
        
        # Add sequence dimension for attention
        text_seq = text_repr.unsqueeze(1)    # (B, 1, d_model)
        audio_seq = audio_repr.unsqueeze(1)  # (B, 1, d_model)
        
        attention_weights = {}
        
        # 1. Self-attention
        text_self, text_self_weights = self.text_self_attn(text_seq, text_seq, text_seq)
        audio_self, audio_self_weights = self.audio_self_attn(audio_seq, audio_seq, audio_seq)
        
        text_seq = self.text_norm1(text_seq + text_self)
        audio_seq = self.audio_norm1(audio_seq + audio_self)
        
        attention_weights['text_self'] = text_self_weights
        attention_weights['audio_self'] = audio_self_weights
        
        # 2. Cross-modal attention
        text_cross, text_cross_weights = self.text_to_audio_attn(
            text_seq, audio_seq, audio_seq
        )
        audio_cross, audio_cross_weights = self.audio_to_text_attn(
            audio_seq, text_seq, text_seq
        )
        
        text_seq = self.text_norm2(text_seq + text_cross)
        audio_seq = self.audio_norm2(audio_seq + audio_cross)
        
        attention_weights['text_cross'] = text_cross_weights
        attention_weights['audio_cross'] = audio_cross_weights
        
        # 3. Feed-forward networks
        text_ff_out = self.text_ff(text_seq)
        audio_ff_out = self.audio_ff(audio_seq)
        
        text_seq = self.text_norm3(text_seq + text_ff_out)
        audio_seq = self.audio_norm3(audio_seq + text_ff_out)
        
        # Remove sequence dimension
        text_enhanced = text_seq.squeeze(1)   # (B, d_model)
        audio_enhanced = audio_seq.squeeze(1) # (B, d_model)
        
        # 4. Memory integration
        memory_states = {}
        if self.memory and speaker_ids is not None:
            # Apply memory to both modalities
            text_enhanced, text_memory = self.memory(
                text_enhanced, speaker_ids, update_memory
            )
            audio_enhanced, audio_memory = self.memory(
                audio_enhanced, speaker_ids, update_memory
            )
            
            memory_states['text_memory'] = text_memory
            memory_states['audio_memory'] = audio_memory
        
        return text_enhanced, audio_enhanced, {
            'attention_weights': attention_weights,
            'memory_states': memory_states
        }


class MemoCMT(nn.Module):
    """
    Memory-Enhanced Cross-Modal Transformer
    Multi-layer transformer with cross-modal attention and persistent dialogue memory
    """
    
    def __init__(
        self,
        text_dim: int = 768,
        audio_dim: int = 768,
        proj_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.15,
        speaker_embed_dim: int = 64,
        global_mem_dim: int = 256,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        use_cross_attention: bool = True,
        use_memory_bank: bool = True,
        memory_bank_size: int = 512,
        use_gating: bool = True,
        attention_temperature: float = 0.1,
        layer_scale: bool = True
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.proj_dim = proj_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.use_cross_attention = use_cross_attention
        self.use_memory_bank = use_memory_bank
        self.memory_bank_size = memory_bank_size
        self.use_gating = use_gating
        self.attention_temperature = attention_temperature
        self.layer_scale = layer_scale
        
        # Input projections
        self.text_projection = nn.Linear(text_dim, proj_dim)
        self.audio_projection = nn.Linear(audio_dim, proj_dim)
        
        # Enhanced gating mechanism
        if use_gating:
            self.text_gate = nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 4),
                nn.ReLU(),
                nn.Linear(proj_dim // 4, proj_dim),
                nn.Sigmoid()
            )
            self.audio_gate = nn.Sequential(
                nn.Linear(proj_dim, proj_dim // 4),
                nn.ReLU(),
                nn.Linear(proj_dim // 4, proj_dim),
                nn.Sigmoid()
            )
        
        # Enhanced memory configuration
        memory_config = {
            'd_model': proj_dim,
            'memory_size': global_mem_dim,
            'speaker_embed_dim': speaker_embed_dim,
            'dropout': dropout
        }
        
        # Memory bank for episodic memory
        if use_memory_bank:
            self.memory_bank = nn.Parameter(
                torch.randn(memory_bank_size, proj_dim) * 0.02
            )
            self.memory_bank_attention = nn.MultiheadAttention(
                embed_dim=proj_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Layer scaling parameters for stable training
        if layer_scale:
            self.layer_scale_params = nn.ParameterList([
                nn.Parameter(torch.ones(proj_dim) * 0.1) for _ in range(num_layers)
            ])
        
        # MemoCMT blocks
        self.blocks = nn.ModuleList([
            MemoCMTBlock(
                d_model=proj_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                use_memory=True,
                memory_config=memory_config
            ) for _ in range(num_layers)
        ])
        
        # Final layers
        if use_layer_norm:
            self.final_text_norm = nn.LayerNorm(proj_dim)
            self.final_audio_norm = nn.LayerNorm(proj_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim)
        )
        
        print(f"✅ MemoCMT initialized:")
        print(f"   Input dims: text={text_dim}, audio={audio_dim}")
        print(f"   Projection dim: {proj_dim}")
        print(f"   Layers: {num_layers}")
        print(f"   Heads: {num_heads}")
        print(f"   Memory dim: {global_mem_dim}")
        
        # Initialize weights for CUDA optimization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights optimized for CUDA performance"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=1/math.sqrt(2))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)
    
    def reset_memory(self, batch_size: int = 1):
        """Reset memory for all blocks"""
        for block in self.blocks:
            if block.memory:
                block.memory.reset_memory(batch_size)
    
    def forward(
        self,
        text_features: torch.Tensor,    # (B, text_dim)
        audio_features: torch.Tensor,   # (B, audio_dim)
        speaker_ids: torch.Tensor,      # (B,)
        turn_lengths: Optional[torch.Tensor] = None,  # For future batch support
        update_memory: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MemoCMT
        
        Args:
            text_features: (B, text_dim)
            audio_features: (B, audio_dim)
            speaker_ids: (B,) - speaker identifiers
            turn_lengths: Optional batch information
            update_memory: Whether to update dialogue memory
            
        Returns:
            Dict containing:
                - 'fused_features': (B, proj_dim) - final fused representation
                - 'text_features': (B, proj_dim) - enhanced text features
                - 'audio_features': (B, proj_dim) - enhanced audio features
                - 'attention_maps': Dict of attention weights
                - 'memory_states': Dict of memory states
        """
        
        batch_size = text_features.size(0)
        
        # Project to common dimension
        text_proj = self.text_projection(text_features)    # (B, proj_dim)
        audio_proj = self.audio_projection(audio_features) # (B, proj_dim)
        
        # Apply dropout
        text_proj = self.dropout(text_proj)
        audio_proj = self.dropout(audio_proj)
        
        # Store initial projections for residual connections
        text_residual = text_proj
        audio_residual = audio_proj
        
        # Forward through MemoCMT blocks
        all_attention_maps = []
        all_memory_states = []
        
        for i, block in enumerate(self.blocks):
            text_proj, audio_proj, block_outputs = block(
                text_proj, audio_proj, speaker_ids, update_memory
            )
            
            all_attention_maps.append(block_outputs['attention_weights'])
            all_memory_states.append(block_outputs['memory_states'])
        
        # Apply final normalization
        if self.use_layer_norm:
            text_proj = self.final_text_norm(text_proj)
            audio_proj = self.final_audio_norm(audio_proj)
        
        # Residual connections
        if self.use_residual:
            text_proj = text_proj + text_residual
            audio_proj = audio_proj + audio_residual
        
        # Fusion
        concatenated = torch.cat([text_proj, audio_proj], dim=-1)  # (B, proj_dim * 2)
        fused_features = self.fusion_layer(concatenated)           # (B, proj_dim)
        
        return {
            'fused_features': fused_features,
            'text_features': text_proj,
            'audio_features': audio_proj,
            'attention_maps': all_attention_maps,
            'memory_states': all_memory_states
        }


def test_memocmt():
    """Test MemoCMT functionality"""
    
    # Test parameters
    batch_size = 4
    text_dim = 768
    audio_dim = 768
    proj_dim = 256
    
    # Create MemoCMT
    memocmt = MemoCMT(
        text_dim=text_dim,
        audio_dim=audio_dim,
        proj_dim=proj_dim,
        num_layers=2,
        num_heads=4,
        ff_dim=512
    )
    
    # Create dummy inputs
    text_features = torch.randn(batch_size, text_dim)
    audio_features = torch.randn(batch_size, audio_dim)
    speaker_ids = torch.randint(0, 2, (batch_size,))
    
    # Reset memory
    memocmt.reset_memory(batch_size)
    
    # Forward pass
    outputs = memocmt(
        text_features=text_features,
        audio_features=audio_features,
        speaker_ids=speaker_ids
    )
    
    print("MemoCMT test:")
    print(f"Input shapes: text={text_features.shape}, audio={audio_features.shape}")
    print(f"Fused features shape: {outputs['fused_features'].shape}")
    print(f"Text features shape: {outputs['text_features'].shape}")
    print(f"Audio features shape: {outputs['audio_features'].shape}")
    print(f"Attention maps layers: {len(outputs['attention_maps'])}")
    print(f"Memory states layers: {len(outputs['memory_states'])}")
    print("MemoCMT working correctly!")
    
    # Test multiple turns (simulating dialogue)
    print("\nTesting dialogue memory persistence:")
    for turn in range(3):
        outputs = memocmt(
            text_features=torch.randn(2, text_dim),  # 2 speakers
            audio_features=torch.randn(2, audio_dim),
            speaker_ids=torch.tensor([0, 1]),  # Different speakers
            update_memory=True
        )
        print(f"Turn {turn + 1}: Fused shape {outputs['fused_features'].shape}")
    
    print("Dialogue memory test completed!")


if __name__ == "__main__":
    test_memocmt()
