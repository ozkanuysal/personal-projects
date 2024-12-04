from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    dropout: float = 0.1
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention mechanism."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.scale = d_model ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor of shape (batch_size, n_heads, seq_len, d_k)
            k: Key tensor of shape (batch_size, n_heads, seq_len, d_k)
            v: Value tensor of shape (batch_size, n_heads, seq_len, d_k)
        """
        scores = torch.matmul(q / self.scale, k.transpose(-2, -1))
        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention layer."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // config.n_heads
        
        self.w_q = nn.Linear(config.d_model, config.d_model)
        self.w_k = nn.Linear(config.d_model, config.d_model)
        self.w_v = nn.Linear(config.d_model, config.d_model)
        self.w_o = nn.Linear(config.d_model, config.d_model)
        
        self.attention = ScaledDotProductAttention(config.d_model, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        batch_size = q.size(0)
        residual = q

        # Linear projections and reshape
        q = self._reshape_to_batches(self.w_q(q), batch_size)
        k = self._reshape_to_batches(self.w_k(k), batch_size)
        v = self._reshape_to_batches(self.w_v(v), batch_size)

        # Apply attention
        output, _ = self.attention(q, k, v)
        
        # Reshape and project back
        output = self._reshape_from_batches(output, batch_size)
        output = self.dropout(self.w_o(output))
        
        # Add residual and normalize
        return self.layer_norm(output + residual)

    def _reshape_to_batches(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

    def _reshape_from_batches(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)


class SimpleTransformer(nn.Module):
    """Simple Transformer model."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.layers = nn.ModuleList([
            MultiHeadAttention(config) for _ in range(config.n_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc = nn.Linear(config.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        for layer in self.layers:
            x = layer(x, x, x)
            
        x = self.final_layer_norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)