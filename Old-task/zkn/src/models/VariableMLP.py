from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariableMLP(nn.Module):
    """Variable-depth Multilayer Perceptron (MLP) implementation.
    
    Args:
        nlayer: Number of hidden layers
        hidden_size: Number of neurons in hidden layers
        input_size: Input dimension
        output_size: Output dimension
        dropout: Dropout probability
        use_layer_norm: Whether to use layer normalization
        activation: Activation function to use
    """
    
    def __init__(
        self,
        nlayer: int = 1,
        hidden_size: int = 256,
        input_size: int = 256,
        output_size: int = 1,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        activation: nn.Module = nn.ReLU()
    ):
        super().__init__()
        
        # Input validation
        if nlayer < 1:
            raise ValueError("Number of layers must be positive")
        if hidden_size < 1 or input_size < 1 or output_size < 1:
            raise ValueError("Layer sizes must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("Dropout must be between 0 and 1")
            
        # Store configuration
        self.nlayer = nlayer
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation
        
        # Define layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        if use_layer_norm:
            self.input_norm = nn.LayerNorm(hidden_size)
        
        # Hidden layers with optional normalization
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(nlayer)
        ])
        if use_layer_norm:
            self.hidden_norms = nn.ModuleList([
                nn.LayerNorm(hidden_size) for _ in range(nlayer)
            ])
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.use_layer_norm = use_layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Input layer
        x = self.input_layer(x)
        if self.use_layer_norm:
            x = self.input_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Hidden layers
        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            if self.use_layer_norm:
                x = self.hidden_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
            
        # Output layer
        return self.output_layer(x)

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return (
            f"VariableMLP(nlayer={self.nlayer}, "
            f"hidden_size={self.hidden_size}, "
            f"input_size={self.input_size}, "
            f"output_size={self.output_size})"
        )