from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CNNConfig:
    """Configuration for CNN models."""
    input_channels: int
    output_size: int
    kernel_size: int = 3
    padding: int = 1
    stride: int = 1
    sparsity: float = 0.1
    

class VariableCNN(nn.Module):
    """CNN with variable number of layers."""
    
    def __init__(
        self, 
        num_layers: int = 1, 
        input_channels: int = 1,
        output_size: int = 10,
        image_size: int = 28
    ):
        super().__init__()
        
        if num_layers < 1:
            raise ValueError("Number of layers must be positive")
            
        self.num_layers = num_layers
        self.image_size = image_size
        
        # Create variable number of convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=input_channels if i == 0 else 2 + (i-1),
                out_channels=2 + i,
                kernel_size=3,
                padding=1
            ) for i in range(num_layers)
        ])
        
        # Calculate output features for fully connected layer
        self.output_features = image_size * image_size * (2 + (num_layers-1))
        self.fc = nn.Linear(self.output_features, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        for conv in self.convs:
            x = F.relu(conv(x))
            
        x = x.view(-1, self.output_features)
        return self.fc(x)


class SparseCNN(nn.Module):
    """CNN with sparse connections."""
    
    def __init__(
        self,
        config: CNNConfig,
        input_size: Tuple[int, int] = (112, 112)
    ):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=20,
            kernel_size=config.kernel_size,
            stride=3
        )
        self.conv2 = nn.Conv2d(
            in_channels=20,
            out_channels=40,
            kernel_size=config.kernel_size,
            stride=3
        )
        
        # Initialize sparse weights
        self._init_sparse_weights(config.sparsity)
        
        # Calculate output size for FC layer
        conv_output_size = self._get_conv_output((1, config.input_channels, *input_size))
        self.fc = nn.Linear(conv_output_size, config.output_size)

    def _init_sparse_weights(self, sparsity: float) -> None:
        """Initialize sparse weights using random mask."""
        with torch.no_grad():
            self.conv1.weight.data *= torch.rand(self.conv1.weight.shape) < sparsity
            self.conv2.weight.data *= torch.rand(self.conv2.weight.shape) < sparsity

    def _get_conv_output(self, shape: Tuple[int, ...]) -> int:
        """Calculate output size of convolutional layers.
        
        Args:
            shape: Input tensor shape (batch_size, channels, height, width)
            
        Returns:
            Number of output features
        """
        batch_size = shape[0]
        input_tensor = torch.rand(batch_size, *shape[1:])
        output_feat = self.conv2(self.conv1(input_tensor))
        return output_feat.view(batch_size, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor after sigmoid activation
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return torch.sigmoid(x)