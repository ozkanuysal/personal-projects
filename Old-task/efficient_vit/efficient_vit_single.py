from typing import Optional, Dict, Any, Tuple
import logging
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import timm

# Configuration
@dataclass
class ModelConfig:
    """Model configuration parameters"""
    in_channels: int = 3
    out_channels: int = 5
    pretrained: bool = False
    model_name: str = 'efficientvit_b3'
    initial_features: int = 32 
    classifier_features: int = 2560

class EfficientVit(nn.Module):
    """EfficientVit model with customizable input/output channels."""
    
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self._build_model()
        
    def _build_model(self) -> None:
        """Builds the model architecture"""
        try:
            self.model = timm.create_model(
                self.config.model_name,
                pretrained=self.config.pretrained,
            )
            
            if self.config.in_channels != 3:
                self._modify_input_layer()
                
            self._modify_classifier()
            
        except Exception as e:
            logging.error(f"Failed to build model: {str(e)}")
            raise
            
    def _modify_input_layer(self) -> None:
        """Modifies input layer for custom channel count"""
        self.model.stem.in_conv.conv = nn.Conv2d(
            self.config.in_channels,
            self.config.initial_features,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            bias=False
        )
        
    def _modify_classifier(self) -> None:
        """Modifies classifier layer for custom output count"""
        self.model.head.classifier[4] = nn.Linear(
            in_features=self.config.classifier_features,
            out_features=self.config.out_channels,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ModelTrainer:
    """Handles model training and validation"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        input_shape: Tuple[int, int] = (512, 512)
    ) -> None:
        self.model = nn.DataParallel(model).to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self._validate_params()
        
    def _validate_params(self) -> None:
        """Validates training parameters"""
        if self.batch_size < 1:
            raise ValueError("Batch size must be positive")
        if any(dim < 1 for dim in self.input_shape):
            raise ValueError("Input dimensions must be positive")
            
    def train_epoch(self) -> float:
        """Runs one training epoch
        
        Returns:
            float: Time taken for epoch in seconds
        """
        start_time = perf_counter()
        
        try:
            with torch.set_grad_enabled(True):
                self.optimizer.zero_grad()
                x = torch.randn(self.batch_size, 3, *self.input_shape).to(self.device)
                y_true = torch.randint(0, 5, (self.batch_size,)).to(self.device)
                
                y_pred = self.model(x)
                loss = nn.functional.cross_entropy(y_pred, y_true)
                loss.backward()
                self.optimizer.step()
                
        except Exception as e:
            logging.error(f"Training failed: {str(e)}")
            raise
            
        return perf_counter() - start_time

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        model = create_model()
        trainer = ModelTrainer(
            model=model,
            device=device,
            batch_size=8,
            input_shape=(512, 512)
        )
        
        n_epochs = 20
        times = []
        
        # Training loop with warmup
        for epoch in range(n_epochs + 2):
            if epoch <= 1:  # Warmup epochs
                trainer.batch_size = 1 if epoch == 0 else trainer.batch_size
                trainer.train_epoch()
                continue
                
            elapsed_time = trainer.train_epoch()
            times.append(elapsed_time)
            logging.info(f"Epoch {epoch}: {elapsed_time:.5f} sec.")
        
        logging.info(f"Average training time: {np.mean(times):.5f} sec.")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()