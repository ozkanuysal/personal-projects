from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import timm
import logging
from dataclasses import dataclass
from time import perf_counter
import numpy as np

@dataclass
class ModelConfig:
    in_channels: int = 3
    out_channels: int = 5
    pretrained: bool = False
    model_name: str = 'efficientvit_b3'
    initial_features: int = 32
    classifier_features: int = 2560

class EfficientVit(nn.Module):
    """EfficientVit model with customizable input/output channels."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.model = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
        )
        
        if config.in_channels != 3:
            self.model.stem.in_conv.conv = nn.Conv2d(
                config.in_channels, 
                config.initial_features,
                kernel_size=(3, 3), 
                stride=(2, 2), 
                padding=(1, 1), 
                bias=False
            )
        
        self.model.head.classifier[4] = nn.Linear(
            in_features=config.classifier_features,
            out_features=config.out_channels,
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        batch_size: int = 8,
        input_shape: tuple = (512, 512)
    ):
        self.model = nn.DataParallel(model).to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.input_shape = input_shape
        
    def train_epoch(self) -> float:
        """Run one training epoch and return elapsed time."""
        start_time = perf_counter()
        
        with torch.set_grad_enabled(True):
            self.optimizer.zero_grad()
            x = torch.randn(self.batch_size, 3, *self.input_shape).to(self.device)
            y_true = torch.randint(0, 5, (self.batch_size,)).to(self.device)
            
            y_pred = self.model(x)
            loss = nn.functional.cross_entropy(y_pred, y_true)
            loss.backward()
            self.optimizer.step()
            
        return perf_counter() - start_time

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count total or trainable parameters in the model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def create_model(config: Optional[ModelConfig] = None) -> EfficientVit:
    """Create and initialize the EfficientVit model."""
    if config is None:
        config = ModelConfig()
        
    model = EfficientVit(config)
    
    # Freeze all layers except classifier
    for name, params in model.named_parameters():
        if 'classifier' in name:
            logging.info(f"Layer requiring gradient: {name}")
            continue
        params.requires_grad = False
    
    logging.info(f"Total parameters: {count_parameters(model)}")
    logging.info(f"Trainable parameters: {count_parameters(model, trainable_only=True)}")
    
    return model

def main():
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_model()
    trainer = ModelTrainer(
        model=model,
        device=device,
        batch_size=8,
        input_shape=(512, 512)
    )
    
    n_epochs = 20
    times = []
    
    for epoch in range(n_epochs + 2):
        if epoch <= 1:  # Warmup epochs
            trainer.batch_size = 1 if epoch == 0 else trainer.batch_size
            trainer.train_epoch()
            continue
            
        elapsed_time = trainer.train_epoch()
        times.append(elapsed_time)
        logging.info(f"Epoch {epoch}: {elapsed_time:.5f} sec.")
    
    logging.info(f"Average training time: {np.mean(times):.5f} sec.")

if __name__ == '__main__':
    main()