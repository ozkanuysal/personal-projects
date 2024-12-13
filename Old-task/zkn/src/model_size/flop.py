import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from thop import profile
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('../..')
from utils.export import export
from models.VariableCNN import VariableCNN
from models.VariableMLP import VariableMLP
from models.VariableLSTM import VariableLSTM
from models.SimpleTransformer import SimpleTransformer


def get_model(modeltype: str, nlayer: int) -> Tuple[nn.Module, torch.Tensor]:
    """Factory method to create models and dummy inputs."""
    model_mapping = {
        'CNN': lambda: (
            VariableCNN(nlayer),
            torch.randn([1, 28, 28])
        ),
        'MLP': lambda: (
            VariableMLP(3 * nlayer, hidden_size=256 + 32 * nlayer),
            torch.randn([1, 256])
        ),
        'Attn': lambda: (
            SimpleTransformer(
                int(np.sqrt(nlayer) / 2) + 1,
                d_model=(32 * nlayer if nlayer < 16 else 32 * (nlayer - 4))
            ),
            torch.randn([1, 16, (32 * nlayer if nlayer < 16 else 32 * (nlayer - 4))])
        ),
        'LSTM': lambda: (
            VariableLSTM(
                nlayer=int(np.sqrt(nlayer) / 2),
                input_size=8 + 8 * (nlayer if nlayer < 16 else nlayer - 4),
                hidden_size=8 + 8 * int(np.sqrt(nlayer) / 2)
            ),
            torch.randn([3 + nlayer, 8 + 8 * (nlayer if nlayer < 16 else nlayer - 4)])
        )
    }
    try:
        return model_mapping[modeltype]()
    except KeyError:
        raise ValueError("modeltype must be one of CNN, MLP, Attn, LSTM")


def main():
    modeltypes = ['Attn', 'CNN', 'LSTM', 'MLP']
    macs_array: Dict[str, List[int]] = {mt: [] for mt in modeltypes}
    params_array: Dict[str, List[int]] = {mt: [] for mt in modeltypes}

    for nlayer in range(1, 21):
        for modeltype in modeltypes:
            print(modeltype, nlayer)
            model, dummy_input = get_model(modeltype, nlayer)
            macs, params = profile(model, inputs=(dummy_input,))
            print(np.log10(macs))
            params_array[modeltype].append(params)
            macs_array[modeltype].append(macs)

    plot_results(macs_array, modeltypes)


def plot_results(macs_array: Dict[str, List[int]], modeltypes: List[str]) -> None:
    """Plot MACs for different model types."""
    plt.figure(figsize=(8, 6))
    for modeltype in modeltypes:
        plt.plot(range(1, len(macs_array[modeltype]) + 1), macs_array[modeltype], label=modeltype)
    plt.legend()
    plt.xlabel('Number of Layers')
    plt.ylabel('MACs (log scale)')
    plt.yscale('log')
    plt.title('MACs vs. Number of Layers for Different Models')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()