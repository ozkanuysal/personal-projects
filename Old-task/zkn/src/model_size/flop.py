import os
import sys
import time
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from thop import profile
import matplotlib.pyplot as plt

sys.path.append('../..')
from utils.export import export
from models.VariableCNN import VariableCNN
from models.VariableMLP import VariableMLP
from models.VariableLSTM import VariableLSTM
from models.SimpleTransformer import SimpleTransformer


def setup_and_prove(modeltype: str, nlayer: int) -> Tuple[nn.Module, torch.Tensor]:
    if modeltype == 'CNN':
        model = VariableCNN(nlayer)
        input_shape = [1, 28, 28]
        dummy_input = torch.randn(input_shape)
    elif modeltype == 'MLP':
        model = VariableMLP(3 * nlayer, hidden_size=256 + 32 * nlayer)
        input_shape = [1, 256]
        dummy_input = torch.randn(input_shape)
    elif modeltype == 'Attn':
        model = SimpleTransformer(int(np.sqrt(nlayer) / 2) + 1, d_model=(32 * nlayer if nlayer < 16 else 32 * (nlayer - 4)))
        input_shape = [1, 16, (32 * nlayer if nlayer < 16 else 32 * (nlayer - 4))]
        dummy_input = torch.randn(input_shape)
    elif modeltype == 'LSTM':
        temp_nlayer = int(np.sqrt(nlayer) / 2)
        temp_extra = (nlayer if nlayer < 16 else nlayer - 4)
        model = VariableLSTM(nlayer=temp_nlayer, input_size=8 + 8 * temp_extra, hidden_size=8 + 8 * temp_nlayer)
        input_shape = [3 + nlayer, 8 + 8 * temp_extra]
        dummy_input = torch.randn(input_shape)
    else:
        raise ValueError("modeltype must be one of CNN, MLP, Attn, LSTM")
    return model, dummy_input


def main():
    modeltypes = ['Attn', 'CNN', 'LSTM', 'MLP']
    macs_array = {modeltype: [] for modeltype in modeltypes}
    params_array = {modeltype: [] for modeltype in modeltypes}

    for nlayer in range(1, 21):
        for modeltype in modeltypes:
            print(modeltype, nlayer)
            model, dummy_input = setup_and_prove(modeltype, nlayer)
            macs, params = profile(model, inputs=(dummy_input,))
            print(np.log10(macs))
            params_array[modeltype].append(params)
            macs_array[modeltype].append(macs)

    plot_results(macs_array, modeltypes)


def plot_results(macs_array: Dict[str, List[int]], modeltypes: List[str]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    for modeltype in modeltypes:
        ax.plot(list(range(len(macs_array[modeltype]))), macs_array[modeltype], label=modeltype)
    ax.legend()
    ax.set_xlabel('Number of layers')
    ax.set_ylabel('log10(MACs)')
    ax.set_yscale('log')
    plt.show()


if __name__ == "__main__":
    main()