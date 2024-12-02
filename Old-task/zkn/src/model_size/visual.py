from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scienceplots

@dataclass
class PlotConfig:
    """Configuration for plots."""
    figsize_single: tuple = (3.5, 2.625)
    figsize_double: tuple = (3.5*2, 2.625*2)
    dpi: int = 500
    colors: list = None
    
    def __post_init__(self):
        plt.style.use('science')
        plt.rcParams.update({'text.usetex': True})
        self.colors = sns.color_palette()[3:6]

class DataLoader:
    """Handle data loading and preprocessing."""
    def __init__(self, filepath: str):
        self.data = pd.read_csv(filepath)
        self.data['setup_time'] = self.data['pk_time'] + self.data['vk_time']
    
    def get_xy_data(self, x_col: str, y_col: str) -> pd.DataFrame:
        return self.data[[x_col, y_col]].dropna()

class ModelVisualizer:
    """Generate visualizations for model analysis."""
    def __init__(self, data: DataLoader, config: PlotConfig):
        self.data = data
        self.config = config
    
    def parameter_scaling(self, fig: plt.Figure, save: bool = True) -> None:
        """Create parameter scaling subplot."""
        spec = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
        
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])
        ax3 = fig.add_subplot(spec[1, :])
        
        self._plot_parameter_count_vs_macs(ax1)
        self._plot_constraints_vs_macs(ax2)
        self._plot_constraints_vs_logrows(ax3)
        
        if save:
            plt.savefig('figs/model_size_parameter_scaling.png', 
                       dpi=self.config.dpi, bbox_inches='tight')
    
    def proof_size_figure(self, fig: plt.Figure, save: bool = True) -> None:
        """Create proof size visualization."""
        ax = fig.add_subplot()
        self._plot_proof_sizes(ax)
        
        if save:
            plt.savefig('figs/proof_size.png', 
                       dpi=self.config.dpi, bbox_inches='tight')
    
    def _plot_parameter_count_vs_macs(self, ax: plt.Axes) -> None:
        sns.scatterplot(data=self.data.data, 
                       x='param_count', 
                       y='macs', 
                       hue='model_type', 
                       ax=ax)
        ax.set_xlabel('Model Parameter Count')