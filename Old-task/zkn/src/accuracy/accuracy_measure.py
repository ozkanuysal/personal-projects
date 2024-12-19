from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import json
import numpy as np
import matplotlib.pyplot as plt
import ezkl

@dataclass
class AccuracyConfig:
    """Configuration for accuracy analysis."""
    sigma: int = 4
    plot_statistical_checks: bool = True
    scale: Optional[int] = None
    settings_file: Optional[Path] = None
    fig_save_path: Optional[Path] = None

class AccuracyVisualizer:
    """Handle visualization of accuracy metrics."""
    
    def __init__(self, config: AccuracyConfig):
        self.config = config
        self._setup_plot_params()
    
    def _setup_plot_params(self):
        plt.rcParams['axes.formatter.useoffset'] = False
        plt.rcParams['axes.formatter.limits'] = (-3, 3)
    
    def plot_accuracy_analysis(
        self, 
        real_values: np.ndarray,
        comparison_values: np.ndarray,
        statistics: dict,
        high_sigma_points: np.ndarray,
        se: np.ndarray
    ) -> None:
        fig, axes = plt.subplots(2, 2)
        self._plot_value_comparison(axes[0][0], real_values, comparison_values, high_sigma_points)
        self._plot_error_homoscedasticity(axes[1][0], real_values, se, high_sigma_points)
        self._plot_error_histogram(axes[0][1], se, high_sigma_points, statistics['mse'])
        self._plot_summary_table(axes[1][1], statistics)
        
        plt.tight_layout()
        if self.config.fig_save_path:
            plt.savefig(self.config.fig_save_path)
        plt.show()
    
    def _plot_value_comparison(self, ax, real_values, comparison_values, high_sigma_points):
        ax.scatter(real_values, comparison_values, s=3)
        ax.scatter(
            real_values[high_sigma_points], 
            comparison_values[high_sigma_points], 
            s=3, c='r', 
            label=f'{self.config.sigma}σ points'
        )
        ax.set_xlabel('Original Values')
        ax.set_ylabel('Proof Values')
        ax.set_title('Actual Value Comparison')
        ax.legend()
    
    def _plot_error_homoscedasticity(self, ax, real_values, se, high_sigma_points):
        ax.scatter(real_values, se, s=3)
        ax.scatter(
            real_values[high_sigma_points], 
            se[high_sigma_points], 
            s=3, c='r', 
            label=f'{self.config.sigma}σ points'
        )
        ax.set_xlabel('Original Values')
        ax.set_ylabel('Standard Error')
        ax.set_title('Error Homoscedasticity')
        ax.legend()
    
    def _plot_error_histogram(self, ax, se, high_sigma_points, mse):
        ax.hist(se, bins=50, alpha=0.7)
        ax.hist(se[high_sigma_points], bins=50, alpha=0.7, color='r', label=f'{self.config.sigma}σ points')
        ax.set_xlabel('Standard Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Error Histogram (MSE: {mse:.4f})')
        ax.legend()
    
    def _plot_summary_table(self, ax, statistics):
        cell_text = [
            ['MSE', f"{statistics['mse']:.4f}"],
            ['Max Error', f"{statistics['max_error']:.4f}"],
            ['Max Error %', f"{statistics['max_error_percent']:.2f}%"],
            ['MSE %', f"{statistics['mse_percent']:.2f}%"]
        ]
        table = ax.table(cellText=cell_text, loc='center')
        table.set_fontsize(14)
        table.scale(1, 4)
        ax.axis('off')
        ax.set_title('Statistical Summary')

class AccuracyAnalyzer:
    """Analyze accuracy between real and comparison values."""
    
    def __init__(self, config: AccuracyConfig):
        self.config = config
        self.visualizer = AccuracyVisualizer(config)
    
    def analyze_files(
        self, 
        proof_file: Path, 
        input_file: Path
    ) -> Tuple[float, float, float, float]:
        """Analyze accuracy between proof and input files."""
        scale = self._get_scale()
        proof_input, proof_output = self._load_proof_file(proof_file, scale)
        input_input, input_output = self._load_input_file(input_file)
        
        print("\nAnalyzing input accuracy:")
        self.analyze_values(input_input, proof_input)
        
        print("\nAnalyzing output accuracy:")
        return self.analyze_values(input_output, proof_output)
    
    def analyze_values(
        self, 
        real_values: np.ndarray, 
        comparison_values: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Analyze accuracy between two sets of values."""
        statistics = self._calculate_statistics(real_values, comparison_values)
        
        if self.config.plot_statistical_checks:
            self.visualizer.plot_accuracy_analysis(
                real_values,
                comparison_values,
                statistics,
                statistics['high_sigma_points'],
                statistics['se']
            )
        
        self._print_statistics(statistics)
        return (
            statistics['mse'],
            statistics['max_error'],
            statistics['max_error_percent'],
            statistics['mse_percent']
        )
    
    def _get_scale(self) -> int:
        """Get scale from settings file or config."""
        if self.config.settings_file:
            with open(self.config.settings_file) as f:
                return json.load(f)['run_args']['input_scale']
        if self.config.scale is None:
            raise ValueError("Must provide either settings_file or scale")
        return self.config.scale
    
    @staticmethod
    def _load_proof_file(
        filename: Path, 
        scale: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and process proof file."""
        with open(filename) as f:
            proof = json.load(f)
        
        if 'instances' not in proof and '.proof' not in str(filename):
            proof_input = proof['inputs'][0]
            proof_output = proof['outputs'][0]
        else:
            proof_input = proof['instances'][0]
            proof_output = proof['instances'][1]
        
        return (
            np.array([ezkl.vecu64_to_float(b, scale) for b in proof_input]).flatten(),
            np.array([ezkl.vecu64_to_float(b, scale) for b in proof_output]).flatten()
        )
    
    @staticmethod
    def _load_input_file(filename: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load and process input file."""
        with open(filename) as f:
            data = json.load(f)
        return (
            np.array(data['input_data']).flatten(),
            np.array(data['output_data']).flatten() if 'output_data' in data else None
        )
    
    def _calculate_statistics(
        self,
        real_values: np.ndarray,
        comparison_values: np.ndarray
    ) -> dict:
        """Calculate statistical metrics."""
        se = real_values - comparison_values
        mse = np.mean(np.square(se))
        max_error = np.max(np.abs(se))
        max_error_percent = np.max(np.abs(se / real_values) * 100)
        mse_percent = np.mean(np.square(se / real_values)) * 100
        sigma = np.std(se)
        high_sigma_points = np.abs(se) > self.config.sigma * sigma
        
        return {
            'se': se,
            'mse': mse,
            'max_error': max_error,
            'max_error_percent': max_error_percent,
            'mse_percent': mse_percent,
            'sigma': sigma,
            'high_sigma_points': high_sigma_points
        }

    def _print_statistics(self, statistics: dict):
        """Print statistical metrics."""
        print(f"MSE: {statistics['mse']}")
        print(f"Max Error: {statistics['max_error']}")
        print(f"Max Error Percent: {statistics['max_error_percent']}%")
        print(f"MSE Percent: {statistics['mse_percent']}%")

def main():
    config = AccuracyConfig(
        sigma=4,
        plot_statistical_checks=True,
        settings_file=Path("settings.json"),
        fig_save_path=Path("accuracy_plot.png")
    )
    
    analyzer = AccuracyAnalyzer(config)
    analyzer.analyze_files(
        proof_file=Path("proof.json"),
        input_file=Path("input.json")
    )

if __name__ == "__main__":
    main()