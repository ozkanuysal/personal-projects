from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import glob
import re
import json
import logging
from datetime import datetime

@dataclass
class ProcessingConfig:
    """Configuration for log processing."""
    base_log_dir: Path = Path('logs')
    model_types: List[str] = None
    
    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ['CNN', 'MLP', 'Attn']
        self.base_log_dir.mkdir(exist_ok=True)

@dataclass
class LogResult:
    """Container for processed log results."""
    model_type: str
    number: int
    scale: float
    bits: float
    logrows: float
    num_constraints: float
    vk_time: float
    pk_time: float
    wall_setup_time: float
    proof_time: float
    wall_prove_time: float

class LogFileProcessor:
    """Process log files for different model types and extract performance metrics."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config.base_log_dir / f'processing_{datetime.now():%Y%m%d}.log'
        )

    def get_files_for_model(self, model_type: str) -> Dict[int, Dict[str, Optional[Path]]]:
        """Organize log files by model number and type."""
        files = glob.glob(f'{self.config.base_log_dir}/{model_type}*.log')
        file_dict: Dict[int, Dict[str, Optional[Path]]] = {}
        
        for file in files:
            path = Path(file)
            file_number = int(path.stem.split('_')[-1])
            file_type = path.stem.split('_')[-2]
            
            if file_number not in file_dict:
                file_dict[file_number] = {'prework': None, 'setup': None, 'prove': None}
            file_dict[file_number][file_type] = path
            
        return file_dict

    def process_prework_file(self, file: Optional[Path]) -> Tuple[float, float, float, float]:
        """Extract metrics from prework log file."""
        if file is None:
            return np.nan, np.nan, np.nan, np.nan
        
        try:
            content = file.read_text()
            mock_time = np.nan
            scale = bits = logrows = num_constraints = np.nan
            
            for line in content.splitlines():
                if '- succeeded' in line:
                    mock_time = float(re.search(r'\d+', line)[0])
                if '{"run_args":' in line:
                    settings = json.loads(line.split('[*]')[0])
                    scale = settings['run_args']['input_scale']
                    bits = settings['run_args']['param_scale']
                    logrows = settings['run_args']['logrows']
                    num_constraints = settings['num_rows']
                    
            return scale, bits, logrows, num_constraints
        except Exception as e:
            self.logger.error(f"Error processing prework file {file}: {str(e)}")
            return np.nan, np.nan, np.nan, np.nan

    def process_setup_file(self, file: Optional[Path]) -> Tuple[float, float, float]:
        """Extract metrics from setup log file."""
        if file is None:
            return np.nan, np.nan, np.nan
        
        try:
            content = file.read_text()
            vk_time = pk_time = wall_setup_time = np.nan
            
            for line in content.splitlines():
                if 'VK took' in line:
                    vk_time = float(re.search(r'VK took (\d+\.\d+)', line)[1])
                if 'PK took' in line:
                    pk_time = float(re.search(r'PK took (\d+\.\d+)', line)[1])
                if 'succeeded' in line:
                    wall_setup_time = float(re.search(r'\d+', line)[0])
                    
            return vk_time, pk_time, wall_setup_time
        except Exception as e:
            self.logger.error(f"Error processing setup file {file}: {str(e)}")
            return np.nan, np.nan, np.nan

    def process_prove_file(self, file: Optional[Path]) -> Tuple[float, float]:
        """Extract metrics from prove log file."""
        if file is None:
            return np.nan, np.nan
        
        try:
            content = file.read_text()
            proof_time = wall_prove_time = np.nan
            
            for line in content.splitlines():
                if 'proof took' in line:
                    proof_time = float(re.search(r'proof took (\d+\.\d+)', line)[1])
                if 'succeeded' in line:
                    wall_prove_time = float(re.search(r'\d+', line)[0])
                    
            return proof_time, wall_prove_time
        except Exception as e:
            self.logger.error(f"Error processing prove file {file}: {str(e)}")
            return np.nan, np.nan

    def process_all_logs(self) -> pd.DataFrame:
        """Process all log files and compile results into a DataFrame."""
        results: List[LogResult] = []
        
        for model_type in self.config.model_types:
            self.logger.info(f"Processing logs for model type: {model_type}")
            file_dict = self.get_files_for_model(model_type)
            
            for number, file_options in file_dict.items():
                try:
                    scale, bits, logrows, num_constraints = self.process_prework_file(file_options['prework'])
                    vk_time, pk_time, wall_setup_time = self.process_setup_file(file_options['setup'])
                    proof_time, wall_prove_time = self.process_prove_file(file_options['prove'])
                    
                    results.append(LogResult(
                        model_type=model_type,
                        number=number,
                        scale=scale,
                        bits=bits,
                        logrows=logrows,
                        num_constraints=num_constraints,
                        vk_time=vk_time,
                        pk_time=pk_time,
                        wall_setup_time=wall_setup_time,
                        proof_time=proof_time,
                        wall_prove_time=wall_prove_time
                    ))
                except Exception as e:
                    self.logger.error(f"Error processing {model_type} model {number}: {str(e)}")
                    continue
                    
        return pd.DataFrame([vars(r) for r in results])

def main():
    config = ProcessingConfig()
    processor = LogFileProcessor(config)
    df = processor.process_all_logs()
    df.to_csv('processed_results.csv', index=False)

if __name__ == "__main__":
    main()