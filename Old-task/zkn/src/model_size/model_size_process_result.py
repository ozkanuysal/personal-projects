import numpy as np
import pandas as pd
import glob
import re
import json
from typing import Dict, List, Optional, Tuple

class LogFileProcessor:
    """Process log files for different model types and extract performance metrics."""
    
    def __init__(self, base_log_dir: str = 'logs'):
        self.base_log_dir = base_log_dir
        self.model_types = ['CNN', 'MLP', 'Attn']
    
    def get_files_for_model(self, model_type: str) -> Dict[int, Dict[str, Optional[str]]]:
        """Organize log files by model number and type."""
        files = glob.glob(f'{self.base_log_dir}/{model_type}*.log')
        file_dict = {}
        
        for file in files:
            file_number = int(file.split('.')[0].split('_')[-1])
            file_type = file.split('.')[0].split('_')[-2]
            
            if file_number not in file_dict:
                file_dict[file_number] = {'prework': None, 'setup': None, 'prove': None}
            file_dict[file_number][file_type] = file
            
        return file_dict

    def process_prework_file(self, file: Optional[str]) -> Tuple[float, float, float, float]:
        """Extract metrics from prework log file."""
        if file is None:
            return np.nan, np.nan, np.nan, np.nan
        
        with open(file) as f:
            lines = f.readlines()
            
        mock_time = np.nan
        scale = bits = logrows = num_constraints = np.nan
        
        for line in lines:
            if '- succeeded' in line:
                mock_time = float(re.search(r'\d+', line)[0])
            if '{"run_args":' in line:
                settings = json.loads(line.split('[*]')[0])
                scale = settings['run_args']['input_scale']
                bits = settings['run_args']['param_scale']
                logrows = settings['run_args']['logrows']
                num_constraints = settings['num_rows']
                
        return scale, bits, logrows, num_constraints

    def process_setup_file(self, file: Optional[str]) -> Tuple[float, float, float]:
        """Extract metrics from setup log file."""
        if file is None:
            return np.nan, np.nan, np.nan
        
        with open(file) as f:
            lines = f.readlines()
            
        vk_time = pk_time = wall_setup_time = np.nan
        
        for line in lines:
            if 'VK took' in line:
                vk_time = float(re.search(r'VK took (\d+\.\d+)', line)[1])
            if 'PK took' in line:
                pk_time = float(re.search(r'PK took (\d+\.\d+)', line)[1])
            if 'succeeded' in line:
                wall_setup_time = float(re.search(r'\d+', line)[0])
                
        return vk_time, pk_time, wall_setup_time

    def process_prove_file(self, file: Optional[str]) -> Tuple[float, float]:
        """Extract metrics from prove log file."""
        if file is None:
            return np.nan, np.nan
        
        with open(file) as f:
            lines = f.readlines()
            
        proof_time = wall_prove_time = np.nan
        
        for line in lines:
            if 'proof took' in line:
                proof_time = float(re.search(r'proof took (\d+\.\d+)', line)[1])
            if 'succeeded' in line:
                wall_prove_time = float(re.search(r'\d+', line)[0])
                
        return proof_time, wall_prove_time

    def process_all_logs(self) -> pd.DataFrame:
        """Process all log files and compile results into a DataFrame."""
        results = []
        
        for model_type in self.model_types:
            file_dict = self.get_files_for_model(model_type)
            
            for number, file_options in file_dict.items():
                # Process prework file
                scale, bits, logrows, num_constraints = self.process_prework_file(file_options['prework'])
                
                # Process setup file
                vk_time, pk_time, wall_setup_time = self.process_setup_file(file_options['setup'])
                
                # Process prove file
                proof_time, wall_prove_time = self.process_prove_file(file_options['prove'])
                
                # Compile results
                results.append([
                    model_type, number, scale, bits