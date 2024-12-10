from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
import timeit
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from thop import profile
import ezkl

from utils.export import export
from models.VariableCNN import VariableCNN
from models.VariableMLP import VariableMLP
from models.VariableLSTM import VariableLSTM
from models.SimpleTransformer import SimpleTransformer

@dataclass
class Config:
    """Configuration for model running and proving."""
    output_file: Path = Path('model_size_results.csv')
    logrows_path: Path = Path('../../../kzgs/kzg%d.srs')
    logging: bool = True
    log_dir: Path = Path('logs')
    proof_dir: Path = Path('proofs')
    runfile_dir: Path = Path('runfiles')
    model_types: List[str] = None

    def __post_init__(self):
        if self.model_types is None:
            self.model_types = ['Attn', 'CNN', 'MLP']
        for directory in [self.log_dir, self.proof_dir, self.runfile_dir]:
            directory.mkdir(exist_ok=True)

class ModelRunner:
    """Handle model setup, running and proving."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=self.config.log_dir / 'model_run.log'
        )

    def _create_model(self, modeltype: str, nlayer: int) -> Tuple[nn.Module, List[int], torch.Tensor]:
        """Create model instance based on type and number of layers."""
        if modeltype == 'CNN':
            model = VariableCNN(nlayer)
            input_shape = [1, 28, 28]
        elif modeltype == 'MLP':
            model = VariableMLP(3 * nlayer, hidden_size=256 + 32 * nlayer)
            input_shape = [1, 256]
        elif modeltype == 'Attn':
            model = SimpleTransformer(
                int(np.sqrt(nlayer) / 2) + 1,
                d_model=(32 * nlayer if nlayer < 16 else 32 * (nlayer - 4))
            )
            input_shape = [1, 16, (32 * nlayer if nlayer < 16 else 32 * (nlayer - 4))]
        elif modeltype == 'LSTM':
            temp_nlayer = int(np.sqrt(nlayer) / 2)
            temp_extra = (nlayer if nlayer < 16 else nlayer - 4)
            model = VariableLSTM(
                nlayer=temp_nlayer,
                input_size=8 + 8 * temp_extra,
                hidden_size=8 + 8 * temp_nlayer
            )
            input_shape = [3 + nlayer, 8 + 8 * temp_extra]
        else:
            raise ValueError("modeltype must be one of CNN, MLP, Attn, LSTM")
            
        dummy_input = torch.randn(input_shape)
        return model, input_shape, dummy_input

    def _run_ezkl_commands(self, modeltype: str, nlayer: int) -> None:
        """Run EZKL setup and proving commands."""
        base_name = f"{modeltype}{nlayer}"
        model_path = self.config.runfile_dir / f"{base_name}.onnx"
        input_path = self.config.runfile_dir / f"input{base_name}.json"
        circuit_path = self.config.runfile_dir / f"{base_name}.ezkl"
        witness_path = self.config.runfile_dir / f"witness_{base_name}.json"
        
        commands = [
            f"ezkl table -M {model_path}",
            f"ezkl gen-settings -M {model_path} --settings-path=settings.json",
            f"ezkl calibrate-settings -M {model_path} -D {input_path} --settings-path=settings.json --target=resources",
            f"ezkl compile-circuit -M {model_path} -S settings.json --compiled-circuit {circuit_path}",
            f"ezkl gen-witness -M {circuit_path} -D {input_path} --output {witness_path}"
        ]
        
        for cmd in commands:
            if self.config.logging:
                cmd += f" >> {self.config.log_dir}/{modeltype}_prework_{nlayer}.log"
            if os.system(cmd) != 0:
                raise RuntimeError(f"Command failed: {cmd}")

    def setup_and_prove(self, modeltype: str, nlayer: int) -> Tuple[str, int, int, int, float, float, int, int, int]:
        """Setup model and run proving process."""
        try:
            model, input_shape, dummy_input = self._create_model(modeltype, nlayer)
            macs, params = profile(model, inputs=(dummy_input,))
            
            # Export model
            export(
                model,
                input_shape=input_shape,
                onnx_filename=self.config.runfile_dir / f'{modeltype}{nlayer}.onnx',
                input_filename=self.config.runfile_dir / f'input{modeltype}{nlayer}.json'
            )

            # Run EZKL commands
            self._run_ezkl_commands(modeltype, nlayer)

            # Get logrows from settings
            with open('settings.json', 'r') as f:
                logrows = json.load(f)['run_args']['logrows']

            # Setup and prove
            time_to_setup = timeit.timeit(
                lambda: self._run_setup_command(modeltype, nlayer, logrows),
                number=1
            )

            proof_file = self.config.proof_dir / f"{modeltype}_proof_{nlayer}.proof"
            time_to_prove = timeit.timeit(
                lambda: self._run_prove_command(modeltype, nlayer, logrows, proof_file),
                number=1
            )

            # Get file sizes
            proof_size = proof_file.stat().st_size
            vk_size = Path('vk.key').stat().st_size
            pk_size = Path('pk.key').stat().st_size

            self.logger.info(
                f"Model type: {modeltype}, nlayer: {nlayer}, "
                f"param_count: {params}, ops_count: {macs}, "
                f"time_to_setup: {time_to_setup}, time_to_prove: {time_to_prove}, "
                f"proof_size: {proof_size}, vk_size: {vk_size}, pk_size: {pk_size}"
            )

            return modeltype, nlayer, params, macs, time_to_setup, time_to_prove, proof_size, vk_size, pk_size
            
        except Exception as e:
            self.logger.error(f"Error in setup_and_prove: {str(e)}")
            raise

    def run_experiments(self):
        """Run experiments for all model types and layers."""
        with open(self.config.output_file, 'w') as f:
            f.write("modeltype,nlayer,param_count,macs,time_to_setup,time_to_prove,proof_size,vk_size,pk_size\n")

        results = []
        for nlayer in tqdm(range(1, 25)):
            for modeltype in self.config.model_types:
                self.logger.info(f"Running {modeltype} with {nlayer} layers")
                result = self.setup_and_prove(modeltype, nlayer)
                results.append(result)
                
                with open(self.config.output_file, 'a') as f:
                    f.write(",".join(map(str, result)) + "\n")

        return results

def main():
    config = Config()
    runner = ModelRunner(config)
    results = runner.run_experiments()
    print("Done with local experiments")
    print(results)

if __name__ == "__main__":
    main()