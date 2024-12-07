from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import subprocess

import torch
import torch.nn as nn
from datasets import load_dataset, Audio
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    AutoTokenizer,
    AutoFeatureExtractor
)
import json
from thop import profile


@dataclass
class Config:
    """Configuration settings for ASR system."""
    output_dir: Path = Path("ASR")
    logs_dir: Path = Path("ASR/logs")
    srs_path: str = "../kzgs/kzg%d.srs"
    logging: bool = False
    model_name: str = "facebook/wav2vec2-base"
    sampling_rate: int = 16_000

    def __post_init__(self):
        self.output_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        if self.logging:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[logging.FileHandler(self.logs_dir / "asr.log")]
            )


class CommandRunner:
    """Handle command execution and logging."""
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def run(self, command: str, log_suffix: str = "") -> subprocess.CompletedProcess:
        """Execute command and handle errors."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            if self.config.logging:
                self.logger.info(f"Command succeeded: {command}")
            return result
        except subprocess.CalledProcessError as e:
            if self.config.logging:
                self.logger.error(f"Command failed: {command}\nError: {e.stderr}")
            raise RuntimeError(f"Command failed: {e.stderr}")


class ASRWrapper(nn.Module):
    """Wrapper for Wav2Vec2 model."""
    def __init__(self, model: Wav2Vec2ForCTC):
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_values).logits[0]
        return torch.argmax(logits, dim=-1)


class ASRSystem:
    """Automatic Speech Recognition System."""
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cmd_runner = CommandRunner(config)
        
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all required models and processors."""
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.config.model_name)
            self.model = ASRWrapper(Wav2Vec2ForCTC.from_pretrained(self.config.model_name))
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.config.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")

    def prepare_dataset(self) -> Dict[str, Any]:
        """Prepare and load dataset."""
        try:
            dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train")
            dataset = dataset.cast_column("audio", Audio(sampling_rate=self.config.sampling_rate))
            return next(iter(dataset))
        except Exception as e:
            raise RuntimeError(f"Failed to prepare dataset: {str(e)}")

    def process_audio(self, audio_sample: Dict[str, Any]) -> torch.Tensor:
        """Process audio sample into model input."""
        return self.feature_extractor(
            audio_sample["audio"]["array"],
            return_tensors="pt",
            sampling_rate=self.config.sampling_rate
        ).input_values

    def export_model(self, input_values: torch.Tensor) -> torch.Tensor:
        """Export model to ONNX format and return predictions."""
        predicted_ids = self.model(input_values)
        
        # Profile model
        macs, params = profile(self.model, inputs=(input_values,))
        self.logger.info(f"Model has {macs} FLOPs and {params} parameters")

        # Export to ONNX
        torch.onnx.export(
            self.model,
            input_values,
            self.config.output_dir / "ASR.onnx",
            export_params=True,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=16
        )

        return predicted_ids

    def export_data(self, input_values: torch.Tensor, predicted_ids: torch.Tensor):
        """Export model input/output data."""
        data = {
            "input_data": [input_values.detach().numpy().reshape([-1]).tolist()],
            "output_data": [p.detach().numpy().reshape([-1]).tolist() for p in predicted_ids]
        }
        with open(self.config.output_dir / "input.json", 'w') as f:
            json.dump(data, f)

    def run_ezkl_setup(self):
        """Run initial EZKL setup commands."""
        commands = [
            f"ezkl table -M {self.config.output_dir}/ASR.onnx",
            f"ezkl gen-settings -M {self.config.output_dir}/ASR.onnx --settings-path={self.config.output_dir}/settings.json",
            f"ezkl calibrate-settings -M {self.config.output_dir}/ASR.onnx -D {self.config.output_dir}/input.json --settings-path={self.config.output_dir}/settings.json",
        ]
        
        for cmd in commands:
            self.cmd_runner.run(cmd)

    def run_ezkl_prove(self, logrows: int):
        """Run EZKL proving commands."""
        commands = [
            f"ezkl compile-circuit -M {self.config.output_dir}/ASR.onnx -S {self.config.output_dir}/settings.json --compiled-circuit {self.config.output_dir}/ASR.ezkl",
            f"ezkl gen-witness -M {self.config.output_dir}/ASR.ezkl -D {self.config.output_dir}/input.json --output {self.config.output_dir}/witnessRandom.json",
            f"ezkl setup -M {self.config.output_dir}/ASR.ezkl --srs-path={self.config.srs_path % logrows} --vk-path={self.config.output_dir}/vk.key --pk-path={self.config.output_dir}/pk.key",
            f"ezkl prove -M {self.config.output_dir}/ASR.ezkl --srs-path={self.config.srs_path % logrows} --pk-path={self.config.output_dir}/pk.key --witness {self.config.output_dir}/witnessRandom.json"
        ]

        for cmd in commands:
            self.cmd_runner.run(cmd)

    def run_ezkl_commands(self):
        """Run all EZKL commands in sequence."""
        self.run_ezkl_setup()
        
        with open(self.config.output_dir / 'settings.json', 'r') as f:
            settings = json.load(f)
            logrows = settings['run_args']['logrows']
        
        self.run_ezkl_prove(logrows)


def main():
    """Main entry point."""
    try:
        config = Config()
        asr_system = ASRSystem(config)
        
        sample = asr_system.prepare_dataset()
        input_values = asr_system.process_audio(sample)
        predicted_ids = asr_system.export_model(input_values)
        asr_system.export_data(input_values, predicted_ids)
        asr_system.run_ezkl_commands()
    except Exception as e:
        logging.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()