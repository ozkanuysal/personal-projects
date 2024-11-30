from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

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
import os
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

    def get_log_path(self, fname: str) -> str:
        return f" >> {self.logs_dir}/{fname}.log" if self.logging else ""


class ASRWrapper(nn.Module):
    """Wrapper for Wav2Vec2 model."""
    def __init__(self, model: Wav2Vec2ForCTC):
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_values).logits[0]
        return torch.argmax(logits, dim=-1)


class ASRSystem:
    def __init__(self, config: Config):
        self.config = config
        self.processor = Wav2Vec2Processor.from_pretrained(config.model_name)
        self.model = ASRWrapper(Wav2Vec2ForCTC.from_pretrained(config.model_name))
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(config.model_name)

    def prepare_dataset(self):
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="train")
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.config.sampling_rate))
        return next(iter(dataset))

    def process_audio(self, audio_sample: Dict[str, Any]) -> torch.Tensor:
        return self.feature_extractor(
            audio_sample["audio"]["array"],
            return_tensors="pt",
            sampling_rate=self.config.sampling_rate
        ).input_values

    def export_model(self, input_values: torch.Tensor):
        predicted_ids = self.model(input_values)
        outputs = self.tokenizer.decode(predicted_ids, output_word_offsets=True)
        transcription = self.processor.batch_decode(predicted_ids)

        # Export model statistics
        macs, params = profile(self.model, inputs=(input_values,))
        print(f"Model has {macs} FLOPs and {params} parameters")

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
        data = {
            "input_data": [input_values.detach().numpy().reshape([-1]).tolist()],
            "output_data": [p.detach().numpy().reshape([-1]).tolist() for p in predicted_ids]
        }
        with open(self.config.output_dir / "input.json", 'w') as f:
            json.dump(data, f)

    def run_ezkl_commands(self):
        commands = [
            f"ezkl table -M {self.config.output_dir}/ASR.onnx",
            f"ezkl gen-settings -M {self.config.output_dir}/ASR.onnx --settings-path={self.config.output_dir}/settings.json",
            f"ezkl calibrate-settings -M {self.config.output_dir}/ASR.onnx -D {self.config.output_dir}/input.json --settings-path={self.config.output_dir}/settings.json",
        ]
        
        for cmd in commands:
            os.system(cmd + self.config.get_log_path('setup'))

        # Load settings and continue with remaining commands
        with open(self.config.output_dir / 'settings.json', 'r') as f:
            settings = json.load(f)
            logrows = settings['run_args']['logrows']

        final_commands = [
            f"ezkl compile-circuit -M {self.config.output_dir}/ASR.onnx -S {self.config.output_dir}/settings.json --compiled-circuit {self.config.output_dir}/ASR.ezkl",
            f"ezkl gen-witness -M {self.config.output_dir}/ASR.ezkl -D {self.config.output_dir}/input.json --output {self.config.output_dir}/witnessRandom.json",
            f"ezkl setup -M {self.config.output_dir}/ASR.ezkl --srs-path={self.config.srs_path % logrows} --vk-path={self.config.output_dir}/vk.key --pk-path={self.config.output_dir}/pk.key",
            f"ezkl prove -M {self.config.output_dir}/ASR.ezkl --srs-path={self.config.srs_path % logrows} --pk-path={self.config.output_dir}/pk.key --witness {self.config.output_dir}/witnessRandom.json"
        ]

        for cmd in final_commands:
            os.system(cmd + self.config.get_log_path('setup'))


def main():
    config = Config()
    asr_system = ASRSystem(config)
    
    sample = asr_system.prepare_dataset()
    input_values = asr_system.process_audio(sample)
    predicted_ids = asr_system.export_model(input_values)
    asr_system.export_data(input_values, predicted_ids)
    asr_system.run_ezkl_commands()


if __name__ == "__main__":
    main()