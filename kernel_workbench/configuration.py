"""Experiment definitions and collection settings; no model code is imported here."""
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class CollectionConfig(BaseModel):
    runner: str = 'tiny_decoder'
    model: str | None = Field(default=None, pattern=r'^[a-z][a-z0-9_]*$')
    gpu: str = 'A100'
    dtype: Literal['bf16', 'fp16', 'fp32'] = 'bf16'
    batch: int = Field(1, ge=1, le=64)
    input_tokens: int = Field(128, ge=1, le=32768)
    output_tokens: int = Field(16, ge=1, le=2048)
    warmup: int = Field(1, ge=0, le=10)
    baseline_repeats: int = Field(5, ge=1, le=30)
    timeout_sec: int = Field(1800, ge=60, le=7200)
    runner_options: dict = Field(default_factory=dict)


class ExperimentConfig(CollectionConfig):
    name: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')
    model: str = Field(pattern=r'^[a-z][a-z0-9_]*$')

    def collection_config(self):
        return {**self.model_dump(exclude={'name'}), 'runner': self.name}


def load_experiment(path):
    path = Path(path)
    if path.is_dir():
        path /= 'config.json'
    return ExperimentConfig(**json.loads(path.read_text()))
