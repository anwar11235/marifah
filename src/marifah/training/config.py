"""Training configuration for the graph DAG pipeline (Session 5)."""

from __future__ import annotations

from typing import Optional

import yaml
from pydantic import BaseModel


class ExperimentConfig(BaseModel):
    name: str = "training_run"
    phase: int = 0
    seed: int = 42


class HMSCConfig(BaseModel):
    K_G: int = 64
    K_R: int = 16
    K_P: int = 16
    d_G: int = 512
    d_R: int = 256
    d_P: int = 128
    num_regions: int = 8
    composition_method: str = "sum"
    p_discreteness: str = "soft"
    num_workflow_types: int = 50
    num_pattern_types: int = 12
    num_primitives: int = 10


class ModelConfig(BaseModel):
    d_model: int = 512
    num_heads: int = 8
    H_cycles: int = 1
    L_cycles: int = 1
    H_layers: int = 2
    L_layers: int = 2
    use_hmsc: bool = False
    hmsc: Optional[HMSCConfig] = None
    vocab_size: int = 10          # number of primitive types
    max_nodes: int = 100          # fixed seq_len for CORAL (batches padded to this)
    halt_max_steps: int = 4
    halt_exploration_prob: float = 0.1
    forward_dtype: str = "float32"   # use float32 on CPU; bfloat16 on GPU


class TrainingPhaseConfig(BaseModel):
    batch_size: int = 64
    max_epochs: int = 100
    eval_interval_epochs: int = 5
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    mixed_precision: str = "bf16"
    lambda_G: float = 0.0
    lambda_R: float = 0.0
    lambda_P: float = 0.0
    main_loss_weight: float = 1.0
    halt_loss_weight: float = 0.1     # down-weighted for single-step training
    save_every_n_steps: Optional[int] = None
    early_stopping_patience: Optional[int] = None


class DataConfig(BaseModel):
    dataset_root: str = "/tmp/marifah_dataset"
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True


class LoggingConfig(BaseModel):
    wandb_mode: str = "disabled"
    wandb_project: str = "marifah-core"
    run_name: Optional[str] = None
    log_interval_steps: int = 100
    utilization_interval_steps: int = 500
    checkpoint_dir: str = "checkpoints/"


class WarmStartConfig(BaseModel):
    checkpoint: Optional[str] = None
    load_optimizer: bool = False


class TrainingConfig(BaseModel):
    experiment: ExperimentConfig = ExperimentConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingPhaseConfig = TrainingPhaseConfig()
    data: DataConfig = DataConfig()
    logging: LoggingConfig = LoggingConfig()
    warm_start: Optional[WarmStartConfig] = None


def load_config(path: str, overrides: Optional[dict] = None) -> TrainingConfig:
    """Load a TrainingConfig from a YAML file with optional dict overrides."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        _deep_update(raw, overrides)
    return TrainingConfig(**raw)


def _deep_update(base: dict, updates: dict) -> None:
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
