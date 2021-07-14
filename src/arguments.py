from pathlib import Path
from typing import List
from dataclasses import dataclass, field
import torch

@dataclass
class Args:

    project_dir: Path
    models_path: Path
    dataset_path: Path
    batch_size: int
    num_experts: int
    id_wandb: str
    initialize: bool = True
    resume: str = "allow"
    n_critic: int = 5
    lambda_gp: int = 10
    epochs_init: int = 10
    epochs: int = 1000
    lr: float = 1e-3
    k: int = 20
    aggr: str = 'max'
    D_channels: List = field(default_factory=lambda: [3,  64,  128, 256, 512, 1024])
    in_channels: int = 3
    num_points: int = 1024
    n_cpu: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    