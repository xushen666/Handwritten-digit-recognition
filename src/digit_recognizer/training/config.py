import random
from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    validation_size: int = 5_000
    batch_size: int = 256
    epochs: int = 15
    learning_rate: float = 0.001
    num_workers: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_indices(size: int, validation_size: int, seed: int) -> tuple[list[int], list[int]]:
    if size <= 1 or validation_size <= 0 or validation_size >= size:
        raise ValueError("validation_size must leave non-empty training and validation sets")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(size, generator=generator).tolist()
    return order[validation_size:], order[:validation_size]
