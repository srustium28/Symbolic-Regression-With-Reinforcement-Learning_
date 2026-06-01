from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


# Locate the project root and default dataset directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "Feynman_with_units"


@dataclass
class EquationDataset:
    """Container for one equation's input/output data and metadata."""

    X: np.ndarray             # input variables, shape (N, d)
    y: np.ndarray             # target output, shape (N,)
    formula_name: str
    equation_id: Optional[str] = None
    source_path: Optional[str] = None

    @property
    def variable_names(self) -> List[str]:
        # Generate labels x1, x2, ..., xd based on number of input columns
        return [f"x{i + 1}" for i in range(self.X.shape[1])]

    def sample_uniform(self, sample_size: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        total_rows = len(self.y)

        if total_rows <= sample_size:
            indices = np.arange(total_rows)
        else:
            # Evenly space indices so the sample covers the full dataset range
            step = max(1, total_rows // sample_size)
            indices = np.arange(0, total_rows, step)[:sample_size]

        return self.X[indices], self.y[indices]


def available_equation_ids(data_dir: Optional[Path] = None) -> List[str]:
    # Return sorted filenames from the data directory
    root = Path(data_dir or DEFAULT_DATA_DIR)
    return sorted(path.name for path in root.iterdir() if path.is_file())


def split_equation_ids(
    equation_ids: Sequence[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    # Shuffle with a fixed seed for reproducibility
    rng = np.random.default_rng(seed)
    shuffled_ids = list(equation_ids)
    rng.shuffle(shuffled_ids)

    total = len(shuffled_ids)
    train_end = int(train_ratio * total)
    val_end = int((train_ratio + val_ratio) * total)

    train_ids = shuffled_ids[:train_end]
    val_ids   = shuffled_ids[train_end:val_end]
    test_ids  = shuffled_ids[val_end:]

    return train_ids, val_ids, test_ids


def load_equation_dataset(
    equation_id: str,
    data_dir: Optional[Path] = None,
    formula_name: Optional[str] = None,
) -> EquationDataset:
    root = Path(data_dir or DEFAULT_DATA_DIR)
    source_path = root / equation_id

    raw = np.loadtxt(source_path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)  # handle single-row files

    # All columns except the last are inputs; the last column is the target
    inputs  = raw[:, :-1].astype(float)
    targets = raw[:, -1].astype(float)

    return EquationDataset(
        X=inputs,
        y=targets,
        formula_name=formula_name or equation_id,
        equation_id=equation_id,
        source_path=str(source_path),
    )


def load_equation_datasets(
    equation_ids: Optional[Sequence[str]] = None,
    data_dir: Optional[Path] = None,
) -> List[EquationDataset]:
    # Default to loading all available datasets if no IDs are provided
    selected_ids: Iterable[str] = equation_ids or available_equation_ids(data_dir)
    return [load_equation_dataset(eq_id, data_dir=data_dir) for eq_id in selected_ids]