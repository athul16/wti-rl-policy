from pathlib import Path
import numpy as np

def load_sim_paths(npz_path: str | Path) -> np.ndarray:
    obj = np.load(npz_path, allow_pickle=True)
    if "paths" not in obj:
        raise KeyError("NPZ must contain array named 'paths'")
    paths = np.asarray(obj["paths"], dtype=np.float32)
    if paths.ndim != 2:
        raise ValueError(f"Expected paths shape (N,T). Got {paths.shape}")
    return paths
