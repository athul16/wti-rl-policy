import numpy as np

def make_state(returns_1d: np.ndarray, t: int, lookback: int) -> np.ndarray:
    start = t - lookback + 1
    window = returns_1d[start : t + 1].astype(np.float32)
    vol = float(np.sqrt(np.mean(window * window) + 1e-8))
    trend = float(np.mean(window) / (vol + 1e-8))
    return np.concatenate([window, np.array([vol, trend], dtype=np.float32)]).astype(np.float32)
