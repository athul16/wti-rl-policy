import numpy as np

def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    m = float(np.mean(x))
    s = float(np.std(x))
    return (x - m) / (s + eps)

def _drawdown(cum: np.ndarray) -> float:
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(np.min(dd))  # negative or 0

def _window_stats(w: np.ndarray) -> np.ndarray:
    vol = float(np.std(w))
    drift = float(np.mean(w))
    mom = float(np.sum(w))
    cum = np.cumsum(w, dtype=np.float64)
    dd = _drawdown(cum)
    return np.array([vol, drift, mom, dd], dtype=np.float32)

def make_state(returns: np.ndarray, t: int, lookback: int, position: int) -> np.ndarray:
    """
    returns: 1D array of per-step returns (float)
    t: current index (we build features using returns[t-lookback : t])
    position: current environment position in {-1, 0, +1}
    """
    # Window
    start = max(0, t - lookback)
    w = returns[start:t].astype(np.float32)

    # Pad on the left if t < lookback
    if w.shape[0] < lookback:
        pad = np.zeros((lookback - w.shape[0],), dtype=np.float32)
        w = np.concatenate([pad, w], axis=0)

    # Normalize the return window (local z-score)
    w_norm = _zscore(w).astype(np.float32)
    w_norm = np.clip(w_norm, -5.0, 5.0)

    # Regime features (computed on raw w, but stabilized)
    vol = float(np.std(w))
    drift = float(np.mean(w))
    k = min(10, lookback)
    mom = float(np.sum(w[-k:]))  # k-step momentum

    cum = np.cumsum(w, dtype=np.float64)
    dd = _drawdown(cum)  # negative number

    # Simple trend proxy: slope of cum return over window
    # (cov(time, cum)/var(time))
    x = np.arange(lookback, dtype=np.float64)
    slope = float(np.cov(x, cum)[0, 1] / (np.var(x) + 1e-12))

    regime = np.array([vol, drift, mom, dd, slope], dtype=np.float32)
    regime = np.clip(regime, -5.0, 5.0)

    # Multi-horizon summaries to support longer-horizon behavior.
    h20 = returns[max(0, t - 20):t].astype(np.float32)
    h60 = returns[max(0, t - 60):t].astype(np.float32)
    h120 = returns[max(0, t - 120):t].astype(np.float32)

    mh = []
    for h in (h20, h60, h120):
        if h.size == 0:
            mh.append(np.zeros((4,), dtype=np.float32))
        else:
            mh.append(_window_stats(h))
    multi_horizon = np.concatenate(mh, axis=0)
    multi_horizon = np.clip(multi_horizon, -5.0, 5.0)

    # Position one-hot in env position order: [-1, 0, +1]
    pos_oh = np.zeros((3,), dtype=np.float32)
    pos_to_idx = {-1: 0, 0: 1, 1: 2}
    idx = pos_to_idx.get(int(position))
    if idx is not None:
        pos_oh[idx] = 1.0

    # Final state = [lookback normalized returns] + [regime 5] + [multi-horizon 12] + [pos one-hot 3]
    state = np.concatenate([w_norm, regime, multi_horizon, pos_oh], axis=0).astype(np.float32)
    return state
