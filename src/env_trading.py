from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np

from src.features import make_state


@dataclass
class EnvConfig:
    lookback: int = 30
    cost: float = 0.0005
    risk_lambda: float = 0.001
    episode_len: int = 252
    start_random: bool = True


class TradingEnv:
    """
    Simple trading env on a fixed return path.

    Actions:
      0 = flat (pos=0)
      1 = long (pos=+1)
      2 = short (pos=-1)

    Reward uses *next* return:
      pnl = pos * r_{t+1}  - cost * |pos - prev_pos|
      reward = pnl - risk_lambda * pos^2
    """

    def __init__(self, returns: np.ndarray, cfg: EnvConfig):
        self.returns = np.asarray(returns, dtype=np.float32)
        self.cfg = cfg

        self.T = int(self.returns.shape[0])
        self.t: int = 0
        self.t0: int = 0
        self.done: bool = False
        self.position: int = 0  # -1,0,+1

        self.reset()

    def reset(self) -> np.ndarray:
        L = int(self.cfg.lookback)

        # pick a start index that allows lookback and episode_len and one-step-ahead reward
        min_start = L
        max_start = self.T - (self.cfg.episode_len + 1)
        if max_start <= min_start:
            raise ValueError(
                f"Path too short for lookback={L} and episode_len={self.cfg.episode_len}. "
                f"T={self.T}, need at least {L + self.cfg.episode_len + 1}."
            )

        if self.cfg.start_random:
            self.t0 = int(np.random.randint(min_start, max_start))
        else:
            self.t0 = min_start

        self.t = self.t0
        self.done = False
        self.position = 0

        return make_state(self.returns, self.t, L, self.position)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("step() called after episode done. Call reset().")

        L = int(self.cfg.lookback)

        if action == 0:
            new_pos = 0
        elif action == 1:
            new_pos = 1
        elif action == 2:
            new_pos = -1
        else:
            raise ValueError(f"Invalid action {action}, expected 0/1/2.")

        prev_pos = self.position
        self.position = new_pos

        # next return for reward
        r_next = float(self.returns[self.t + 1])

        trade_cost = float(self.cfg.cost) * abs(self.position - prev_pos)
        pnl = self.position * r_next - trade_cost
        risk_pen = float(self.cfg.risk_lambda) * (self.position ** 2)
        reward = pnl - risk_pen

        # advance time
        self.t += 1

        # done when we’ve taken episode_len steps
        if (self.t - self.t0) >= int(self.cfg.episode_len):
            self.done = True

        obs = make_state(self.returns, self.t, L, self.position)
        info = {"pnl": pnl, "trade_cost": trade_cost, "risk_pen": risk_pen, "pos": self.position}

        return obs, float(reward), self.done, info
