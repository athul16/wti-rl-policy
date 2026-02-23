import numpy as np
from dataclasses import dataclass
from src.features import make_state

ACTIONS = np.array([-1.0, 0.0, 1.0], dtype=np.float32)

@dataclass
class EnvConfig:
    lookback: int = 30
    cost: float = 0.0005
    risk_lambda: float = 0.001
    episode_len: int = 252
    start_random: bool = True

class TradingEnv:
    def __init__(self, returns_1d: np.ndarray, cfg: EnvConfig):
        self.returns = np.asarray(returns_1d, dtype=np.float32)
        self.cfg = cfg
        self.T = len(self.returns)
        min_len = cfg.lookback + cfg.episode_len + 2
        if self.T < min_len:
            raise ValueError(f"Return series too short. Need at least {min_len}, got {self.T}.")
        self.done = False
        self.reset()

    def reset(self):
        L = self.cfg.lookback
        ep = self.cfg.episode_len

        max_start = self.T - (L + ep + 2)
        if max_start < 0:
            raise ValueError("Not enough data for episode with current lookback/episode_len.")

        if self.cfg.start_random:
            start_idx = np.random.randint(0, max_start + 1)
            self.start = start_idx + L
        else:
            self.start = L

        self.t = self.start + L - 1
        self.steps = 0
        self.prev_w = 0.0
        self.done = False
        return make_state(self.returns, self.t, L)

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Call reset() before stepping after done=True.")

        if self.t + 1 >= self.T:
            self.done = True
            obs = make_state(self.returns, self.t, self.cfg.lookback)
            return obs, 0.0, True, {"w": float(ACTIONS[action]), "r_next": 0.0, "turnover": 0.0}

        w = float(ACTIONS[action])
        r_next = float(self.returns[self.t + 1])

        turnover = abs(w - self.prev_w)
        reward = w * r_next - self.cfg.cost * turnover - self.cfg.risk_lambda * (w * w)

        self.prev_w = w
        self.t += 1
        self.steps += 1
        self.done = self.steps >= self.cfg.episode_len

        obs = make_state(self.returns, self.t, self.cfg.lookback)
        info = {"w": w, "r_next": r_next, "turnover": turnover}
        return obs, reward, self.done, info
