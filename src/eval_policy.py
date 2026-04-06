from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.device import get_device, device_info
from src.io_sims import load_sim_paths
from src.env_trading import TradingEnv, EnvConfig
from src.models import QNetwork
from src.policy import select_action_eps_greedy

def run_episode(env, q=None, device=None, greedy=True):
    obs = env.reset()
    done = False
    pnl = []
    actions = []

    while not done:
        if q is None:
            # baseline expects env to have .position updated by env.step
            a = 1  # placeholder; caller sets by passing different env/logic
        else:
            eps = 0.0 if greedy else 0.05
            a = select_action_eps_greedy(q, obs, eps, device, n_actions=3)
        obs, r, done, _ = env.step(a)
        pnl.append(float(r))
        actions.append(int(a))
    return np.array(pnl, dtype=np.float64), np.array(actions, dtype=np.int64)

def run_baseline(env, action_id):
    obs = env.reset()
    done = False
    pnl = []
    while not done:
        obs, r, done, _ = env.step(action_id)
        pnl.append(float(r))
    return np.array(pnl, dtype=np.float64)

def main():
    # Use the held-out real test episodes, matching metrics_eval.py
    ep_path = Path("data/sims/wti_real_test.npz")
    if not ep_path.exists():
        raise FileNotFoundError("Missing data/sims/wti_real_test.npz (run: python3 -m src.split_real_episodes)")

    paths = load_sim_paths(ep_path)
    print("Loaded held-out test episodes:", paths.shape)
    print(device_info())

    cfg = EnvConfig(lookback=30, cost=0.0005, risk_lambda=0.001, episode_len=252, start_random=False)

    ckpt_path = Path("outputs/checkpoints/dqn_ep50.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError("Missing checkpoint outputs/checkpoints/dqn_ep50.pt")

    obj = torch.load(ckpt_path, map_location="cpu")
    obs_dim = int(obj.get("obs_dim", cfg.lookback + 2))
    n_actions = int(obj.get("n_actions", 3))

    device = get_device()
    q = QNetwork(obs_dim=obs_dim, hidden=128, n_actions=n_actions).to(device)
    q.load_state_dict(obj["q_state_dict"])
    q.eval()

    # Pick a deterministic episode index so plot + metrics match
    idx = 0
    rets = paths[idx]

    env_rl = TradingEnv(rets, cfg)
    pnl_rl, acts = run_episode(env_rl, q=q, device=device, greedy=True)

    env_long = TradingEnv(rets, cfg)
    pnl_long = run_baseline(env_long, action_id=1)   # always long

    env_flat = TradingEnv(rets, cfg)
    pnl_flat = run_baseline(env_flat, action_id=0)   # always flat (adjust if your mapping differs)

    env_short = TradingEnv(rets, cfg)
    pnl_short = run_baseline(env_short, action_id=2)  # always short

    eq_rl = np.cumsum(pnl_rl)
    eq_long = np.cumsum(pnl_long)
    eq_flat = np.cumsum(pnl_flat)
    eq_short = np.cumsum(pnl_short)

    out_dir = Path("outputs/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(eq_rl, label="RL Policy")
    plt.plot(eq_flat, label="Always Flat")
    plt.plot(eq_long, label="Always Long")
    plt.plot(eq_short, label="Always Short")
    plt.title("Held-Out Policy Comparison on 1 Episode")
    plt.xlabel("Step")
    plt.ylabel("Cumulative PnL")
    plt.legend()
    out = out_dir / "heldout_policy_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("saved", out)

if __name__ == "__main__":
    main()
