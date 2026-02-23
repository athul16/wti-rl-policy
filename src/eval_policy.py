from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.device import get_device
from src.io_sims import load_sim_paths
from src.env_trading import TradingEnv, EnvConfig, ACTIONS
from src.models import QNetwork

def run_episode(qnet, returns_1d, cfg: EnvConfig, device, eps: float = 0.0):
    env = TradingEnv(returns_1d, cfg)
    obs = env.reset()
    done = False

    ws = []
    rs = []
    rewards = []

    while not done:
        if np.random.rand() < eps:
            a = int(np.random.randint(0, len(ACTIONS)))
        else:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                q = qnet(x)
                a = int(torch.argmax(q, dim=-1).item())

        obs, rew, done, info = env.step(a)
        ws.append(info["w"])
        rs.append(info["r_next"])
        rewards.append(rew)

    ws = np.array(ws, dtype=np.float32)
    rs = np.array(rs, dtype=np.float32)
    pnl = ws * rs
    equity = np.cumsum(pnl)
    return {
        "equity": equity,
        "pnl": pnl,
        "ws": ws,
        "rs": rs,
        "rewards": np.array(rewards, dtype=np.float32),
    }

def main():
    device = get_device()

    ckpt_path = Path("outputs/checkpoints/dqn_ep50.pt")
    if not ckpt_path.exists():
        ckpts = sorted(Path("outputs/checkpoints").glob("dqn_ep*.pt"))
        if not ckpts:
            raise FileNotFoundError("No checkpoints found in outputs/checkpoints/")
        ckpt_path = ckpts[-1]

    ckpt = torch.load(ckpt_path, map_location="cpu")
    obs_dim = int(ckpt["obs_dim"])
    n_actions = int(ckpt["n_actions"])

    qnet = QNetwork(obs_dim=obs_dim, hidden=128, n_actions=n_actions).to(device)
    qnet.load_state_dict(ckpt["q_state_dict"])
    qnet.eval()

    cfg = EnvConfig(**ckpt["cfg"])

    sim_path = Path("data/sims/wti_simulated_paths.npz")
    paths = load_sim_paths(sim_path)
    N = paths.shape[0]

    idx = np.random.randint(0, N)
    returns_1d = paths[idx]

    out_rl = run_episode(qnet, returns_1d, cfg, device=device, eps=0.0)

    env2 = TradingEnv(returns_1d, cfg)
    obs = env2.reset()
    done = False
    rs = []
    while not done:
        obs, rew, done, info = env2.step(2)
        rs.append(info["r_next"])
    rs = np.array(rs, dtype=np.float32)
    out_long = np.cumsum(rs)

    env3 = TradingEnv(returns_1d, cfg)
    obs = env3.reset()
    done = False
    rs2 = []
    while not done:
        obs, rew, done, info = env3.step(1)
        rs2.append(info["r_next"])
    rs2 = np.array(rs2, dtype=np.float32)
    out_flat = np.cumsum(0.0 * rs2)

    fig = plt.figure()
    plt.plot(out_rl["equity"], label="RL policy")
    plt.plot(out_long, label="Always long")
    plt.plot(out_flat, label="Always flat")
    plt.title("Equity curve on 1 simulated WTI path")
    plt.xlabel("Step")
    plt.ylabel("Cumulative PnL (log-return units)")
    plt.legend()
    Path("outputs/figures").mkdir(parents=True, exist_ok=True)
    out_file = Path("outputs/figures/equity_curve.png")
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    print("saved", out_file)

if __name__ == "__main__":
    main()
