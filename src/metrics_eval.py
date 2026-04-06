import argparse
import json
from pathlib import Path
import re
import numpy as np
import torch

from src.device import get_device, device_info
from src.io_sims import load_sim_paths
from src.env_trading import TradingEnv, EnvConfig
from src.models import QNetwork

def latest_ckpt(ckpt_dir: Path) -> Path:
    pts = list(ckpt_dir.glob("dqn_ep*.pt"))
    if not pts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}.")
    def ep_num(p: Path) -> int:
        m = re.search(r"dqn_ep(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1
    return sorted(pts, key=ep_num)[-1]

def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())

def sharpe(returns: np.ndarray, ann_factor: float = 252.0) -> float:
    r = returns[np.isfinite(returns)]
    if r.size < 2:
        return float("nan")
    mu = r.mean()
    sig = r.std(ddof=1)
    if sig == 0:
        return float("nan")
    return float(np.sqrt(ann_factor) * mu / sig)

def sortino(returns: np.ndarray, ann_factor: float = 252.0) -> float:
    r = returns[np.isfinite(returns)]
    if r.size < 2:
        return float("nan")
    mu = r.mean()
    downside = r[r < 0]
    if downside.size < 2:
        return float("nan")
    ds = downside.std(ddof=1)
    if ds == 0:
        return float("nan")
    return float(np.sqrt(ann_factor) * mu / ds)

def safe_nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    valid = x[np.isfinite(x)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())

def safe_nanstd(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    valid = x[np.isfinite(x)]
    if valid.size == 0:
        return float("nan")
    return float(valid.std())

def summarize_policy(episodes, n_actions: int, runner):
    episode_metrics = []
    action_counts = np.zeros(n_actions, dtype=np.int64)

    for path in episodes:
        pnl, eq, acts = runner(path)
        total_pnl = float(eq[-1]) if eq.size else 0.0
        episode_metrics.append(
            {
                "total_pnl": total_pnl,
                "sharpe": sharpe(pnl),
                "max_drawdown": max_drawdown(eq) if eq.size else float("nan"),
                "win": total_pnl > 0.0,
            }
        )
        for a in range(n_actions):
            action_counts[a] += int((acts == a).sum())

    result = {"num_episodes": int(len(episodes)), "action_counts": dict(zip(range(n_actions), action_counts.tolist()))}
    if not episode_metrics:
        result.update({"mean_total_pnl": float("nan"), "std_total_pnl": float("nan"), "mean_sharpe(252)": float("nan"), "std_sharpe(252)": float("nan"), "mean_max_drawdown": float("nan"), "win_rate_across_episodes": float("nan")})
        return result

    total_pnls = np.asarray([m["total_pnl"] for m in episode_metrics], dtype=np.float64)
    sharpes = np.asarray([m["sharpe"] for m in episode_metrics], dtype=np.float64)
    max_drawdowns = np.asarray([m["max_drawdown"] for m in episode_metrics], dtype=np.float64)
    wins = np.asarray([m["win"] for m in episode_metrics], dtype=np.float64)

    result.update(
        {
            "mean_total_pnl": float(np.mean(total_pnls)),
            "std_total_pnl": float(np.std(total_pnls)),
            "mean_sharpe(252)": safe_nanmean(sharpes),
            "std_sharpe(252)": safe_nanstd(sharpes),
            "mean_max_drawdown": safe_nanmean(max_drawdowns),
            "win_rate_across_episodes": float(np.mean(wins)),
        }
    )
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save aggregated evaluation results to outputs/eval/metrics_eval.json",
    )
    args = parser.parse_args()

    episodes = load_sim_paths(Path("data/sims/wti_real_test.npz"))
    print("Loaded held-out test episodes:", episodes.shape)
    print(device_info())

    ckpt = latest_ckpt(Path("outputs/checkpoints"))
    obj = torch.load(ckpt, map_location="cpu")
    print("Loading checkpoint:", ckpt)

    cfg = EnvConfig(**obj["cfg"])
    # Make held-out evaluation deterministic across runs.
    # Training may use randomized starts, but evaluation should not.
    cfg.start_random = False

    obs_dim = int(obj["obs_dim"])
    n_actions = int(obj["n_actions"])

    device = get_device()
    q = QNetwork(obs_dim=obs_dim, hidden=128, n_actions=n_actions).to(device)
    q.load_state_dict(obj["q_state_dict"])
    q.eval()

    # Greedy RL rollout on one path
    def run_episode(path):
        env = TradingEnv(path, cfg)
        obs = env.reset()
        done = False
        pnl = []
        acts = []
        while not done:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                qs = q(x).squeeze(0)
                a = int(torch.argmax(qs).item())
            obs, r, done, info = env.step(a)
            pnl.append(float(info.get("pnl", r)))
            acts.append(a)
        pnl = np.asarray(pnl, dtype=np.float32)
        eq = np.cumsum(pnl)
        return pnl, eq, np.asarray(acts, dtype=np.int64)

    # Fixed-action baseline rollout on one path
    def run_fixed_policy(path, action_id: int):
        env = TradingEnv(path, cfg)
        obs = env.reset()
        done = False
        pnl = []
        acts = []
        while not done:
            obs, r, done, info = env.step(action_id)
            pnl.append(float(info.get("pnl", r)))
            acts.append(int(action_id))
        pnl = np.asarray(pnl, dtype=np.float32)
        eq = np.cumsum(pnl)
        return pnl, eq, np.asarray(acts, dtype=np.int64)

    results = {
        "RL Policy": summarize_policy(episodes, n_actions, run_episode),
        "Always Flat": summarize_policy(episodes, n_actions, lambda path: run_fixed_policy(path, action_id=0)),
        "Always Long": summarize_policy(episodes, n_actions, lambda path: run_fixed_policy(path, action_id=1)),
        "Always Short": summarize_policy(episodes, n_actions, lambda path: run_fixed_policy(path, action_id=2)),
    }

    metrics = [
        "num_episodes",
        "mean_total_pnl",
        "std_total_pnl",
        "mean_sharpe(252)",
        "std_sharpe(252)",
        "mean_max_drawdown",
        "win_rate_across_episodes",
        "action_counts",
    ]
    policies = ["RL Policy", "Always Flat", "Always Long", "Always Short"]

    print("\n=== Held-Out Comparison ===")
    for metric in metrics:
        row = [metric.ljust(24)]
        for policy in policies:
            value = results[policy][metric]
            if isinstance(value, float):
                cell = f"{value:.6f}" if np.isfinite(value) else "nan"
            else:
                cell = str(value)
            row.append(cell.ljust(24))
        print(" | ".join(row))

    if args.save_json:
        out_dir = Path("outputs/eval")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "metrics_eval.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print("\nsaved", out_path)

if __name__ == "__main__":
    main()
