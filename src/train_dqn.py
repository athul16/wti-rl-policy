from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from src.device import get_device, device_info
from src.io_sims import load_sim_paths
from src.env_trading import TradingEnv, EnvConfig
from src.models import QNetwork
from src.replay import ReplayBuffer
from src.policy import select_action_eps_greedy


def soft_update(target, online, tau: float):
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)


def main():
    sim_path = Path("data/sims/wti_real_train.npz")
    if not sim_path.exists():
        raise FileNotFoundError(
            "Put file at data/sims/wti_real_train.npz (must contain 'paths')."
        )

    paths = load_sim_paths(sim_path)
    N, T = paths.shape
    print("Loaded sims:", paths.shape)
    print(device_info())

    cfg = EnvConfig(
        lookback=30,
        cost=0.0005,
        risk_lambda=0.001,
        episode_len=252,
        start_random=True,
    )
    n_actions = 3
    device = get_device()

    rng = np.random.default_rng(0)

    # Infer obs_dim from environment (auto-adapts if you change features)
    idx0 = int(rng.integers(0, N))
    env0 = TradingEnv(paths[idx0], cfg)
    obs0 = env0.reset()
    obs_dim = int(len(obs0))
    print(f"obs_dim inferred from env.reset(): {obs_dim}")

    q = QNetwork(obs_dim=obs_dim, hidden=128, n_actions=n_actions).to(device)
    q_targ = QNetwork(obs_dim=obs_dim, hidden=128, n_actions=n_actions).to(device)
    q_targ.load_state_dict(q.state_dict())

    opt = torch.optim.Adam(q.parameters(), lr=3e-4)
    rb = ReplayBuffer(capacity=200_000)

    gamma = 0.99
    batch_size = 256
    warmup_steps = 10_000
    train_steps_per_epoch = 2_000
    epochs = 50

    eps_start, eps_end = 1.0, 0.05
    eps_decay_steps = 200_000

    tau = 0.005
    global_step = 0

    out_dir = Path("outputs/checkpoints")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep in trange(1, epochs + 1):
        q.train()

        idx = int(rng.integers(0, N))
        env = TradingEnv(paths[idx], cfg)
        obs = env.reset()

        ep_ret = 0.0
        done = False

        # Rollout 1 episode
        while not done:
            eps = max(
                eps_end,
                eps_start - (eps_start - eps_end) * (global_step / eps_decay_steps),
            )
            a = select_action_eps_greedy(q, obs, eps, device, n_actions=n_actions)
            obs2, r, done, _ = env.step(a)

            rb.add(obs, a, r, obs2, done)
            obs = obs2
            ep_ret += r
            global_step += 1

        # Train from replay after warmup
        avg_loss = float("nan")
        if len(rb) >= warmup_steps:
            losses = []
            for _ in range(train_steps_per_epoch):
                s, a, r, s2, d = rb.sample(batch_size)

                s = torch.tensor(s, dtype=torch.float32, device=device)
                a = torch.tensor(a, dtype=torch.int64, device=device).unsqueeze(1)
                r = torch.tensor(r, dtype=torch.float32, device=device)
                s2 = torch.tensor(s2, dtype=torch.float32, device=device)
                d = torch.tensor(d, dtype=torch.float32, device=device)

                q_sa = q(s).gather(1, a).squeeze(1)

                with torch.no_grad():
                    q_next = q_targ(s2).max(dim=1).values
                    target = r + gamma * (1.0 - d) * q_next

                loss = F.mse_loss(q_sa, target)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                opt.step()

                soft_update(q_targ, q, tau)
                losses.append(loss.item())

            avg_loss = float(np.mean(losses))

        warm = "WARMUP" if len(rb) < warmup_steps else "TRAIN"
        print(
            f"\nEpoch {ep:03d} | mode {warm} | eps {eps:.3f} | ep_return {ep_ret:.6f} | "
            f"buffer {len(rb)} | loss {avg_loss}"
        )

        if ep % 5 == 0:
            torch.save(
                {
                    "q_state_dict": q.state_dict(),
                    "cfg": cfg.__dict__,
                    "obs_dim": obs_dim,
                    "n_actions": n_actions,
                },
                out_dir / f"dqn_ep{ep}.pt",
            )


if __name__ == "__main__":
    main()
