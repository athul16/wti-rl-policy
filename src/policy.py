import numpy as np
import torch

def select_action_eps_greedy(qnet, obs_np: np.ndarray, eps: float, device: torch.device, n_actions: int = 3) -> int:
    if np.random.rand() < eps:
        return int(np.random.randint(0, n_actions))
    with torch.no_grad():
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0)
        q = qnet(obs)
        return int(torch.argmax(q, dim=-1).item())
