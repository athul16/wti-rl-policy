import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128, n_actions: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
