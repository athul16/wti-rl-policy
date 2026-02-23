# WTI RL Policy (DQN)

Train a DQN trading policy (short/flat/long) on crude oil return paths.

## Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Data
Place simulated paths at:
data/sims/wti_simulated_paths.npz
The NPZ must contain an array named `paths` with shape (N, T). Values should be log returns (preferred) or prices.

## Train
python -m src.train_dqn
