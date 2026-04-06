# WTI RL Policy — Project Context

## Project Role
This repository is Project 2 in a two-project system.

It trains a **DQN trading agent** on simulated WTI paths.

These simulated paths come from:
../wti-signature-cvae

This repo does NOT generate data.
It consumes synthetic market scenarios.

---

# System Architecture

Real WTI data
    ↓
wti-signature-cvae
    ↓ generates synthetic paths
wti-rl-policy (THIS REPO)
    ↓ trains RL trader
Output:
robust trading policy

---

# RL Objective

Learn a trading policy that maximizes returns across:

- trending markets
- mean reverting markets
- high volatility
- crash regimes
- synthetic stress scenarios

---

# Action Space

Discrete:

0 = short
1 = flat
2 = long

Positions persist across timesteps.

---

# Environment Design

State may include:
- recent returns window
- current position
- rolling volatility
- time index
- optional signature features (future idea)

---

# Reward Function (Baseline)

reward_t =
position_{t-1} * return_t
- transaction_cost * |position change|

Important:
Reward must reflect PnL.

---

# Data Input

Expected:

NPZ file:
paths shape (N, T)

Each path:
- return series OR
- price series

This must be explicitly handled.

---

# Training Loop

For each path:
    reset environment
    step through time
    choose action
    observe reward
    store transition
    train DQN

---

# DQN Components

- Q network
- target network
- replay buffer
- epsilon-greedy exploration
- TD loss
- periodic target update

---

# Evaluation Metrics

Must include:

- cumulative PnL
- Sharpe ratio
- max drawdown
- volatility
- win rate
- average trade return
- reward curve
- buy & hold comparison
- random policy comparison

---

# Plots Needed

1. training reward curve
2. episode PnL
3. cumulative returns
4. action distribution
5. Q value evolution
6. drawdown chart

---

# Why Synthetic Data Matters

Training only on historical data:
- overfits
- limited regimes

Training on generated paths:
- broader distribution
- more crashes
- more volatility patterns
- better generalization

---

# Future Extensions

- risk-adjusted reward
- Sharpe-based reward
- policy gradient comparison
- regime-conditioned policy
- ensemble training
- signature-based state

---

# DO NOT

- assume Gaussian returns
- ignore transaction costs
- evaluate only reward
- train on one path only

---

# This Repo Exists To

Learn a **robust trading policy**
from synthetic WTI market scenarios.
