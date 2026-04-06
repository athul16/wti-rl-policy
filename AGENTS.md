# AGENTS.md

## Project
WTI RL Policy (DQN)

## Purpose
This repository is Project 2 of a two-project system.

Its job is to train a DQN-based trading policy on WTI crude oil simulated paths.

The agent likely chooses among:
- short
- flat
- long

The training data for this repo is intended to come from the upstream repository:
`wti-signature-cvae`

That upstream repo generates synthetic WTI paths that should preserve realistic market behavior.

## System-level context
Overall pipeline:

1. Real WTI data is used in `wti-signature-cvae`
2. That repo learns a signature-based generative model
3. It exports simulated paths
4. This repo loads those simulated paths
5. A trading environment is built over the paths
6. A DQN agent is trained to learn trading behavior

## Core objective
Learn a trading policy that performs well across many realistic simulated WTI market scenarios, not just a narrow slice of historical data.

This repo should be designed with the assumption that path diversity and robustness matter.

## Data assumptions
Expected input is something like:
- `data/sims/wti_simulated_paths.npz`
- contains array `paths`
- shape `(N, T)`

The values may be:
- log returns (preferred), or
- prices

The code should make this explicit and not silently assume one if the other is provided.

## Expectations for the codebase
When analyzing or editing this repo:

- Identify how data is loaded
- Identify whether inputs are treated as returns or prices
- Identify the environment state representation
- Identify the action space and exact mapping
- Identify the reward function
- Identify transaction cost / slippage assumptions
- Identify the DQN architecture
- Identify replay buffer / target network / epsilon schedule logic
- Identify training loop and evaluation loop
- Identify plotting and metrics outputs

## Key design questions
Any analysis should explicitly answer:
1. What exactly is the state?
2. What are the actions?
3. How is reward defined?
4. Is reward raw PnL, risk-adjusted PnL, or something else?
5. Are position changes penalized?
6. Are transaction costs included?
7. How is generalization tested across simulated paths?

## Evaluation expectations
This repo should support or eventually support:
- training reward over time
- episode PnL over time
- cumulative return curves
- Sharpe ratio
- volatility
- max drawdown
- win rate / trade stats
- comparison vs buy-and-hold
- comparison vs simple baselines
- action distribution over time
- diagnostics by market regime if available

## Relationship to upstream generator
This repo depends on the quality of the generated paths.

When debugging poor RL performance, consider:
- unrealistic simulated data
- bad reward design
- weak state representation
- unstable DQN training
- mismatch between returns/prices format
- insufficient evaluation metrics

## Editing guidance
- Prefer small, high-confidence changes
- Keep data interfaces clean
- Make assumptions explicit
- Add evaluation before overcomplicating the agent
- Prioritize reproducibility and clear metrics
