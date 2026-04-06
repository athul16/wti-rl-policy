# AGENTS.md

## Project
WTI RL Policy (DQN)

## Purpose
This repository is Project 2 of a two-project system.

Its intended long-term job is to train a DQN-based trading policy on synthetic WTI crude oil paths produced by `../wti-signature-cvae`.

Its actual current job is narrower:
- download real WTI data
- convert prices to log returns
- slice those returns into fixed-length overlapping episodes
- train a DQN trading policy on those real-data episodes

The repo is therefore currently a real-data episodization -> DQN scaffold, not yet a full synthetic-data consumer.

## True Current Architecture
Current implemented pipeline:

1. `src/data_real_wti.py`
   - downloads WTI futures (`CL=F`) from Yahoo Finance
   - computes log prices and first differences
   - saves `data/real/wti_returns.npz` with keys:
     - `returns`
     - `dates`

2. `src/split_real_episodes.py`
   - loads `data/real/wti_returns.npz`
   - removes non-finite values
   - splits the return history by time into:
     - first 80% train
     - last 20% test
   - converts each split into overlapping windows of length `T=600` with `stride=5`
   - saves:
     - `data/sims/wti_real_train.npz`
     - `data/sims/wti_real_test.npz`

3. `src/train_dqn.py`
   - loads `data/sims/wti_real_train.npz`
   - samples one path per epoch
   - creates a `TradingEnv` on that path
   - rolls out one episode per epoch
   - stores transitions in replay
   - trains a DQN with a target network and epsilon-greedy exploration
   - saves checkpoints to `outputs/checkpoints/`

4. `src/eval_policy.py` and `src/metrics_eval.py`
   - load `data/sims/wti_real_episodes.npz` or the latest checkpoint
   - evaluate only one episode, not the full held-out test set
   - produce a simple equity plot or single-episode summary metrics

## Actual Data Flow
Exact data flow from raw input to environment step:

1. Raw input starts as daily WTI price history from Yahoo Finance in `src/data_real_wti.py`.
2. Prices are transformed into log returns:
   - `logp = log(Close)`
   - `ret = diff(logp)`
3. Returns are stored in `data/real/wti_returns.npz` under key `returns`.
4. `src/split_real_episodes.py` slices the 1D return series into many overlapping paths of shape `(600,)`.
5. Those paths are saved into NPZ files under key `paths`, yielding shape `(N, 600)`.
6. `src/train_dqn.py` loads `paths` through `src/io_sims.py`.
7. A single row `paths[idx]` is passed into `TradingEnv(returns=path, cfg=...)`.
8. `TradingEnv.reset()` chooses a start index `t0` and calls `make_state(returns, t0, lookback, position)`.
9. `make_state(...)` builds the observation from the trailing return window and current position encoding.
10. `select_action_eps_greedy(...)` chooses an action from the Q-network or uniformly at random.
11. `TradingEnv.step(action)` maps the action to a position, reads `returns[t+1]`, computes reward, advances time, and emits the next state.

## Paths: Returns vs Prices
The code currently treats paths as returns at every implemented stage that matters.

Actual current semantics by stage:

- `src/data_real_wti.py`
  - input: prices
  - output: log returns

- `src/data_fred_wti.py`
  - input: prices
  - output: log returns

- `src/split_real_episodes.py`
  - input: return series
  - output: `(N, T)` return paths

- `src/make_real_episodes.py`
  - input: return series
  - output: `(N, T)` return paths

- `src/make_dummy_sims.py`
  - directly generates return paths

- `src/io_sims.py`
  - loads generic `paths`
  - does not validate whether they are returns or prices

- `src/features.py`
  - assumes the series contains per-step returns

- `src/env_trading.py`
  - assumes the series contains per-step returns
  - reward uses `returns[t+1]`

Important consequence:
- although README-level text says paths may be returns or prices, the implemented environment and features only make sense for return paths
- price paths are not supported correctly today

## Current State Representation
The observation is created in `src/features.py`.

For default `lookback=30`, the state is:

1. `30` trailing returns from `returns[t-lookback:t]`
2. local z-score normalization of that return window
3. `5` regime summary features computed on the raw window:
   - volatility
   - mean drift
   - short-horizon momentum
   - drawdown of cumulative returns
   - slope of cumulative returns over the window
4. `3` position one-hot features

Default state size:
- `30 + 5 + 3 = 38`

Current status:
- the position one-hot block is aligned with environment positions in `{-1, 0, +1}`

## Action Semantics
The true environment mapping is defined in `src/env_trading.py`:

- action `0` -> flat -> position `0`
- action `1` -> long -> position `+1`
- action `2` -> short -> position `-1`

`src/policy.py` is agnostic and only selects an integer action in `[0, n_actions)`.

Evaluation scripts assume the same mapping in some places:
- `eval_policy.py` uses `action_id=1` for always-long
- `eval_policy.py` uses `action_id=0` for always-flat

But repository-level prose is inconsistent:
- some docs describe `0=short, 1=flat, 2=long`
- actual environment code does not do that

## Reward Definition
Reward is defined in `src/env_trading.py` using the next return:

- `r_next = returns[t+1]`
- `trade_cost = cost * abs(new_position - previous_position)`
- `pnl = new_position * r_next - trade_cost`
- `risk_penalty = risk_lambda * new_position^2`
- `reward = pnl - risk_penalty`

Interpretation:
- reward is not raw PnL
- reward is next-step PnL minus transaction cost minus an inventory penalty

Current defaults in training:
- `cost = 0.0005`
- `risk_lambda = 0.002`
- `episode_len = 504`
- `train_steps_per_epoch = 4000`
- `lookback = 30`

## DQN Implementation
Current DQN components:

- Q-network:
  - `src/models.py`
  - two hidden layers of size `128`
  - ReLU activations
  - output dimension `3`

- Replay buffer:
  - `src/replay.py`
  - uniform random sampling

- Exploration:
  - epsilon-greedy in `src/policy.py`
  - epsilon decays linearly from `1.0` to `0.05`

- Target network:
  - soft update with `tau=0.005`

- Optimization:
  - Adam
  - MSE TD loss
  - gradient clipping at `1.0`

Training loop behavior:
- one random training path per epoch
- one environment rollout per epoch
- transitions appended to replay
- replay updates begin only after `warmup_steps=10_000`

## Evaluation Reality
The current evaluation setup is limited.

What it currently does:

- `src/eval_policy.py`
  - loads `data/sims/wti_real_test.npz`
  - evaluates one deterministic held-out episode
  - compares RL policy to always-flat, always-long, and always-short
  - saves `outputs/eval/heldout_policy_comparison.png`

- `src/metrics_eval.py`
  - loads `data/sims/wti_real_test.npz`
  - evaluates all held-out test episodes
  - uses deterministic episode starts during evaluation
  - prints:
    - mean total pnl
    - std total pnl
    - mean Sharpe
    - std Sharpe
    - mean max drawdown
    - win rate across episodes
    - action counts
  - compares:
    - RL Policy
    - Always Flat
    - Always Long
    - Always Short
  - can save machine-readable output to `outputs/eval/metrics_eval.json`

What it does not currently do:

- compare against always-short
- compare against random or buy-and-hold benchmarks
- aggregate baseline metrics across the held-out distribution
- report trade-level statistics
- test generalization on upstream synthetic paths

## Known Bugs And Inconsistencies
Current issues that should be treated as real code-level facts:

1. Action mapping inconsistency across docs
   - repository prose and comments disagree on whether `0` is short or flat
   - environment code is the authoritative mapping today:
     - `0 flat`
     - `1 long`
     - `2 short`

2. Return-vs-price contract is ambiguous in documentation but not in code
   - docs say paths may be returns or prices
   - implemented environment assumes returns only

3. Evaluation is still incomplete even after the held-out metrics baseline
   - `metrics_eval.py` now aggregates RL and fixed-baseline metrics across held-out episodes
   - `eval_policy.py` still plots only one illustrative episode
   - evaluation is deterministic but still not sliced by regime, trend, or scenario type

4. Generalization testing is still narrow
   - held-out testing exists for real-data episodes
   - there is still no evaluation on upstream synthetic paths

5. Unused or placeholder modules exist
   - `src/io_real.py` is empty
   - `src/train_a2c.py` is empty
   - `src/normalizer.py` is currently unused

## Data Contract For Upstream Generator
The downstream loader contract in this repo is currently minimal:

- file format: `.npz`
- required key: `paths`
- required shape: `(N, T)`
- numeric dtype compatible with `np.float32`

The practical contract required by the environment is stricter:

- each path must be a 1D sequence of per-step returns
- log returns are the intended format
- price levels are not supported correctly at present
- path length must satisfy:
  - `T >= lookback + episode_len + 1`
  - with defaults, `T >= 283`
- current local datasets use `T = 600`

Recommended upstream output for future integration:
- `data/sims/wti_simulated_paths.npz`
- key `paths`
- shape `(N, T)`
- values are log returns

Optional future metadata:
- generator config
- seed
- regime labels
- split labels
- notes declaring the values are returns, not prices

## Relationship To `wti-signature-cvae`
Intended future architecture:

1. `wti-signature-cvae` learns a generative model on real WTI data
2. it exports synthetic return paths
3. this repo loads those synthetic paths through `src/io_sims.py`
4. `TradingEnv` runs over those synthetic return paths
5. DQN is trained and later evaluated on held-out synthetic and real benchmarks

Current reality:
- the upstream repo exists as a design scaffold
- its implementation files are currently empty
- this repo is not yet wired to consume real synthetic outputs from upstream
- training today uses real historical episodes, not generator-produced simulations

## Audit Expectations
When analyzing or editing this repo:

- identify the exact path format being used in code, not just in docs
- treat `src/env_trading.py` as the source of truth for action semantics
- check the position one-hot block carefully when debugging policy behavior
- distinguish reward from PnL in any evaluation discussion
- note whether results come from:
  - train episodes
  - full real episodes
  - held-out test episodes
  - future synthetic episodes

## Editing Guidance
- Prefer small, high-confidence changes
- Keep data interfaces explicit
- Do not silently accept price paths as if they were returns
- Fix evaluation before adding algorithmic complexity
- Prioritize reproducibility and clear metrics
- Preserve a clean downstream contract for future `wti-signature-cvae` integration

## Version Control
After ANY file modification:

- `git add -A`
- `git commit` with a descriptive message
- `git push origin main`
- show commit hash
- confirm push success

Commit message format:

`<scope>: <short summary>`

Details:
- what changed
- why it changed
- impact on training/eval/plotting

Scopes to use:
- `eval`
- `train`
- `env`
- `plot`
- `feat`
- `fix`
- `data`
- `agents`

Example messages:
- `eval: deterministic held-out evaluation`
- `fix: position encoding bug in features`
- `plot: add RL vs baseline comparison graph`
