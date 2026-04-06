# WTI RL Policy — Session Handoff Context

## What Project 2 Currently Does
This repository is Project 2 in a two-project system.

Its current implemented pipeline is:

1. download real WTI futures prices
2. convert prices to log returns
3. slice returns into overlapping fixed-length episodes
4. train a single-asset DQN trading policy on those episodes
5. evaluate the policy on deterministic held-out episodes against simple fixed baselines

The repo does not yet consume real synthetic paths from `../wti-signature-cvae`.

## What Was Fixed This Session
- corrected position encoding so the state matches env positions `{-1, 0, +1}`
- made held-out evaluation deterministic by disabling randomized episode starts in evaluation
- upgraded `metrics_eval.py` to compare:
  - RL Policy
  - Always Flat
  - Always Long
  - Always Short
- added compact comparison output and optional JSON export
- updated `eval_policy.py` to generate held-out comparison plots at:
  - `outputs/eval/heldout_policy_comparison.png`
- documented repository workflow and current project state in `AGENTS.md`

## Current Best Metrics And What They Mean
Best credible Project 2 takeaway from this session:
- under the stronger deterministic held-out evaluation setup, RL can beat trivial always-long and always-short baselines

Important note:
- the later 504-step experiments did not improve quality
- the strongest conclusion is that Project 2 now has a credible evaluation baseline, not that the latest checkpoint is the best policy

Representative stronger-baseline result before the failed feature expansion:
- deterministic held-out comparison across 136 episodes
- RL mean total pnl was positive
- RL mean Sharpe was positive
- RL beat always-long and always-short on mean total pnl

## Visualizations That Now Exist
- held-out single-episode policy comparison plot:
  - `outputs/eval/heldout_policy_comparison.png`
- machine-readable aggregated evaluation output:
  - `outputs/eval/metrics_eval.json`

## What Failed
Failed experiment:
- adding raw multi-horizon regime-aware state features in `src/features.py`
- windows used: `20`, `60`, `120`
- this increased state size from `38` to `50`

Observed result under the 504-step setup:
- severe held-out regression
- mean total pnl became strongly negative
- Sharpe became strongly negative
- max drawdown worsened materially
- win rate fell to zero

Likely interpretation:
- Project 2 is now bottlenecked more by data/regime quality than by more ad hoc manual state expansion
- richer state alone did not solve the long-horizon instability problem

## Why The Next Likely Focus Is Project 1
Project 2 now has:
- a working real-data training loop
- deterministic held-out evaluation
- baseline comparisons
- plotting

What it lacks is a better distribution of training scenarios.

The next likely source of progress is `wti-signature-cvae`:
- build or inspect the upstream generator
- create regime-aware synthetic paths
- define a clean export contract
- feed those paths back into Project 2

## Exact Suggested Next-Session Priority
1. inspect and implement `wti-signature-cvae`
2. define a clean synthetic-path export contract
3. reconnect Project 1 output into Project 2 input
