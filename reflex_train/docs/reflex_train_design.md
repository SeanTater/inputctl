# Reflex Train Design Notes

This document describes how reflex_train moves from behavior cloning to offline RL-style training.

## Goals

- Learn a reflex policy that can execute key presses from vision + intent.
- Use offline signals (win/death events) to compute returns.
- Let a value function bias the policy toward higher-return behavior.
- Keep the model compact by sharing the vision backbone.

## Data flow (offline)

1. Record sessions into `recording.mp4`, `frames.jsonl`, `inputs.jsonl`.
2. `precompute_intents.py` writes:
   - `intent.jsonl` (weak intents)
   - `events.jsonl` (death/win markers)
   - `episodes.jsonl` + `returns.jsonl` (terminal rewards, per-frame returns)
3. Training streams frames per video so decoding stays sequential.

## Model

`ReflexNet` is a shared backbone with multiple heads:

- Policy (keys): multi-label logits over tracked keys.
- Policy (mouse): regression head (currently supervised by dummy labels).
- Intent: predicts intent from visual features.
- Value: predicts expected return for state/goal.
- Inverse dynamics (aux): predicts the current action from a (state, next_state) pair.

All heads share the same ResNet-18 backbone. The goal vector is fused only for policy/value heads.

## Offline RL flavor (IQL-style)

This code uses an IQL-style recipe, which works well for offline datasets.

- **Value loss**: expectile regression to fit a pessimistic value estimate.
- **Policy update**: advantage-weighted regression (AWR) with `exp((return - value) / temperature)`.
- **Why it helps**: pushes the policy toward high-return slices of the dataset without requiring online rollouts.

This is still offline: it does not explore or generate new trajectories. It improves imitation by
emphasizing better outcomes.

## Inverse dynamics (auxiliary)

We add an inverse dynamics head to make visual features action-aware:

- Input: features at time t and features at time t+1.
- Target: key vector at time t (or action-horizon target if configured).
- Benefit: encourages the backbone to encode changes caused by actions.

The dataset provides `next_pixels` and `current_keys` for this loss.

## How value influences actions

The value head does not directly pick actions at inference time. It helps during training:

- AWR/IQL uses value to reweight behavior cloning loss.
- Higher-return transitions pull the policy closer.

To use value directly for action selection, you would need a Q-head or rollout model
that can compare alternative actions. That is not implemented yet.

## Known gaps

- No explicit Q(s,a) head; value is state-only.
- Mouse head is placeholder (supervision is neutral).
- Inverse dynamics is an auxiliary loss only.
- Returns depend on win/death events; sparse reward can be noisy.

## Next steps (if needed)

- Add a Q-head or advantage head to compare actions.
- Add denser rewards (progress/score) to improve value signal.
- Add confidence filters or curriculum runs to reduce label noise.
