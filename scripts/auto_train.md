# Auto-Training Monitor Prompt

Copy this entire prompt into a new Claude Code chat in this project directory.

---

## Task

You are an autonomous RL training engineer. Your job is to train a Balatro poker roguelike agent to consistently beat the game (ante 8 = win). You have full authority to modify code, architecture, reward shaping, observations, and hyperparameters — whatever it takes. You will:

1. Start a training run
2. Monitor progress periodically
3. Diagnose problems (entropy collapse, plateau, instability, wasted actions)
4. Make code/config changes and restart training
5. Iterate until the agent averages ante 8 (game win)

This is a multi-day task. Be patient, methodical, and take big swings when plateaus demand it.

## Project overview

- **Project**: `g:\01_Active\Code\personal\jackdaw-balatro`
- **Game**: Balatro — a poker roguelike where you build a deck with jokers and try to beat 8 antes
- **Algorithm**: PPO with factored action space (type → entity/card selection)
- **Training script**: `python scripts/train_factored.py`
- **Evaluation**: `python scripts/play.py <checkpoint> --n-games 20 --verbose`
- **Live game**: `python scripts/play.py <checkpoint> --live` (requires balatrobot running)

## Current state

- **Best result**: ante ~2.5 average after 47M steps with old action space
- **Recent change**: removed Sort/Swap actions (were 95% of all actions, pure waste)
- **Current issue**: entropy collapses rapidly with the reduced action space — needs tuning
- **Network**: 916K params (ENTITY_EMBED=96, STATE_EMBED=384)
- **Observations**: semantic joker features (29-dim per joker), 20-dim shop items, 235-dim global

## What you CAN modify (anything in the RL pipeline)

- `scripts/train_factored.py` — hyperparameters, training loop config
- `jackdaw/rl/trainer.py` — PPO implementation, loss functions, KL thresholds, entropy targeting
- `jackdaw/rl/network.py` — network architecture, dimensions, attention layers
- `jackdaw/rl/env_wrapper.py` — reward shaping, episode truncation, observation remapping
- `jackdaw/rl/rollout.py` — rollout buffer, GAE computation
- `jackdaw/env/observation.py` — observation encoding (what the agent sees)
- `jackdaw/env/action_space.py` — action masking (which actions are legal)

## What you MUST NOT modify (the game engine and live game interface)

- `jackdaw/engine/` — the game engine (must match real Balatro)
- `jackdaw/bridge/` — live game connection
- `jackdaw/env/game_interface.py` — adapter interface
- `jackdaw/env/balatro_env.py` — base environment (use env_wrapper instead)
- `jackdaw/env/game_spec.py` — spec definitions

## Current hyperparameters

```
--lr 1e-4                    # learning rate (cosine decay)
--ent-coef 1.0               # entropy targeting coefficient
--entropy-target 1.2          # target entropy level
--clip-range 0.15             # PPO clip range
--n-steps 4096                # rollout length
--n-epochs 10                 # PPO epochs per update
--batch-size 1024             # minibatch size
--n-envs 4                    # parallel environments
--total-timesteps 50000000    # total training steps
```

## Key metrics and healthy ranges

| Metric | Healthy | Problem |
|--------|---------|---------|
| `ent` (entropy) | 0.8 - 1.5 | < 0.5 collapsed, > 2.5 not learning |
| `ante` | Increasing over time | Stuck 1000+ updates = plateau |
| `kl` | -0.1 to 0.15 | > 0.3 frequent = updates too large |
| `ep_len` | 15-60 | > 100 stalling, < 8 dying immediately |
| `ep_rew` | Increasing | Negative and stuck = not learning |
| `epochs` | 3-10 | Always 1 = too aggressive, always 10 = too conservative |
| `ploss` | -0.05 to 0.1 | > 1.0 = exploding |
| `vloss` | 0.01 - 0.15 | > 0.5 = value function broken |

## Monitoring procedure

1. **Start training** in background:
```bash
python scripts/train_factored.py --n-envs 4 --total-timesteps 50000000 --log-dir "runs/balatro_factoredv2" > train.log 2>&1 &
```

2. **Check every 15 minutes** by reading the last 50 lines:
```bash
tail -50 train.log
```

3. **If a problem is detected**, stop training and fix it:
```bash
kill $(pgrep -f train_factored)
# ... make changes ...
# Resume from latest checkpoint:
python scripts/train_factored.py --resume runs/balatro_factoredv2/checkpoint_XXXXX.pt --n-envs 4 --total-timesteps 50000000 --log-dir "runs/balatro_factoredv2" > train.log 2>&1 &
```

4. **When ante plateaus** for 1000+ updates with healthy entropy, the bottleneck is likely architectural — consider bigger changes (reward shaping, observation features, network design).

5. **Every few hours**, run a quick evaluation:
```bash
python scripts/play.py runs/balatro_factoredv2/checkpoint_LATEST.pt --n-games 10 --verbose
```

## Diagnosis and fix playbook

### Entropy collapse (ent < 0.5 within first 200 updates)
The policy locks onto one action. The reduced action space (15 types, ~3-6 available per state) makes this likely.
- Increase `--ent-coef` (try 2.0, 5.0, 10.0)
- Lower `--lr` (try 5e-5, 3e-5)
- Lower `--clip-range` (try 0.1, 0.08)
- Raise `--entropy-target` (try 1.5)
- Consider switching entropy targeting from quadratic to linear: `ent_coef * abs(entropy - target)` in trainer.py

### KL divergence spikes (kl > 0.3 frequently)
Policy updates are too large, destabilizing learning.
- Lower `--lr`
- In `trainer.py`, lower per-minibatch KL threshold (line ~420, currently 0.1)
- Lower `--clip-range`

### Ante plateau (stuck at same level for 1000+ updates)
The model has converged at current capacity/information.
- **If entropy is healthy**: the bottleneck is observation quality or network capacity
  - Add new observation features (e.g., remaining deck composition, joker synergy scores)
  - Increase network size (ENTITY_EMBED, STATE_EMBED, attention layers)
  - Improve reward shaping (bigger bonuses for ante progression, penalize bad purchases)
- **If LR is very low**: cosine schedule decayed too far — restart with fresh schedule
- **If entropy is low**: see entropy collapse fixes above

### Episode length explosion (ep_len > 100)
Agent stalling in shop or making no progress.
- Increase shop step cost in `env_wrapper.py`
- Lower `max_episode_steps` (currently 200)
- Add negative reward for being in shop too long

### Agent not buying jokers / not using shop
Missing strategic learning. Consider:
- Add reward for buying jokers that synergize with hand types
- Add observation features about shop item value
- Increase reward for ante progression to incentivize long-term planning

## Architecture reference

### Network (network.py)
- Entity encoders: MLP per entity type → ENTITY_EMBED dim
- Cross-entity transformer: N_ATTN_LAYERS layers, N_HEADS heads
- Attention pooling per entity type → concatenate with global features
- State combiner → STATE_EMBED dim
- Heads: action type (linear), entity pointer (dot-product), card selection (MLP scorer), value (MLP)

### Observation (observation.py)
- Global context: 235 features (phase, economy, hand levels, blinds, strategic features)
- Jokers: 29 features each (17 semantic from centers.json + 12 runtime)
- Hand cards: 15 features each (rank, suit, enhancements, scoring status)
- Shop items: 20 features each (8 base + joker category features)
- Consumables: 7 features each
- Pack cards: 15 features each (same as hand cards)

### Reward (env_wrapper.py)
- Step cost: -0.001 (shop: -0.002)
- Blind beaten: +0.15 * max(ante/4, 0.5)
- Ante increased: +0.2 * ante
- Efficient clear: +0.01 * hands_remaining
- Score progress: +0.02 * min(progress, 1.0) + overclear bonus
- Economy: +0.001 * interest brackets
- Terminal: +0.5 win / -0.2 loss

### Trainer (trainer.py)
- PPO with clipped value loss
- Entropy targeting: `ent_coef * (entropy - target)^2`
- Per-minibatch KL check at 0.1 (breaks mid-epoch)
- Per-epoch KL check at 0.1 (breaks between epochs)
- CosineAnnealingLR (lr → lr/10 over total updates)
- Gradient clipping at 0.5
- Log-ratio clamp at (-5, 5)

## Success criteria

Training is complete when:
- **Ante average >= 8** in evaluation (20 games)
- Win rate > 50%
- Training is stable (no collapse)
- Report the final hyperparameters, any code changes made, and evaluation results

## Important notes

- **Resume from checkpoints** after hyperparameter changes — don't waste trained weights
- **Train from scratch** only if you change network dimensions or observation dims
- Checkpoints save every 50 updates in the log-dir
- The cosine LR schedule is tied to `--total-timesteps` — keep consistent when resuming
- Balatro has 8 antes. Ante 1-3 is early game, 4-5 mid game, 6-8 late game. Late game requires joker synergies and strategic deck building
- The entropy targeting formula: `loss += ent_coef * (entropy - target)^2` — quadratic penalty in both directions
- Read existing code before making changes — understand patterns first
- Commit meaningful changes with descriptive messages so progress is tracked
