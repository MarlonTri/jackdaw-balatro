"""Iterative self-improvement via filtered behavioral cloning.

Collects episodes from the current agent (and optionally a scripted agent),
filters for the best episodes (highest ante), and trains BC on those.
Repeat to iteratively improve.

Usage::

    # Round 1: from scripted agent baseline
    uv run python scripts/self_improve.py --source scripted --episodes 20000 --ante-threshold 3

    # Round 2: from previous BC weights + RL checkpoint
    uv run python scripts/self_improve.py --source both --checkpoint runs/v19_long/checkpoint_*.pt --episodes 20000 --ante-threshold 3

    # Round 3+: raise threshold as agent improves
    uv run python scripts/self_improve.py --source network --checkpoint runs/si_round2.pt --episodes 20000 --ante-threshold 4
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from jackdaw.env.game_interface import DirectAdapter
from jackdaw.rl.env_wrapper import FactoredBalatroEnv
from jackdaw.rl.network import (
    ENTITY_MAX_COUNTS,
    NEEDS_CARDS,
    NEEDS_ENTITY,
    FactoredPolicy,
)
from jackdaw.rl.scripted_agent import ScriptedBalatroAgent
from jackdaw.rl.trainer import HAND_CARD_MAX, _masks_to_numpy

# Reuse BC training infrastructure
from train_bc import _encode_card_target, build_tensors, train_bc


# ---------------------------------------------------------------------------
# Data collection from neural network
# ---------------------------------------------------------------------------


def _obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Single observation → batch of 1."""
    return {k: torch.from_numpy(v).float().unsqueeze(0).to(device) for k, v in obs.items()}


def _mask_to_tensor(mask_np, device: torch.device) -> dict:
    """Single mask tuple → batch of 1."""
    type_mask, card_mask, entity_masks, min_cs, max_cs = mask_np
    result = {
        "type_mask": torch.from_numpy(type_mask).bool().unsqueeze(0).to(device),
        "card_mask": torch.from_numpy(card_mask).bool().unsqueeze(0).to(device),
        "entity_masks": {},
        "min_card_select": torch.tensor([min_cs], dtype=torch.long, device=device),
        "max_card_select": torch.tensor([max_cs], dtype=torch.long, device=device),
    }
    for atype, emask in entity_masks.items():
        result["entity_masks"][atype] = torch.from_numpy(emask).bool().unsqueeze(0).to(device)
    return result


def collect_network_episodes(
    network: FactoredPolicy,
    n_episodes: int,
    device: torch.device,
    ante_threshold: int = 3,
) -> tuple[list[dict], list[int]]:
    """Collect episodes from neural network, return filtered transitions."""
    env = FactoredBalatroEnv(adapter_factory=DirectAdapter, reward_shaping=True)
    network.eval()

    all_transitions: list[dict] = []
    all_antes: list[int] = []
    kept_episodes = 0
    t0 = time.time()

    for ep in range(n_episodes):
        obs, mask, info = env.reset()
        episode_buf: list[dict] = []
        done = False

        while not done:
            mask_np = _masks_to_numpy(mask)
            obs_t = _obs_to_tensor(obs, device)
            masks_t = _mask_to_tensor(mask_np, device)

            with torch.no_grad():
                out = network(obs_t, masks_t)

            at = int(out["action_type"][0].item())
            et = int(out["entity_target"][0].item())
            ct = out["card_target"][0].cpu().numpy()

            # Build FactoredAction
            from jackdaw.env.game_spec import FactoredAction

            et_val = et if at in NEEDS_ENTITY and et >= 0 else None
            ct_val = None
            if at in NEEDS_CARDS:
                selected = np.nonzero(ct)[0]
                if len(selected) > 0:
                    ct_val = tuple(int(j) for j in selected)

            action = FactoredAction(action_type=at, card_target=ct_val, entity_target=et_val)

            # Store transition
            episode_buf.append({
                "obs": obs,
                "type_mask": mask_np[0],
                "card_mask": mask_np[1],
                "entity_masks": mask_np[2],
                "min_card_select": mask_np[3],
                "max_card_select": mask_np[4],
                "action_type": at,
                "entity_target": et if et >= 0 else -1,
                "card_target": ct.astype(bool),
            })

            obs, reward, terminated, truncated, mask, info = env.step(action)
            done = terminated or truncated

        ante = info.get("balatro/ante_reached", 1)
        all_antes.append(ante)

        # Keep episode if above threshold
        if ante >= ante_threshold:
            all_transitions.extend(episode_buf)
            kept_episodes += 1

        if (ep + 1) % 1000 == 0:
            dt = time.time() - t0
            mean_ante = np.mean(all_antes[-1000:])
            keep_rate = kept_episodes / (ep + 1)
            print(
                f"  Episode {ep+1}/{n_episodes} | "
                f"ante {mean_ante:.2f} | "
                f"kept {kept_episodes} ({keep_rate:.1%}) | "
                f"transitions {len(all_transitions):,} | "
                f"{dt:.0f}s"
            )

    dt = time.time() - t0
    print(
        f"\nNetwork: {n_episodes} episodes in {dt:.0f}s | "
        f"mean ante {np.mean(all_antes):.2f} | max {np.max(all_antes)} | "
        f"kept {kept_episodes} episodes ({kept_episodes/n_episodes:.1%}), "
        f"{len(all_transitions):,} transitions"
    )
    return all_transitions, all_antes


def collect_scripted_episodes(
    n_episodes: int,
    ante_threshold: int = 3,
) -> tuple[list[dict], list[int]]:
    """Collect episodes from scripted agent, return filtered transitions."""
    env = FactoredBalatroEnv(adapter_factory=DirectAdapter, reward_shaping=True)
    agent = ScriptedBalatroAgent()

    all_transitions: list[dict] = []
    all_antes: list[int] = []
    kept_episodes = 0
    t0 = time.time()

    for ep in range(n_episodes):
        obs, mask, info = env.reset()
        agent.reset()
        episode_buf: list[dict] = []
        done = False

        while not done:
            action = agent.act(info.get("raw_state", {}), mask, info)
            mask_np = _masks_to_numpy(mask)

            episode_buf.append({
                "obs": obs,
                "type_mask": mask_np[0],
                "card_mask": mask_np[1],
                "entity_masks": mask_np[2],
                "min_card_select": mask_np[3],
                "max_card_select": mask_np[4],
                "action_type": action.action_type,
                "entity_target": action.entity_target if action.entity_target is not None else -1,
                "card_target": _encode_card_target(action),
            })

            obs, reward, terminated, truncated, mask, info = env.step(action)
            done = terminated or truncated

        ante = info.get("balatro/ante_reached", 1)
        all_antes.append(ante)

        if ante >= ante_threshold:
            all_transitions.extend(episode_buf)
            kept_episodes += 1

        if (ep + 1) % 1000 == 0:
            dt = time.time() - t0
            mean_ante = np.mean(all_antes[-1000:])
            keep_rate = kept_episodes / (ep + 1)
            print(
                f"  Episode {ep+1}/{n_episodes} | "
                f"ante {mean_ante:.2f} | "
                f"kept {kept_episodes} ({keep_rate:.1%}) | "
                f"transitions {len(all_transitions):,} | "
                f"{dt:.0f}s"
            )

    dt = time.time() - t0
    print(
        f"\nScripted: {n_episodes} episodes in {dt:.0f}s | "
        f"mean ante {np.mean(all_antes):.2f} | max {np.max(all_antes)} | "
        f"kept {kept_episodes} episodes ({kept_episodes/n_episodes:.1%}), "
        f"{len(all_transitions):,} transitions"
    )
    return all_transitions, all_antes


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-improvement via filtered BC")
    parser.add_argument(
        "--source", choices=["scripted", "network", "both"], default="both",
        help="Data source: scripted agent, neural network, or both",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Network checkpoint path")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--ante-threshold", type=int, default=3, help="Min ante to keep episode")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Source: {args.source} | Episodes: {args.episodes} | Ante threshold: {args.ante_threshold}")

    all_transitions: list[dict] = []

    # Collect from scripted agent
    if args.source in ("scripted", "both"):
        n = args.episodes if args.source == "scripted" else args.episodes // 2
        print(f"\n=== Collecting {n} scripted episodes ===")
        scripted_trans, scripted_antes = collect_scripted_episodes(n, args.ante_threshold)
        all_transitions.extend(scripted_trans)

    # Collect from neural network
    if args.source in ("network", "both"):
        if args.checkpoint is None:
            parser.error("--checkpoint required for network/both source")
        n = args.episodes if args.source == "network" else args.episodes // 2

        print(f"\n=== Collecting {n} network episodes ===")
        network = FactoredPolicy()
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "network" in ckpt:
            network.load_state_dict(ckpt["network"])
        else:
            network.load_state_dict(ckpt)
        network.to(device)
        print(f"Loaded: {args.checkpoint}")

        net_trans, net_antes = collect_network_episodes(network, n, device, args.ante_threshold)
        all_transitions.extend(net_trans)
        del network
        torch.cuda.empty_cache()

    if not all_transitions:
        print(f"\nNo episodes passed ante threshold {args.ante_threshold}! Lower the threshold.")
        return

    print(f"\n=== Total filtered data: {len(all_transitions):,} transitions ===")

    # Build tensors and train
    print("\n=== Building tensors ===")
    data = build_tensors(all_transitions, device)
    del all_transitions

    print("\n=== Training BC on filtered episodes ===")
    network = FactoredPolicy()

    # Optionally initialize from checkpoint
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "network" in ckpt:
            network.load_state_dict(ckpt["network"])
        else:
            network.load_state_dict(ckpt)
        print(f"Initialized from: {args.checkpoint}")

    train_bc(
        network, data,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # Save
    save_path = args.save_path or "runs/si_round.pt"
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), save_path)
    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    main()
