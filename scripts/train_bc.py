"""Behavioral cloning: collect expert data from scripted agent, then train.

Usage::

    uv run python scripts/train_bc.py --episodes 10000 --epochs 30
    uv run python scripts/train_bc.py --episodes 10000 --epochs 30 --resume runs/v17c/checkpoint_9830400.pt
"""

from __future__ import annotations

import argparse
import functools
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from jackdaw.env.game_interface import DirectAdapter
from jackdaw.rl.env_wrapper import FactoredBalatroEnv
from jackdaw.rl.network import (
    ENTITY_MAX_COUNTS,
    NEEDS_CARDS,
    NEEDS_ENTITY,
    NUM_ACTION_TYPES,
    FactoredPolicy,
)
from jackdaw.rl.scripted_agent import ScriptedBalatroAgent
from jackdaw.rl.trainer import HAND_CARD_MAX, _masks_to_numpy


def collect_data(
    n_episodes: int,
    max_steps: int = 10_000,
) -> list[dict]:
    """Collect transitions from the scripted agent."""
    env = FactoredBalatroEnv(
        adapter_factory=DirectAdapter,
        reward_shaping=True,
        max_steps=max_steps,
    )
    agent = ScriptedBalatroAgent()
    transitions: list[dict] = []
    antes: list[int] = []

    t0 = time.time()
    for ep in range(n_episodes):
        obs, mask, info = env.reset()
        agent.reset()
        done = False
        ep_steps = 0

        while not done:
            action = agent.act(info.get("raw_state", {}), mask, info)

            # Record transition
            type_mask, card_mask, entity_masks, min_cs, max_cs = _masks_to_numpy(mask)
            transitions.append({
                "obs": obs,
                "type_mask": type_mask,
                "card_mask": card_mask,
                "entity_masks": entity_masks,
                "min_card_select": min_cs,
                "max_card_select": max_cs,
                "action_type": action.action_type,
                "entity_target": action.entity_target if action.entity_target is not None else -1,
                "card_target": _encode_card_target(action),
            })

            obs, reward, terminated, truncated, mask, info = env.step(action)
            done = terminated or truncated
            ep_steps += 1

        ante = info.get("balatro/ante_reached", 1)
        antes.append(ante)

        if (ep + 1) % 500 == 0:
            dt = time.time() - t0
            mean_ante = np.mean(antes[-500:])
            print(
                f"  Episode {ep + 1}/{n_episodes} | "
                f"ante {mean_ante:.2f} | "
                f"transitions {len(transitions):,} | "
                f"{dt:.0f}s"
            )

    dt = time.time() - t0
    print(
        f"\nCollected {len(transitions):,} transitions from {n_episodes} episodes "
        f"in {dt:.0f}s"
    )
    print(f"Mean ante: {np.mean(antes):.2f} | Max ante: {np.max(antes)}")
    return transitions


def _encode_card_target(action) -> np.ndarray:
    """Encode card_target as bool array."""
    ct = np.zeros(HAND_CARD_MAX, dtype=bool)
    if action.card_target is not None:
        for idx in action.card_target:
            if idx < HAND_CARD_MAX:
                ct[idx] = True
    return ct


def build_tensors(
    transitions: list[dict], device: torch.device
) -> dict[str, torch.Tensor | dict]:
    """Convert transitions to batched tensors."""
    N = len(transitions)

    # Observations
    obs_keys = list(transitions[0]["obs"].keys())
    obs = {
        k: torch.from_numpy(np.stack([t["obs"][k] for t in transitions]))
        .float()
        .to(device)
        for k in obs_keys
    }

    # Actions
    action_type = torch.tensor(
        [t["action_type"] for t in transitions], dtype=torch.long, device=device
    )
    entity_target = torch.tensor(
        [t["entity_target"] for t in transitions], dtype=torch.long, device=device
    )
    card_target = (
        torch.from_numpy(np.stack([t["card_target"] for t in transitions]))
        .bool()
        .to(device)
    )

    # Masks
    type_mask = (
        torch.from_numpy(np.stack([t["type_mask"] for t in transitions]))
        .bool()
        .to(device)
    )
    card_mask = (
        torch.from_numpy(np.stack([t["card_mask"] for t in transitions]))
        .bool()
        .to(device)
    )
    min_card_select = torch.tensor(
        [t["min_card_select"] for t in transitions], dtype=torch.long, device=device
    )
    max_card_select = torch.tensor(
        [t["max_card_select"] for t in transitions], dtype=torch.long, device=device
    )

    # Entity masks
    all_keys: set[int] = set()
    for t in transitions:
        all_keys.update(t["entity_masks"].keys())
    entity_masks_t: dict[int, torch.Tensor] = {}
    for atype in all_keys:
        arrs = []
        ref_shape = None
        for t in transitions:
            if atype in t["entity_masks"]:
                arrs.append(t["entity_masks"][atype])
                if ref_shape is None:
                    ref_shape = t["entity_masks"][atype].shape
            else:
                arrs.append(None)
        assert ref_shape is not None
        filled = [a if a is not None else np.zeros(ref_shape, dtype=bool) for a in arrs]
        entity_masks_t[atype] = torch.from_numpy(np.stack(filled)).bool().to(device)

    masks = {
        "type_mask": type_mask,
        "card_mask": card_mask,
        "entity_masks": entity_masks_t,
        "min_card_select": min_card_select,
        "max_card_select": max_card_select,
    }

    return {
        "obs": obs,
        "action_type": action_type,
        "entity_target": entity_target,
        "card_target": card_target,
        "masks": masks,
    }


def train_bc(
    network: FactoredPolicy,
    data: dict,
    n_epochs: int = 30,
    batch_size: int = 2048,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Train network via behavioral cloning (maximize log-prob of expert actions)."""
    network.to(device)
    network.train()
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, eps=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr / 10
    )

    N = data["action_type"].shape[0]
    print(f"\nBC training: {N:,} samples, {n_epochs} epochs, batch_size={batch_size}")

    for epoch in range(1, n_epochs + 1):
        indices = np.arange(N)
        np.random.shuffle(indices)

        total_loss = 0.0
        total_type_acc = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            idx = torch.from_numpy(indices[start:end]).long().to(device)

            obs_b = {k: v[idx] for k, v in data["obs"].items()}
            at_b = data["action_type"][idx]
            et_b = data["entity_target"][idx]
            ct_b = data["card_target"][idx]
            masks_b = {
                "type_mask": data["masks"]["type_mask"][idx],
                "card_mask": data["masks"]["card_mask"][idx],
                "entity_masks": {
                    k: v[idx] for k, v in data["masks"]["entity_masks"].items()
                },
                "min_card_select": data["masks"]["min_card_select"][idx],
                "max_card_select": data["masks"]["max_card_select"][idx],
            }

            # Forward: evaluate log-prob of expert actions
            log_prob, value, type_entropy, card_entropy = network.evaluate(
                obs_b, masks_b, at_b, et_b, ct_b
            )

            # Skip NaN batches
            if torch.isnan(log_prob).any():
                continue

            # BC loss: maximize log-prob (= minimize negative log-prob)
            loss = -log_prob.mean()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()

            # Accuracy: check if highest-prob action type matches expert
            with torch.no_grad():
                state, _ = network._encode(obs_b)
                type_logits = network.action_type_head(state)
                type_logits = type_logits + network._compute_action_biases(obs_b)
                type_logits = type_logits.masked_fill(~masks_b["type_mask"], -1e4)
                pred_type = type_logits.argmax(dim=-1)
                type_acc = (pred_type == at_b).float().mean().item()

            total_loss += loss.item()
            total_type_acc += type_acc
            n_batches += 1

        scheduler.step()
        n_batches = max(n_batches, 1)
        avg_loss = total_loss / n_batches
        avg_acc = total_type_acc / n_batches
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"  Epoch {epoch:>3}/{n_epochs} | "
            f"loss {avg_loss:.4f} | "
            f"type_acc {avg_acc:.1%} | "
            f"lr {current_lr:.2e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavioral cloning for Balatro")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default="runs/bc_init.pt")
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Initialize from existing checkpoint before BC training",
    )
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

    # Phase 1: Collect expert data
    print("\n=== Phase 1: Collecting expert data ===")
    transitions = collect_data(args.episodes)

    # Phase 2: Build tensors
    print("\n=== Phase 2: Building tensors ===")
    data = build_tensors(transitions, device)
    del transitions  # free memory

    # Phase 3: Train BC
    print("\n=== Phase 3: Behavioral cloning ===")
    network = FactoredPolicy()

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        network.load_state_dict(ckpt["network"])
        print(f"Initialized from: {args.resume}")

    train_bc(
        network, data,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )

    # Save
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), save_path)
    print(f"\nBC weights saved to: {save_path}")
    print(f"Resume PPO with: python scripts/train_factored.py --resume {save_path}")


if __name__ == "__main__":
    main()
