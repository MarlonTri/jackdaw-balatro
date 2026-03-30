"""Run a trained factored policy on a single Balatro game.

Usage::

    # Against the internal engine (fast, no Balatro needed)
    python scripts/play.py runs/balatro_factored/checkpoint_1000000.pt
    python scripts/play.py checkpoint.pt --n-games 10 --verbose

    # Against live Balatro (requires balatrobot running)
    python scripts/play.py checkpoint.pt --live
    python scripts/play.py checkpoint.pt --live --host 127.0.0.1 --port 12346

    # Live + engine validation (runs both, compares states)
    python scripts/play.py checkpoint.pt --live --validate
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import torch

from jackdaw.env.action_space import ActionType, factored_to_engine_action
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.game_spec import FactoredAction
from jackdaw.rl.env_wrapper import FactoredBalatroEnv
from jackdaw.rl.network import (
    ENTITY_MAX_COUNTS,
    NEEDS_CARDS,
    NEEDS_ENTITY,
    FactoredPolicy,
)

_SPEC = balatro_game_spec()
HAND_CARD_MAX = _SPEC.entity_types[0].max_count


def _pad_mask(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) >= target_len:
        return arr[:target_len].astype(bool)
    padded = np.zeros(target_len, dtype=bool)
    padded[: len(arr)] = arr
    return padded


def _obs_to_device(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).float().unsqueeze(0).to(device) for k, v in obs.items()}


def _masks_to_device(mask, device: torch.device) -> dict:
    type_mask = mask.type_mask.astype(bool)
    card_mask = _pad_mask(mask.card_mask, HAND_CARD_MAX)

    entity_masks = {}
    for atype, emask in mask.entity_masks.items():
        etype_idx = _SPEC.entity_type_for_action(atype)
        if etype_idx >= 0:
            entity_masks[atype] = (
                torch.from_numpy(_pad_mask(emask, ENTITY_MAX_COUNTS[etype_idx]))
                .bool()
                .unsqueeze(0)
                .to(device)
            )

    return {
        "type_mask": torch.from_numpy(type_mask).bool().unsqueeze(0).to(device),
        "card_mask": torch.from_numpy(card_mask).bool().unsqueeze(0).to(device),
        "entity_masks": entity_masks,
        "min_card_select": torch.tensor([mask.min_card_select], dtype=torch.long, device=device),
        "max_card_select": torch.tensor([mask.max_card_select], dtype=torch.long, device=device),
    }


# ---------------------------------------------------------------------------
# State comparison (engine vs live game)
# ---------------------------------------------------------------------------


def _hand_keys(gs: dict[str, Any]) -> set[str]:
    return {c.card_key for c in gs.get("hand", []) if hasattr(c, "card_key") and c.card_key}


def _joker_keys(gs: dict[str, Any]) -> list[str]:
    return [c.center_key for c in gs.get("jokers", []) if hasattr(c, "center_key")]


def _consumable_keys(gs: dict[str, Any]) -> list[str]:
    return [c.center_key for c in gs.get("consumables", []) if hasattr(c, "center_key")]


def compare_states(
    sim_gs: dict[str, Any],
    bot: dict[str, Any],
    step_label: str,
) -> list[str]:
    """Compare engine state to balatrobot live state. Returns list of diffs."""
    diffs: list[str] = []

    def cmp(name: str, s: Any, live: Any) -> None:
        if s != live:
            diffs.append(f"{name}: engine={s} live={live}")

    cmp("money", sim_gs.get("dollars", 0), bot.get("money", 0))
    cmp("ante", sim_gs.get("round_resets", {}).get("ante", 1), bot.get("ante_num", 1))

    cr = sim_gs.get("current_round", {})
    br = bot.get("round", {})

    from jackdaw.engine.actions import GamePhase
    phase = sim_gs.get("phase")
    # Only compare round-specific fields during active round phases;
    # they are stale in SHOP/ROUND_EVAL/BLIND_SELECT.
    _round_phases = (GamePhase.SELECTING_HAND,)
    if phase in _round_phases:
        cmp("chips", sim_gs.get("chips", 0), br.get("chips", 0))
        cmp("hands_left", cr.get("hands_left", 0), br.get("hands_left", 0))
        cmp("discards_left", cr.get("discards_left", 0), br.get("discards_left", 0))

    # Hand cards (as sets — order may differ)
    sim_hand = _hand_keys(sim_gs)
    live_hand = {c["key"] for c in bot.get("hand", {}).get("cards", [])}
    cmp("hand_cards", sim_hand, live_hand)

    # Deck size
    cmp("deck_size", len(sim_gs.get("deck", [])), bot.get("cards", {}).get("count", 0))

    # Jokers
    sim_jokers = _joker_keys(sim_gs)
    live_jokers = [c["key"] for c in bot.get("jokers", {}).get("cards", [])]
    cmp("jokers", sim_jokers, live_jokers)

    # Consumables
    sim_cons = _consumable_keys(sim_gs)
    live_cons = [c["key"] for c in bot.get("consumables", {}).get("cards", [])]
    cmp("consumables", sim_cons, live_cons)

    status = "OK" if not diffs else f"{len(diffs)} DIFFS"
    print(f"  [{step_label}] {status}")
    for d in diffs:
        print(f"    {d}")
    return diffs


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------


def play_game(
    network: FactoredPolicy,
    device: torch.device,
    adapter_factory=None,
    seed: str = "PLAY_0",
    max_steps: int = 10_000,
    verbose: bool = False,
    live: bool = False,
    validator=None,
) -> dict:
    """Play a single game with the trained policy. Returns game stats."""
    if adapter_factory is None:
        adapter_factory = DirectAdapter
    env = FactoredBalatroEnv(
        adapter_factory=adapter_factory,
        reward_shaping=False,
        max_steps=max_steps,
        seed_prefix=seed,
    )
    env.max_episode_steps = max_steps

    network.eval()
    obs, mask, info = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    prev_ante = 1
    prev_round = 0
    errors_in_a_row = 0

    if validator:
        validator.on_reset(info)

    while True:
        obs_t = _obs_to_device(obs, device)
        masks_t = _masks_to_device(mask, device)

        with torch.no_grad():
            out = network(obs_t, masks_t)

        action_type = out["action_type"].item()
        entity_target = out["entity_target"].item()
        card_target_arr = out["card_target"][0].cpu().numpy()
        value = out["value"].item()

        ct = None
        et = None
        if action_type in NEEDS_ENTITY and entity_target >= 0:
            et = entity_target
        if action_type in NEEDS_CARDS:
            selected = np.nonzero(card_target_arr)[0]
            if len(selected) > 0:
                ct = tuple(int(i) for i in selected)

        fa = FactoredAction(action_type=action_type, card_target=ct, entity_target=et)

        if live and verbose:
            at_name = ActionType(action_type).name if action_type < 21 else str(action_type)
            print(f"    action={at_name} entity={et} cards={ct}")

        try:
            next_obs, reward, terminated, truncated, next_mask, info = env.step(fa)
            errors_in_a_row = 0
        except Exception as e:
            if not live:
                raise
            errors_in_a_row += 1
            if verbose:
                print(f"  [!] Action rejected: {e} (retrying with re-fetched state)")
            if errors_in_a_row > 10:
                print("  Too many consecutive errors, aborting game")
                return {
                    "won": False,
                    "ante_reached": prev_ante,
                    "rounds_beaten": prev_round,
                    "steps": step_count,
                    "reward": total_reward,
                    "validation_diffs": validator.total_diffs if validator else 0,
                }
            try:
                reobs, remask, reinfo = env._inner.reobserve()
                from jackdaw.rl.env_wrapper import _remap_shop_masks

                shop_splits = reinfo.get("shop_splits", (0, 0, 0))
                remask = _remap_shop_masks(remask, shop_splits)
                obs = env._build_obs(reobs)
                mask = remask
            except Exception:
                pass
            time.sleep(0.2)
            continue

        if validator:
            validator.on_step(fa, info, step_count)

        total_reward += reward
        step_count += 1
        done = terminated or truncated

        gs = info.get("raw_state", {})
        ante = gs.get("round_resets", {}).get("ante", prev_ante)
        round_num = gs.get("round", prev_round)

        if verbose and (ante > prev_ante or round_num > prev_round):
            phase = gs.get("phase", "?")
            dollars = gs.get("dollars", 0)
            print(
                f"  Step {step_count}: ante={ante} round={round_num} "
                f"phase={phase} ${dollars} value={value:.3f}"
            )
            prev_ante = ante
            prev_round = round_num

        if done:
            won = env.episode_won
            final_ante = info.get("balatro/ante_reached", ante)
            final_rounds = info.get("balatro/rounds_beaten", round_num)
            if verbose:
                result = "WON!" if won else "Lost"
                print(
                    f"  {result} at ante {final_ante} "
                    f"(rounds beaten: {final_rounds}, steps: {step_count})"
                )
            return {
                "won": won,
                "ante_reached": final_ante,
                "rounds_beaten": final_rounds,
                "steps": step_count,
                "reward": total_reward,
                "validation_diffs": validator.total_diffs if validator else 0,
            }

        obs = next_obs
        mask = next_mask

        if live:
            time.sleep(0.15)


# ---------------------------------------------------------------------------
# Dual-execution validator: runs engine in parallel with live game
# ---------------------------------------------------------------------------


class DualValidator:
    """Runs the engine alongside the live game, comparing states after each step."""

    def __init__(self, backend, seed: str, back_key: str = "b_red", stake: int = 1):
        from jackdaw.engine.actions import GamePhase
        from jackdaw.engine.game import step as engine_step
        from jackdaw.engine.run_init import initialize_run

        self._engine_step = engine_step
        self._backend = backend
        self._gs: dict[str, Any] = {}
        self.total_diffs = 0
        self.all_diffs: list[list[str]] = []

        self._gs = initialize_run(back_key, stake, seed)
        self._gs["phase"] = GamePhase.BLIND_SELECT
        self._gs["blind_on_deck"] = "Small"

    def on_reset(self, info: dict[str, Any]) -> None:
        """Compare initial states after reset."""
        try:
            bot = self._backend.handle("gamestate", {})
            diffs = compare_states(self._gs, bot, "init")
            self.all_diffs.append(diffs)
            self.total_diffs += len(diffs)
        except Exception as e:
            print(f"  [validate] Failed to compare init state: {e}")

    def _shop_splits(self) -> tuple[int, int, int]:
        """Return (n_cards, n_vouchers, n_boosters) from engine state."""
        return (
            len(self._gs.get("shop_cards", [])),
            len(self._gs.get("shop_vouchers", [])),
            len(self._gs.get("shop_boosters", [])),
        )

    def on_step(self, fa: FactoredAction, info: dict[str, Any], step_num: int) -> None:
        """Apply the same action to the engine and compare with live state."""
        # The FactoredAction from the network uses global shop_item indices.
        # Unmap to sub-list indices before converting to engine action.
        from jackdaw.rl.env_wrapper import _unmap_shop_action

        fa = _unmap_shop_action(fa, self._shop_splits())

        # Convert FactoredAction → engine Action using the engine's state
        try:
            engine_action = factored_to_engine_action(fa, self._gs)
        except Exception as e:
            print(f"  [validate] step {step_num}: can't convert action: {e}")
            return

        # Step the engine
        try:
            self._engine_step(self._gs, engine_action)
        except Exception as e:
            print(f"  [validate] step {step_num}: engine error: {e}")
            return

        # Fetch live state and compare
        try:
            # CashOut triggers earnings animations; wait longer for money to settle.
            from jackdaw.engine.actions import CashOut as EngineCashOut

            if isinstance(engine_action, EngineCashOut):
                time.sleep(3)
                # Debug: print engine earnings breakdown
                earnings = self._gs.get("round_earnings")
                if earnings:
                    print(
                        f"    [earnings] blind={earnings.blind_reward} "
                        f"hands={earnings.unused_hands_bonus} "
                        f"discards={earnings.unused_discards_bonus} "
                        f"interest={earnings.interest} "
                        f"jokers={earnings.joker_dollars} "
                        f"rental={earnings.rental_cost} "
                        f"total={earnings.total}"
                    )
            else:
                time.sleep(0.3)
            bot = self._backend.handle("gamestate", {})
            diffs = compare_states(self._gs, bot, f"step_{step_num}")
            self.all_diffs.append(diffs)
            self.total_diffs += len(diffs)
        except Exception as e:
            print(f"  [validate] step {step_num}: can't fetch live state: {e}")

    def summary(self) -> None:
        """Print validation summary."""
        total_steps = len(self.all_diffs)
        clean = sum(1 for d in self.all_diffs if not d)
        print(f"\nValidation: {clean}/{total_steps} steps match, {self.total_diffs} total diffs")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Balatro with a trained policy")
    parser.add_argument("model_path", help="Path to .pt checkpoint or model file")
    parser.add_argument("--seed", default="PLAY_0", help="Game seed prefix")
    parser.add_argument("--n-games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--verbose", action="store_true", help="Print game progress")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument(
        "--live", action="store_true", help="Play against live Balatro via balatrobot"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run engine in parallel and compare states"
    )
    parser.add_argument("--host", default="127.0.0.1", help="Balatrobot host (with --live)")
    parser.add_argument("--port", type=int, default=12346, help="Balatrobot port (with --live)")
    args = parser.parse_args()

    if args.validate and not args.live:
        print("--validate requires --live")
        return

    device = torch.device(args.device)
    network = FactoredPolicy()

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "network" in ckpt:
        network.load_state_dict(ckpt["network"])
        print(f"Loaded checkpoint (step {ckpt.get('global_step', '?')})")
    else:
        network.load_state_dict(ckpt)
        print("Loaded model state dict")

    network.to(device)

    # Set up adapter factory and validator
    adapter_factory = None
    backend = None
    if args.live:
        from jackdaw.bridge.backend import LiveBackend
        from jackdaw.env.game_interface import BridgeAdapter

        backend = LiveBackend(host=args.host, port=args.port)
        try:
            backend.handle("health", {})
            print(f"Connected to balatrobot at {args.host}:{args.port}")
            backend.handle("menu", {})
            print("Returned to menu")
        except Exception as e:
            print(f"Cannot connect to balatrobot at {args.host}:{args.port}: {e}")
            print("Make sure balatrobot is running:")
            print("  uvx balatrobot serve --fast --no-audio --love-path <path>")
            return

        def adapter_factory():
            return BridgeAdapter(backend)

    results = []
    for i in range(args.n_games):
        seed = f"{args.seed}_{i}"
        if args.verbose or args.live:
            print(f"\n--- Game {i + 1}/{args.n_games} (seed: {seed}) ---")

        validator = None
        if args.validate and backend:
            validator = DualValidator(backend, seed)

        result = play_game(
            network,
            device,
            adapter_factory=adapter_factory,
            seed=seed,
            verbose=args.verbose or args.live,
            live=args.live,
            validator=validator,
        )
        results.append(result)

        if validator:
            validator.summary()

    # Summary
    antes = [r["ante_reached"] for r in results]
    wins = sum(1 for r in results if r["won"])
    steps = [r["steps"] for r in results]
    print(f"\n{'=' * 40}")
    print(f"Games: {len(results)}")
    print(f"Win rate: {wins}/{len(results)} ({100 * wins / len(results):.1f}%)")
    print(f"Ante reached: mean={np.mean(antes):.1f} max={np.max(antes)} min={np.min(antes)}")
    print(f"Steps: mean={np.mean(steps):.0f}")
    if args.validate:
        total_vdiffs = sum(r.get("validation_diffs", 0) for r in results)
        print(f"Validation diffs: {total_vdiffs}")


if __name__ == "__main__":
    main()
