"""Rule-based Balatro agent for behavioral cloning data generation.

Plays a simple but effective strategy:
- Hand selection: play strongest hand, discard weak cards
- Shop: buy jokers > open boosters > use planets > next round
- Never sell jokers, never skip blinds
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from jackdaw.engine.actions import GamePhase
from jackdaw.engine.hand_eval import get_best_hand
from jackdaw.env.game_spec import FactoredAction, GameActionMask

# Action type IDs (from action_space.py)
PLAY_HAND = 0
DISCARD = 1
SELECT_BLIND = 2
SKIP_BLIND = 3
CASH_OUT = 4
REROLL = 5
NEXT_ROUND = 6
SKIP_PACK = 7
BUY_CARD = 8
SELL_JOKER = 9
SELL_CONSUMABLE = 10
USE_CONSUMABLE = 11
REDEEM_VOUCHER = 12
OPEN_BOOSTER = 13
PICK_PACK_CARD = 14

# Hand type ranking (higher = better)
_HAND_RANK = {
    "High Card": 0,
    "Pair": 1,
    "Two Pair": 2,
    "Three of a Kind": 3,
    "Straight": 4,
    "Flush": 5,
    "Full House": 6,
    "Four of a Kind": 7,
    "Straight Flush": 8,
    "Five of a Kind": 9,
    "Flush House": 10,
    "Flush Five": 11,
}


class ScriptedBalatroAgent:
    """Rule-based agent that plays Balatro at a basic-competent level.

    Designed to consistently reach ante 3-5 for behavioral cloning.
    """

    def reset(self) -> None:
        """Call at episode start."""
        pass

    def act(
        self,
        raw_state: dict[str, Any],
        mask: GameActionMask,
        info: dict[str, Any],
    ) -> FactoredAction:
        """Choose an action given raw game state and legal action mask."""
        phase = raw_state.get("phase")

        if phase == GamePhase.BLIND_SELECT:
            return self._blind_select(mask)
        elif phase == GamePhase.SELECTING_HAND:
            return self._selecting_hand(raw_state, mask)
        elif phase == GamePhase.ROUND_EVAL:
            return self._round_eval(mask)
        elif phase == GamePhase.SHOP:
            return self._shop(raw_state, mask, info)
        elif phase == GamePhase.PACK_OPENING:
            return self._pack_opening(raw_state, mask)
        else:
            # Fallback: pick first legal action type
            return self._fallback(mask)

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _blind_select(self, mask: GameActionMask) -> FactoredAction:
        """Always select blind."""
        if mask.type_mask[SELECT_BLIND]:
            return FactoredAction(action_type=SELECT_BLIND)
        # Can't select (shouldn't happen), skip
        return FactoredAction(action_type=SKIP_BLIND)

    def _round_eval(self, mask: GameActionMask) -> FactoredAction:
        """Always cash out."""
        return FactoredAction(action_type=CASH_OUT)

    def _selecting_hand(
        self, gs: dict[str, Any], mask: GameActionMask
    ) -> FactoredAction:
        """Play or discard based on hand quality."""
        hand = gs.get("hand", [])
        if not hand:
            return self._fallback(mask)

        cr = gs.get("current_round", {})
        discards_left = cr.get("discards_left", 0)

        # Check for usable consumables first (planets especially)
        action = self._try_use_consumable(gs, mask)
        if action is not None:
            return action

        # Evaluate hand
        hand_name, scoring_cards, _ = get_best_hand(hand)
        hand_rank = _HAND_RANK.get(hand_name, 0)

        # Get indices of scoring cards in hand
        scoring_ids = set(id(c) for c in scoring_cards)
        scoring_indices = [i for i, c in enumerate(hand) if id(c) in scoring_ids]
        non_scoring_indices = [i for i, c in enumerate(hand) if id(c) not in scoring_ids]

        # Check for flush draw (4 of same suit)
        flush_draw = self._check_flush_draw(hand)

        # Decision logic
        if hand_rank >= _HAND_RANK["Two Pair"]:
            # Strong hand — play immediately
            return self._play_cards(scoring_indices, mask)

        if flush_draw is not None and discards_left > 0:
            # 4 to a flush — discard non-suited cards
            suit_indices, _ = flush_draw
            discard_indices = [i for i in range(len(hand)) if i not in suit_indices]
            if discard_indices and mask.type_mask[DISCARD]:
                return self._discard_cards(discard_indices[:5], mask)

        if hand_rank >= _HAND_RANK["Pair"]:
            if discards_left > 0 and len(non_scoring_indices) >= 3:
                # Pair — discard worst non-scoring cards to improve
                worst = self._worst_cards(hand, non_scoring_indices, max_discard=3)
                if worst and mask.type_mask[DISCARD]:
                    return self._discard_cards(worst, mask)
            # Play the pair
            return self._play_cards(scoring_indices, mask)

        # High Card — discard if possible
        if discards_left > 0 and mask.type_mask[DISCARD]:
            # Keep the 2-3 highest rank cards, discard rest
            ranked = sorted(range(len(hand)), key=lambda i: self._card_rank(hand[i]), reverse=True)
            keep = set(ranked[:3])
            discard_indices = [i for i in range(len(hand)) if i not in keep]
            if discard_indices:
                return self._discard_cards(discard_indices[:5], mask)

        # No discards left — play whatever we have
        return self._play_cards(scoring_indices or list(range(min(5, len(hand)))), mask)

    def _shop(
        self, gs: dict[str, Any], mask: GameActionMask, info: dict[str, Any]
    ) -> FactoredAction:
        """Buy jokers, open boosters, use consumables, then leave."""
        shop_cards = gs.get("shop_cards", [])
        shop_boosters = gs.get("shop_boosters", [])
        shop_splits = info.get("shop_splits", (len(shop_cards), 0, len(shop_boosters)))
        n_cards, n_vouchers, n_boosters = shop_splits
        n_jokers = len(gs.get("jokers", []))
        joker_slots = gs.get("joker_slots", 5)
        dollars = gs.get("dollars", 0)

        # Priority 1: Use consumable planets (if in hand phase or shop phase)
        action = self._try_use_consumable(gs, mask)
        if action is not None:
            return action

        # Priority 2: Buy jokers from shop
        if mask.type_mask[BUY_CARD] and BUY_CARD in mask.entity_masks:
            buy_mask = mask.entity_masks[BUY_CARD]
            for i in range(min(n_cards, len(buy_mask))):
                if not buy_mask[i]:
                    continue
                card = shop_cards[i]
                card_set = card.ability.get("set", "")
                if card_set == "Joker" and n_jokers < joker_slots:
                    return FactoredAction(action_type=BUY_CARD, entity_target=i)
                # Also buy planet/tarot cards if we have consumable slots
                cons_slots = gs.get("consumable_slots", 2)
                n_cons = len(gs.get("consumables", []))
                if card_set in ("Planet", "Tarot") and n_cons < cons_slots:
                    return FactoredAction(action_type=BUY_CARD, entity_target=i)

        # Priority 3: Open boosters
        if mask.type_mask[OPEN_BOOSTER] and OPEN_BOOSTER in mask.entity_masks:
            booster_mask = mask.entity_masks[OPEN_BOOSTER]
            for i in range(len(booster_mask)):
                if booster_mask[i]:
                    return FactoredAction(action_type=OPEN_BOOSTER, entity_target=i)

        # Priority 4: Redeem vouchers
        if mask.type_mask[REDEEM_VOUCHER] and REDEEM_VOUCHER in mask.entity_masks:
            vouch_mask = mask.entity_masks[REDEEM_VOUCHER]
            for i in range(len(vouch_mask)):
                if vouch_mask[i]:
                    return FactoredAction(action_type=REDEEM_VOUCHER, entity_target=i)

        # Priority 5: Reroll if rich ($15+ above interest floor)
        interest_floor = min(dollars // 5, 5) * 5
        if mask.type_mask[REROLL] and dollars - interest_floor >= 10:
            return FactoredAction(action_type=REROLL)

        # Default: next round
        if mask.type_mask[NEXT_ROUND]:
            return FactoredAction(action_type=NEXT_ROUND)

        return self._fallback(mask)

    def _pack_opening(
        self, gs: dict[str, Any], mask: GameActionMask
    ) -> FactoredAction:
        """Pick best card from pack: jokers > planets > tarots > any."""
        if not (mask.type_mask[PICK_PACK_CARD] and PICK_PACK_CARD in mask.entity_masks):
            if mask.type_mask[SKIP_PACK]:
                return FactoredAction(action_type=SKIP_PACK)
            return self._fallback(mask)

        pack_mask = mask.entity_masks[PICK_PACK_CARD]
        pack_cards = gs.get("pack_cards", [])
        n_jokers = len(gs.get("jokers", []))
        joker_slots = gs.get("joker_slots", 5)
        cons_slots = gs.get("consumable_slots", 2)
        n_cons = len(gs.get("consumables", []))

        # Priority picks: Joker > Planet > Tarot > any
        best_idx = None
        best_priority = -1
        for i in range(len(pack_mask)):
            if not pack_mask[i] or i >= len(pack_cards):
                continue
            card = pack_cards[i]
            card_set = card.ability.get("set", "")
            if card_set == "Joker" and n_jokers < joker_slots:
                priority = 10
            elif card_set == "Planet" and n_cons < cons_slots:
                priority = 8
            elif card_set == "Tarot" and n_cons < cons_slots:
                priority = 6
            elif card_set == "Spectral" and n_cons < cons_slots:
                priority = 4
            elif card.base is not None:
                priority = 2  # playing card
            else:
                priority = 1
            if priority > best_priority:
                best_priority = priority
                best_idx = i

        if best_idx is not None:
            return FactoredAction(action_type=PICK_PACK_CARD, entity_target=best_idx)

        if mask.type_mask[SKIP_PACK]:
            return FactoredAction(action_type=SKIP_PACK)
        return self._fallback(mask)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _try_use_consumable(
        self, gs: dict[str, Any], mask: GameActionMask
    ) -> FactoredAction | None:
        """Try to use a planet card (levels up hand types)."""
        if not mask.type_mask[USE_CONSUMABLE]:
            return None
        if USE_CONSUMABLE not in mask.entity_masks:
            return None

        cons_mask = mask.entity_masks[USE_CONSUMABLE]
        consumables = gs.get("consumables", [])

        for i in range(len(cons_mask)):
            if not cons_mask[i] or i >= len(consumables):
                continue
            card = consumables[i]
            card_set = card.ability.get("set", "")
            # Prioritize planet cards (no card targets needed)
            if card_set == "Planet":
                return FactoredAction(action_type=USE_CONSUMABLE, entity_target=i)

        return None

    def _play_cards(
        self, indices: list[int], mask: GameActionMask
    ) -> FactoredAction:
        """Play the given card indices."""
        if not indices:
            # Fallback: play first legal card
            legal = np.nonzero(mask.card_mask)[0]
            indices = [int(legal[0])] if len(legal) > 0 else [0]

        # Clamp to [min_card_select, max_card_select]
        indices = indices[: mask.max_card_select]
        while len(indices) < mask.min_card_select:
            # Add more legal cards
            legal = np.nonzero(mask.card_mask)[0]
            for j in legal:
                if int(j) not in indices:
                    indices.append(int(j))
                    break
            else:
                break

        return FactoredAction(action_type=PLAY_HAND, card_target=tuple(indices))

    def _discard_cards(
        self, indices: list[int], mask: GameActionMask
    ) -> FactoredAction:
        """Discard the given card indices."""
        # Filter to legal cards
        legal_set = set(np.nonzero(mask.card_mask)[0].tolist())
        indices = [i for i in indices if i in legal_set]

        if not indices:
            # Nothing to discard — play instead
            return self._play_cards([], mask)

        indices = indices[: mask.max_card_select]
        while len(indices) < mask.min_card_select:
            for j in legal_set:
                if j not in indices:
                    indices.append(j)
                    break
            else:
                break

        return FactoredAction(action_type=DISCARD, card_target=tuple(indices))

    def _check_flush_draw(
        self, hand: list
    ) -> tuple[list[int], str] | None:
        """Check if hand has 4+ cards of one suit (flush draw)."""
        suit_groups: dict[str, list[int]] = {}
        for i, card in enumerate(hand):
            base = getattr(card, "base", None)
            if base is None:
                continue
            suit = base.suit.value if hasattr(base.suit, "value") else str(base.suit)
            suit_groups.setdefault(suit, []).append(i)

        for suit, indices in suit_groups.items():
            if len(indices) >= 4:
                return indices, suit
        return None

    def _worst_cards(
        self, hand: list, indices: list[int], max_discard: int
    ) -> list[int]:
        """Return the worst N cards from the given indices (lowest rank)."""
        ranked = sorted(indices, key=lambda i: self._card_rank(hand[i]))
        return ranked[:max_discard]

    @staticmethod
    def _card_rank(card) -> int:
        """Get numeric rank for sorting (higher = better to keep)."""
        base = getattr(card, "base", None)
        if base is None:
            return 0
        return getattr(base, "id", 0)  # 2-14, Ace=14

    def _fallback(self, mask: GameActionMask) -> FactoredAction:
        """Pick the first legal action type with no targets."""
        for t in range(len(mask.type_mask)):
            if mask.type_mask[t]:
                return FactoredAction(action_type=t)
        return FactoredAction(action_type=0)
