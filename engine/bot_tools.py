from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Protocol

from engine.icm import calculate_exact_icm

try:
    from treys import Card
except ImportError:  # pragma: no cover - treys is a runtime dependency, but keep config tooling importable.
    Card = None


@dataclass(frozen=True)
class DecisionContext:
    equity: float
    required_equity: float
    call_margin: float
    raise_threshold: float
    jam_threshold: float
    street: int
    active_players: int
    stack_bb: float
    spot_type: str
    position: str
    call_amount: int
    stack_size: int
    min_raise: int
    big_blind: int
    payout_pressure: float
    open_size: int
    pot_size: int
    max_raise_extra: int
    forced_action: tuple[str, int] | None = None
    tool_event: Dict[str, Any] | None = None
    tool_events: tuple = ()

    def with_raise_threshold(self, value: float) -> "DecisionContext":
        return replace(self, raise_threshold=max(0.36, min(0.90, float(value))))

    def with_call_margin(self, value: float) -> "DecisionContext":
        return replace(self, call_margin=max(-0.02, min(0.25, float(value))))

    def with_jam_threshold(self, value: float) -> "DecisionContext":
        return replace(self, jam_threshold=max(0.45, min(0.94, float(value))))

    def with_payout_pressure(self, value: float) -> "DecisionContext":
        return replace(self, payout_pressure=max(0.0, min(1.0, float(value))))

    def with_forced_action(self, action: str, amount: int = 0) -> "DecisionContext":
        return replace(self, forced_action=(str(action), max(0, int(amount))))

    def with_tool_event(self, event: Mapping[str, Any]) -> "DecisionContext":
        payload = dict(event)
        return replace(self, tool_event=payload, tool_events=self.tool_events + (payload,))


class BotTool(Protocol):
    name: str
    priority: int

    def apply(self, context: DecisionContext, game_state: Mapping[str, Any]) -> DecisionContext:
        ...


class ICMPressureTool:
    name = "icm_pressure"

    def __init__(
        self,
        *,
        priority: int = 20,
        enabled: bool = True,
        strength: float = 1.0,
        exact_when_available: bool = True,
        call_margin_weight: float = 0.10,
        raise_threshold_weight: float = 0.08,
        jam_threshold_weight: float = 0.12,
    ):
        self.priority = int(priority)
        self.enabled = bool(enabled)
        self.strength = max(0.0, float(strength))
        self.exact_when_available = bool(exact_when_available)
        self.call_margin_weight = max(0.0, float(call_margin_weight))
        self.raise_threshold_weight = max(0.0, float(raise_threshold_weight))
        self.jam_threshold_weight = max(0.0, float(jam_threshold_weight))

    def apply(self, context: DecisionContext, game_state: Mapping[str, Any]) -> DecisionContext:
        if not self.enabled or self.strength <= 0:
            return context
        pressure = self._pressure(game_state)
        if pressure <= 0:
            return context
        context = context.with_payout_pressure(max(context.payout_pressure, pressure))
        context = context.with_call_margin(context.call_margin + pressure * self.call_margin_weight)
        context = context.with_raise_threshold(context.raise_threshold + pressure * self.raise_threshold_weight)
        return context.with_jam_threshold(context.jam_threshold + pressure * self.jam_threshold_weight)

    def _pressure(self, game_state: Mapping[str, Any]) -> float:
        if self.exact_when_available:
            exact_pressure = self._exact_icm_pressure(game_state)
            if exact_pressure is not None:
                return min(1.0, exact_pressure * self.strength)
        return min(1.0, self._fallback_icm_pressure(game_state) * self.strength)

    def _exact_icm_pressure(self, game_state: Mapping[str, Any]) -> float | None:
        table_stacks = game_state.get("table_stacks")
        payouts = game_state.get("payouts")
        hero_index = game_state.get("hero_table_index")
        players_left = int(game_state.get("players_left", 0) or 0)
        if not isinstance(table_stacks, list) or not table_stacks:
            return None
        if players_left != len(table_stacks):
            return None
        if hero_index is None:
            return None

        try:
            hero_index = int(hero_index)
            stacks = [float(stack) for stack in table_stacks]
            payout_values = self._payout_values(payouts)
            hero_stack = max(0.0, float(game_state.get("stack_size", 0) or 0.0))
        except (TypeError, ValueError):
            return None
        if hero_index < 0 or hero_index >= len(stacks) or hero_stack <= 0 or not payout_values:
            return None

        total_chips = sum(stacks)
        if total_chips <= 0:
            return None

        try:
            current_icm = calculate_exact_icm(stacks, payout_values)[hero_index]
            loss_stacks = list(stacks)
            loss_stacks[hero_index] = 0.0
            loss_icm = calculate_exact_icm(loss_stacks, payout_values)[hero_index]
            gain_stacks = list(stacks)
            gain_stacks[hero_index] += min(hero_stack, max(1.0, total_chips * 0.05))
            gain_icm = calculate_exact_icm(gain_stacks, payout_values)[hero_index]
        except ValueError:
            return None

        downside = max(0.0, current_icm - loss_icm)
        upside = max(0.0, gain_icm - current_icm)
        if downside <= 0:
            return 0.0

        bubble_bonus = self._heuristic_payout_pressure(game_state) * 0.30
        asymmetry = downside / max(downside + upside, 1e-9)
        return max(0.0, min(1.0, (asymmetry - 0.50) * 1.45 + bubble_bonus))

    def _payout_values(self, payouts: Any) -> list[float]:
        if isinstance(payouts, dict):
            return [float(value) for _place, value in sorted(payouts.items(), key=lambda item: int(item[0]))]
        if isinstance(payouts, list):
            return [float(value) for value in payouts]
        return []

    def _fallback_icm_pressure(self, game_state: Mapping[str, Any]) -> float:
        heuristic = self._heuristic_payout_pressure(game_state)
        itm_distance = max(0.0, min(1.0, float(game_state.get("itm_distance", 1.0) or 0.0)))
        players_left = int(game_state.get("players_left", 0) or 0)
        paid_places = int(game_state.get("paid_places", 0) or 0)

        pressure = heuristic
        if paid_places > 0 and players_left > paid_places:
            pressure += max(0.0, 1.0 - itm_distance) * 0.20
        return max(0.0, min(1.0, pressure))

    def _heuristic_payout_pressure(self, game_state: Mapping[str, Any]) -> float:
        next_prize_gain = max(0.0, float(game_state.get("next_prize_gain_pct", 0.0) or 0.0))
        players_left = int(game_state.get("players_left", 0) or 0)
        paid_places = int(game_state.get("paid_places", 0) or 0)
        if paid_places <= 0 or players_left <= 0:
            return min(1.0, next_prize_gain * 4.0)
        bubble_pressure = 1.0 if paid_places < players_left <= paid_places + 2 else 0.0
        final_table_pressure = 0.5 if players_left <= 10 else 0.0
        return min(1.0, bubble_pressure + final_table_pressure + next_prize_gain * 4.0)


class PreflopReraiseTightnessTool:
    name = "preflop_reraise_tightness"

    def __init__(self, *, priority: int = 30, enabled: bool = True, tightness: float = 0.08):
        self.priority = int(priority)
        self.enabled = bool(enabled)
        self.tightness = max(0.0, float(tightness))

    def apply(self, context: DecisionContext, game_state: Mapping[str, Any]) -> DecisionContext:
        if not self.enabled:
            return context
        if context.street == 0 and context.call_amount > 0 and context.spot_type in {"srp", "single_raised"}:
            return context.with_raise_threshold(context.raise_threshold + self.tightness)
        return context


class TableAdaptationTool:
    name = "table_adaptation"

    def __init__(
        self,
        *,
        priority: int = 35,
        enabled: bool = True,
        sample_quality_min: float = 0.50,
        loose_passive_vpip: float = 0.44,
        loose_passive_pfr: float = 0.14,
        aggressive_pfr: float = 0.32,
        aggressive_three_bet_rate: float = 0.22,
    ):
        self.priority = int(priority)
        self.enabled = bool(enabled)
        self.sample_quality_min = max(0.0, min(1.0, float(sample_quality_min)))
        self.loose_passive_vpip = max(0.0, min(1.0, float(loose_passive_vpip)))
        self.loose_passive_pfr = max(0.0, min(1.0, float(loose_passive_pfr)))
        self.aggressive_pfr = max(0.0, min(1.0, float(aggressive_pfr)))
        self.aggressive_three_bet_rate = max(0.0, min(1.0, float(aggressive_three_bet_rate)))

    def apply(self, context: DecisionContext, game_state: Mapping[str, Any]) -> DecisionContext:
        if not self.enabled:
            return context
        stats = self._table_stats(game_state)
        if not stats:
            return context

        quality = stats["sample_quality"]
        pressure_spot = context.spot_type in {
            "srp",
            "single_raised",
            "three_bet",
            "3bet",
            "four_bet",
            "4bet",
            "all_in_pressure",
        }
        loose_passive_score = max(0.0, stats["vpip"] - 0.42) * max(0.0, 0.16 - stats["pfr"])
        aggressive_score = max(0.0, stats["three_bet_rate"] - 0.20)
        loose_passive = stats["vpip"] >= self.loose_passive_vpip and stats["pfr"] <= self.loose_passive_pfr
        aggressive = stats["three_bet_rate"] >= self.aggressive_three_bet_rate or stats["pfr"] >= self.aggressive_pfr

        margin = context.call_margin - min(0.010, loose_passive_score * 0.20) * quality
        if pressure_spot:
            margin += min(0.020, aggressive_score * 0.20) * quality
        if context.street == 0 and context.position in {"BTN", "CO", "BB"} and loose_passive:
            margin -= 0.004 * quality

        threshold = context.raise_threshold
        if loose_passive:
            threshold -= 0.010 * quality
        if context.street == 0 and context.call_amount > 0 and aggressive:
            threshold += 0.030 * quality

        return context.with_call_margin(margin).with_raise_threshold(threshold)

    def _table_stats(self, game_state: Mapping[str, Any]) -> Dict[str, float] | None:
        stats = game_state.get("table_stats")
        if not isinstance(stats, dict):
            return None
        try:
            quality = max(0.0, min(1.0, float(stats.get("sample_quality", 0.0) or 0.0)))
            if quality < self.sample_quality_min:
                return None
            return {
                "sample_quality": quality,
                "vpip": max(0.0, min(1.0, float(stats.get("vpip", 0.0) or 0.0))),
                "pfr": max(0.0, min(1.0, float(stats.get("pfr", 0.0) or 0.0))),
                "three_bet_rate": max(0.0, min(1.0, float(stats.get("three_bet_rate", 0.0) or 0.0))),
            }
        except (TypeError, ValueError):
            return None


class ButtonStealTool:
    name = "button_steal"

    def __init__(
        self,
        *,
        priority: int = 40,
        enabled: bool = True,
        sample_quality_min: float = 0.50,
        max_vpip: float = 0.30,
        max_pfr: float = 0.16,
        max_three_bet_rate: float = 0.08,
        base_discount: float = 0.018,
        tightness_multiplier: float = 0.12,
        max_discount: float = 0.040,
        max_open_stack_fraction: float = 0.14,
        max_active_players: int = 4,
        min_stack_bb: float = 16.0,
        max_payout_pressure: float = 0.45,
        requires_table_sample: bool = True,
    ):
        self.priority = int(priority)
        self.enabled = bool(enabled)
        self.sample_quality_min = max(0.0, min(1.0, float(sample_quality_min)))
        self.max_vpip = max(0.0, min(1.0, float(max_vpip)))
        self.max_pfr = max(0.0, min(1.0, float(max_pfr)))
        self.max_three_bet_rate = max(0.0, min(1.0, float(max_three_bet_rate)))
        self.base_discount = max(0.0, float(base_discount))
        self.tightness_multiplier = max(0.0, float(tightness_multiplier))
        self.max_discount = max(0.0, float(max_discount))
        self.max_open_stack_fraction = max(0.0, float(max_open_stack_fraction))
        self.max_active_players = int(max_active_players)
        self.min_stack_bb = max(0.0, float(min_stack_bb))
        self.max_payout_pressure = max(0.0, float(max_payout_pressure))
        self.requires_table_sample = bool(requires_table_sample)

    def apply(self, context: DecisionContext, game_state: Mapping[str, Any]) -> DecisionContext:
        if not self.enabled or not self._is_structural_steal_spot(context):
            return context

        stats = self._table_stats(game_state)
        if self.requires_table_sample and not stats:
            return context
        if stats and (
            stats["vpip"] > self.max_vpip
            or stats["pfr"] > self.max_pfr
            or stats["three_bet_rate"] > self.max_three_bet_rate
        ):
            return context

        tightness = 0.0
        quality = 1.0
        if stats:
            quality = stats["sample_quality"]
            tightness = max(0.0, self.max_vpip - stats["vpip"]) + max(0.0, self.max_pfr - stats["pfr"])
        discount = min(self.max_discount, self.base_discount + tightness * self.tightness_multiplier) * quality
        return context.with_raise_threshold(context.raise_threshold - discount)

    def _is_structural_steal_spot(self, context: DecisionContext) -> bool:
        if context.street != 0 or context.position != "BTN":
            return False
        if context.call_amount != 0:
            return False
        if context.spot_type not in {"unknown", "limped"}:
            return False
        if context.active_players > self.max_active_players:
            return False
        if context.stack_bb < self.min_stack_bb:
            return False
        if context.payout_pressure > self.max_payout_pressure:
            return False
        if context.stack_size <= 0 or context.open_size <= 0:
            return False
        return context.open_size / float(context.stack_size) <= self.max_open_stack_fraction

    def _table_stats(self, game_state: Mapping[str, Any]) -> Dict[str, float] | None:
        stats = game_state.get("table_stats")
        if not isinstance(stats, dict):
            return None
        try:
            quality = max(0.0, min(1.0, float(stats.get("sample_quality", 0.0) or 0.0)))
            if quality < self.sample_quality_min:
                return None
            return {
                "sample_quality": quality,
                "vpip": max(0.0, min(1.0, float(stats.get("vpip", 0.0) or 0.0))),
                "pfr": max(0.0, min(1.0, float(stats.get("pfr", 0.0) or 0.0))),
                "three_bet_rate": max(0.0, min(1.0, float(stats.get("three_bet_rate", 0.0) or 0.0))),
            }
        except (TypeError, ValueError):
            return None


class EndgameConversionTool:
    name = "endgame_conversion"

    def __init__(
        self,
        *,
        priority: int = 45,
        enabled: bool = True,
        final_table_players: int = 9,
        top3_players: int = 3,
        min_stack_bb: float = 18.0,
        big_stack_avg_multiplier: float = 1.10,
        cover_fraction_min: float = 0.50,
        raise_discount: float = 0.018,
        jam_discount: float = 0.018,
        call_margin_discount: float = 0.004,
        top3_multiplier: float = 1.60,
        heads_up_multiplier: float = 2.00,
        protect_bubble: bool = True,
    ):
        self.priority = int(priority)
        self.enabled = bool(enabled)
        self.final_table_players = max(2, int(final_table_players))
        self.top3_players = max(2, int(top3_players))
        self.min_stack_bb = max(0.0, float(min_stack_bb))
        self.big_stack_avg_multiplier = max(0.0, float(big_stack_avg_multiplier))
        self.cover_fraction_min = max(0.0, min(1.0, float(cover_fraction_min)))
        self.raise_discount = max(0.0, float(raise_discount))
        self.jam_discount = max(0.0, float(jam_discount))
        self.call_margin_discount = max(0.0, float(call_margin_discount))
        self.top3_multiplier = max(0.0, float(top3_multiplier))
        self.heads_up_multiplier = max(0.0, float(heads_up_multiplier))
        self.protect_bubble = bool(protect_bubble)

    def apply(self, context: DecisionContext, game_state: Mapping[str, Any]) -> DecisionContext:
        if not self.enabled:
            return context
        multiplier = self._conversion_multiplier(context, game_state)
        if multiplier <= 0:
            return context
        return (
            context.with_call_margin(context.call_margin - self.call_margin_discount * multiplier)
            .with_raise_threshold(context.raise_threshold - self.raise_discount * multiplier)
            .with_jam_threshold(context.jam_threshold - self.jam_discount * multiplier)
        )

    def _conversion_multiplier(self, context: DecisionContext, game_state: Mapping[str, Any]) -> float:
        players_left = self._int(game_state.get("players_left"))
        paid_places = self._int(game_state.get("paid_places"))
        if players_left <= 0 or players_left > self.final_table_players:
            return 0.0
        if self.protect_bubble and paid_places > 0 and paid_places < players_left <= paid_places + 2:
            return 0.0
        if context.stack_bb < self.min_stack_bb:
            return 0.0
        if not self._has_stack_leverage(context, game_state):
            return 0.0

        if players_left <= 2:
            return self.heads_up_multiplier
        if players_left <= self.top3_players:
            return self.top3_multiplier
        return 1.0

    def _has_stack_leverage(self, context: DecisionContext, game_state: Mapping[str, Any]) -> bool:
        if self._cover_fraction(context, game_state) >= self.cover_fraction_min:
            return True
        avg_stack = self._float(game_state.get("avg_table_stack"))
        if avg_stack <= 0:
            avg_stack = self._average_table_stack(game_state)
        if avg_stack <= 0:
            return False
        return context.stack_size >= avg_stack * self.big_stack_avg_multiplier

    def _cover_fraction(self, context: DecisionContext, game_state: Mapping[str, Any]) -> float:
        table_stacks = game_state.get("table_stacks")
        if not isinstance(table_stacks, list) or len(table_stacks) <= 1:
            return 0.0
        hero_index = self._int(game_state.get("hero_table_index"), default=-1)
        covered = 0
        opponents = 0
        for index, raw_stack in enumerate(table_stacks):
            if index == hero_index:
                continue
            stack = self._float(raw_stack)
            if stack <= 0:
                continue
            opponents += 1
            if context.stack_size >= stack:
                covered += 1
        if opponents <= 0:
            return 0.0
        return covered / float(opponents)

    def _average_table_stack(self, game_state: Mapping[str, Any]) -> float:
        table_stacks = game_state.get("table_stacks")
        if not isinstance(table_stacks, list) or not table_stacks:
            return 0.0
        stacks = [self._float(stack) for stack in table_stacks]
        stacks = [stack for stack in stacks if stack > 0]
        return sum(stacks) / len(stacks) if stacks else 0.0

    def _int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


class ContinuationPressureTool:
    name = "cbet_pressure"

    def __init__(
        self,
        *,
        priority: int = 46,
        enabled: bool = True,
        sample_quality_min: float = 0.50,
        min_equity: float = 0.42,
        max_payout_pressure: float = 0.70,
        min_stack_bb: float = 14.0,
        max_active_players: int = 2,
        min_spr: float = 1.75,
        threshold_discount: float = 0.035,
        max_threshold_discount: float = 0.060,
        fold_equity_weight: float = 0.040,
        min_fold_equity: float = 0.38,
        flop_size: float = 0.33,
        max_stack_fraction: float = 0.18,
        require_initiative: bool = True,
        allow_position_proxy: bool = True,
    ):
        self.priority = int(priority)
        self.enabled = bool(enabled)
        self.sample_quality_min = max(0.0, min(1.0, float(sample_quality_min)))
        self.min_equity = max(0.0, min(1.0, float(min_equity)))
        self.max_payout_pressure = max(0.0, min(1.0, float(max_payout_pressure)))
        self.min_stack_bb = max(0.0, float(min_stack_bb))
        self.max_active_players = max(2, int(max_active_players))
        self.min_spr = max(0.0, float(min_spr))
        self.threshold_discount = max(0.0, min(0.20, float(threshold_discount)))
        self.max_threshold_discount = max(self.threshold_discount, min(0.20, float(max_threshold_discount)))
        self.fold_equity_weight = max(0.0, min(0.20, float(fold_equity_weight)))
        self.min_fold_equity = max(0.0, min(1.0, float(min_fold_equity)))
        self.flop_size = max(0.05, float(flop_size))
        self.max_stack_fraction = max(0.0, min(1.0, float(max_stack_fraction)))
        self.require_initiative = bool(require_initiative)
        self.allow_position_proxy = bool(allow_position_proxy)

    def apply(self, context: DecisionContext, game_state: Mapping[str, Any]) -> DecisionContext:
        if not self.enabled:
            return context
        reject = self._rejection_reason(context, game_state)
        if reject:
            return self._with_event(context, "reject", reason=reject)

        fold_equity = self._fold_equity(game_state)
        if fold_equity < self.min_fold_equity:
            return self._with_event(context, "reject", reason="low_fold_equity", fold_equity=fold_equity)

        amount = self._raise_amount(context)
        if amount <= 0:
            return self._with_event(context, "reject", reason="no_raise_amount", fold_equity=fold_equity)

        fold_bonus = max(0.0, fold_equity - self.min_fold_equity) * self.fold_equity_weight
        discount = min(self.max_threshold_discount, self.threshold_discount + fold_bonus)
        context = context.with_raise_threshold(context.raise_threshold - discount)
        return self._with_event(
            context,
            "threshold_discount",
            amount=amount,
            fold_equity=fold_equity,
            threshold_discount=discount,
        )

    def _rejection_reason(self, context: DecisionContext, game_state: Mapping[str, Any]) -> str:
        if context.forced_action is not None:
            return "already_forced"
        if context.street != 3:
            return "street"
        if context.call_amount != 0:
            return "facing_bet"
        if context.active_players > self.max_active_players:
            return "multiway"
        if context.stack_bb < self.min_stack_bb:
            return "short_stack"
        if context.payout_pressure > self.max_payout_pressure:
            return "payout_pressure"
        if context.pot_size <= 0 or context.max_raise_extra < max(1, context.min_raise):
            return "cannot_raise"
        if context.equity < self.min_equity:
            return "low_equity"
        if self.min_spr > 0 and context.stack_size / float(context.pot_size) < self.min_spr:
            return "low_spr"
        if self.require_initiative and not self._has_initiative(context, game_state):
            return "no_initiative"
        return ""

    def _has_initiative(self, context: DecisionContext, game_state: Mapping[str, Any]) -> bool:
        if "has_initiative" in game_state:
            return bool(game_state.get("has_initiative"))
        if context.spot_type in {"three_bet", "3bet", "four_bet", "4bet"}:
            return True
        if context.spot_type in {"srp", "single_raised"} and context.position in {"BTN", "CO", "HJ"}:
            return True
        return self.allow_position_proxy and context.position in {"BTN", "CO"} and context.spot_type == "unknown"

    def _fold_equity(self, game_state: Mapping[str, Any]) -> float:
        stats = self._stats(game_state.get("opponent_stats")) or self._stats(game_state.get("table_stats"))
        if not stats:
            return self.min_fold_equity
        tightness = max(0.0, 0.34 - stats["vpip"]) * 0.55
        passivity = max(0.0, 0.22 - stats["pfr"]) * 0.30
        low_reraise = max(0.0, 0.10 - stats["three_bet_rate"]) * 0.15
        return max(0.0, min(1.0, (0.34 + tightness + passivity + low_reraise) * stats["sample_quality"]))

    def _raise_amount(self, context: DecisionContext) -> int:
        target = int(max(context.big_blind, context.pot_size * self.flop_size))
        target = max(context.min_raise, target)
        if self.max_stack_fraction > 0:
            target = min(target, int(context.stack_size * self.max_stack_fraction))
        return min(context.max_raise_extra, max(context.min_raise, target))

    def _with_event(
        self,
        context: DecisionContext,
        decision: str,
        *,
        reason: str = "",
        amount: int = 0,
        fold_equity: float | None = None,
        threshold_discount: float = 0.0,
    ) -> DecisionContext:
        event = {
            "tool": self.name,
            "decision": decision,
            "amount": int(amount),
            "street": context.street,
            "position": context.position,
            "equity": context.equity,
            "raise_threshold": context.raise_threshold,
            "threshold_gap": context.raise_threshold - context.equity,
            "threshold_discount": threshold_discount,
            "pot_size": context.pot_size,
            "stack_bb": context.stack_bb,
            "payout_pressure": context.payout_pressure,
            "active_players": context.active_players,
        }
        if reason:
            event["reason"] = reason
        if fold_equity is not None:
            event["fold_equity"] = fold_equity
        return context.with_tool_event(event)

    def _stats(self, raw_stats: Any) -> Dict[str, float] | None:
        if not isinstance(raw_stats, dict):
            return None
        try:
            quality = max(0.0, min(1.0, float(raw_stats.get("sample_quality", 0.0) or 0.0)))
            if quality < self.sample_quality_min:
                return None
            return {
                "sample_quality": quality,
                "vpip": max(0.0, min(1.0, float(raw_stats.get("vpip", 0.25) or 0.0))),
                "pfr": max(0.0, min(1.0, float(raw_stats.get("pfr", 0.16) or 0.0))),
                "three_bet_rate": max(0.0, min(1.0, float(raw_stats.get("three_bet_rate", 0.08) or 0.0))),
            }
        except (TypeError, ValueError):
            return None


class BluffPressureTool:
    name = "bluff_pressure"

    def __init__(
        self,
        *,
        priority: int = 46,
        enabled: bool = True,
        sample_quality_min: float = 0.65,
        min_fold_equity: float = 0.50,
        min_equity: float = 0.42,
        max_equity: float = 0.54,
        max_threshold_gap: float = 0.07,
        max_payout_pressure: float = 0.65,
        min_stack_bb: float = 14.0,
        max_active_players: int = 2,
        avoid_low_spr: bool = True,
        min_spr: float = 2.0,
        allowed_streets: Iterable[int] | None = None,
        flop_size: float = 0.33,
        turn_size: float = 0.50,
        river_size_min: float | None = None,
        river_size_max: float | None = None,
        max_stack_fraction: float = 0.18,
        position_bonus: float = 0.06,
        initiative_bonus: float = 0.05,
        require_position_or_initiative: bool = True,
        min_ev_edge_pot_fraction: float = 0.03,
        required_fold_equity_safety_margin: float = 0.03,
        equity_fold_equity_credit: float = 0.15,
        max_equity_fold_equity_credit: float = 0.06,
        mode: str = "force_raise",
        threshold_discount: float = 0.035,
        max_threshold_discount: float = 0.055,
        leverage_max_payout_pressure: float | None = None,
        leverage_cover_fraction_min: float = 0.50,
        leverage_stack_ratio_min: float = 1.10,
        require_stack_advantage: bool = False,
        stack_advantage_cover_fraction_min: float = 0.50,
        stack_advantage_stack_ratio_min: float = 1.25,
        require_river_scare_card: bool = False,
        survival_payout_pressure_min: float = 1.0,
        survival_max_risk_stack_fraction: float = 1.0,
        survival_cover_fraction_min: float = 0.50,
        survival_stack_ratio_min: float = 1.10,
    ):
        self.priority = int(priority)
        self.enabled = bool(enabled)
        self.sample_quality_min = max(0.0, min(1.0, float(sample_quality_min)))
        self.min_fold_equity = max(0.0, min(1.0, float(min_fold_equity)))
        self.min_equity = max(0.0, min(1.0, float(min_equity)))
        self.max_equity = max(self.min_equity, min(1.0, float(max_equity)))
        self.max_threshold_gap = max(0.0, min(1.0, float(max_threshold_gap)))
        self.max_payout_pressure = max(0.0, min(1.0, float(max_payout_pressure)))
        self.min_stack_bb = max(0.0, float(min_stack_bb))
        self.max_active_players = max(2, int(max_active_players))
        self.avoid_low_spr = bool(avoid_low_spr)
        self.min_spr = max(0.0, float(min_spr))
        self.allowed_streets = set(allowed_streets or (3, 4))
        self.flop_size = max(0.05, float(flop_size))
        self.turn_size = max(0.05, float(turn_size))
        self.river_size_min = None if river_size_min is None else max(0.05, float(river_size_min))
        raw_river_size_max = river_size_max if river_size_max is not None else river_size_min
        self.river_size_max = None if raw_river_size_max is None else max(0.05, float(raw_river_size_max))
        if self.river_size_min is not None and self.river_size_max is not None:
            self.river_size_max = max(self.river_size_min, self.river_size_max)
        self.max_stack_fraction = max(0.0, min(1.0, float(max_stack_fraction)))
        self.position_bonus = max(0.0, float(position_bonus))
        self.initiative_bonus = max(0.0, float(initiative_bonus))
        self.require_position_or_initiative = bool(require_position_or_initiative)
        self.min_ev_edge_pot_fraction = max(0.0, float(min_ev_edge_pot_fraction))
        self.required_fold_equity_safety_margin = max(0.0, min(1.0, float(required_fold_equity_safety_margin)))
        self.equity_fold_equity_credit = max(0.0, min(1.0, float(equity_fold_equity_credit)))
        self.max_equity_fold_equity_credit = max(0.0, min(1.0, float(max_equity_fold_equity_credit)))
        self.mode = str(mode)
        if self.mode not in {"force_raise", "threshold"}:
            raise ValueError("bluff_pressure mode must be 'force_raise' or 'threshold'")
        self.threshold_discount = max(0.0, min(0.20, float(threshold_discount)))
        self.max_threshold_discount = max(self.threshold_discount, min(0.20, float(max_threshold_discount)))
        raw_leverage_max = self.max_payout_pressure if leverage_max_payout_pressure is None else leverage_max_payout_pressure
        self.leverage_max_payout_pressure = max(self.max_payout_pressure, min(1.0, float(raw_leverage_max)))
        self.leverage_cover_fraction_min = max(0.0, min(1.0, float(leverage_cover_fraction_min)))
        self.leverage_stack_ratio_min = max(0.0, float(leverage_stack_ratio_min))
        self.require_stack_advantage = bool(require_stack_advantage)
        self.stack_advantage_cover_fraction_min = max(0.0, min(1.0, float(stack_advantage_cover_fraction_min)))
        self.stack_advantage_stack_ratio_min = max(0.0, float(stack_advantage_stack_ratio_min))
        self.require_river_scare_card = bool(require_river_scare_card)
        self.survival_payout_pressure_min = max(0.0, min(1.0, float(survival_payout_pressure_min)))
        self.survival_max_risk_stack_fraction = max(0.0, min(1.0, float(survival_max_risk_stack_fraction)))
        self.survival_cover_fraction_min = max(0.0, min(1.0, float(survival_cover_fraction_min)))
        self.survival_stack_ratio_min = max(0.0, float(survival_stack_ratio_min))

    def apply(self, context: DecisionContext, game_state: Mapping[str, Any]) -> DecisionContext:
        if not self.enabled:
            return context
        structural_reject = self._structural_rejection_reason(context, game_state)
        if structural_reject:
            return self._with_reject_event(context, structural_reject)
        if self.require_position_or_initiative and not self._has_position_or_initiative(context, game_state):
            return self._with_reject_event(context, "no_position_or_initiative")
        fold_equity = self._fold_equity(context, game_state)
        if fold_equity < self.min_fold_equity:
            return self._with_reject_event(context, "low_fold_equity", fold_equity=fold_equity)
        amount = self._raise_amount(context)
        if amount <= 0:
            return self._with_reject_event(context, "no_raise_amount", fold_equity=fold_equity)
        if self.require_river_scare_card and not self._has_river_scare_card(context, game_state):
            return self._with_reject_event(context, "river_texture", amount=amount, fold_equity=fold_equity)
        if self.require_stack_advantage and not self._has_stack_advantage(context, game_state):
            return self._with_reject_event(context, "stack_disadvantage", amount=amount, fold_equity=fold_equity)
        if self._has_survival_risk(context, game_state, amount):
            return self._with_reject_event(context, "survival_risk", amount=amount, fold_equity=fold_equity)
        required_fold_equity = self._required_fold_equity(context, amount)
        if fold_equity < required_fold_equity:
            return self._with_reject_event(
                context,
                "required_fold_equity",
                amount=amount,
                fold_equity=fold_equity,
                required_fold_equity=required_fold_equity,
            )
        if not self._has_positive_bluff_ev(context, amount, fold_equity):
            return self._with_reject_event(
                context,
                "negative_bluff_ev",
                amount=amount,
                fold_equity=fold_equity,
                required_fold_equity=required_fold_equity,
            )
        if self.mode == "threshold":
            discount = min(self.max_threshold_discount, self.threshold_discount + max(0.0, fold_equity - self.min_fold_equity) * 0.05)
            context = context.with_raise_threshold(context.raise_threshold - discount)
            return context.with_tool_event(
                {
                    "tool": self.name,
                    "decision": "threshold_discount",
                    "amount": int(amount),
                    "street": context.street,
                    "position": context.position,
                    "equity": context.equity,
                    "raise_threshold": context.raise_threshold,
                    "threshold_gap": context.raise_threshold - context.equity,
                    "threshold_discount": discount,
                    "fold_equity": fold_equity,
                    "required_fold_equity": required_fold_equity,
                    "pot_size": context.pot_size,
                    "stack_bb": context.stack_bb,
                    "payout_pressure": context.payout_pressure,
                    "active_players": context.active_players,
                }
            )
        return context.with_forced_action("raise", amount).with_tool_event(
            {
                "tool": self.name,
                "decision": "force_raise",
                "amount": amount,
                "street": context.street,
                "position": context.position,
                "equity": context.equity,
                "raise_threshold": context.raise_threshold,
                "threshold_gap": context.raise_threshold - context.equity,
                "fold_equity": fold_equity,
                "required_fold_equity": required_fold_equity,
                "pot_size": context.pot_size,
                "stack_bb": context.stack_bb,
                "payout_pressure": context.payout_pressure,
                "active_players": context.active_players,
            }
        )

    def _structural_rejection_reason(self, context: DecisionContext, game_state: Mapping[str, Any]) -> str:
        if context.forced_action is not None:
            return "already_forced"
        if context.street not in self.allowed_streets:
            return "street"
        if context.call_amount != 0:
            return "facing_bet"
        if context.active_players > self.max_active_players:
            return "multiway"
        if context.stack_bb < self.min_stack_bb:
            return "short_stack"
        if context.payout_pressure > self.max_payout_pressure and not self._can_apply_leverage_pressure(
            context, game_state
        ):
            return "payout_pressure"
        if context.pot_size <= 0 or context.max_raise_extra < max(1, context.min_raise):
            return "cannot_raise"
        if not (self.min_equity <= context.equity <= min(self.max_equity, context.raise_threshold - 0.01)):
            return "equity_window"
        if context.raise_threshold - context.equity > self.max_threshold_gap:
            return "threshold_gap"
        if self.avoid_low_spr and context.pot_size > 0:
            spr = context.stack_size / float(context.pot_size)
            if spr < self.min_spr:
                return "low_spr"
        return ""

    def _can_apply_leverage_pressure(self, context: DecisionContext, game_state: Mapping[str, Any]) -> bool:
        if context.payout_pressure > self.leverage_max_payout_pressure:
            return False
        if self.leverage_max_payout_pressure <= self.max_payout_pressure:
            return False
        if self._cover_fraction(context, game_state) >= self.leverage_cover_fraction_min:
            return True
        avg_stack = self._float(game_state.get("avg_table_stack"))
        if avg_stack <= 0:
            avg_stack = self._average_table_stack(game_state)
        if avg_stack <= 0:
            return False
        return context.stack_size >= avg_stack * self.leverage_stack_ratio_min

    def _has_survival_risk(self, context: DecisionContext, game_state: Mapping[str, Any], amount: int) -> bool:
        if self.survival_payout_pressure_min >= 1.0:
            return False
        if context.payout_pressure < self.survival_payout_pressure_min:
            return False
        if context.stack_size <= 0:
            return False
        risk_fraction = float(amount) / float(context.stack_size)
        if risk_fraction <= self.survival_max_risk_stack_fraction:
            return False
        if self._cover_fraction(context, game_state) >= self.survival_cover_fraction_min:
            return False
        avg_stack = self._float(game_state.get("avg_table_stack"))
        if avg_stack <= 0:
            avg_stack = self._average_table_stack(game_state)
        if avg_stack > 0 and context.stack_size >= avg_stack * self.survival_stack_ratio_min:
            return False
        return True

    def _has_stack_advantage(self, context: DecisionContext, game_state: Mapping[str, Any]) -> bool:
        if self._cover_fraction(context, game_state) >= self.stack_advantage_cover_fraction_min:
            return True
        avg_stack = self._float(game_state.get("avg_table_stack"))
        if avg_stack <= 0:
            avg_stack = self._average_table_stack(game_state)
        if avg_stack <= 0:
            return False
        return context.stack_size >= avg_stack * self.stack_advantage_stack_ratio_min

    def _cover_fraction(self, context: DecisionContext, game_state: Mapping[str, Any]) -> float:
        table_stacks = game_state.get("table_stacks")
        if not isinstance(table_stacks, list) or len(table_stacks) <= 1:
            return 0.0
        hero_index = self._int(game_state.get("hero_table_index"), default=-1)
        covered = 0
        opponents = 0
        for index, raw_stack in enumerate(table_stacks):
            if index == hero_index:
                continue
            stack = self._float(raw_stack)
            if stack <= 0:
                continue
            opponents += 1
            if context.stack_size >= stack:
                covered += 1
        if opponents <= 0:
            return 0.0
        return covered / float(opponents)

    def _average_table_stack(self, game_state: Mapping[str, Any]) -> float:
        table_stacks = game_state.get("table_stacks")
        if not isinstance(table_stacks, list) or not table_stacks:
            return 0.0
        stacks = [self._float(stack) for stack in table_stacks]
        stacks = [stack for stack in stacks if stack > 0]
        return sum(stacks) / len(stacks) if stacks else 0.0

    def _has_river_scare_card(self, context: DecisionContext, game_state: Mapping[str, Any]) -> bool:
        if context.street != 5:
            return True
        board_cards = game_state.get("board_cards")
        if not isinstance(board_cards, list) or len(board_cards) < 5:
            return False
        board = [self._card_texture(card) for card in board_cards[:5]]
        if any(card is None for card in board):
            return False
        prior = board[:4]
        river = board[4]
        assert river is not None
        prior_ranks = [card[0] for card in prior if card is not None]
        prior_suits = [card[1] for card in prior if card is not None and card[1] is not None]
        river_rank, river_suit = river

        if river_rank >= 12:
            return True
        if prior_ranks and river_rank > max(prior_ranks):
            return True
        if self._river_completes_flush(prior_suits, river_suit):
            return True
        if self._straight_window_count(prior_ranks) < self._straight_window_count(prior_ranks + [river_rank]):
            return True
        rank_counts = {rank: prior_ranks.count(rank) for rank in set(prior_ranks)}
        if rank_counts.get(river_rank, 0) > 0 and river_rank >= 10:
            return True
        return False

    def _card_texture(self, raw_card: Any) -> tuple[int, int | None] | None:
        try:
            card_int = int(raw_card)
        except (TypeError, ValueError):
            return None
        if 2 <= card_int <= 14:
            return card_int, None
        if Card is None:
            return None
        try:
            rank = int(Card.get_rank_int(card_int)) + 2
            suit = int(Card.get_suit_int(card_int))
        except Exception:
            return None
        if not (2 <= rank <= 14):
            return None
        return rank, suit

    def _river_completes_flush(self, prior_suits: list[int], river_suit: int | None) -> bool:
        if river_suit is None:
            return False
        return prior_suits.count(river_suit) == 2

    def _straight_window_count(self, ranks: Iterable[int]) -> int:
        rank_set = set(ranks)
        if 14 in rank_set:
            rank_set.add(1)
        count = 0
        for low in range(1, 11):
            if len(rank_set.intersection(range(low, low + 5))) >= 4:
                count += 1
        return count

    def _int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _float(self, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _with_reject_event(
        self,
        context: DecisionContext,
        reason: str,
        *,
        amount: int = 0,
        fold_equity: float | None = None,
        required_fold_equity: float | None = None,
    ) -> DecisionContext:
        event = {
            "tool": self.name,
            "decision": "reject",
            "reason": reason,
            "amount": int(amount),
            "street": context.street,
            "position": context.position,
            "equity": context.equity,
            "raise_threshold": context.raise_threshold,
            "threshold_gap": context.raise_threshold - context.equity,
            "pot_size": context.pot_size,
            "stack_bb": context.stack_bb,
            "payout_pressure": context.payout_pressure,
            "active_players": context.active_players,
        }
        if fold_equity is not None:
            event["fold_equity"] = fold_equity
        if required_fold_equity is not None:
            event["required_fold_equity"] = required_fold_equity
        return context.with_tool_event(event)

    def _fold_equity(self, context: DecisionContext, game_state: Mapping[str, Any]) -> float:
        stats = self._stats(game_state.get("opponent_stats")) or self._stats(game_state.get("table_stats"))
        if not stats:
            return 0.0
        quality = stats["sample_quality"]
        vpip = stats["vpip"]
        pfr = stats["pfr"]
        three_bet = stats["three_bet_rate"]

        tightness = max(0.0, 0.30 - vpip) * 0.85
        passivity = max(0.0, 0.18 - pfr) * 0.45
        low_reraise = max(0.0, 0.10 - three_bet) * 0.35
        score = 0.32 + tightness + passivity + low_reraise
        if context.position in {"BTN", "CO"}:
            score += self.position_bonus
        if self._has_initiative(context, game_state):
            score += self.initiative_bonus
        return max(0.0, min(1.0, score * quality))

    def _has_initiative(self, context: DecisionContext, game_state: Mapping[str, Any]) -> bool:
        if "has_initiative" in game_state:
            return bool(game_state.get("has_initiative"))
        return context.spot_type in {"srp", "single_raised"} and context.position in {"BTN", "CO", "HJ"}

    def _has_position_or_initiative(self, context: DecisionContext, game_state: Mapping[str, Any]) -> bool:
        return context.position in {"BTN", "CO"} or self._has_initiative(context, game_state)

    def _has_positive_bluff_ev(self, context: DecisionContext, amount: int, fold_equity: float) -> bool:
        pot = float(context.pot_size)
        bet = float(amount)
        if pot <= 0 or bet <= 0:
            return False
        called_ev = context.equity * (pot + bet) - (1.0 - context.equity) * bet
        total_ev = fold_equity * pot + (1.0 - fold_equity) * called_ev
        return total_ev >= pot * self.min_ev_edge_pot_fraction

    def _required_fold_equity(self, context: DecisionContext, amount: int) -> float:
        pot = float(context.pot_size)
        risk = float(amount)
        if pot <= 0.0 or risk <= 0.0:
            return 1.0
        pure_bluff_required = risk / (risk + pot)
        equity_credit = min(
            self.max_equity_fold_equity_credit,
            max(0.0, context.equity) * self.equity_fold_equity_credit,
        )
        required = pure_bluff_required + self.required_fold_equity_safety_margin - equity_credit
        return max(self.min_fold_equity, min(1.0, required))

    def _raise_amount(self, context: DecisionContext) -> int:
        fraction = self._bet_fraction(context)
        target = int(max(context.big_blind, context.pot_size * fraction))
        target = max(context.min_raise, target)
        if self.max_stack_fraction > 0:
            target = min(target, int(context.stack_size * self.max_stack_fraction))
        return min(context.max_raise_extra, max(context.min_raise, target))

    def _bet_fraction(self, context: DecisionContext) -> float:
        if context.street == 3:
            return self.flop_size
        if context.street == 5 and self.river_size_min is not None and self.river_size_max is not None:
            if self.river_size_max <= self.river_size_min:
                return self.river_size_min
            gap = max(0.0, context.raise_threshold - context.equity)
            pressure = gap / self.max_threshold_gap if self.max_threshold_gap > 0 else 1.0
            pressure = max(0.0, min(1.0, pressure))
            return self.river_size_min + (self.river_size_max - self.river_size_min) * pressure
        return self.turn_size

    def _stats(self, raw_stats: Any) -> Dict[str, float] | None:
        if not isinstance(raw_stats, dict):
            return None
        try:
            quality = max(0.0, min(1.0, float(raw_stats.get("sample_quality", 0.0) or 0.0)))
            if quality < self.sample_quality_min:
                return None
            return {
                "sample_quality": quality,
                "vpip": max(0.0, min(1.0, float(raw_stats.get("vpip", 0.25) or 0.0))),
                "pfr": max(0.0, min(1.0, float(raw_stats.get("pfr", 0.16) or 0.0))),
                "three_bet_rate": max(0.0, min(1.0, float(raw_stats.get("three_bet_rate", 0.08) or 0.0))),
            }
        except (TypeError, ValueError):
            return None


TOOL_REGISTRY = {
    "icm_pressure": ICMPressureTool,
    "ICMPressureTool": ICMPressureTool,
    "preflop_reraise_tightness": PreflopReraiseTightnessTool,
    "PreflopReraiseTightnessTool": PreflopReraiseTightnessTool,
    "table_adaptation": TableAdaptationTool,
    "TableAdaptationTool": TableAdaptationTool,
    "button_steal": ButtonStealTool,
    "ButtonStealTool": ButtonStealTool,
    "endgame_conversion": EndgameConversionTool,
    "EndgameConversionTool": EndgameConversionTool,
    "cbet_pressure": ContinuationPressureTool,
    "ContinuationPressureTool": ContinuationPressureTool,
    "bluff_pressure": BluffPressureTool,
    "BluffPressureTool": BluffPressureTool,
}


def register_bot_tool(cls: type, *, override: bool = False) -> type:
    """Register a BotTool class so it is usable from config `params.tools[]`.

    Registers under both the snake-case tool type (``cls.name``) and the class
    name (``cls.__name__``). Idempotent: re-registering the same class is a
    no-op. Raises on a genuine key clash unless ``override=True``. Returns
    ``cls`` so it can be used as a decorator.
    """
    snake = getattr(cls, "name", None)
    pascal = cls.__name__
    if not snake:
        raise ValueError(f"{pascal} must define a class attr `name` (snake_case tool type)")
    for key in (snake, pascal):
        existing = TOOL_REGISTRY.get(key)
        if existing is not None and existing is not cls and not override:
            raise ValueError(
                f"bot tool key {key!r} already registered to {existing.__name__}; "
                f"pass override=True to replace"
            )
        TOOL_REGISTRY[key] = cls
    return cls


def _tool_param_names(cls: type) -> list[str]:
    """Keyword param names accepted by a tool's ``__init__`` (excluding self)."""
    signature = inspect.signature(cls.__init__)
    return [
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    ]


def available_decision_tools() -> Dict[str, Dict[str, Any]]:
    catalog: Dict[str, Dict[str, Any]] = {}
    seen: set[type] = set()
    for cls in TOOL_REGISTRY.values():
        if cls in seen:
            continue
        seen.add(cls)
        catalog[cls.name] = {"class": cls.__name__, "params": _tool_param_names(cls)}
    return catalog


def build_bot_tools(specs: Iterable[Mapping[str, Any] | str] | None) -> list[BotTool]:
    tools: list[BotTool] = []
    for raw_spec in specs or []:
        if isinstance(raw_spec, str):
            spec: Mapping[str, Any] = {"type": raw_spec}
        else:
            spec = dict(raw_spec)
        if not bool(spec.get("enabled", True)):
            continue
        tool_type = str(spec.get("type") or spec.get("tool") or spec.get("name") or "")
        try:
            tool_class = TOOL_REGISTRY[tool_type]
        except KeyError as exc:
            available = ", ".join(sorted(TOOL_REGISTRY))
            raise ValueError(f"unknown bot tool {tool_type!r}; available tools: {available}") from exc
        params = dict(spec.get("params", {}) or {})
        params.update({
            key: value
            for key, value in spec.items()
            if key not in {"type", "tool", "name", "params"} and value is not None
        })
        tools.append(tool_class(**params))
    return sorted(tools, key=lambda tool: tool.priority)
