from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Mapping, Protocol

from engine.icm import calculate_exact_icm


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
        return replace(self, tool_event=dict(event))


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
        max_stack_fraction: float = 0.18,
        position_bonus: float = 0.06,
        initiative_bonus: float = 0.05,
        require_position_or_initiative: bool = True,
        min_ev_edge_pot_fraction: float = 0.03,
        mode: str = "force_raise",
        threshold_discount: float = 0.035,
        max_threshold_discount: float = 0.055,
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
        self.max_stack_fraction = max(0.0, min(1.0, float(max_stack_fraction)))
        self.position_bonus = max(0.0, float(position_bonus))
        self.initiative_bonus = max(0.0, float(initiative_bonus))
        self.require_position_or_initiative = bool(require_position_or_initiative)
        self.min_ev_edge_pot_fraction = max(0.0, float(min_ev_edge_pot_fraction))
        self.mode = str(mode)
        if self.mode not in {"force_raise", "threshold"}:
            raise ValueError("bluff_pressure mode must be 'force_raise' or 'threshold'")
        self.threshold_discount = max(0.0, min(0.20, float(threshold_discount)))
        self.max_threshold_discount = max(self.threshold_discount, min(0.20, float(max_threshold_discount)))

    def apply(self, context: DecisionContext, game_state: Mapping[str, Any]) -> DecisionContext:
        if not self.enabled:
            return context
        structural_reject = self._structural_rejection_reason(context)
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
        if not self._has_positive_bluff_ev(context, amount, fold_equity):
            return self._with_reject_event(context, "negative_bluff_ev", amount=amount, fold_equity=fold_equity)
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
                "pot_size": context.pot_size,
                "stack_bb": context.stack_bb,
                "payout_pressure": context.payout_pressure,
                "active_players": context.active_players,
            }
        )

    def _structural_rejection_reason(self, context: DecisionContext) -> str:
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
        if context.payout_pressure > self.max_payout_pressure:
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

    def _with_reject_event(
        self,
        context: DecisionContext,
        reason: str,
        *,
        amount: int = 0,
        fold_equity: float | None = None,
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

    def _raise_amount(self, context: DecisionContext) -> int:
        fraction = self.flop_size if context.street == 3 else self.turn_size
        target = int(max(context.big_blind, context.pot_size * fraction))
        target = max(context.min_raise, target)
        if self.max_stack_fraction > 0:
            target = min(target, int(context.stack_size * self.max_stack_fraction))
        return min(context.max_raise_extra, max(context.min_raise, target))

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
    "bluff_pressure": BluffPressureTool,
    "BluffPressureTool": BluffPressureTool,
}


def available_decision_tools() -> Dict[str, Dict[str, Any]]:
    return {
        "icm_pressure": {
            "class": ICMPressureTool.__name__,
            "params": [
                "priority",
                "enabled",
                "strength",
                "exact_when_available",
                "call_margin_weight",
                "raise_threshold_weight",
                "jam_threshold_weight",
            ],
        },
        "preflop_reraise_tightness": {
            "class": PreflopReraiseTightnessTool.__name__,
            "params": ["priority", "enabled", "tightness"],
        },
        "table_adaptation": {
            "class": TableAdaptationTool.__name__,
            "params": [
                "priority",
                "enabled",
                "sample_quality_min",
                "loose_passive_vpip",
                "loose_passive_pfr",
                "aggressive_pfr",
                "aggressive_three_bet_rate",
            ],
        },
        "button_steal": {
            "class": ButtonStealTool.__name__,
            "params": [
                "priority",
                "enabled",
                "sample_quality_min",
                "max_vpip",
                "max_pfr",
                "max_three_bet_rate",
                "base_discount",
                "tightness_multiplier",
                "max_discount",
                "max_open_stack_fraction",
                "max_active_players",
                "min_stack_bb",
                "max_payout_pressure",
                "requires_table_sample",
            ],
        },
        "endgame_conversion": {
            "class": EndgameConversionTool.__name__,
            "params": [
                "priority",
                "enabled",
                "final_table_players",
                "top3_players",
                "min_stack_bb",
                "big_stack_avg_multiplier",
                "cover_fraction_min",
                "raise_discount",
                "jam_discount",
                "call_margin_discount",
                "top3_multiplier",
                "heads_up_multiplier",
                "protect_bubble",
            ],
        },
        "bluff_pressure": {
            "class": BluffPressureTool.__name__,
            "params": [
                "priority",
                "enabled",
                "sample_quality_min",
                "min_fold_equity",
                "min_equity",
                "max_equity",
                "max_threshold_gap",
                "max_payout_pressure",
                "min_stack_bb",
                "max_active_players",
                "avoid_low_spr",
                "min_spr",
                "allowed_streets",
                "flop_size",
                "turn_size",
                "max_stack_fraction",
                "position_bonus",
                "initiative_bonus",
                "require_position_or_initiative",
                "min_ev_edge_pot_fraction",
                "mode",
                "threshold_discount",
                "max_threshold_discount",
            ],
        },
    }


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
