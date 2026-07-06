"""Evolutionary MTT experiment for reduced basemodel checkpoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from engine.baseline_model_bot import add_basemodel_to_path
from engine.phase22_small_mtt_engine_simulation import (
    _engine_overrides,
    _seed_everything,
    _write_json,
    temporary_engine_config,
)
from engine.reduced_model_bot import ReducedModelEngineBot
from engine.tournament import Tournament


DEFAULT_BLINDS = [
    {"small": 25, "big": 50},
    {"small": 50, "big": 100},
    {"small": 75, "big": 150},
    {"small": 100, "big": 200},
    {"small": 150, "big": 300},
    {"small": 200, "big": 400},
    {"small": 250, "big": 500},
    {"small": 300, "big": 600},
    {"small": 400, "big": 800},
    {"small": 500, "big": 1000},
    {"small": 600, "big": 1200},
    {"small": 800, "big": 1600},
    {"small": 1000, "big": 2000},
    {"small": 1500, "big": 3000},
    {"small": 2000, "big": 4000},
    {"small": 3000, "big": 6000},
    {"small": 4000, "big": 8000},
]

DEFAULT_PAYOUTS = {
    "1": 0.237,
    "2": 0.150,
    "3": 0.103,
    "4": 0.074,
    "5": 0.055,
    "6": 0.043,
    "7": 0.034,
    "8": 0.028,
    "9": 0.023,
    "10": 0.019,
    "11": 0.019,
    "12": 0.016,
    "13": 0.016,
    "14": 0.014,
    "15": 0.014,
    "16": 0.012,
    "17": 0.012,
    "18": 0.011,
    "19": 0.011,
    "20": 0.010,
    "21": 0.010,
    "22": 0.009,
    "23": 0.009,
    "24": 0.008,
    "25": 0.008,
    "26": 0.007,
}


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    checkpoint_path: Path
    generation: int = 0
    parent_id: str = ""
    root_id: str = ""


def candidate_lineage(candidate: Candidate) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "generation": int(candidate.generation),
        "parent_id": candidate.parent_id,
        "root_id": candidate.root_id or candidate.candidate_id,
        "checkpoint_path": str(candidate.checkpoint_path),
    }


def candidate_number(candidate_id: str) -> int:
    prefix = "candidate_"
    if not candidate_id.startswith(prefix):
        return 0
    try:
        return int(candidate_id.removeprefix(prefix))
    except ValueError:
        return 0


def next_candidate_index(candidates: Iterable[Candidate]) -> int:
    return max((candidate_number(candidate.candidate_id) for candidate in candidates), default=0) + 1


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def default_engine_config() -> Dict[str, Any]:
    return {
        "bot_decision_timeout_ms": 500,
        "starting_stack": 10000,
        "max_players_per_table": 9,
        "hands_per_level": 12,
        "max_hands_per_tournament": 3000,
        "blinds_schedule": DEFAULT_BLINDS,
        "payouts": DEFAULT_PAYOUTS,
    }


def load_json_config(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def basemodel_root_from_engine(engine_root: Path, configured: str | Path = "../poker-ai-basemodel") -> Path:
    root = Path(configured).expanduser()
    if not root.is_absolute():
        root = (engine_root / root).resolve()
    add_basemodel_to_path(root)
    return root


def observation_contract(basemodel_root: Path, schema: str) -> tuple[int, tuple[str, ...]]:
    add_basemodel_to_path(basemodel_root)
    from poker_ai.reduced_observations import reduced_observation_contract

    size, fields, _encoder = reduced_observation_contract(schema)
    return int(size), tuple(fields)


def create_random_checkpoint(
    path: str | Path,
    *,
    basemodel_root: Path,
    observation_size: int,
    seed: int,
) -> Path:
    add_basemodel_to_path(basemodel_root)
    import torch

    from poker_ai.checkpoint import save_checkpoint
    from poker_ai.model import PokerActorCritic

    torch.manual_seed(int(seed))
    model = PokerActorCritic(observation_size=observation_size)
    return save_checkpoint(path, model, config={"source": "evolutionary_random", "seed": int(seed)})


def mutate_checkpoint(
    source_path: str | Path,
    target_path: str | Path,
    *,
    basemodel_root: Path,
    observation_size: int,
    seed: int,
    sigma: float = 0.02,
    policy_head_only: bool = True,
) -> Path:
    add_basemodel_to_path(basemodel_root)
    import torch

    from poker_ai.checkpoint import load_checkpoint, save_checkpoint
    from poker_ai.model import PokerActorCritic

    torch.manual_seed(int(seed))
    model = PokerActorCritic(observation_size=observation_size)
    payload = load_checkpoint(source_path, model, map_location="cpu")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if policy_head_only and not name.startswith("policy_head."):
                continue
            parameter.add_(torch.randn_like(parameter) * float(sigma))
    config = dict(payload.get("config") or {})
    config.update(
        {
            "source": "evolutionary_mutation",
            "parent_checkpoint": str(source_path),
            "seed": int(seed),
            "sigma": float(sigma),
            "policy_head_only": bool(policy_head_only),
        }
    )
    return save_checkpoint(target_path, model, config=config)


def build_candidate_lineup(
    candidates: Iterable[Candidate],
    *,
    entries_per_candidate: int,
    basemodel_root: Path,
    engine_root: Path,
    observation_schema: str,
    observation_size: int,
    deterministic: bool,
    decision_timeout_ms: int,
    equity_source: str,
    equity_fallback_source: str | None,
    equity_iterations: int | None,
    use_preflop_spot_range: bool,
) -> tuple[List[Any], Dict[str, str]]:
    bots: List[Any] = []
    name_to_candidate: Dict[str, str] = {}
    for candidate in candidates:
        prototype = ReducedModelEngineBot(
            candidate.checkpoint_path,
            basemodel_root=basemodel_root,
            engine_root=engine_root,
            name=f"{candidate.candidate_id}_entry_000",
            deterministic=deterministic,
            decision_timeout_ms=decision_timeout_ms,
            equity_source=equity_source,
            equity_fallback_source=equity_fallback_source,
            equity_iterations=equity_iterations,
            use_preflop_spot_range=use_preflop_spot_range,
            observation_size=observation_size,
            observation_schema=observation_schema,
            require_checkpoint=True,
        )
        for entry_index in range(entries_per_candidate):
            bot = copy.copy(prototype)
            bot.name = f"{candidate.candidate_id}_entry_{entry_index + 1:03d}"
            bot.fallback_counts = {"no_model": 0, "inference_errors": 0, "illegal_actions": 0, "timeouts": 0}
            bot.equity_counts = {"computed": 0, "fallbacks": 0}
            bots.append(bot)
            name_to_candidate[bot.name] = candidate.candidate_id
    return bots, name_to_candidate


def summarize_candidate_results(results: Iterable[Dict[str, Any]], name_to_candidate: Dict[str, str]) -> Dict[str, Any]:
    stats: Dict[str, Dict[str, Any]] = {}
    for row in results:
        name = str(row.get("name", ""))
        candidate_id = name_to_candidate.get(name)
        if not candidate_id:
            continue
        position = int(row.get("position", 0) or 0)
        payout = float(row.get("payout_pct", 0.0) or 0.0)
        summary = stats.setdefault(
            candidate_id,
            {
                "entries": 0,
                "wins": 0,
                "top3": 0,
                "final_table": 0,
                "itm": 0,
                "position_sum": 0.0,
                "total_payout_pct": 0.0,
            },
        )
        summary["entries"] += 1
        summary["wins"] += 1 if position == 1 else 0
        summary["top3"] += 1 if 1 <= position <= 3 else 0
        summary["final_table"] += 1 if 1 <= position <= 9 else 0
        summary["itm"] += 1 if payout > 0.0 else 0
        summary["position_sum"] += position
        summary["total_payout_pct"] += payout

    for summary in stats.values():
        entries = max(1, int(summary["entries"]))
        summary["average_position"] = summary["position_sum"] / entries
        summary["win_rate"] = summary["wins"] / entries
        summary["top3_rate"] = summary["top3"] / entries
        summary["final_table_rate"] = summary["final_table"] / entries
        summary["itm_rate"] = summary["itm"] / entries
        summary["average_payout_pct"] = summary["total_payout_pct"] / entries
        summary["score"] = float(summary["final_table"]) + 3.0 * float(summary["top3"]) + 8.0 * float(summary["wins"])
    return stats


def summarize_population_results(results: Iterable[Dict[str, Any]], name_to_population: Dict[str, str]) -> Dict[str, Any]:
    return summarize_candidate_results(results, name_to_population)


def summarize_candidate_actions(events: Iterable[Dict[str, Any]], name_to_candidate: Dict[str, str]) -> Dict[str, Any]:
    stats: Dict[str, Dict[str, Any]] = {}
    dealt_hands: Dict[str, set] = {}
    vpip_hands: Dict[str, set] = {}
    pfr_hands: Dict[str, set] = {}
    preflop_call_hands: Dict[str, set] = {}
    three_bet_opportunity_hands: Dict[str, set] = {}
    three_bet_hands: Dict[str, set] = {}
    faced_three_bet_hands: Dict[str, set] = {}
    folded_to_three_bet_hands: Dict[str, set] = {}
    cbet_opportunity_hands: Dict[str, set] = {}
    cbet_hands: Dict[str, set] = {}
    saw_flop_hands: Dict[str, set] = {}
    showdown_hands: Dict[str, set] = {}
    won_showdown_hands: Dict[str, set] = {}
    preflop_raise_count_by_hand: Dict[tuple, int] = {}
    preflop_opener_by_hand: Dict[tuple, str] = {}
    preflop_aggressor_by_hand: Dict[tuple, str] = {}
    flop_bet_seen_by_hand: Dict[tuple, bool] = {}
    pending_tool_raises: List[Dict[str, Any]] = []

    def default_summary() -> Dict[str, Any]:
        return {
            "action_total": 0,
            "action_counts": {},
            "street_action_counts": {},
            "amount_total": 0,
            "raise_amount_total": 0,
            "raise_count": 0,
            "postflop_raise_count": 0,
            "postflop_call_count": 0,
            "vpip_hands": 0,
            "vpip_count": 0,
            "pfr_count": 0,
            "preflop_call_count": 0,
            "three_bet_opportunity_count": 0,
            "three_bet_count": 0,
            "faced_three_bet_count": 0,
            "folded_to_three_bet_count": 0,
            "cbet_opportunity_count": 0,
            "cbet_count": 0,
            "saw_flop_count": 0,
            "showdown_count": 0,
            "won_showdown_count": 0,
            "tool_event_counts": {},
            "tool_decision_counts": {},
            "tool_reject_reason_counts": {},
            "tool_action_counts": {},
            "tool_action_amount_total": {},
            "tool_action_context_totals": {},
            "tool_action_context_averages": {},
            "tool_response_counts": {},
            "tool_response_amounts": {},
            "tool_response_rates": {},
            "river_betting_range_counts": {},
            "river_betting_range_rates": {},
        }

    def river_betting_range_counts(summary: Dict[str, Any]) -> Dict[str, Any]:
        return summary.setdefault(
            "river_betting_range_counts",
            {
                "no_facing_raise_total": 0,
                "no_facing_bluff_raise": 0,
                "no_facing_non_bluff_raise": 0,
                "size_buckets": {},
            },
        )

    def river_size_bucket(amount: int, pot_size: int) -> str:
        if pot_size <= 0:
            return "unknown"
        size_ratio = float(amount) / float(pot_size)
        if size_ratio <= 0.50:
            return "le_50"
        if size_ratio <= 0.75:
            return "le_75"
        if size_ratio <= 1.00:
            return "le_100"
        return "gt_100"

    def response_counts(summary: Dict[str, Any], tool_name: str) -> Dict[str, int]:
        return summary["tool_response_counts"].setdefault(
            tool_name,
            {
                "raises": 0,
                "opponent_responses": 0,
                "opponents_folded": 0,
                "opponents_called": 0,
                "opponents_raised_over": 0,
                "won_immediately": 0,
                "called_or_raised": 0,
                "no_response": 0,
                "unresolved": 0,
            },
        )

    def response_amounts(summary: Dict[str, Any], tool_name: str) -> Dict[str, float]:
        return summary.setdefault("tool_response_amounts", {}).setdefault(
            tool_name,
            {
                "risk_amount_total": 0.0,
                "pot_before_total": 0.0,
                "won_immediately_pot_total": 0.0,
                "called_risk_amount_total": 0.0,
                "raised_over_risk_amount_total": 0.0,
                "opponent_call_amount_total": 0.0,
                "opponent_raise_over_amount_total": 0.0,
            },
        )

    def finalize_tool_raise(pending: Dict[str, Any], reason: str) -> None:
        if pending.get("closed"):
            return
        pending["closed"] = True
        candidate_id = str(pending.get("candidate_id", ""))
        tool_name = str(pending.get("tool_name", ""))
        if not candidate_id or not tool_name:
            return
        summary = stats.setdefault(candidate_id, default_summary())
        counts = response_counts(summary, tool_name)
        amounts = response_amounts(summary, tool_name)
        responses = int(pending.get("responses", 0) or 0)
        calls = int(pending.get("calls", 0) or 0)
        raises_over = int(pending.get("raises_over", 0) or 0)
        folds = int(pending.get("folds", 0) or 0)
        risk_amount = float(pending.get("risk_amount", 0.0) or 0.0)
        pot_before = float(pending.get("pot_before", 0.0) or 0.0)
        if responses <= 0:
            counts["no_response"] += 1
        elif calls == 0 and raises_over == 0 and folds == responses:
            counts["won_immediately"] += 1
            amounts["won_immediately_pot_total"] += pot_before
        elif calls > 0 or raises_over > 0:
            counts["called_or_raised"] += 1
            if calls > 0:
                amounts["called_risk_amount_total"] += risk_amount
            if raises_over > 0:
                amounts["raised_over_risk_amount_total"] += risk_amount
        else:
            counts["unresolved"] += 1

    def flush_tool_raises_until(hand_key: tuple, street: str) -> None:
        for pending in pending_tool_raises:
            if pending.get("closed"):
                continue
            if pending.get("hand_key") != hand_key or pending.get("street") != street:
                finalize_tool_raise(pending, "street_changed")

    for event in events:
        event_type = str(event.get("type", ""))
        if event_type not in {"action", "deal", "showdown", "award_pot"}:
            continue
        hand_key = (
            int(event.get("tournament_id", 0) or 0),
            int(event.get("table_id", 0) or 0),
            int(event.get("hand_id", 0) or 0),
        )
        player_name = str(event.get("player", ""))
        player_hand_key = (*hand_key, player_name)
        if event_type != "action":
            flush_tool_raises_until(hand_key, "")
        candidate_id = name_to_candidate.get(player_name)
        if not candidate_id:
            continue
        if event_type == "deal":
            dealt_hands.setdefault(candidate_id, set()).add(player_hand_key)
            continue
        if event_type == "showdown":
            saw_flop_hands.setdefault(candidate_id, set()).add(player_hand_key)
            showdown_hands.setdefault(candidate_id, set()).add(player_hand_key)
            continue
        if event_type == "award_pot":
            if bool(event.get("showdown", False)):
                won_showdown_hands.setdefault(candidate_id, set()).add(player_hand_key)
            continue

        action = str(event.get("action", "unknown") or "unknown")
        amount = int(event.get("amount", 0) or 0)
        if action == "call" and amount == 0:
            action = "check"
        street = str(event.get("street", ""))
        flush_tool_raises_until(hand_key, street)
        for pending in pending_tool_raises:
            if pending.get("closed"):
                continue
            if pending.get("hand_key") != hand_key or pending.get("street") != street:
                continue
            if pending.get("player") == player_name:
                continue
            if player_name in pending.setdefault("responded_players", set()):
                continue
            if int(event.get("call_amount", 0) or 0) <= 0:
                continue
            pending["responded_players"].add(player_name)
            pending["responses"] = int(pending.get("responses", 0) or 0) + 1
            pending_candidate_id = str(pending.get("candidate_id", ""))
            pending_tool_name = str(pending.get("tool_name", ""))
            if pending_candidate_id and pending_tool_name:
                pending_summary = stats.setdefault(pending_candidate_id, default_summary())
                counts = response_counts(pending_summary, pending_tool_name)
                amounts = response_amounts(pending_summary, pending_tool_name)
                counts["opponent_responses"] += 1
                if action == "fold":
                    pending["folds"] = int(pending.get("folds", 0) or 0) + 1
                    counts["opponents_folded"] += 1
                elif action == "raise":
                    pending["raises_over"] = int(pending.get("raises_over", 0) or 0) + 1
                    counts["opponents_raised_over"] += 1
                    amounts["opponent_raise_over_amount_total"] += amount
                    finalize_tool_raise(pending, "raised_over")
                else:
                    pending["calls"] = int(pending.get("calls", 0) or 0) + 1
                    counts["opponents_called"] += 1
                    amounts["opponent_call_amount_total"] += amount
        if street == "preflop" and action in {"call", "raise"} and amount > 0:
            vpip_hands.setdefault(candidate_id, set()).add(player_hand_key)
            if action == "raise":
                pfr_hands.setdefault(candidate_id, set()).add(player_hand_key)
            elif action == "call":
                preflop_call_hands.setdefault(candidate_id, set()).add(player_hand_key)
        if street == "preflop":
            raise_count_before = int(preflop_raise_count_by_hand.get(hand_key, 0))
            opener_player = preflop_opener_by_hand.get(hand_key)
            if raise_count_before == 1 and int(event.get("call_amount", 0) or 0) > 0 and player_name != opener_player:
                three_bet_opportunity_hands.setdefault(candidate_id, set()).add(player_hand_key)
            if action == "fold" and player_hand_key in faced_three_bet_hands.get(candidate_id, set()):
                folded_to_three_bet_hands.setdefault(candidate_id, set()).add(player_hand_key)
            if action == "raise" and amount > 0:
                if raise_count_before == 0:
                    preflop_opener_by_hand[hand_key] = player_name
                elif raise_count_before == 1:
                    three_bet_hands.setdefault(candidate_id, set()).add(player_hand_key)
                    if opener_player and opener_player != player_name:
                        opener_candidate_id = name_to_candidate.get(opener_player)
                        if opener_candidate_id:
                            faced_three_bet_hands.setdefault(opener_candidate_id, set()).add((*hand_key, opener_player))
                preflop_raise_count_by_hand[hand_key] = raise_count_before + 1
                preflop_aggressor_by_hand[hand_key] = player_name
        elif street in {"flop", "turn", "river"}:
            saw_flop_hands.setdefault(candidate_id, set()).add(player_hand_key)
            if street == "flop":
                preflop_aggressor_player = preflop_aggressor_by_hand.get(hand_key)
                flop_bet_seen = bool(flop_bet_seen_by_hand.get(hand_key, False))
                if preflop_aggressor_player and player_name == preflop_aggressor_player and not flop_bet_seen and int(event.get("call_amount", 0) or 0) == 0:
                    cbet_opportunity_hands.setdefault(candidate_id, set()).add(player_hand_key)
                    if action == "raise" and amount > 0:
                        cbet_hands.setdefault(candidate_id, set()).add(player_hand_key)
                if action == "raise" and amount > 0:
                    flop_bet_seen_by_hand[hand_key] = True

        summary = stats.setdefault(candidate_id, default_summary())
        summary["action_total"] += 1
        summary["action_counts"][action] = int(summary["action_counts"].get(action, 0)) + 1
        street_counts = summary["street_action_counts"].setdefault(street or "unknown", {})
        street_counts[action] = int(street_counts.get(action, 0)) + 1
        summary["amount_total"] += amount
        if action == "raise":
            summary["raise_count"] += 1
            summary["raise_amount_total"] += amount
            if street != "preflop":
                summary["postflop_raise_count"] += 1
        elif action == "call" and street != "preflop":
            summary["postflop_call_count"] += 1

        tool_event = event.get("tool_event")
        tool_events = event.get("tool_events")
        if not isinstance(tool_events, list) or not tool_events:
            tool_events = [tool_event] if isinstance(tool_event, dict) else []
        is_tool_bluff_raise = any(
            isinstance(item, dict)
            and str(item.get("tool", "") or "") == "bluff_pressure"
            and str(item.get("decision", "") or "") == "force_raise"
            for item in tool_events
        )
        if street == "river" and action == "raise" and amount > 0 and int(event.get("call_amount", 0) or 0) == 0:
            river_counts = river_betting_range_counts(summary)
            river_counts["no_facing_raise_total"] = int(river_counts.get("no_facing_raise_total", 0)) + 1
            bluff_key = "no_facing_bluff_raise" if is_tool_bluff_raise else "no_facing_non_bluff_raise"
            river_counts[bluff_key] = int(river_counts.get(bluff_key, 0)) + 1
            bucket = river_size_bucket(amount, int(event.get("pot_size", 0) or 0))
            bucket_counts = river_counts.setdefault("size_buckets", {}).setdefault(
                bucket,
                {
                    "no_facing_raise_total": 0,
                    "no_facing_bluff_raise": 0,
                    "no_facing_non_bluff_raise": 0,
                },
            )
            bucket_counts["no_facing_raise_total"] = int(bucket_counts.get("no_facing_raise_total", 0)) + 1
            bucket_counts[bluff_key] = int(bucket_counts.get(bluff_key, 0)) + 1

        for tool_event in tool_events:
            if not isinstance(tool_event, dict):
                continue
            tool_name = str(tool_event.get("tool", "") or "")
            if tool_name:
                decision = str(tool_event.get("decision", "") or "unknown")
                reason = str(tool_event.get("reason", "") or "")
                summary["tool_event_counts"][tool_name] = int(summary["tool_event_counts"].get(tool_name, 0)) + 1
                tool_decisions = summary["tool_decision_counts"].setdefault(tool_name, {})
                tool_decisions[decision] = int(tool_decisions.get(decision, 0)) + 1
                if reason:
                    tool_reasons = summary["tool_reject_reason_counts"].setdefault(tool_name, {})
                    tool_reasons[reason] = int(tool_reasons.get(reason, 0)) + 1
                if decision == "force_raise":
                    summary["tool_action_counts"][tool_name] = int(summary["tool_action_counts"].get(tool_name, 0)) + 1
                    summary["tool_action_amount_total"][tool_name] = int(summary["tool_action_amount_total"].get(tool_name, 0)) + amount
                    tool_totals = summary["tool_action_context_totals"].setdefault(tool_name, {})
                    for key in (
                        "equity",
                        "raise_threshold",
                        "threshold_gap",
                        "fold_equity",
                        "pot_size",
                        "stack_bb",
                        "payout_pressure",
                        "active_players",
                    ):
                        try:
                            tool_totals[key] = float(tool_totals.get(key, 0.0)) + float(tool_event.get(key, 0.0) or 0.0)
                        except (TypeError, ValueError):
                            pass
                    pending_tool_raises.append(
                        {
                            "candidate_id": candidate_id,
                            "tool_name": tool_name,
                            "hand_key": hand_key,
                            "street": street,
                            "player": player_name,
                            "risk_amount": amount,
                            "pot_before": int(event.get("pot_size", 0) or 0),
                            "responded_players": set(),
                            "responses": 0,
                            "folds": 0,
                            "calls": 0,
                            "raises_over": 0,
                            "closed": False,
                        }
                    )
                    response_counts(summary, tool_name)["raises"] += 1
                    response_amount = response_amounts(summary, tool_name)
                    response_amount["risk_amount_total"] += amount
                    response_amount["pot_before_total"] += int(event.get("pot_size", 0) or 0)

    for pending in pending_tool_raises:
        if not pending.get("closed"):
            finalize_tool_raise(pending, "end_of_events")

    action_candidate_ids = (
        set(stats)
        | set(dealt_hands)
        | set(vpip_hands)
        | set(pfr_hands)
        | set(preflop_call_hands)
        | set(three_bet_opportunity_hands)
        | set(three_bet_hands)
        | set(faced_three_bet_hands)
        | set(folded_to_three_bet_hands)
        | set(cbet_opportunity_hands)
        | set(cbet_hands)
        | set(saw_flop_hands)
        | set(showdown_hands)
        | set(won_showdown_hands)
    )
    for candidate_id in action_candidate_ids:
        summary = stats.setdefault(candidate_id, default_summary())
        summary["vpip_hands"] = len(dealt_hands.get(candidate_id, set()))
        summary["vpip_count"] = len(vpip_hands.get(candidate_id, set()))
        summary["pfr_count"] = len(pfr_hands.get(candidate_id, set()))
        summary["preflop_call_count"] = len(preflop_call_hands.get(candidate_id, set()))
        summary["three_bet_opportunity_count"] = len(three_bet_opportunity_hands.get(candidate_id, set()))
        summary["three_bet_count"] = len(three_bet_hands.get(candidate_id, set()))
        summary["faced_three_bet_count"] = len(faced_three_bet_hands.get(candidate_id, set()))
        summary["folded_to_three_bet_count"] = len(folded_to_three_bet_hands.get(candidate_id, set()))
        summary["cbet_opportunity_count"] = len(cbet_opportunity_hands.get(candidate_id, set()))
        summary["cbet_count"] = len(cbet_hands.get(candidate_id, set()))
        summary["saw_flop_count"] = len(saw_flop_hands.get(candidate_id, set()))
        summary["showdown_count"] = len(showdown_hands.get(candidate_id, set()))
        summary["won_showdown_count"] = len(won_showdown_hands.get(candidate_id, set()))

    for summary in stats.values():
        total = max(1, int(summary["action_total"]))
        counts = dict(summary["action_counts"])
        summary["action_rates"] = {action: count / total for action, count in sorted(counts.items())}
        summary["average_amount"] = float(summary["amount_total"]) / total
        summary["average_raise_amount"] = (
            float(summary["raise_amount_total"]) / int(summary["raise_count"])
            if int(summary["raise_count"]) > 0
            else 0.0
        )
        summary["vpip"] = (
            float(summary["vpip_count"]) / int(summary["vpip_hands"])
            if int(summary["vpip_hands"]) > 0
            else 0.0
        )
        summary["pfr"] = (
            float(summary["pfr_count"]) / int(summary["vpip_hands"])
            if int(summary["vpip_hands"]) > 0
            else 0.0
        )
        summary["preflop_call_rate"] = (
            float(summary["preflop_call_count"]) / int(summary["vpip_hands"])
            if int(summary["vpip_hands"]) > 0
            else 0.0
        )
        postflop_call_count = int(summary["postflop_call_count"])
        summary["postflop_aggression_factor"] = (
            float(summary["postflop_raise_count"]) / postflop_call_count
            if postflop_call_count > 0
            else float(summary["postflop_raise_count"])
        )
        three_bet_opportunities = int(summary["three_bet_opportunity_count"])
        summary["three_bet_rate"] = (
            float(summary["three_bet_count"]) / three_bet_opportunities
            if three_bet_opportunities > 0
            else 0.0
        )
        faced_three_bets = int(summary["faced_three_bet_count"])
        summary["fold_to_three_bet_rate"] = (
            float(summary["folded_to_three_bet_count"]) / faced_three_bets
            if faced_three_bets > 0
            else 0.0
        )
        cbet_opportunities = int(summary["cbet_opportunity_count"])
        summary["cbet_rate"] = float(summary["cbet_count"]) / cbet_opportunities if cbet_opportunities > 0 else 0.0
        saw_flop_count = int(summary["saw_flop_count"])
        summary["wtsd"] = float(summary["showdown_count"]) / saw_flop_count if saw_flop_count > 0 else 0.0
        showdown_count = int(summary["showdown_count"])
        summary["wsd"] = float(summary["won_showdown_count"]) / showdown_count if showdown_count > 0 else 0.0
        summary["tool_action_context_averages"] = {}
        for tool_name, count in dict(summary.get("tool_action_counts", {})).items():
            total_count = max(1, int(count))
            totals = dict(summary.get("tool_action_context_totals", {}).get(tool_name, {}))
            summary["tool_action_context_averages"][tool_name] = {
                key: float(value) / total_count for key, value in sorted(totals.items())
            }
        summary["tool_response_rates"] = {}
        for tool_name, counts in dict(summary.get("tool_response_counts", {})).items():
            raises = int(dict(counts).get("raises", 0) or 0)
            responses = int(dict(counts).get("opponent_responses", 0) or 0)
            amounts = dict(summary.get("tool_response_amounts", {}).get(tool_name, {}))
            risk_total = float(amounts.get("risk_amount_total", 0.0) or 0.0)
            won_pot_total = float(amounts.get("won_immediately_pot_total", 0.0) or 0.0)
            failed_risk_total = (
                float(amounts.get("called_risk_amount_total", 0.0) or 0.0)
                + float(amounts.get("raised_over_risk_amount_total", 0.0) or 0.0)
            )
            summary["tool_response_rates"][tool_name] = {
                "won_immediately_rate": float(dict(counts).get("won_immediately", 0) or 0) / raises if raises else 0.0,
                "called_or_raised_rate": float(dict(counts).get("called_or_raised", 0) or 0) / raises if raises else 0.0,
                "opponent_fold_rate": float(dict(counts).get("opponents_folded", 0) or 0) / responses if responses else 0.0,
                "opponent_call_rate": float(dict(counts).get("opponents_called", 0) or 0) / responses if responses else 0.0,
                "opponent_raise_over_rate": float(dict(counts).get("opponents_raised_over", 0) or 0) / responses if responses else 0.0,
                "average_risk_amount": risk_total / raises if raises else 0.0,
                "average_pot_before": float(amounts.get("pot_before_total", 0.0) or 0.0) / raises if raises else 0.0,
                "immediate_chip_delta_estimate": won_pot_total - failed_risk_total,
                "immediate_chip_delta_per_raise_estimate": (won_pot_total - failed_risk_total) / raises if raises else 0.0,
            }
        river_counts = river_betting_range_counts(summary)
        river_total = int(river_counts.get("no_facing_raise_total", 0) or 0)
        river_bluffs = int(river_counts.get("no_facing_bluff_raise", 0) or 0)
        summary["river_betting_range_rates"] = {
            "no_facing_bluff_share": float(river_bluffs) / river_total if river_total else 0.0,
            "no_facing_non_bluff_share": (
                float(river_counts.get("no_facing_non_bluff_raise", 0) or 0) / river_total if river_total else 0.0
            ),
            "size_buckets": {},
        }
        for bucket, bucket_counts in dict(river_counts.get("size_buckets", {})).items():
            bucket_total = int(dict(bucket_counts).get("no_facing_raise_total", 0) or 0)
            bucket_bluffs = int(dict(bucket_counts).get("no_facing_bluff_raise", 0) or 0)
            summary["river_betting_range_rates"]["size_buckets"][str(bucket)] = {
                "no_facing_bluff_share": float(bucket_bluffs) / bucket_total if bucket_total else 0.0,
                "no_facing_non_bluff_share": (
                    float(dict(bucket_counts).get("no_facing_non_bluff_raise", 0) or 0) / bucket_total
                    if bucket_total
                    else 0.0
                ),
            }
    return stats


def merge_candidate_action_summaries(summaries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Dict[str, Any]] = {}

    def default_merged_summary() -> Dict[str, Any]:
        return {
            "action_total": 0,
            "action_counts": {},
            "street_action_counts": {},
            "amount_total": 0,
            "raise_amount_total": 0,
            "raise_count": 0,
            "postflop_raise_count": 0,
            "postflop_call_count": 0,
            "vpip_hands": 0,
            "vpip_count": 0,
            "pfr_count": 0,
            "preflop_call_count": 0,
            "three_bet_opportunity_count": 0,
            "three_bet_count": 0,
            "faced_three_bet_count": 0,
            "folded_to_three_bet_count": 0,
            "cbet_opportunity_count": 0,
            "cbet_count": 0,
            "saw_flop_count": 0,
            "showdown_count": 0,
            "won_showdown_count": 0,
            "tool_event_counts": {},
            "tool_decision_counts": {},
            "tool_reject_reason_counts": {},
            "tool_action_counts": {},
            "tool_action_amount_total": {},
            "tool_action_context_totals": {},
            "tool_action_context_averages": {},
            "tool_response_counts": {},
            "tool_response_amounts": {},
            "tool_response_rates": {},
            "river_betting_range_counts": {},
            "river_betting_range_rates": {},
        }

    for action_summary in summaries:
        for candidate_id, row in dict(action_summary).items():
            target = merged.setdefault(str(candidate_id), default_merged_summary())
            target["action_total"] += int(row.get("action_total", 0) or 0)
            target["amount_total"] += int(row.get("amount_total", 0) or 0)
            target["raise_amount_total"] += int(row.get("raise_amount_total", 0) or 0)
            target["raise_count"] += int(row.get("raise_count", 0) or 0)
            target["postflop_raise_count"] += int(row.get("postflop_raise_count", 0) or 0)
            target["postflop_call_count"] += int(row.get("postflop_call_count", 0) or 0)
            target["vpip_hands"] += int(row.get("vpip_hands", 0) or 0)
            target["vpip_count"] += int(row.get("vpip_count", 0) or 0)
            target["pfr_count"] += int(row.get("pfr_count", 0) or 0)
            target["preflop_call_count"] += int(row.get("preflop_call_count", 0) or 0)
            target["three_bet_opportunity_count"] += int(row.get("three_bet_opportunity_count", 0) or 0)
            target["three_bet_count"] += int(row.get("three_bet_count", 0) or 0)
            target["faced_three_bet_count"] += int(row.get("faced_three_bet_count", 0) or 0)
            target["folded_to_three_bet_count"] += int(row.get("folded_to_three_bet_count", 0) or 0)
            target["cbet_opportunity_count"] += int(row.get("cbet_opportunity_count", 0) or 0)
            target["cbet_count"] += int(row.get("cbet_count", 0) or 0)
            target["saw_flop_count"] += int(row.get("saw_flop_count", 0) or 0)
            target["showdown_count"] += int(row.get("showdown_count", 0) or 0)
            target["won_showdown_count"] += int(row.get("won_showdown_count", 0) or 0)
            for action, count in dict(row.get("action_counts", {})).items():
                target["action_counts"][str(action)] = int(target["action_counts"].get(str(action), 0)) + int(count)
            for street, street_counts in dict(row.get("street_action_counts", {})).items():
                target_street = target["street_action_counts"].setdefault(str(street), {})
                for action, count in dict(street_counts).items():
                    target_street[str(action)] = int(target_street.get(str(action), 0)) + int(count)
            for tool_name, count in dict(row.get("tool_event_counts", {})).items():
                tool_name = str(tool_name)
                target["tool_event_counts"][tool_name] = int(target["tool_event_counts"].get(tool_name, 0)) + int(count)
            for tool_name, decision_counts in dict(row.get("tool_decision_counts", {})).items():
                target_decisions = target["tool_decision_counts"].setdefault(str(tool_name), {})
                for decision, count in dict(decision_counts).items():
                    target_decisions[str(decision)] = int(target_decisions.get(str(decision), 0)) + int(count)
            for tool_name, reason_counts in dict(row.get("tool_reject_reason_counts", {})).items():
                target_reasons = target["tool_reject_reason_counts"].setdefault(str(tool_name), {})
                for reason, count in dict(reason_counts).items():
                    target_reasons[str(reason)] = int(target_reasons.get(str(reason), 0)) + int(count)
            for tool_name, count in dict(row.get("tool_action_counts", {})).items():
                tool_name = str(tool_name)
                target["tool_action_counts"][tool_name] = int(target["tool_action_counts"].get(tool_name, 0)) + int(count)
            for tool_name, amount in dict(row.get("tool_action_amount_total", {})).items():
                tool_name = str(tool_name)
                target["tool_action_amount_total"][tool_name] = (
                    int(target["tool_action_amount_total"].get(tool_name, 0)) + int(amount)
                )
            for tool_name, totals in dict(row.get("tool_action_context_totals", {})).items():
                target_totals = target["tool_action_context_totals"].setdefault(str(tool_name), {})
                for key, value in dict(totals).items():
                    try:
                        target_totals[str(key)] = float(target_totals.get(str(key), 0.0)) + float(value)
                    except (TypeError, ValueError):
                        pass
            for tool_name, counts in dict(row.get("tool_response_counts", {})).items():
                target_counts = target["tool_response_counts"].setdefault(str(tool_name), {})
                for key, value in dict(counts).items():
                    target_counts[str(key)] = int(target_counts.get(str(key), 0)) + int(value)
            for tool_name, amounts in dict(row.get("tool_response_amounts", {})).items():
                target_amounts = target["tool_response_amounts"].setdefault(str(tool_name), {})
                for key, value in dict(amounts).items():
                    try:
                        target_amounts[str(key)] = float(target_amounts.get(str(key), 0.0)) + float(value)
                    except (TypeError, ValueError):
                        pass
            river_counts = dict(row.get("river_betting_range_counts", {}))
            if river_counts:
                target_river = target["river_betting_range_counts"]
                for key in ("no_facing_raise_total", "no_facing_bluff_raise", "no_facing_non_bluff_raise"):
                    target_river[key] = int(target_river.get(key, 0)) + int(river_counts.get(key, 0) or 0)
                target_buckets = target_river.setdefault("size_buckets", {})
                for bucket, bucket_counts in dict(river_counts.get("size_buckets", {})).items():
                    target_bucket = target_buckets.setdefault(
                        str(bucket),
                        {
                            "no_facing_raise_total": 0,
                            "no_facing_bluff_raise": 0,
                            "no_facing_non_bluff_raise": 0,
                        },
                    )
                    for key in ("no_facing_raise_total", "no_facing_bluff_raise", "no_facing_non_bluff_raise"):
                        target_bucket[key] = int(target_bucket.get(key, 0)) + int(dict(bucket_counts).get(key, 0) or 0)

    for row in merged.values():
        total = max(1, int(row["action_total"]))
        row["action_rates"] = {action: count / total for action, count in sorted(dict(row["action_counts"]).items())}
        row["average_amount"] = float(row["amount_total"]) / total
        row["average_raise_amount"] = (
            float(row["raise_amount_total"]) / int(row["raise_count"]) if int(row["raise_count"]) > 0 else 0.0
        )
        row["vpip"] = (
            float(row["vpip_count"]) / int(row["vpip_hands"]) if int(row["vpip_hands"]) > 0 else 0.0
        )
        row["pfr"] = float(row["pfr_count"]) / int(row["vpip_hands"]) if int(row["vpip_hands"]) > 0 else 0.0
        row["preflop_call_rate"] = (
            float(row["preflop_call_count"]) / int(row["vpip_hands"]) if int(row["vpip_hands"]) > 0 else 0.0
        )
        postflop_call_count = int(row["postflop_call_count"])
        row["postflop_aggression_factor"] = (
            float(row["postflop_raise_count"]) / postflop_call_count
            if postflop_call_count > 0
            else float(row["postflop_raise_count"])
        )
        three_bet_opportunities = int(row["three_bet_opportunity_count"])
        row["three_bet_rate"] = (
            float(row["three_bet_count"]) / three_bet_opportunities if three_bet_opportunities > 0 else 0.0
        )
        faced_three_bets = int(row["faced_three_bet_count"])
        row["fold_to_three_bet_rate"] = (
            float(row["folded_to_three_bet_count"]) / faced_three_bets if faced_three_bets > 0 else 0.0
        )
        cbet_opportunities = int(row["cbet_opportunity_count"])
        row["cbet_rate"] = float(row["cbet_count"]) / cbet_opportunities if cbet_opportunities > 0 else 0.0
        saw_flop_count = int(row["saw_flop_count"])
        row["wtsd"] = float(row["showdown_count"]) / saw_flop_count if saw_flop_count > 0 else 0.0
        showdown_count = int(row["showdown_count"])
        row["wsd"] = float(row["won_showdown_count"]) / showdown_count if showdown_count > 0 else 0.0
        row["tool_action_context_averages"] = {}
        for tool_name, count in dict(row.get("tool_action_counts", {})).items():
            total_count = max(1, int(count))
            totals = dict(row.get("tool_action_context_totals", {}).get(tool_name, {}))
            row["tool_action_context_averages"][tool_name] = {
                key: float(value) / total_count for key, value in sorted(totals.items())
            }
        row["tool_response_rates"] = {}
        for tool_name, counts in dict(row.get("tool_response_counts", {})).items():
            counts = dict(counts)
            raises = int(counts.get("raises", 0) or 0)
            responses = int(counts.get("opponent_responses", 0) or 0)
            amounts = dict(row.get("tool_response_amounts", {}).get(tool_name, {}))
            risk_total = float(amounts.get("risk_amount_total", 0.0) or 0.0)
            won_pot_total = float(amounts.get("won_immediately_pot_total", 0.0) or 0.0)
            failed_risk_total = (
                float(amounts.get("called_risk_amount_total", 0.0) or 0.0)
                + float(amounts.get("raised_over_risk_amount_total", 0.0) or 0.0)
            )
            row["tool_response_rates"][tool_name] = {
                "won_immediately_rate": float(counts.get("won_immediately", 0) or 0) / raises if raises else 0.0,
                "called_or_raised_rate": float(counts.get("called_or_raised", 0) or 0) / raises if raises else 0.0,
                "opponent_fold_rate": float(counts.get("opponents_folded", 0) or 0) / responses if responses else 0.0,
                "opponent_call_rate": float(counts.get("opponents_called", 0) or 0) / responses if responses else 0.0,
                "opponent_raise_over_rate": float(counts.get("opponents_raised_over", 0) or 0) / responses if responses else 0.0,
                "average_risk_amount": risk_total / raises if raises else 0.0,
                "average_pot_before": float(amounts.get("pot_before_total", 0.0) or 0.0) / raises if raises else 0.0,
                "immediate_chip_delta_estimate": won_pot_total - failed_risk_total,
                "immediate_chip_delta_per_raise_estimate": (won_pot_total - failed_risk_total) / raises if raises else 0.0,
            }
        river_counts = row.setdefault("river_betting_range_counts", {})
        river_total = int(river_counts.get("no_facing_raise_total", 0) or 0)
        river_bluffs = int(river_counts.get("no_facing_bluff_raise", 0) or 0)
        row["river_betting_range_rates"] = {
            "no_facing_bluff_share": float(river_bluffs) / river_total if river_total else 0.0,
            "no_facing_non_bluff_share": (
                float(river_counts.get("no_facing_non_bluff_raise", 0) or 0) / river_total if river_total else 0.0
            ),
            "size_buckets": {},
        }
        for bucket, bucket_counts in dict(river_counts.get("size_buckets", {})).items():
            bucket_counts = dict(bucket_counts)
            bucket_total = int(bucket_counts.get("no_facing_raise_total", 0) or 0)
            bucket_bluffs = int(bucket_counts.get("no_facing_bluff_raise", 0) or 0)
            row["river_betting_range_rates"]["size_buckets"][str(bucket)] = {
                "no_facing_bluff_share": float(bucket_bluffs) / bucket_total if bucket_total else 0.0,
                "no_facing_non_bluff_share": (
                    float(bucket_counts.get("no_facing_non_bluff_raise", 0) or 0) / bucket_total
                    if bucket_total
                    else 0.0
                ),
            }
    return merged


def _name_candidate_map_from_results(results: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    return {str(row.get("name", "")): str(row.get("name", "")).split("_entry_", 1)[0] for row in results}


def run_candidate_mtts(
    candidates: List[Candidate],
    *,
    mtt_count: int,
    entries_per_candidate: int,
    seed: int,
    engine_config: Dict[str, Any],
    basemodel_root: Path,
    engine_root: Path,
    observation_schema: str,
    observation_size: int,
    deterministic: bool,
    artifact_root: Path,
    write_events: bool = False,
    partial_summary_path: Path | None = None,
) -> Dict[str, Any]:
    all_results: List[Dict[str, Any]] = []
    tournament_summaries: List[Dict[str, Any]] = []
    action_summaries: List[Dict[str, Any]] = []
    failures: List[str] = []
    started = time.perf_counter()

    with temporary_engine_config(_engine_overrides(engine_config)):
        for index in range(int(mtt_count)):
            tournament_id = index + 1
            _seed_everything(int(seed) + index)
            try:
                bots, name_to_candidate = build_candidate_lineup(
                    candidates,
                    entries_per_candidate=entries_per_candidate,
                    basemodel_root=basemodel_root,
                    engine_root=engine_root,
                    observation_schema=observation_schema,
                    observation_size=observation_size,
                    deterministic=deterministic,
                    decision_timeout_ms=int(engine_config.get("bot_decision_timeout_ms", 500)),
                    equity_source=str(engine_config.get("equity_source", "pokerstove")),
                    equity_fallback_source=engine_config.get("equity_fallback_source", "constant"),
                    equity_iterations=engine_config.get("equity_iterations"),
                    use_preflop_spot_range=bool(engine_config.get("model_use_preflop_spot_range", False)),
                )
                tournament = Tournament(bots, tournament_id=tournament_id)
                results, events = tournament.play()
                all_results.extend(results)
                candidate_action_summary = summarize_candidate_actions(events, name_to_candidate)
                action_summaries.append(candidate_action_summary)
                tournament_summary = {
                    "tournament_id": tournament_id,
                    "candidate_summary": summarize_candidate_results(results, name_to_candidate),
                    "candidate_action_summary": candidate_action_summary,
                    "stopped_max_hands": any(event.get("type") == "tournament_stopped_max_hands" for event in events),
                    "event_count": len(events),
                }
                if write_events:
                    event_path = artifact_root / f"tournament_{tournament_id:04d}_events.json"
                    _write_json(event_path, events)
                    tournament_summary["event_log_path"] = str(event_path)
                tournament_summaries.append(tournament_summary)
                if partial_summary_path is not None:
                    _write_json(
                        partial_summary_path,
                        {
                            "completed_tournament_count": len(tournament_summaries),
                            "candidate_summary": summarize_candidate_results(
                                all_results,
                                _name_candidate_map_from_results(all_results),
                            ),
                            "candidate_action_summary": merge_candidate_action_summaries(action_summaries),
                            "failures": failures,
                        },
                    )
            except Exception as exc:
                failures.append(f"tournament {tournament_id} failed: {exc}")
                if partial_summary_path is not None:
                    _write_json(
                        partial_summary_path,
                        {
                            "completed_tournament_count": len(tournament_summaries),
                            "candidate_summary": summarize_candidate_results(
                                all_results,
                                _name_candidate_map_from_results(all_results),
                            ),
                            "candidate_action_summary": merge_candidate_action_summaries(action_summaries),
                            "failures": failures,
                        },
                    )

    return {
        "candidate_summary": summarize_candidate_results(
            all_results,
            _name_candidate_map_from_results(all_results),
        ),
        "candidate_action_summary": merge_candidate_action_summaries(action_summaries),
        "tournaments": tournament_summaries,
        "failures": failures,
        "runtime_seconds": time.perf_counter() - started,
    }


def _run_candidate_mtt_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    tournament_id = int(payload["tournament_id"])
    candidates = [
        Candidate(candidate_id=str(row["candidate_id"]), checkpoint_path=Path(str(row["checkpoint_path"])))
        for row in payload["candidates"]
    ]
    engine_config = dict(payload["engine_config"])
    artifact_root = Path(str(payload["artifact_root"]))
    try:
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:
            pass
        _seed_everything(int(payload["seed"]))
        with temporary_engine_config(_engine_overrides(engine_config)):
            bots, name_to_candidate = build_candidate_lineup(
                candidates,
                entries_per_candidate=int(payload["entries_per_candidate"]),
                basemodel_root=Path(str(payload["basemodel_root"])),
                engine_root=Path(str(payload["engine_root"])),
                observation_schema=str(payload["observation_schema"]),
                observation_size=int(payload["observation_size"]),
                deterministic=bool(payload["deterministic"]),
                decision_timeout_ms=int(engine_config.get("bot_decision_timeout_ms", 500)),
                equity_source=str(engine_config.get("equity_source", "pokerstove")),
                equity_fallback_source=engine_config.get("equity_fallback_source", "constant"),
                equity_iterations=engine_config.get("equity_iterations"),
                use_preflop_spot_range=bool(engine_config.get("model_use_preflop_spot_range", False)),
            )
            tournament = Tournament(bots, tournament_id=tournament_id)
            results, events = tournament.play()
            candidate_action_summary = summarize_candidate_actions(events, name_to_candidate)
            tournament_summary = {
                "tournament_id": tournament_id,
                "candidate_summary": summarize_candidate_results(results, name_to_candidate),
                "candidate_action_summary": candidate_action_summary,
                "stopped_max_hands": any(event.get("type") == "tournament_stopped_max_hands" for event in events),
                "event_count": len(events),
            }
            if bool(payload.get("write_events", False)):
                event_path = artifact_root / f"tournament_{tournament_id:04d}_events.json"
                _write_json(event_path, events)
                tournament_summary["event_log_path"] = str(event_path)
            return {
                "tournament_id": tournament_id,
                "results": results,
                "tournament_summary": tournament_summary,
                "failure": "",
            }
    except Exception as exc:
        return {"tournament_id": tournament_id, "results": [], "tournament_summary": {}, "failure": str(exc)}


def run_candidate_mtts_parallel(
    candidates: List[Candidate],
    *,
    mtt_count: int,
    entries_per_candidate: int,
    seed: int,
    workers: int,
    engine_config: Dict[str, Any],
    basemodel_root: Path,
    engine_root: Path,
    observation_schema: str,
    observation_size: int,
    deterministic: bool,
    artifact_root: Path,
    write_events: bool = False,
    partial_summary_path: Path | None = None,
) -> Dict[str, Any]:
    all_results: List[Dict[str, Any]] = []
    tournament_summaries: List[Dict[str, Any]] = []
    action_summaries: List[Dict[str, Any]] = []
    failures: List[str] = []
    started = time.perf_counter()
    candidate_payload = [
        {"candidate_id": candidate.candidate_id, "checkpoint_path": str(candidate.checkpoint_path)}
        for candidate in candidates
    ]
    max_workers = max(1, min(int(workers), int(mtt_count)))

    def write_partial() -> None:
        if partial_summary_path is None:
            return
        _write_json(
            partial_summary_path,
            {
                "completed_tournament_count": len(tournament_summaries),
                "candidate_summary": summarize_candidate_results(
                    all_results,
                    _name_candidate_map_from_results(all_results),
                ),
                "candidate_action_summary": merge_candidate_action_summaries(action_summaries),
                "failures": failures,
            },
        )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {}
        for index in range(int(mtt_count)):
            tournament_id = index + 1
            payload = {
                "tournament_id": tournament_id,
                "seed": int(seed) + index,
                "candidates": candidate_payload,
                "entries_per_candidate": entries_per_candidate,
                "engine_config": engine_config,
                "basemodel_root": str(basemodel_root),
                "engine_root": str(engine_root),
                "observation_schema": observation_schema,
                "observation_size": observation_size,
                "deterministic": deterministic,
                "artifact_root": str(artifact_root),
                "write_events": write_events,
            }
            future_to_id[executor.submit(_run_candidate_mtt_worker, payload)] = tournament_id

        for future in concurrent.futures.as_completed(future_to_id):
            tournament_id = future_to_id[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append(f"tournament {tournament_id} failed: {exc}")
                write_partial()
                continue
            failure = str(result.get("failure", "") or "")
            if failure:
                failures.append(f"tournament {tournament_id} failed: {failure}")
            else:
                all_results.extend(list(result.get("results", [])))
                tournament_summary = dict(result.get("tournament_summary", {}))
                tournament_summaries.append(tournament_summary)
                action_summaries.append(dict(tournament_summary.get("candidate_action_summary", {})))
            write_partial()

    tournament_summaries.sort(key=lambda row: int(row.get("tournament_id", 0)))
    return {
        "candidate_summary": summarize_candidate_results(
            all_results,
            _name_candidate_map_from_results(all_results),
        ),
        "candidate_action_summary": merge_candidate_action_summaries(action_summaries),
        "tournaments": tournament_summaries,
        "failures": failures,
        "runtime_seconds": time.perf_counter() - started,
        "worker_count": max_workers,
    }


def ranked_candidates(candidate_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [{"candidate_id": candidate_id, **summary} for candidate_id, summary in candidate_summary.items()]
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("score", 0.0)),
            float(row.get("total_payout_pct", 0.0)),
            -float(row.get("average_position", 999999.0)),
        ),
        reverse=True,
    )


def attach_lineage_to_ranking(ranking: List[Dict[str, Any]], candidates: Iterable[Candidate]) -> List[Dict[str, Any]]:
    lineage_by_id = {candidate.candidate_id: candidate_lineage(candidate) for candidate in candidates}
    annotated = []
    for row in ranking:
        candidate_id = str(row.get("candidate_id", ""))
        lineage = lineage_by_id.get(candidate_id, {})
        annotated.append({**row, "lineage": lineage})
    return annotated


def attach_actions_to_ranking(ranking: List[Dict[str, Any]], action_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    annotated = []
    for row in ranking:
        candidate_id = str(row.get("candidate_id", ""))
        annotated.append({**row, "action_summary": dict(action_summary.get(candidate_id, {}))})
    return annotated


def create_initial_candidates(
    *,
    artifact_root: Path,
    basemodel_root: Path,
    observation_size: int,
    candidate_count: int,
    seed: int,
) -> List[Candidate]:
    checkpoint_root = artifact_root / "checkpoints" / "initial"
    candidates = []
    for index in range(int(candidate_count)):
        candidate_id = f"candidate_{index + 1:03d}"
        checkpoint_path = checkpoint_root / candidate_id / "latest.pt"
        create_random_checkpoint(
            checkpoint_path,
            basemodel_root=basemodel_root,
            observation_size=observation_size,
            seed=int(seed) + index,
        )
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                checkpoint_path=checkpoint_path,
                generation=0,
                parent_id="",
                root_id=candidate_id,
            )
        )
    return candidates


def resolve_path_from_engine(engine_root: Path, path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (engine_root / resolved).resolve()
    return resolved


def load_population_manifest(path: str | Path, *, engine_root: Path) -> List[Candidate]:
    manifest_path = resolve_path_from_engine(engine_root, path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    checkpoint_rows = manifest.get("next_population_checkpoints")
    if checkpoint_rows is None:
        checkpoint_rows = manifest.get("candidate_checkpoints")
    if not isinstance(checkpoint_rows, dict):
        raise ValueError(f"population manifest has no checkpoint map: {manifest_path}")
    lineage_rows = manifest.get("lineage")
    if not isinstance(lineage_rows, dict):
        lineage_rows = {}

    candidates = []
    for candidate_id, checkpoint_value in sorted(checkpoint_rows.items()):
        checkpoint_path = resolve_path_from_engine(engine_root, str(checkpoint_value))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"population checkpoint does not exist: {checkpoint_path}")
        lineage = dict(lineage_rows.get(str(candidate_id), {}) or {})
        candidates.append(
            Candidate(
                candidate_id=str(candidate_id),
                checkpoint_path=checkpoint_path,
                generation=int(lineage.get("generation", 0) or 0),
                parent_id=str(lineage.get("parent_id", "") or ""),
                root_id=str(lineage.get("root_id", "") or str(candidate_id)),
            )
        )
    return candidates


def load_population_next_candidate_index(path: str | Path, *, engine_root: Path, candidates: List[Candidate]) -> int:
    manifest_path = resolve_path_from_engine(engine_root, path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    stored_next_id = manifest.get("next_candidate_index")
    if stored_next_id is None:
        return next_candidate_index(candidates)
    return max(int(stored_next_id), next_candidate_index(candidates))


def create_mutation_candidates(
    *,
    artifact_root: Path,
    champion_checkpoint_path: Path,
    basemodel_root: Path,
    observation_size: int,
    variant_count: int,
    seed: int,
    sigma: float,
    policy_head_only: bool,
) -> List[Candidate]:
    checkpoint_root = artifact_root / "checkpoints" / "mutations"
    candidates = []
    for index in range(int(variant_count)):
        candidate_id = f"challenger_{index + 1:03d}"
        checkpoint_path = checkpoint_root / candidate_id / "latest.pt"
        mutate_checkpoint(
            champion_checkpoint_path,
            checkpoint_path,
            basemodel_root=basemodel_root,
            observation_size=observation_size,
            seed=int(seed) + index,
            sigma=sigma,
            policy_head_only=policy_head_only,
        )
        candidates.append(
            Candidate(
                candidate_id=candidate_id,
                checkpoint_path=checkpoint_path,
                generation=1,
                parent_id=str(champion_checkpoint_path),
                root_id=candidate_id,
            )
        )
    return candidates


def create_replaced_population(
    *,
    artifact_root: Path,
    candidates: List[Candidate],
    ranking: List[Dict[str, Any]],
    basemodel_root: Path,
    observation_size: int,
    replace_count: int,
    seed: int,
    sigma: float,
    policy_head_only: bool,
    next_id: int | None = None,
    parent_candidate_ids: List[str] | None = None,
) -> Dict[str, Any]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    ranked_ids = [str(row.get("candidate_id", "")) for row in ranking]
    selected_parent_ids = [str(candidate_id) for candidate_id in (parent_candidate_ids or []) if str(candidate_id) in by_id]
    best_ids = selected_parent_ids or [candidate_id for candidate_id in ranked_ids if candidate_id in by_id][:replace_count]
    worst_ids = [candidate_id for candidate_id in reversed(ranked_ids) if candidate_id in by_id][:replace_count]
    replaced = set(worst_ids)
    next_root = artifact_root / "checkpoints" / "next_population"
    next_candidates: Dict[str, str] = {}
    next_lineage: Dict[str, Dict[str, Any]] = {}
    replacements = []
    allocated_next_id = int(next_id or next_candidate_index(candidates))

    for candidate in candidates:
        if candidate.candidate_id not in replaced:
            next_candidates[candidate.candidate_id] = str(candidate.checkpoint_path)
            next_lineage[candidate.candidate_id] = candidate_lineage(candidate)

    for index, replaced_id in enumerate(worst_ids):
        parent_id = best_ids[index % len(best_ids)] if best_ids else ranked_ids[0]
        parent = by_id[parent_id]
        child_id = f"candidate_{allocated_next_id:03d}"
        allocated_next_id += 1
        checkpoint_path = next_root / child_id / "latest.pt"
        mutate_checkpoint(
            parent.checkpoint_path,
            checkpoint_path,
            basemodel_root=basemodel_root,
            observation_size=observation_size,
            seed=int(seed) + index,
            sigma=sigma,
            policy_head_only=policy_head_only,
        )
        next_candidates[child_id] = str(checkpoint_path)
        child_lineage = {
            "candidate_id": child_id,
            "generation": int(parent.generation) + 1,
            "parent_id": parent_id,
            "root_id": parent.root_id or parent_id,
            "checkpoint_path": str(checkpoint_path),
            "mutation_seed": int(seed) + index,
        }
        next_lineage[child_id] = child_lineage
        replacements.append(
            {
                "replaced_candidate_id": replaced_id,
                "new_candidate_id": child_id,
                "parent_candidate_id": parent_id,
                "parent_root_id": parent.root_id or parent_id,
                "child_generation": child_lineage["generation"],
                "checkpoint_path": str(checkpoint_path),
            }
        )

    return {
        "replace_count": len(replacements),
        "best_candidate_ids": best_ids,
        "worst_candidate_ids": worst_ids,
        "replacements": replacements,
        "next_population_checkpoints": dict(sorted(next_candidates.items())),
        "lineage": dict(sorted(next_lineage.items())),
        "next_candidate_index": allocated_next_id,
    }


def tournament_winner_candidate_ids(simulation: Dict[str, Any]) -> List[str]:
    winner_ids = []
    for tournament in list(simulation.get("tournaments", [])):
        winner_id = ""
        winner_position = None
        for candidate_id, row in dict(tournament.get("candidate_summary", {})).items():
            position = int(float(row.get("average_position", 0) or 0))
            if winner_position is None or position < winner_position:
                winner_position = position
                winner_id = str(candidate_id)
        if winner_position == 1 and winner_id:
            winner_ids.append(winner_id)
    return winner_ids


def run_initial_generation(config: Dict[str, Any], *, engine_root: Path) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(config.get("run_id", "evolutionary_reduced_mtt"))
    artifact_root = Path(str(config.get("artifact_root") or Path("runs") / run_id / utc_stamp()))
    artifact_root.mkdir(parents=True, exist_ok=True)
    basemodel_root = basemodel_root_from_engine(engine_root, config.get("basemodel_root", "../poker-ai-basemodel"))
    observation_schema = str(config.get("observation_schema", "reduced_v6"))
    observation_size, observation_fields = observation_contract(basemodel_root, observation_schema)
    candidate_count = int(config.get("candidate_count", 20))
    entries_per_candidate = int(config.get("entries_per_candidate", 10))
    mtt_count = int(config.get("mtt_count", 8))
    seed = int(config.get("random_seed", 7001))
    engine_config = default_engine_config()
    engine_config.update(dict(config.get("engine", {})))

    candidates = create_initial_candidates(
        artifact_root=artifact_root,
        basemodel_root=basemodel_root,
        observation_size=observation_size,
        candidate_count=candidate_count,
        seed=seed,
    )
    simulation = run_candidate_mtts(
        candidates,
        mtt_count=mtt_count,
        entries_per_candidate=entries_per_candidate,
        seed=seed + 100000,
        engine_config=engine_config,
        basemodel_root=basemodel_root,
        engine_root=engine_root,
        observation_schema=observation_schema,
        observation_size=observation_size,
        deterministic=bool(config.get("deterministic", True)),
        artifact_root=artifact_root,
        write_events=bool(config.get("write_events", False)),
        partial_summary_path=artifact_root / "partial_summary.json",
    )
    ranking = attach_actions_to_ranking(
        attach_lineage_to_ranking(ranked_candidates(dict(simulation.get("candidate_summary", {}))), candidates),
        dict(simulation.get("candidate_action_summary", {})),
    )
    champion = ranking[0] if ranking else {}
    report = {
        "run_id": run_id,
        "mode": "initial",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "basemodel_root": str(basemodel_root),
        "observation_schema": observation_schema,
        "observation_size": observation_size,
        "observation_fields": observation_fields,
        "candidate_count": candidate_count,
        "entries_per_candidate": entries_per_candidate,
        "mtt_count": mtt_count,
        "random_seed": seed,
        "engine": engine_config,
        "candidate_checkpoints": {candidate.candidate_id: str(candidate.checkpoint_path) for candidate in candidates},
        "lineage": {candidate.candidate_id: candidate_lineage(candidate) for candidate in candidates},
        "champion_candidate_id": champion.get("candidate_id", ""),
        "champion_checkpoint_path": str(
            next((candidate.checkpoint_path for candidate in candidates if candidate.candidate_id == champion.get("candidate_id")), "")
        ),
        "ranking": ranking,
        "simulation": simulation,
    }
    report_path = artifact_root / "initial_generation_report.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def run_steady_generation(config: Dict[str, Any], *, engine_root: Path) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(config.get("run_id", "evolutionary_reduced_steady"))
    artifact_root = Path(str(config.get("artifact_root") or Path("runs") / run_id / utc_stamp()))
    artifact_root.mkdir(parents=True, exist_ok=True)
    basemodel_root = basemodel_root_from_engine(engine_root, config.get("basemodel_root", "../poker-ai-basemodel"))
    observation_schema = str(config.get("observation_schema", "reduced_v6"))
    observation_size, observation_fields = observation_contract(basemodel_root, observation_schema)
    population_size = int(config.get("population_size", 200))
    mtt_count = int(config.get("mtt_count", 4))
    workers = int(config.get("workers", 1))
    replace_count = int(config.get("replace_count", 20))
    seed = int(config.get("random_seed", 7201))
    population_path = str(config.get("population_path", "") or "")
    engine_config = default_engine_config()
    engine_config.update(dict(config.get("engine", {})))

    if population_path:
        candidates = load_population_manifest(population_path, engine_root=engine_root)
        next_id = load_population_next_candidate_index(population_path, engine_root=engine_root, candidates=candidates)
        population_size = len(candidates)
    else:
        candidates = create_initial_candidates(
            artifact_root=artifact_root,
            basemodel_root=basemodel_root,
            observation_size=observation_size,
            candidate_count=population_size,
            seed=seed,
        )
        next_id = next_candidate_index(candidates)
    runner = run_candidate_mtts_parallel if workers > 1 else run_candidate_mtts
    simulation = runner(
        candidates,
        mtt_count=mtt_count,
        entries_per_candidate=1,
        seed=seed + 100000,
        engine_config=engine_config,
        basemodel_root=basemodel_root,
        engine_root=engine_root,
        observation_schema=observation_schema,
        observation_size=observation_size,
        deterministic=bool(config.get("deterministic", True)),
        artifact_root=artifact_root,
        write_events=bool(config.get("write_events", False)),
        partial_summary_path=artifact_root / "partial_summary.json",
        **({"workers": workers} if workers > 1 else {}),
    )
    ranking = attach_actions_to_ranking(
        attach_lineage_to_ranking(ranked_candidates(dict(simulation.get("candidate_summary", {}))), candidates),
        dict(simulation.get("candidate_action_summary", {})),
    )
    selection_mode = str(config.get("selection_mode", "score"))
    parent_candidate_ids = None
    if selection_mode == "winners_only":
        parent_candidate_ids = tournament_winner_candidate_ids(dict(simulation))
        replace_count = len(parent_candidate_ids)
    next_population = create_replaced_population(
        artifact_root=artifact_root,
        candidates=candidates,
        ranking=ranking,
        basemodel_root=basemodel_root,
        observation_size=observation_size,
        replace_count=replace_count,
        seed=seed + 200000,
        sigma=float(config.get("mutation_sigma", 0.02)),
        policy_head_only=bool(config.get("policy_head_only", True)),
        next_id=next_id,
        parent_candidate_ids=parent_candidate_ids,
    )
    report = {
        "run_id": run_id,
        "mode": "steady",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "basemodel_root": str(basemodel_root),
        "observation_schema": observation_schema,
        "observation_size": observation_size,
        "observation_fields": observation_fields,
        "population_size": population_size,
        "entries_per_candidate": 1,
        "mtt_count": mtt_count,
        "workers": workers,
        "replace_count": replace_count,
        "selection_mode": selection_mode,
        "selected_parent_candidate_ids": parent_candidate_ids or next_population.get("best_candidate_ids", []),
        "random_seed": seed,
        "source_population_path": population_path,
        "engine": engine_config,
        "candidate_checkpoints": {candidate.candidate_id: str(candidate.checkpoint_path) for candidate in candidates},
        "lineage": {candidate.candidate_id: candidate_lineage(candidate) for candidate in candidates},
        "ranking": ranking,
        "next_population": next_population,
        "simulation": simulation,
    }
    report_path = artifact_root / "steady_generation_report.json"
    _write_json(report_path, report)
    _write_json(artifact_root / "next_population.json", next_population)
    report["report_path"] = str(report_path)
    return report


def run_mutation_generation(config: Dict[str, Any], *, engine_root: Path) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(config.get("run_id", "evolutionary_reduced_mtt_mutation"))
    artifact_root = Path(str(config.get("artifact_root") or Path("runs") / run_id / utc_stamp()))
    artifact_root.mkdir(parents=True, exist_ok=True)
    basemodel_root = basemodel_root_from_engine(engine_root, config.get("basemodel_root", "../poker-ai-basemodel"))
    observation_schema = str(config.get("observation_schema", "reduced_v6"))
    observation_size, observation_fields = observation_contract(basemodel_root, observation_schema)
    champion_checkpoint_path = Path(str(config.get("champion_checkpoint_path", ""))).expanduser()
    if not champion_checkpoint_path.is_absolute():
        champion_checkpoint_path = (engine_root / champion_checkpoint_path).resolve()
    if not champion_checkpoint_path.is_file():
        raise FileNotFoundError(f"champion checkpoint does not exist: {champion_checkpoint_path}")

    variant_count = int(config.get("variant_count", 10))
    champion_entries = int(config.get("champion_entries", 100))
    entries_per_variant = int(config.get("entries_per_variant", 10))
    mtt_count = int(config.get("mtt_count", 8))
    seed = int(config.get("random_seed", 7101))
    engine_config = default_engine_config()
    engine_config.update(dict(config.get("engine", {})))

    challengers = create_mutation_candidates(
        artifact_root=artifact_root,
        champion_checkpoint_path=champion_checkpoint_path,
        basemodel_root=basemodel_root,
        observation_size=observation_size,
        variant_count=variant_count,
        seed=seed,
        sigma=float(config.get("mutation_sigma", 0.02)),
        policy_head_only=bool(config.get("policy_head_only", True)),
    )
    champion = Candidate(candidate_id="champion", checkpoint_path=champion_checkpoint_path)
    candidates = [champion, *challengers]

    champion_simulation = run_candidate_mtts(
        [champion],
        mtt_count=0,
        entries_per_candidate=champion_entries,
        seed=seed,
        engine_config=engine_config,
        basemodel_root=basemodel_root,
        engine_root=engine_root,
        observation_schema=observation_schema,
        observation_size=observation_size,
        deterministic=bool(config.get("deterministic", True)),
        artifact_root=artifact_root,
    )
    del champion_simulation

    # The lineup builder supports one entry count, so build the 50/50 field explicitly.
    all_results: List[Dict[str, Any]] = []
    tournament_summaries: List[Dict[str, Any]] = []
    failures: List[str] = []
    started = time.perf_counter()
    partial_summary_path = artifact_root / "partial_summary.json"
    with temporary_engine_config(_engine_overrides(engine_config)):
        for index in range(mtt_count):
            tournament_id = index + 1
            _seed_everything(seed + 100000 + index)
            try:
                champion_bots, champion_names = build_candidate_lineup(
                    [champion],
                    entries_per_candidate=champion_entries,
                    basemodel_root=basemodel_root,
                    engine_root=engine_root,
                    observation_schema=observation_schema,
                    observation_size=observation_size,
                    deterministic=bool(config.get("deterministic", True)),
                    decision_timeout_ms=int(engine_config.get("bot_decision_timeout_ms", 500)),
                    equity_source=str(engine_config.get("equity_source", "pokerstove")),
                    equity_fallback_source=engine_config.get("equity_fallback_source", "constant"),
                    equity_iterations=engine_config.get("equity_iterations"),
                    use_preflop_spot_range=bool(engine_config.get("model_use_preflop_spot_range", False)),
                )
                challenger_bots, challenger_names = build_candidate_lineup(
                    challengers,
                    entries_per_candidate=entries_per_variant,
                    basemodel_root=basemodel_root,
                    engine_root=engine_root,
                    observation_schema=observation_schema,
                    observation_size=observation_size,
                    deterministic=bool(config.get("deterministic", True)),
                    decision_timeout_ms=int(engine_config.get("bot_decision_timeout_ms", 500)),
                    equity_source=str(engine_config.get("equity_source", "pokerstove")),
                    equity_fallback_source=engine_config.get("equity_fallback_source", "constant"),
                    equity_iterations=engine_config.get("equity_iterations"),
                    use_preflop_spot_range=bool(engine_config.get("model_use_preflop_spot_range", False)),
                )
                name_to_candidate = {**champion_names, **challenger_names}
                name_to_population = {
                    **{name: "champion" for name in champion_names},
                    **{name: "challenger" for name in challenger_names},
                }
                tournament = Tournament([*champion_bots, *challenger_bots], tournament_id=tournament_id)
                results, events = tournament.play()
                all_results.extend(results)
                tournament_summaries.append(
                    {
                        "tournament_id": tournament_id,
                        "candidate_summary": summarize_candidate_results(results, name_to_candidate),
                        "population_summary": summarize_population_results(results, name_to_population),
                        "stopped_max_hands": any(event.get("type") == "tournament_stopped_max_hands" for event in events),
                        "event_count": len(events),
                    }
                )
                aggregate_candidate = summarize_candidate_results(all_results, _name_candidate_map_from_results(all_results))
                aggregate_population = summarize_population_results(
                    all_results,
                    {
                        str(row.get("name", "")): "champion"
                        if str(row.get("name", "")).startswith("champion_entry_")
                        else "challenger"
                        for row in all_results
                    },
                )
                _write_json(
                    partial_summary_path,
                    {
                        "completed_tournament_count": len(tournament_summaries),
                        "candidate_summary": aggregate_candidate,
                        "population_summary": aggregate_population,
                        "failures": failures,
                    },
                )
            except Exception as exc:
                failures.append(f"tournament {tournament_id} failed: {exc}")

    candidate_summary = summarize_candidate_results(all_results, _name_candidate_map_from_results(all_results))
    population_summary = summarize_population_results(
        all_results,
        {
            str(row.get("name", "")): "champion"
            if str(row.get("name", "")).startswith("champion_entry_")
            else "challenger"
            for row in all_results
        },
    )
    challenger_ranking = [
        row for row in ranked_candidates(candidate_summary) if str(row.get("candidate_id", "")).startswith("challenger_")
    ]
    best_challenger = challenger_ranking[0] if challenger_ranking else {}
    report = {
        "run_id": run_id,
        "mode": "mutation",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "basemodel_root": str(basemodel_root),
        "observation_schema": observation_schema,
        "observation_size": observation_size,
        "observation_fields": observation_fields,
        "mtt_count": mtt_count,
        "random_seed": seed,
        "champion_entries": champion_entries,
        "variant_count": variant_count,
        "entries_per_variant": entries_per_variant,
        "champion_checkpoint_path": str(champion_checkpoint_path),
        "challenger_checkpoints": {candidate.candidate_id: str(candidate.checkpoint_path) for candidate in challengers},
        "best_challenger_candidate_id": best_challenger.get("candidate_id", ""),
        "best_challenger_checkpoint_path": str(
            next((candidate.checkpoint_path for candidate in challengers if candidate.candidate_id == best_challenger.get("candidate_id")), "")
        ),
        "candidate_ranking": ranked_candidates(candidate_summary),
        "challenger_ranking": challenger_ranking,
        "population_summary": population_summary,
        "simulation": {
            "candidate_summary": candidate_summary,
            "population_summary": population_summary,
            "tournaments": tournament_summaries,
            "failures": failures,
            "runtime_seconds": time.perf_counter() - started,
        },
    }
    report_path = artifact_root / "mutation_generation_report.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    parser.add_argument("--mode", choices=("initial", "mutation", "steady"), default=None)
    parser.add_argument("--champion-checkpoint", default="")
    parser.add_argument("--mtts", type=int, default=None)
    parser.add_argument("--candidates", type=int, default=None)
    parser.add_argument("--entries-per-candidate", type=int, default=None)
    parser.add_argument("--artifact-root", default="")
    parser.add_argument("--population", default="")
    args = parser.parse_args(argv)

    config = load_json_config(args.config)
    if args.mode is not None:
        config["mode"] = args.mode
    if args.champion_checkpoint:
        config["champion_checkpoint_path"] = args.champion_checkpoint
    if args.mtts is not None:
        config["mtt_count"] = args.mtts
    if args.candidates is not None:
        config["candidate_count"] = args.candidates
    if args.entries_per_candidate is not None:
        config["entries_per_candidate"] = args.entries_per_candidate
    if args.artifact_root:
        config["artifact_root"] = args.artifact_root
    if args.population:
        config["population_path"] = args.population

    engine_root = Path(__file__).resolve().parents[1]
    mode = str(config.get("mode", "initial"))
    if mode == "mutation":
        report = run_mutation_generation(config, engine_root=engine_root)
    elif mode == "steady":
        report = run_steady_generation(config, engine_root=engine_root)
    else:
        report = run_initial_generation(config, engine_root=engine_root)
    print(
        json.dumps(
            {
                "report_path": report["report_path"],
                "mode": report["mode"],
                "champion_candidate_id": report.get("champion_candidate_id", ""),
                "champion_checkpoint_path": report.get("champion_checkpoint_path", ""),
                "best_challenger_candidate_id": report.get("best_challenger_candidate_id", ""),
                "best_challenger_checkpoint_path": report.get("best_challenger_checkpoint_path", ""),
                "top5": report.get("ranking", report.get("candidate_ranking", []))[:5],
                "next_population": {
                    "replace_count": report.get("next_population", {}).get("replace_count", 0),
                    "best_candidate_ids": report.get("next_population", {}).get("best_candidate_ids", []),
                    "worst_candidate_ids": report.get("next_population", {}).get("worst_candidate_ids", []),
                },
                "population_summary": report.get("population_summary", {}),
                "failures": report["simulation"]["failures"],
                "runtime_seconds": report["simulation"]["runtime_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["simulation"]["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
