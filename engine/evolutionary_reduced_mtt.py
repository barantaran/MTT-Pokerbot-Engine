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
    for event in events:
        event_type = str(event.get("type", ""))
        if event_type not in {"action", "deal"}:
            continue
        candidate_id = name_to_candidate.get(str(event.get("player", "")))
        if not candidate_id:
            continue
        hand_key = (
            int(event.get("tournament_id", 0) or 0),
            int(event.get("table_id", 0) or 0),
            int(event.get("hand_id", 0) or 0),
        )
        if event_type == "deal":
            dealt_hands.setdefault(candidate_id, set()).add(hand_key)
            continue
        action = str(event.get("action", "unknown") or "unknown")
        amount = int(event.get("amount", 0) or 0)
        if action == "call" and amount == 0:
            action = "check"
        if str(event.get("street", "")) == "preflop" and action in {"call", "raise"} and amount > 0:
            vpip_hands.setdefault(candidate_id, set()).add(hand_key)
        summary = stats.setdefault(
            candidate_id,
            {
                "action_total": 0,
                "action_counts": {},
                "amount_total": 0,
                "raise_amount_total": 0,
                "raise_count": 0,
                "vpip_hands": 0,
                "vpip_count": 0,
            },
        )
        summary["action_total"] += 1
        summary["action_counts"][action] = int(summary["action_counts"].get(action, 0)) + 1
        summary["amount_total"] += amount
        if action == "raise":
            summary["raise_count"] += 1
            summary["raise_amount_total"] += amount

    for candidate_id in set(stats) | set(dealt_hands) | set(vpip_hands):
        summary = stats.setdefault(
            candidate_id,
            {
                "action_total": 0,
                "action_counts": {},
                "amount_total": 0,
                "raise_amount_total": 0,
                "raise_count": 0,
                "vpip_hands": 0,
                "vpip_count": 0,
            },
        )
        summary["vpip_hands"] = len(dealt_hands.get(candidate_id, set()))
        summary["vpip_count"] = len(vpip_hands.get(candidate_id, set()))

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
    return stats


def merge_candidate_action_summaries(summaries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Dict[str, Any]] = {}
    for action_summary in summaries:
        for candidate_id, row in dict(action_summary).items():
            target = merged.setdefault(
                str(candidate_id),
                {
                    "action_total": 0,
                    "action_counts": {},
                    "amount_total": 0,
                    "raise_amount_total": 0,
                    "raise_count": 0,
                    "vpip_hands": 0,
                    "vpip_count": 0,
                },
            )
            target["action_total"] += int(row.get("action_total", 0) or 0)
            target["amount_total"] += int(row.get("amount_total", 0) or 0)
            target["raise_amount_total"] += int(row.get("raise_amount_total", 0) or 0)
            target["raise_count"] += int(row.get("raise_count", 0) or 0)
            target["vpip_hands"] += int(row.get("vpip_hands", 0) or 0)
            target["vpip_count"] += int(row.get("vpip_count", 0) or 0)
            for action, count in dict(row.get("action_counts", {})).items():
                target["action_counts"][str(action)] = int(target["action_counts"].get(str(action), 0)) + int(count)

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
