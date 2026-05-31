"""Paired engine evaluation for reduced checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from engine.phase22_small_mtt_engine_simulation import (
    ACCEPTED,
    REJECTED,
    _engine_overrides,
    _read_json,
    _run_stamp,
    _seed_everything,
    _subtract_counts,
    _sum_count_dicts,
    _utc_now,
    _write_json,
    build_lineup,
    lineup_summary,
    temporary_engine_config,
)
from engine.phase26_engine_retraining_evaluation import should_write_full_event_log
from engine.phase29_reduced_clone_engine_evaluation import (
    _resolve_report_artifact,
    resolve_phase28_training_report,
)
from engine.reduced_model_bot import ReducedModelEngineBot
from engine.tournament import Tournament


OPTIONAL_MODEL_PREFIXES = ("v3", "v4", "v5")
SCHEMA_BY_PREFIX = {"v3": "reduced_v3", "v4": "reduced_v4", "v5": "reduced_v5"}
SIZE_BY_SCHEMA = {"reduced_v3": 9, "reduced_v4": 10, "reduced_v5": 11}


def load_config(path: str | Path) -> Dict[str, Any]:
    loaded = _read_json(path)
    loaded["config_path"] = str(path)
    return loaded


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(config)
    run_id = str(resolved.get("phase32_reduced_v1_v2_engine_evaluation_id", "phase32_reduced_v1_v2_engine_evaluation"))
    artifact_root = Path(str(resolved.get("artifact_root") or Path(str(resolved.get("output_dir", Path("runs") / run_id))) / _run_stamp()))
    resolved["artifact_root"] = str(artifact_root)
    resolved.setdefault("campaign_results_dir", str(artifact_root / "campaign"))
    resolved.setdefault("reduced_v1_v2_engine_evaluation_report_path", str(artifact_root / "reduced_v1_v2_engine_evaluation_report.json"))
    return resolved


def _training_prerequisite(config: Dict[str, Any], *, prefix: str, engine_root: Path) -> Dict[str, Any]:
    scoped = {
        "phase28_training_report_path": config.get(f"{prefix}_training_report_path", ""),
        "phase28_training_report_glob": config.get(f"{prefix}_training_report_glob", ""),
        "expected_phase28_training_status": config.get("expected_phase28_training_status", ACCEPTED),
        "required_phase29_reduced_clone_evaluation_allowed": True,
    }
    report_path, report, failure = resolve_phase28_training_report(scoped)
    basemodel_root = Path(str(config.get("basemodel_root", "../poker-ai-basemodel"))).expanduser()
    if not basemodel_root.is_absolute():
        basemodel_root = (engine_root / basemodel_root).resolve()
    raw_checkpoint = str(config.get(f"{prefix}_checkpoint_path") or report.get("candidate_checkpoint_path") or "")
    checkpoint = (
        _resolve_report_artifact(raw_checkpoint, report_path, basemodel_root=basemodel_root, engine_root=engine_root)
        if raw_checkpoint and report_path
        else Path("")
    )
    failures = []
    if failure:
        failures.append(f"{prefix}: {failure}")
    if report.get("phase28_equity_imitation_training_status") != config.get("expected_phase28_training_status", ACCEPTED):
        failures.append(f"{prefix}: training report is not accepted")
    if not bool(report.get("phase29_reduced_clone_evaluation_allowed", False)):
        failures.append(f"{prefix}: training report does not allow reduced evaluation")
    if not raw_checkpoint:
        failures.append(f"{prefix}: candidate checkpoint is missing")
    if raw_checkpoint and not checkpoint.is_file():
        failures.append(f"{prefix}: candidate checkpoint does not exist: {checkpoint}")
    return {
        "prefix": prefix,
        "training_report_path": report_path,
        "training_status": str(report.get("phase28_equity_imitation_training_status", "")),
        "collection_report_path": str(report.get("source_collection_report_path", "")),
        "dataset_path": str(report.get("dataset_path", "")),
        "checkpoint_path": str(checkpoint) if raw_checkpoint else "",
        "checkpoint_path_from_report": raw_checkpoint,
        "observation_size": int(report.get("observation_size", config.get(f"{prefix}_reduced_observation_size", 8)) or 8),
        "reduced_observation_schema": str(report.get("reduced_observation_schema", config.get(f"{prefix}_reduced_observation_schema", "reduced_v1"))),
        "failures": failures,
        "passed": not failures,
        "basemodel_root": str(basemodel_root),
    }


def verify_prerequisites(config: Dict[str, Any], engine_root: Path) -> Dict[str, Any]:
    v1 = _training_prerequisite(config, prefix="v1", engine_root=engine_root)
    v2 = _training_prerequisite(config, prefix="v2", engine_root=engine_root)
    model_prefixes = ["v1", "v2"]
    prerequisite_by_prefix = {"v1": v1, "v2": v2}
    lineup = dict(config.get("lineup", {}))
    for prefix in OPTIONAL_MODEL_PREFIXES:
        has_prefix = int(lineup.get(f"{prefix}_model", 0) or 0) > 0 or any(
            config.get(key)
            for key in (f"{prefix}_training_report_path", f"{prefix}_training_report_glob", f"{prefix}_checkpoint_path")
        )
        if has_prefix:
            prerequisite_by_prefix[prefix] = _training_prerequisite(config, prefix=prefix, engine_root=engine_root)
            model_prefixes.append(prefix)
    failures: List[str] = []
    for prefix in model_prefixes:
        failures.extend(prerequisite_by_prefix[prefix]["failures"])
    basemodel_root = ""
    for prefix in reversed(model_prefixes):
        basemodel_root = prerequisite_by_prefix[prefix]["basemodel_root"] or basemodel_root
    return {
        **prerequisite_by_prefix,
        "model_prefixes": model_prefixes,
        "basemodel_root": basemodel_root,
        "prerequisite_failures": failures,
        "prerequisite_passed": not failures,
    }


def _make_reduced_bot(
    config: Dict[str, Any],
    *,
    prefix: str,
    checkpoint_path: str,
    basemodel_root: str,
    engine_root: Path,
    name: str,
    prerequisite: Dict[str, Any],
) -> ReducedModelEngineBot:
    default_schema = SCHEMA_BY_PREFIX.get(prefix, "reduced_v1")
    observation_schema = str(
        config.get(f"{prefix}_reduced_observation_schema")
        or prerequisite.get("reduced_observation_schema")
        or default_schema
    )
    observation_size = int(
        config.get(f"{prefix}_reduced_observation_size")
        or prerequisite.get("observation_size")
        or SIZE_BY_SCHEMA.get(observation_schema, 8)
    )
    return ReducedModelEngineBot(
        checkpoint_path,
        basemodel_root=basemodel_root,
        engine_root=engine_root,
        name=name,
        deterministic=bool(config.get("deterministic", True)),
        decision_timeout_ms=int(config.get("bot_decision_timeout_ms", 500)),
        equity_source=str(config.get("equity_source", "pokerstove")),
        equity_fallback_source=config.get("equity_fallback_source", "constant"),
        equity_iterations=config.get("equity_iterations"),
        observation_size=observation_size,
        observation_schema=observation_schema,
        require_checkpoint=True,
    )


def build_phase32_lineup(config: Dict[str, Any], prerequisite: Dict[str, Any], engine_root: Path) -> List[Any]:
    lineup = dict(config.get("lineup", {}))
    v1_count = int(lineup.get("v1_model", 1))
    v2_count = int(lineup.get("v2_model", 1))
    v3_count = int(lineup.get("v3_model", 0))
    v4_count = int(lineup.get("v4_model", 0))
    v5_count = int(lineup.get("v5_model", 0))
    patched = dict(config)
    patched_lineup = dict(lineup)
    patched_lineup["model"] = 0
    patched_lineup.pop("v1_model", None)
    patched_lineup.pop("v2_model", None)
    patched_lineup.pop("v3_model", None)
    patched_lineup.pop("v4_model", None)
    patched_lineup.pop("v5_model", None)
    patched["lineup"] = patched_lineup
    bots: List[Any] = []
    for index in range(v1_count):
        bots.append(
            _make_reduced_bot(
                config,
                prefix="v1",
                checkpoint_path=prerequisite["v1"]["checkpoint_path"],
                basemodel_root=prerequisite["basemodel_root"],
                engine_root=engine_root,
                name=f"ReducedV1ModelBot_{index + 1}",
                prerequisite=prerequisite["v1"],
            )
        )
    for index in range(v2_count):
        bots.append(
            _make_reduced_bot(
                config,
                prefix="v2",
                checkpoint_path=prerequisite["v2"]["checkpoint_path"],
                basemodel_root=prerequisite["basemodel_root"],
                engine_root=engine_root,
                name=f"ReducedV2ModelBot_{index + 1}",
                prerequisite=prerequisite["v2"],
            )
        )
    for index in range(v3_count):
        bots.append(
            _make_reduced_bot(
                config,
                prefix="v3",
                checkpoint_path=prerequisite["v3"]["checkpoint_path"],
                basemodel_root=prerequisite["basemodel_root"],
                engine_root=engine_root,
                name=f"ReducedV3ModelBot_{index + 1}",
                prerequisite=prerequisite["v3"],
            )
        )
    for index in range(v4_count):
        bots.append(
            _make_reduced_bot(
                config,
                prefix="v4",
                checkpoint_path=prerequisite["v4"]["checkpoint_path"],
                basemodel_root=prerequisite["basemodel_root"],
                engine_root=engine_root,
                name=f"ReducedV4ModelBot_{index + 1}",
                prerequisite=prerequisite["v4"],
            )
        )
    for index in range(v5_count):
        bots.append(
            _make_reduced_bot(
                config,
                prefix="v5",
                checkpoint_path=prerequisite["v5"]["checkpoint_path"],
                basemodel_root=prerequisite["basemodel_root"],
                engine_root=engine_root,
                name=f"ReducedV5ModelBot_{index + 1}",
                prerequisite=prerequisite["v5"],
            )
        )
    bots.extend(build_lineup(patched, checkpoint_path="", basemodel_root=prerequisite["basemodel_root"]))
    return bots


def _group_for_result(row: Dict[str, Any]) -> str:
    name = str(row.get("name", ""))
    bot_class = str(row.get("bot_class", "unknown"))
    if name.startswith("ReducedV1ModelBot_"):
        return "reduced_v1"
    if name.startswith("ReducedV2ModelBot_"):
        return "reduced_v2"
    if name.startswith("ReducedV3ModelBot_"):
        return "reduced_v3"
    if name.startswith("ReducedV4ModelBot_"):
        return "reduced_v4"
    if name.startswith("ReducedV5ModelBot_"):
        return "reduced_v5"
    if bot_class == "TightEquityBot":
        return "tight_equity"
    if bot_class == "AggressiveBot":
        return "equity_aggressive"
    if bot_class == "RandomBot":
        return "random"
    return bot_class


def _summarize_results(results_by_tournament: List[Dict[str, Any]], *, final_table_size: int = 9) -> Dict[str, Any]:
    groups: Dict[str, Dict[str, Any]] = {}
    completed = 0
    stopped = 0
    final_table_size = max(1, int(final_table_size or 9))
    for tournament in results_by_tournament:
        results = list(tournament.get("results", []))
        completed += 1 if results else 0
        stopped += 1 if tournament.get("stopped_max_hands") else 0
        for row in results:
            group = _group_for_result(row)
            position = int(row.get("position", 0) or 0)
            payout = float(row.get("payout_pct", 0.0) or 0.0)
            summary = groups.setdefault(
                group,
                {
                    "entries": 0,
                    "position_sum": 0.0,
                    "wins": 0,
                    "top_3": 0,
                    "final_table": 0,
                    "itm": 0,
                    "total_payout_pct": 0.0,
                },
            )
            summary["entries"] += 1
            summary["position_sum"] += position
            summary["wins"] += 1 if position == 1 else 0
            summary["top_3"] += 1 if 0 < position <= 3 else 0
            summary["final_table"] += 1 if 0 < position <= final_table_size else 0
            summary["itm"] += 1 if payout > 0 else 0
            summary["total_payout_pct"] += payout
    for summary in groups.values():
        entries = max(1, int(summary["entries"]))
        summary["average_position"] = summary["position_sum"] / entries
        summary["win_rate"] = summary["wins"] / entries
        summary["top_3_rate"] = summary["top_3"] / entries
        summary["final_table_rate"] = summary["final_table"] / entries
        summary["itm_rate"] = summary["itm"] / entries
        summary["average_payout_pct"] = summary["total_payout_pct"] / entries
    return {"completed_tournament_count": completed, "stopped_max_hands_count": stopped, "groups": groups}


def _summarize_events(tournament_id: int, events: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    action_counts = {"reduced_v1": {}, "reduced_v2": {}, "reduced_v3": {}, "reduced_v4": {}, "reduced_v5": {}}
    type_counts: Dict[str, int] = {}
    for event in events:
        event_type = str(event.get("type", "unknown"))
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
        if event_type != "action":
            continue
        player = str(event.get("player", ""))
        action = str(event.get("action", "unknown"))
        if player.startswith("ReducedV1ModelBot_"):
            action_counts["reduced_v1"][action] = action_counts["reduced_v1"].get(action, 0) + 1
        if player.startswith("ReducedV2ModelBot_"):
            action_counts["reduced_v2"][action] = action_counts["reduced_v2"].get(action, 0) + 1
        if player.startswith("ReducedV3ModelBot_"):
            action_counts["reduced_v3"][action] = action_counts["reduced_v3"].get(action, 0) + 1
        if player.startswith("ReducedV4ModelBot_"):
            action_counts["reduced_v4"][action] = action_counts["reduced_v4"].get(action, 0) + 1
        if player.startswith("ReducedV5ModelBot_"):
            action_counts["reduced_v5"][action] = action_counts["reduced_v5"].get(action, 0) + 1
    return {
        "tournament_id": tournament_id,
        "event_count": len(events),
        "result_count": len(results),
        "stopped_max_hands": any(event.get("type") == "tournament_stopped_max_hands" for event in events),
        "hand_count": int(type_counts.get("hand_end", 0)),
        "action_counts_by_model": action_counts,
    }


def _action_mix(event_summaries: Iterable[Dict[str, Any]], group: str) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for summary in event_summaries:
        for action, count in dict(summary.get("action_counts_by_model", {}).get(group, {})).items():
            counts[action] = counts.get(action, 0) + int(count)
    total = sum(counts.values())
    top_action = max(counts, key=counts.get) if counts else None
    return {
        "model_action_counts": counts,
        "model_action_total": total,
        "distinct_model_actions": sum(1 for count in counts.values() if count > 0),
        "top_model_action": top_action,
        "top_model_action_rate": (counts.get(top_action, 0) / total) if top_action and total else 0.0,
    }


def _model_count_deltas(model_bots: List[ReducedModelEngineBot], before: List[Dict[str, int]], attr: str) -> Dict[str, Any]:
    after = [dict(getattr(bot, attr, {})) for bot in model_bots]
    deltas = [_subtract_counts(new, old) for old, new in zip(before, after)]
    return {"per_model_bot": deltas, "totals": _sum_count_dicts(deltas)}


def run_campaign(config: Dict[str, Any], prerequisite: Dict[str, Any], engine_root: Path) -> Dict[str, Any]:
    tournament_count = int(config.get("tournament_count", 1))
    seed = int(config.get("random_seed", 0))
    campaign_root = Path(str(config["campaign_results_dir"]))
    bots = build_phase32_lineup(config, prerequisite, engine_root)
    model_bot_groups = {
        "reduced_v1": [bot for bot in bots if isinstance(bot, ReducedModelEngineBot) and bot.name.startswith("ReducedV1ModelBot_")],
        "reduced_v2": [bot for bot in bots if isinstance(bot, ReducedModelEngineBot) and bot.name.startswith("ReducedV2ModelBot_")],
        "reduced_v3": [bot for bot in bots if isinstance(bot, ReducedModelEngineBot) and bot.name.startswith("ReducedV3ModelBot_")],
        "reduced_v4": [bot for bot in bots if isinstance(bot, ReducedModelEngineBot) and bot.name.startswith("ReducedV4ModelBot_")],
        "reduced_v5": [bot for bot in bots if isinstance(bot, ReducedModelEngineBot) and bot.name.startswith("ReducedV5ModelBot_")],
    }
    model_bot_groups = {key: value for key, value in model_bot_groups.items() if value}
    fallback_before = {group: [dict(bot.fallback_counts) for bot in group_bots] for group, group_bots in model_bot_groups.items()}
    equity_before = {group: [dict(bot.equity_counts) for bot in group_bots] for group, group_bots in model_bot_groups.items()}
    result_rows = []
    event_summaries = []
    event_log_paths = []
    failures = []
    with temporary_engine_config(_engine_overrides(config)):
        for index in range(tournament_count):
            tournament_id = index + 1
            _seed_everything(seed + index)
            try:
                tournament = Tournament(bots, tournament_id=tournament_id)
                results, events = tournament.play()
                stopped = any(event.get("type") == "tournament_stopped_max_hands" for event in events)
                event_summaries.append(_summarize_events(tournament_id, events, results))
                if should_write_full_event_log(config, tournament_id):
                    event_log_path = campaign_root / "events" / f"tournament_{tournament_id}_events.json"
                    _write_json(event_log_path, events)
                    event_log_paths.append(str(event_log_path))
                result_rows.append({"tournament_id": tournament_id, "results": results, "event_count": len(events), "stopped_max_hands": stopped})
            except Exception as exc:
                failures.append(f"tournament {tournament_id} failed: {exc}")
    campaign = {
        "lineup_summary": lineup_summary(bots),
        "tournament_results": result_rows,
        "event_summaries": event_summaries,
        "event_log_paths": event_log_paths,
        "tournament_failures": failures,
        "summary": _summarize_results(
            result_rows,
            final_table_size=int(config.get("final_table_size", config.get("max_players_per_table", 9)) or 9),
        ),
        "action_mix_summary": {
            group: _action_mix(event_summaries, group) for group in model_bot_groups
        },
        "bot_fallback_summary": {
            group: _model_count_deltas(group_bots, fallback_before[group], "fallback_counts")
            for group, group_bots in model_bot_groups.items()
        },
        "model_equity_summary": {
            group: _model_count_deltas(group_bots, equity_before[group], "equity_counts")
            for group, group_bots in model_bot_groups.items()
        },
    }
    _write_json(campaign_root / "tournament_results.json", result_rows)
    _write_json(campaign_root / "event_summaries.json", event_summaries)
    _write_json(campaign_root / "campaign_summary.json", {key: value for key, value in campaign.items() if key not in {"tournament_results", "event_summaries"}})
    return campaign


def _delta(lhs: Dict[str, Any], rhs: Dict[str, Any], key: str) -> float | None:
    if lhs.get(key) is None or rhs.get(key) is None:
        return None
    return float(lhs[key]) - float(rhs[key])


def build_report(config: Dict[str, Any], prerequisite: Dict[str, Any], campaign: Dict[str, Any], started_at: str, runtime_seconds: float) -> Dict[str, Any]:
    acceptance = dict(config.get("acceptance", {}))
    summary = dict(campaign.get("summary", {}))
    groups = dict(summary.get("groups", {}))
    v1 = dict(groups.get("reduced_v1", {}))
    v2 = dict(groups.get("reduced_v2", {}))
    v3 = dict(groups.get("reduced_v3", {}))
    v4 = dict(groups.get("reduced_v4", {}))
    v5 = dict(groups.get("reduced_v5", {}))
    random_group = dict(groups.get("random", {}))
    model_groups = ["reduced_v1", "reduced_v2"]
    for prefix, group in (("v3", "reduced_v3"), ("v4", "reduced_v4"), ("v5", "reduced_v5")):
        if group in groups or prefix in prerequisite:
            model_groups.append(group)
    fallback_totals = [dict(campaign.get("bot_fallback_summary", {}).get(group, {}).get("totals", {})) for group in model_groups]
    equity_totals = [dict(campaign.get("model_equity_summary", {}).get(group, {}).get("totals", {})) for group in model_groups]
    tournament_count = int(config.get("tournament_count", 0))
    stopped_rate = (int(summary.get("stopped_max_hands_count", 0)) / max(1, tournament_count))
    require_v2_beats_random = bool(acceptance.get("require_v2_beats_random", "random" in groups))
    gates = {
        "prerequisite_gate_passed": bool(prerequisite.get("prerequisite_passed", False)),
        "campaign_runtime_gate_passed": not campaign.get("tournament_failures"),
        "completed_tournament_gate_passed": int(summary.get("completed_tournament_count", 0)) >= int(acceptance.get("min_completed_tournaments", 1)),
        "v1_entry_gate_passed": int(v1.get("entries", 0) or 0) >= int(acceptance.get("min_v1_entries", 1)),
        "v2_entry_gate_passed": int(v2.get("entries", 0) or 0) >= int(acceptance.get("min_v2_entries", 1)),
        "v3_entry_gate_passed": (
            int(v3.get("entries", 0) or 0) >= int(acceptance.get("min_v3_entries", 1))
            if "reduced_v3" in model_groups
            else True
        ),
        "v4_entry_gate_passed": (
            int(v4.get("entries", 0) or 0) >= int(acceptance.get("min_v4_entries", 1))
            if "reduced_v4" in model_groups
            else True
        ),
        "v5_entry_gate_passed": (
            int(v5.get("entries", 0) or 0) >= int(acceptance.get("min_v5_entries", 1))
            if "reduced_v5" in model_groups
            else True
        ),
        "stopped_max_hands_rate_gate_passed": stopped_rate <= float(acceptance.get("max_stopped_max_hands_rate", 0.2)),
        "model_runtime_gate_passed": all(
            int(total.get(key, 0)) <= int(acceptance.get(f"max_model_{key}", 0))
            for total in fallback_totals
            for key in ("inference_errors", "timeouts", "illegal_actions")
        ),
        "model_equity_fallback_gate_passed": all(
            int(total.get("fallbacks", 0)) <= int(acceptance.get("max_model_equity_fallbacks", 0)) for total in equity_totals
        ),
        "v2_beats_random_gate_passed": (
            v2.get("average_position") is not None
            and random_group.get("average_position") is not None
            and float(v2["average_position"]) <= float(random_group["average_position"])
            if require_v2_beats_random
            else True
        ),
    }
    failures = list(prerequisite.get("prerequisite_failures", [])) + list(campaign.get("tournament_failures", []))
    failures.extend(f"{gate} failed" for gate, passed in gates.items() if not passed)
    status = ACCEPTED if not failures else REJECTED
    comparison = {
        "v1_average_position": v1.get("average_position"),
        "v2_average_position": v2.get("average_position"),
        "v2_minus_v1_average_position": _delta(v2, v1, "average_position"),
        "v1_itm_rate": v1.get("itm_rate"),
        "v2_itm_rate": v2.get("itm_rate"),
        "v2_minus_v1_itm_rate": _delta(v2, v1, "itm_rate"),
        "v1_total_payout_pct": v1.get("total_payout_pct"),
        "v2_total_payout_pct": v2.get("total_payout_pct"),
        "v2_minus_v1_total_payout_pct": _delta(v2, v1, "total_payout_pct"),
        "v1_final_table_rate": v1.get("final_table_rate"),
        "v2_final_table_rate": v2.get("final_table_rate"),
        "v2_minus_v1_final_table_rate": _delta(v2, v1, "final_table_rate"),
        "v3_average_position": v3.get("average_position"),
        "v3_minus_v2_average_position": _delta(v3, v2, "average_position"),
        "v3_itm_rate": v3.get("itm_rate"),
        "v3_minus_v2_itm_rate": _delta(v3, v2, "itm_rate"),
        "v3_total_payout_pct": v3.get("total_payout_pct"),
        "v3_minus_v2_total_payout_pct": _delta(v3, v2, "total_payout_pct"),
        "v3_final_table_rate": v3.get("final_table_rate"),
        "v3_minus_v2_final_table_rate": _delta(v3, v2, "final_table_rate"),
        "v4_average_position": v4.get("average_position"),
        "v4_minus_v3_average_position": _delta(v4, v3, "average_position"),
        "v4_itm_rate": v4.get("itm_rate"),
        "v4_minus_v3_itm_rate": _delta(v4, v3, "itm_rate"),
        "v4_total_payout_pct": v4.get("total_payout_pct"),
        "v4_minus_v3_total_payout_pct": _delta(v4, v3, "total_payout_pct"),
        "v4_final_table_rate": v4.get("final_table_rate"),
        "v4_minus_v3_final_table_rate": _delta(v4, v3, "final_table_rate"),
        "v5_average_position": v5.get("average_position"),
        "v5_minus_v4_average_position": _delta(v5, v4, "average_position"),
        "v5_itm_rate": v5.get("itm_rate"),
        "v5_minus_v4_itm_rate": _delta(v5, v4, "itm_rate"),
        "v5_total_payout_pct": v5.get("total_payout_pct"),
        "v5_minus_v4_total_payout_pct": _delta(v5, v4, "total_payout_pct"),
        "v5_final_table_rate": v5.get("final_table_rate"),
        "v5_minus_v4_final_table_rate": _delta(v5, v4, "final_table_rate"),
        "random_average_position": random_group.get("average_position"),
    }
    return {
        "phase32_reduced_v1_v2_engine_evaluation_id": config.get("phase32_reduced_v1_v2_engine_evaluation_id", "phase32_reduced_v1_v2_engine_evaluation"),
        "phase32_reduced_v1_v2_engine_evaluation_status": status,
        "phase32_reduced_v1_v2_engine_evaluation_failures": list(dict.fromkeys(failures)),
        "phase32_reduced_v1_v2_engine_evaluation_report_path": str(config.get("reduced_v1_v2_engine_evaluation_report_path", "")),
        "phase32_reduced_v1_v2_engine_evaluation_started_at": started_at,
        "phase32_reduced_v1_v2_engine_evaluation_finished_at": _utc_now(),
        "phase32_gate_results": gates,
        "comparison_summary": comparison,
        "group_stats": groups,
        "lineup_summary": campaign.get("lineup_summary", {}),
        "completed_tournament_count": int(summary.get("completed_tournament_count", 0)),
        "stopped_max_hands_rate": stopped_rate,
        "action_mix_summary": campaign.get("action_mix_summary", {}),
        "bot_fallback_summary": campaign.get("bot_fallback_summary", {}),
        "model_equity_summary": campaign.get("model_equity_summary", {}),
        "v1_checkpoint_path": prerequisite.get("v1", {}).get("checkpoint_path", ""),
        "v2_checkpoint_path": prerequisite.get("v2", {}).get("checkpoint_path", ""),
        "v3_checkpoint_path": prerequisite.get("v3", {}).get("checkpoint_path", ""),
        "v4_checkpoint_path": prerequisite.get("v4", {}).get("checkpoint_path", ""),
        "v5_checkpoint_path": prerequisite.get("v5", {}).get("checkpoint_path", ""),
        "source_reports": {
            "v1_training_report_path": prerequisite.get("v1", {}).get("training_report_path", ""),
            "v2_training_report_path": prerequisite.get("v2", {}).get("training_report_path", ""),
            "v3_training_report_path": prerequisite.get("v3", {}).get("training_report_path", ""),
            "v4_training_report_path": prerequisite.get("v4", {}).get("training_report_path", ""),
            "v5_training_report_path": prerequisite.get("v5", {}).get("training_report_path", ""),
        },
        "artifact_paths": {
            "artifact_root": str(config.get("artifact_root", "")),
            "campaign_results_dir": str(config.get("campaign_results_dir", "")),
            "event_log_paths": list(campaign.get("event_log_paths", [])),
        },
        "runtime_summary": {"runtime_seconds": runtime_seconds, "event_summary_count": len(campaign.get("event_summaries", []))},
        "next_phase_recommendation": "inspect_v1_v2_pairing" if status != ACCEPTED else "decide_reduced_v2_next_step",
        "config_path": config.get("config_path", ""),
    }


def run_phase32_reduced_v1_v2_engine_evaluation(config: Dict[str, Any], config_path: str | Path | None = None) -> Dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    config = _resolve_paths(dict(config))
    if config_path is not None:
        config["config_path"] = str(config_path)
    engine_root = Path(__file__).resolve().parents[1]
    Path(str(config["artifact_root"])).mkdir(parents=True, exist_ok=True)
    Path(str(config["campaign_results_dir"])).mkdir(parents=True, exist_ok=True)
    prerequisite = verify_prerequisites(config, engine_root)
    campaign: Dict[str, Any] = {}
    if prerequisite.get("prerequisite_passed"):
        try:
            campaign = run_campaign(config, prerequisite, engine_root)
        except Exception as exc:
            campaign = {"tournament_failures": [f"evaluation setup failed: {exc}"]}
    report = build_report(config, prerequisite, campaign, started_at, time.perf_counter() - started)
    _write_json(config["reduced_v1_v2_engine_evaluation_report_path"], report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase32_reduced_v1_v2_engine_evaluation.json")
    args = parser.parse_args(argv)
    report = run_phase32_reduced_v1_v2_engine_evaluation(load_config(args.config), config_path=args.config)
    print(
        json.dumps(
            {
                "phase32_reduced_v1_v2_engine_evaluation_status": report["phase32_reduced_v1_v2_engine_evaluation_status"],
                "completed_tournament_count": report["completed_tournament_count"],
                "comparison_summary": report["comparison_summary"],
                "group_stats": report["group_stats"],
                "phase32_reduced_v1_v2_engine_evaluation_report_path": report["phase32_reduced_v1_v2_engine_evaluation_report_path"],
                "next_phase_recommendation": report["next_phase_recommendation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["phase32_reduced_v1_v2_engine_evaluation_status"] != ACCEPTED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
