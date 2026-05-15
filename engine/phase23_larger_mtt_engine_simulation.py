"""Phase 23 larger MTT engine simulation runner."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from engine.baseline_model_bot import BaselineModelEngineBot, resolve_checkpoint_path
from engine.phase22_small_mtt_engine_simulation import (
    ACCEPTED,
    REJECTED,
    _engine_overrides,
    _equity_counts,
    _fallback_counts,
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
from engine.tournament import Tournament


def load_config(path: str | Path) -> Dict[str, Any]:
    loaded = _read_json(path)
    loaded["config_path"] = str(path)
    return loaded


def _load_report(path: str | Path) -> Dict[str, Any] | None:
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def resolve_phase22_report(config: Dict[str, Any]) -> tuple[str, Dict[str, Any], str]:
    configured_path = str(config.get("phase22_report_path", "") or "")
    required_status = str(config.get("expected_phase22_status", ACCEPTED))
    if configured_path:
        report = _load_report(configured_path)
        if report is None:
            return configured_path, {}, "source Phase 22 report is missing or malformed"
        return configured_path, report, ""

    pattern = str(config.get("phase22_report_glob", "") or "")
    if not pattern:
        return "", {}, "phase22_report_path or phase22_report_glob is required"

    matches = sorted(glob.glob(pattern), key=lambda item: Path(item).stat().st_mtime, reverse=True)
    malformed_count = 0
    for path in matches:
        report = _load_report(path)
        if report is None:
            malformed_count += 1
            continue
        if report.get("phase22_small_engine_simulation_status") == required_status and bool(
            report.get("larger_mtt_engine_simulation_allowed", False)
        ):
            return path, report, ""
    if matches:
        suffix = f"; {malformed_count} malformed candidates skipped" if malformed_count else ""
        return matches[0], _load_report(matches[0]) or {}, f"no accepted Phase 22 report matched {pattern}{suffix}"
    return "", {}, f"no Phase 22 reports matched {pattern}"


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(config)
    run_id = str(resolved.get("phase23_larger_engine_simulation_id", "phase23_larger_mtt_engine_simulation"))
    artifact_root_value = str(resolved.get("artifact_root", "") or "")
    if artifact_root_value:
        artifact_root = Path(artifact_root_value)
    else:
        artifact_root = Path(str(resolved.get("output_dir", Path("runs") / run_id))) / _run_stamp()
    resolved["artifact_root"] = str(artifact_root)
    resolved.setdefault(
        "larger_engine_simulation_report_path",
        str(artifact_root / "larger_mtt_engine_simulation_report.json"),
    )
    resolved.setdefault("tournament_results_path", str(artifact_root / "tournament_results.json"))
    resolved.setdefault("event_summaries_path", str(artifact_root / "event_summaries.json"))
    resolved.setdefault("event_log_dir", str(artifact_root / "events"))
    return resolved


def verify_prerequisite(config: Dict[str, Any], engine_root: Path) -> Dict[str, Any]:
    source_path, source, load_failure = resolve_phase22_report(config)
    promoted = str(source.get("promoted_checkpoint_path") or "")
    basemodel_root = Path(str(config.get("basemodel_root", "../poker-ai-basemodel"))).expanduser()
    if not basemodel_root.is_absolute():
        basemodel_root = (engine_root / basemodel_root).resolve()
    checkpoint = resolve_checkpoint_path(promoted, basemodel_root=basemodel_root, engine_root=engine_root) if promoted else Path("")

    required_status = str(config.get("expected_phase22_status", ACCEPTED))
    require_allowed = bool(config.get("required_larger_mtt_engine_simulation_allowed", True))
    failures = []
    if load_failure:
        failures.append(load_failure)
    if source.get("phase22_small_engine_simulation_status") != required_status:
        failures.append(
            f"source Phase 22 status is {source.get('phase22_small_engine_simulation_status')!r}, "
            f"expected {required_status!r}"
        )
    if require_allowed and not bool(source.get("larger_mtt_engine_simulation_allowed", False)):
        failures.append("source Phase 22 did not allow larger MTT engine simulation")
    if not promoted:
        failures.append("source Phase 22 report does not include promoted_checkpoint_path")
    if promoted and not checkpoint.is_file():
        failures.append(f"promoted checkpoint does not exist: {checkpoint}")

    return {
        "source_phase22_report_path": source_path,
        "source_phase22_status": str(source.get("phase22_small_engine_simulation_status", "")),
        "source_phase22_failures": source.get("phase22_small_engine_simulation_failures", []),
        "source_larger_mtt_engine_simulation_allowed": bool(source.get("larger_mtt_engine_simulation_allowed", False)),
        "source_phase21_report_path": source.get("source_phase21_report_path", ""),
        "source_phase21_status": source.get("source_phase21_status", ""),
        "source_phase22_lineup_summary": source.get("lineup_summary", {}),
        "source_phase22_bot_fallback_summary": source.get("bot_fallback_summary", {}),
        "source_promoted_checkpoint_path": source.get("promoted_checkpoint_path", ""),
        "promoted_checkpoint_path": str(checkpoint) if promoted else "",
        "promoted_checkpoint_path_from_report": promoted,
        "basemodel_root": str(basemodel_root),
        "prerequisite_failures": failures,
        "prerequisite_passed": not failures,
    }


def build_phase23_lineup(config: Dict[str, Any], checkpoint_path: str, basemodel_root: str) -> List[Any]:
    bots = build_lineup(config, checkpoint_path, basemodel_root)
    model_index = 0
    for bot in bots:
        if isinstance(bot, BaselineModelEngineBot):
            model_index += 1
            bot.name = f"Phase23ModelBot_{model_index}"
    return bots


def should_write_full_event_log(config: Dict[str, Any], tournament_id: int) -> bool:
    event_logging = dict(config.get("event_logging", {}))
    if bool(event_logging.get("write_full_event_logs", False)):
        return True
    return tournament_id <= int(event_logging.get("write_full_event_logs_for_first_n", 0))


def summarize_events(tournament_id: int, events: List[Dict[str, Any]], results: List[Dict[str, Any]]) -> Dict[str, Any]:
    type_counts: Dict[str, int] = {}
    action_counts: Dict[str, int] = {}
    model_action_counts: Dict[str, int] = {}
    for event in events:
        event_type = str(event.get("type", "unknown"))
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
        if event_type == "action":
            action = str(event.get("action", "unknown"))
            action_counts[action] = action_counts.get(action, 0) + 1
            if str(event.get("player", "")).startswith("Phase23ModelBot_"):
                model_action_counts[action] = model_action_counts.get(action, 0) + 1
    winner = next((str(event.get("player", "")) for event in events if event.get("type") == "tournament_win"), "")
    stopped = any(event.get("type") == "tournament_stopped_max_hands" for event in events)
    return {
        "tournament_id": tournament_id,
        "event_count": len(events),
        "result_count": len(results),
        "winner": winner,
        "stopped_max_hands": stopped,
        "hand_count": int(type_counts.get("hand_end", 0)),
        "knockout_count": int(type_counts.get("knockout", 0)),
        "level_up_count": int(type_counts.get("level_up", 0)),
        "table_broken_count": int(type_counts.get("table_broken", 0)),
        "event_type_counts": type_counts,
        "action_counts": action_counts,
        "model_action_counts": model_action_counts,
    }


def run_tournaments(config: Dict[str, Any], bots: List[Any], artifact_root: Path) -> Dict[str, Any]:
    tournament_count = int(config.get("tournament_count", 1))
    seed = int(config.get("random_seed", 0))
    result_rows = []
    event_summaries = []
    event_log_paths = []
    tournament_failures = []
    model_bots = [bot for bot in bots if isinstance(bot, BaselineModelEngineBot)]
    fallback_before = [_fallback_counts(bot) for bot in model_bots]
    equity_before = [_equity_counts(bot) for bot in model_bots]

    with temporary_engine_config(_engine_overrides(config)):
        for index in range(tournament_count):
            tournament_id = index + 1
            _seed_everything(seed + index)
            try:
                tournament = Tournament(bots, tournament_id=tournament_id)
                results, events = tournament.play()
                stopped = any(event.get("type") == "tournament_stopped_max_hands" for event in events)
                event_summary = summarize_events(tournament_id, events, results)
                event_summaries.append(event_summary)
                if should_write_full_event_log(config, tournament_id):
                    event_log_path = Path(str(config["event_log_dir"])) / f"tournament_{tournament_id}_events.json"
                    _write_json(event_log_path, events)
                    event_log_paths.append(str(event_log_path))
                result_rows.append(
                    {
                        "tournament_id": tournament_id,
                        "results": results,
                        "event_count": len(events),
                        "stopped_max_hands": stopped,
                    }
                )
            except Exception as exc:
                tournament_failures.append(f"tournament {tournament_id} failed: {exc}")

    fallback_after = [_fallback_counts(bot) for bot in model_bots]
    equity_after = [_equity_counts(bot) for bot in model_bots]
    model_fallback_deltas = [_subtract_counts(after, before) for before, after in zip(fallback_before, fallback_after)]
    model_equity_deltas = [_subtract_counts(after, before) for before, after in zip(equity_before, equity_after)]

    return {
        "tournament_results": result_rows,
        "event_summaries": event_summaries,
        "event_log_paths": event_log_paths,
        "tournament_failures": tournament_failures,
        "bot_fallback_summary": {
            "model_bot_count": len(model_bots),
            "per_model_bot": model_fallback_deltas,
            "totals": _sum_count_dicts(model_fallback_deltas),
        },
        "model_equity_summary": {
            "per_model_bot": model_equity_deltas,
            "totals": _sum_count_dicts(model_equity_deltas),
        },
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def summarize_results(results_by_tournament: List[Dict[str, Any]]) -> Dict[str, Any]:
    placement_summary: Dict[str, Dict[str, float]] = {}
    payout_summary: Dict[str, Dict[str, float]] = {}
    model_rows = []
    valid_tournaments = 0
    stopped_max_hands = 0

    for tournament in results_by_tournament:
        results = list(tournament.get("results", []))
        if results:
            valid_tournaments += 1
        if tournament.get("stopped_max_hands"):
            stopped_max_hands += 1
        for row in results:
            class_name = str(row.get("bot_class", "unknown"))
            position = int(row.get("position", 0) or 0)
            payout = float(row.get("payout_pct", 0.0) or 0.0)
            placement = placement_summary.setdefault(
                class_name,
                {"count": 0, "position_sum": 0.0, "wins": 0, "top_3": 0, "itm": 0},
            )
            payout_row = payout_summary.setdefault(class_name, {"count": 0, "payout_sum": 0.0})
            placement["count"] += 1
            placement["position_sum"] += position
            placement["wins"] += 1 if position == 1 else 0
            placement["top_3"] += 1 if 0 < position <= 3 else 0
            placement["itm"] += 1 if payout > 0 else 0
            payout_row["count"] += 1
            payout_row["payout_sum"] += payout
            if class_name == "BaselineModelEngineBot":
                model_rows.append(dict(row, tournament_id=tournament.get("tournament_id")))

    for class_name, placement in placement_summary.items():
        count = max(1, int(placement["count"]))
        placement["average_position"] = placement["position_sum"] / count
        placement["win_rate"] = placement["wins"] / count
        placement["top_3_rate"] = placement["top_3"] / count
        placement["itm_rate"] = placement["itm"] / count
        payout = payout_summary[class_name]
        payout["average_payout_pct"] = payout["payout_sum"] / count

    model_positions = [int(row.get("position", 0) or 0) for row in model_rows]
    model_payouts = [float(row.get("payout_pct", 0.0) or 0.0) for row in model_rows]
    model_entries = len(model_rows)
    model_summary = {
        "entries": model_entries,
        "placements": model_rows,
        "best_position": min(model_positions) if model_positions else None,
        "worst_position": max(model_positions) if model_positions else None,
        "average_position": sum(model_positions) / model_entries if model_entries else None,
        "median_position": statistics.median(model_positions) if model_positions else None,
        "top_3_rate": _rate(sum(1 for position in model_positions if position <= 3), model_entries),
        "itm_rate": _rate(sum(1 for payout in model_payouts if payout > 0), model_entries),
        "win_rate": _rate(sum(1 for position in model_positions if position == 1), model_entries),
        "total_payout_pct": sum(model_payouts),
    }

    return {
        "completed_tournament_count": valid_tournaments,
        "stopped_max_hands_count": stopped_max_hands,
        "placement_summary_by_bot_class": placement_summary,
        "payout_summary_by_bot_class": payout_summary,
        "model_bot_summary": model_summary,
    }


def _decision(config: Dict[str, Any], status: str, summary: Dict[str, Any], lineup: Dict[str, Any]) -> tuple[bool, str]:
    if status != ACCEPTED:
        return False, "resolve_phase23_engine_simulation_failures"
    thresholds = dict(config.get("decision_thresholds", {}))
    model = dict(summary.get("model_bot_summary", {}))
    min_entries = int(thresholds.get("min_model_entries_for_decision", 1))
    entries = int(model.get("entries", 0) or 0)
    if entries < min_entries:
        return False, "run_additional_phase23_larger_mtt_engine_simulation"

    field_size = int(lineup.get("total_bots", 0) or 0)
    weak_quantile = float(thresholds.get("weak_model_average_position_quantile", 0.6))
    positive_itm = float(thresholds.get("positive_model_itm_rate", 0.15))
    average_position = model.get("average_position")
    model_itm_rate = float(model.get("itm_rate", 0.0) or 0.0)
    weak_by_position = average_position is not None and field_size > 0 and float(average_position) > field_size * weak_quantile
    weak_by_itm = model_itm_rate < positive_itm
    if weak_by_position or weak_by_itm:
        return True, "plan_phase24_engine_rollout_collection_for_retraining"
    return True, "launch_phase24_engine_rollout_collection"


def build_report(
    config: Dict[str, Any],
    prerequisite: Dict[str, Any],
    simulation: Dict[str, Any],
    started_at: str,
    runtime_seconds: float,
) -> Dict[str, Any]:
    acceptance = dict(config.get("acceptance", {}))
    summary = summarize_results(list(simulation.get("tournament_results", [])))
    result_artifact_path = str(config.get("tournament_results_path", ""))
    event_summary_path = str(config.get("event_summaries_path", ""))
    fallback_totals = dict(simulation.get("bot_fallback_summary", {}).get("totals", {}))
    completed = int(summary["completed_tournament_count"])
    tournament_count = int(config.get("tournament_count", 0))
    stopped_rate = _rate(int(summary["stopped_max_hands_count"]), max(1, tournament_count))
    gates = {
        "source_phase22_accepted_gate_passed": prerequisite.get("source_phase22_status")
        == config.get("expected_phase22_status", ACCEPTED),
        "source_larger_sim_allowed_gate_passed": bool(
            prerequisite.get("source_larger_mtt_engine_simulation_allowed", False)
        ),
        "promoted_checkpoint_exists_gate_passed": bool(prerequisite.get("promoted_checkpoint_path"))
        and Path(str(prerequisite.get("promoted_checkpoint_path"))).is_file(),
        "lineup_gate_passed": simulation.get("lineup_summary", {}).get("total_bots", 0)
        > int(config.get("max_players_per_table", 9)),
        "tournament_runtime_gate_passed": not simulation.get("tournament_failures"),
        "completed_tournament_gate_passed": completed >= int(acceptance.get("min_completed_tournaments", 1)),
        "stopped_max_hands_rate_gate_passed": stopped_rate <= float(acceptance.get("max_stopped_max_hands_rate", 1.0)),
        "model_inference_error_gate_passed": int(fallback_totals.get("inference_errors", 0))
        <= int(acceptance.get("max_model_inference_errors", 0)),
        "model_illegal_action_gate_passed": int(fallback_totals.get("illegal_actions", 0))
        <= int(acceptance.get("max_model_illegal_actions", 0)),
        "model_timeout_fallback_gate_passed": int(fallback_totals.get("timeouts", 0))
        <= int(acceptance.get("max_model_timeout_fallbacks", 0)),
        "event_summary_artifact_gate_passed": (not bool(acceptance.get("require_event_summary_artifact", True)))
        or (bool(event_summary_path) and Path(event_summary_path).is_file()),
        "result_artifact_gate_passed": (not bool(acceptance.get("require_result_artifact", True)))
        or (bool(result_artifact_path) and Path(result_artifact_path).is_file()),
    }
    failures = list(prerequisite.get("prerequisite_failures", []))
    failures.extend(str(failure) for failure in simulation.get("tournament_failures", []))
    for gate, passed in gates.items():
        if not passed:
            failures.append(f"{gate} failed")

    status = ACCEPTED if not failures else REJECTED
    rollout_allowed, recommendation = _decision(config, status, summary, simulation.get("lineup_summary", {}))
    artifact_paths = {
        "artifact_root": str(config.get("artifact_root", "")),
        "larger_engine_simulation_report_path": str(config.get("larger_engine_simulation_report_path", "")),
        "tournament_results_path": result_artifact_path,
        "event_summaries_path": event_summary_path,
        "event_log_paths": list(simulation.get("event_log_paths", [])),
    }
    model_summary = summary["model_bot_summary"]
    return {
        "phase23_larger_engine_simulation_id": config.get(
            "phase23_larger_engine_simulation_id", "phase23_larger_mtt_engine_simulation"
        ),
        "phase23_larger_engine_simulation_status": status,
        "phase23_larger_engine_simulation_failures": failures,
        "phase23_larger_engine_simulation_report_path": str(config.get("larger_engine_simulation_report_path", "")),
        "phase23_larger_engine_simulation_started_at": started_at,
        "phase23_larger_engine_simulation_finished_at": _utc_now(),
        "source_phase22_report_path": prerequisite.get("source_phase22_report_path", ""),
        "source_phase22_status": prerequisite.get("source_phase22_status", ""),
        "source_phase22_failures": prerequisite.get("source_phase22_failures", []),
        "source_larger_mtt_engine_simulation_allowed": prerequisite.get(
            "source_larger_mtt_engine_simulation_allowed", False
        ),
        "source_phase21_report_path": prerequisite.get("source_phase21_report_path", ""),
        "source_phase21_status": prerequisite.get("source_phase21_status", ""),
        "promoted_checkpoint_path": prerequisite.get("promoted_checkpoint_path", ""),
        "lineup_summary": simulation.get("lineup_summary", {}),
        "tournament_count": tournament_count,
        "completed_tournament_count": completed,
        "stopped_max_hands_count": summary["stopped_max_hands_count"],
        "stopped_max_hands_rate": stopped_rate,
        "model_entry_count": model_summary["entries"],
        "model_average_position": model_summary["average_position"],
        "model_median_position": model_summary["median_position"],
        "model_top_3_rate": model_summary["top_3_rate"],
        "model_itm_rate": model_summary["itm_rate"],
        "model_win_rate": model_summary["win_rate"],
        "model_total_payout_pct": model_summary["total_payout_pct"],
        "placement_summary_by_bot_class": summary["placement_summary_by_bot_class"],
        "payout_summary_by_bot_class": summary["payout_summary_by_bot_class"],
        "model_bot_summary": model_summary,
        "bustout_summary": {
            "event_knockout_count": sum(int(row.get("knockout_count", 0)) for row in simulation.get("event_summaries", [])),
            "model_non_cash_count": sum(
                1 for row in model_summary["placements"] if float(row.get("payout_pct", 0.0) or 0.0) <= 0.0
            ),
        },
        "runtime_summary": {
            "runtime_seconds": runtime_seconds,
            "event_summary_count": len(simulation.get("event_summaries", [])),
            "sampled_event_log_count": len(simulation.get("event_log_paths", [])),
            "tournament_failures": simulation.get("tournament_failures", []),
        },
        "bot_fallback_summary": simulation.get("bot_fallback_summary", {}),
        "model_equity_summary": simulation.get("model_equity_summary", {}),
        "artifact_paths": artifact_paths,
        "phase23_gate_results": gates,
        "engine_rollout_collection_allowed": rollout_allowed,
        "next_phase_recommendation": recommendation,
        "config_path": config.get("config_path", ""),
    }


def run_phase23_larger_mtt_engine_simulation(config: Dict[str, Any], config_path: str | Path | None = None) -> Dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    config = _resolve_paths(config)
    if config_path is not None:
        config["config_path"] = str(config_path)
    engine_root = Path(__file__).resolve().parents[1]
    artifact_root = Path(str(config["artifact_root"]))
    artifact_root.mkdir(parents=True, exist_ok=True)
    prerequisite = verify_prerequisite(config, engine_root)
    simulation: Dict[str, Any] = {
        "lineup_summary": {},
        "tournament_results": [],
        "event_summaries": [],
        "event_log_paths": [],
        "tournament_failures": [],
        "bot_fallback_summary": {"model_bot_count": 0, "per_model_bot": [], "totals": {}},
        "model_equity_summary": {"per_model_bot": [], "totals": {}},
    }

    if prerequisite.get("prerequisite_passed"):
        try:
            bots = build_phase23_lineup(
                config,
                str(prerequisite["promoted_checkpoint_path"]),
                str(prerequisite["basemodel_root"]),
            )
            simulation["lineup_summary"] = lineup_summary(bots)
            simulation.update(run_tournaments(config, bots, artifact_root))
        except Exception as exc:
            simulation["tournament_failures"] = [f"simulation setup failed: {exc}"]

    _write_json(str(config["tournament_results_path"]), simulation.get("tournament_results", []))
    _write_json(str(config["event_summaries_path"]), simulation.get("event_summaries", []))
    report = build_report(config, prerequisite, simulation, started_at, time.perf_counter() - started)
    _write_json(str(config["larger_engine_simulation_report_path"]), report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase23_larger_mtt_engine_simulation.json")
    args = parser.parse_args(argv)

    report = run_phase23_larger_mtt_engine_simulation(load_config(args.config), config_path=args.config)
    print(
        json.dumps(
            {
                "phase23_larger_engine_simulation_status": report["phase23_larger_engine_simulation_status"],
                "promoted_checkpoint_path": report["promoted_checkpoint_path"],
                "completed_tournament_count": report["completed_tournament_count"],
                "model_average_position": report["model_average_position"],
                "model_itm_rate": report["model_itm_rate"],
                "engine_rollout_collection_allowed": report["engine_rollout_collection_allowed"],
                "next_phase_recommendation": report["next_phase_recommendation"],
                "phase23_larger_engine_simulation_report_path": report[
                    "phase23_larger_engine_simulation_report_path"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["phase23_larger_engine_simulation_status"] != ACCEPTED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
