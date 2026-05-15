"""Phase 29 engine evaluation for the Phase 28 reduced imitation clone."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
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
from engine.phase26_engine_retraining_evaluation import _action_mix_summary, should_write_full_event_log, summarize_phase26_events
from engine.reduced_model_bot import ReducedModelEngineBot
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


def resolve_phase28_training_report(config: Dict[str, Any]) -> tuple[str, Dict[str, Any], str]:
    configured_path = str(config.get("phase28_training_report_path", "") or "")
    required_status = str(config.get("expected_phase28_training_status", ACCEPTED))
    if configured_path:
        report = _load_report(configured_path)
        if report is None:
            return configured_path, {}, "source Phase 28 training report is missing or malformed"
        return configured_path, report, ""

    pattern = str(config.get("phase28_training_report_glob", "") or "")
    if not pattern:
        return "", {}, "phase28_training_report_path or phase28_training_report_glob is required"

    matches = sorted(glob.glob(pattern), key=lambda item: Path(item).stat().st_mtime, reverse=True)
    malformed_count = 0
    for path in matches:
        report = _load_report(path)
        if report is None:
            malformed_count += 1
            continue
        if report.get("phase28_equity_imitation_training_status") == required_status and bool(
            report.get("phase29_reduced_clone_evaluation_allowed", False)
        ):
            return path, report, ""
    if matches:
        suffix = f"; {malformed_count} malformed candidates skipped" if malformed_count else ""
        return matches[0], _load_report(matches[0]) or {}, f"no accepted Phase 28 training report matched {pattern}{suffix}"
    return "", {}, f"no Phase 28 training reports matched {pattern}"


def _root_before_runs(path: str | Path) -> Path | None:
    parts = Path(path).resolve().parts
    if "runs" not in parts:
        return None
    return Path(*parts[: parts.index("runs")])


def _resolve_report_artifact(raw_path: str, report_path: str | Path, *, basemodel_root: Path, engine_root: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute() or path.is_file():
        return path
    candidates = [basemodel_root / path, engine_root / path, Path.cwd() / path]
    runs_root = _root_before_runs(report_path)
    if runs_root is not None:
        candidates.insert(0, runs_root / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(config)
    run_id = str(resolved.get("phase29_reduced_clone_engine_evaluation_id", "phase29_reduced_clone_engine_evaluation"))
    artifact_root_value = str(resolved.get("artifact_root", "") or "")
    if artifact_root_value:
        artifact_root = Path(artifact_root_value)
    else:
        artifact_root = Path(str(resolved.get("output_dir", Path("runs") / run_id))) / _run_stamp()
    resolved["artifact_root"] = str(artifact_root)
    resolved.setdefault("reduced_clone_engine_evaluation_report_path", str(artifact_root / "reduced_clone_engine_evaluation_report.json"))
    resolved.setdefault("campaign_results_dir", str(artifact_root / "campaign"))
    return resolved


def verify_prerequisite(config: Dict[str, Any], engine_root: Path) -> Dict[str, Any]:
    source_path, source, load_failure = resolve_phase28_training_report(config)
    basemodel_root = Path(str(config.get("basemodel_root", "../poker-ai-basemodel"))).expanduser()
    if not basemodel_root.is_absolute():
        basemodel_root = (engine_root / basemodel_root).resolve()

    checkpoint_raw = str(config.get("candidate_checkpoint_path") or source.get("candidate_checkpoint_path") or "")
    checkpoint = (
        _resolve_report_artifact(checkpoint_raw, source_path, basemodel_root=basemodel_root, engine_root=engine_root)
        if checkpoint_raw and source_path
        else Path("")
    )
    required_status = str(config.get("expected_phase28_training_status", ACCEPTED))
    require_allowed = bool(config.get("required_phase29_reduced_clone_evaluation_allowed", True))
    failures = []
    if load_failure:
        failures.append(load_failure)
    if source.get("phase28_equity_imitation_training_status") != required_status:
        failures.append(
            f"source Phase 28 training status is {source.get('phase28_equity_imitation_training_status')!r}, "
            f"expected {required_status!r}"
        )
    if require_allowed and not bool(source.get("phase29_reduced_clone_evaluation_allowed", False)):
        failures.append("source Phase 28 training did not allow Phase 29 reduced clone evaluation")
    if not checkpoint_raw:
        failures.append("source Phase 28 training report does not include candidate_checkpoint_path")
    if checkpoint_raw and not checkpoint.is_file():
        failures.append(f"candidate checkpoint does not exist: {checkpoint}")
    return {
        "source_phase28_training_report_path": source_path,
        "source_phase28_training_status": str(source.get("phase28_equity_imitation_training_status", "")),
        "source_phase28_training_failures": source.get("phase28_equity_imitation_training_failures", []),
        "source_phase29_reduced_clone_evaluation_allowed": bool(source.get("phase29_reduced_clone_evaluation_allowed", False)),
        "source_collection_report_path": source.get("source_collection_report_path", ""),
        "source_dataset_path": source.get("dataset_path", ""),
        "candidate_checkpoint_path": str(checkpoint) if checkpoint_raw else "",
        "candidate_checkpoint_path_from_report": checkpoint_raw,
        "basemodel_root": str(basemodel_root),
        "prerequisite_failures": failures,
        "prerequisite_passed": not failures,
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def summarize_reduced_results(results_by_tournament: List[Dict[str, Any]]) -> Dict[str, Any]:
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
            if class_name == "ReducedModelEngineBot":
                model_rows.append(dict(row, tournament_id=tournament.get("tournament_id")))

    for class_name, placement in placement_summary.items():
        count = max(1, int(placement["count"]))
        placement["average_position"] = placement["position_sum"] / count
        placement["win_rate"] = placement["wins"] / count
        placement["top_3_rate"] = placement["top_3"] / count
        placement["itm_rate"] = placement["itm"] / count
        payout_summary[class_name]["average_payout_pct"] = payout_summary[class_name]["payout_sum"] / count

    model_positions = [int(row.get("position", 0) or 0) for row in model_rows]
    model_payouts = [float(row.get("payout_pct", 0.0) or 0.0) for row in model_rows]
    model_entries = len(model_rows)
    return {
        "completed_tournament_count": valid_tournaments,
        "stopped_max_hands_count": stopped_max_hands,
        "placement_summary_by_bot_class": placement_summary,
        "payout_summary_by_bot_class": payout_summary,
        "model_bot_summary": {
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
        },
    }


def build_bot_class_stats_table(
    placement_by_class: Dict[str, Dict[str, Any]],
    payout_by_class: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows = []
    for bot_class in sorted(placement_by_class):
        placement = dict(placement_by_class.get(bot_class, {}))
        payout = dict(payout_by_class.get(bot_class, {}))
        count = int(placement.get("count", payout.get("count", 0)) or 0)
        rows.append(
            {
                "bot_class": bot_class,
                "entries": count,
                "average_position": placement.get("average_position"),
                "itm_rate": float(placement.get("itm_rate", 0.0) or 0.0),
                "win_rate": float(placement.get("win_rate", 0.0) or 0.0),
                "top_3_rate": float(placement.get("top_3_rate", 0.0) or 0.0),
                "total_payout_pct": float(payout.get("payout_sum", 0.0) or 0.0),
                "average_payout_pct": float(payout.get("average_payout_pct", 0.0) or 0.0),
            }
        )
    return rows


def build_phase29_lineup(config: Dict[str, Any], checkpoint_path: str, basemodel_root: str, engine_root: Path) -> List[Any]:
    patched = dict(config)
    patched_lineup = dict(config.get("lineup", {}))
    model_count = int(patched_lineup.get("model", 1))
    patched_lineup["model"] = 0
    patched["lineup"] = patched_lineup
    bots: List[Any] = []
    for index in range(model_count):
        bots.append(
            ReducedModelEngineBot(
                checkpoint_path,
                basemodel_root=basemodel_root,
                engine_root=engine_root,
                name=f"Phase29ReducedModelBot_{index + 1}",
                deterministic=bool(config.get("deterministic", True)),
                decision_timeout_ms=int(config.get("bot_decision_timeout_ms", 500)),
                equity_source=str(config.get("equity_source", "treys")),
                equity_fallback_source=config.get("equity_fallback_source", "constant"),
                equity_iterations=config.get("equity_iterations"),
                observation_size=int(config.get("reduced_observation_size", 8)),
                require_checkpoint=True,
            )
        )
    bots.extend(build_lineup(patched, checkpoint_path="", basemodel_root=basemodel_root))
    return bots


def run_campaign(config: Dict[str, Any], prerequisite: Dict[str, Any], engine_root: Path) -> Dict[str, Any]:
    tournament_count = int(config.get("tournament_count", 1))
    seed = int(config.get("random_seed", 0))
    campaign_root = Path(str(config["campaign_results_dir"]))
    model_name_prefix = "Phase29ReducedModelBot_"
    bots = build_phase29_lineup(config, prerequisite["candidate_checkpoint_path"], prerequisite["basemodel_root"], engine_root)
    model_bots = [bot for bot in bots if isinstance(bot, ReducedModelEngineBot)]
    fallback_before = [dict(getattr(bot, "fallback_counts", {})) for bot in model_bots]
    equity_before = [dict(getattr(bot, "equity_counts", {})) for bot in model_bots]
    result_rows = []
    event_summaries = []
    event_log_paths = []
    tournament_failures = []
    with temporary_engine_config(_engine_overrides(config)):
        for index in range(tournament_count):
            tournament_id = index + 1
            _seed_everything(seed + index)
            try:
                tournament = Tournament(bots, tournament_id=tournament_id)
                results, events = tournament.play()
                stopped = any(event.get("type") == "tournament_stopped_max_hands" for event in events)
                event_summaries.append(summarize_phase26_events(tournament_id, events, results, model_name_prefix))
                if should_write_full_event_log(config, tournament_id):
                    event_log_path = campaign_root / "events" / f"tournament_{tournament_id}_events.json"
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
    fallback_after = [dict(getattr(bot, "fallback_counts", {})) for bot in model_bots]
    equity_after = [dict(getattr(bot, "equity_counts", {})) for bot in model_bots]
    fallback_deltas = [_subtract_counts(after, before) for before, after in zip(fallback_before, fallback_after)]
    equity_deltas = [_subtract_counts(after, before) for before, after in zip(equity_before, equity_after)]
    campaign = {
        "checkpoint_path": prerequisite["candidate_checkpoint_path"],
        "lineup_summary": lineup_summary(bots),
        "tournament_results": result_rows,
        "event_summaries": event_summaries,
        "event_log_paths": event_log_paths,
        "tournament_failures": tournament_failures,
        "summary": summarize_reduced_results(result_rows),
        "action_mix_summary": _action_mix_summary(event_summaries),
        "bot_fallback_summary": {
            "model_bot_count": len(model_bots),
            "per_model_bot": fallback_deltas,
            "totals": _sum_count_dicts(fallback_deltas),
        },
        "model_equity_summary": {
            "per_model_bot": equity_deltas,
            "totals": _sum_count_dicts(equity_deltas),
        },
    }
    _write_json(campaign_root / "tournament_results.json", result_rows)
    _write_json(campaign_root / "event_summaries.json", event_summaries)
    _write_json(campaign_root / "campaign_summary.json", {key: value for key, value in campaign.items() if key not in {"tournament_results", "event_summaries"}})
    return campaign


def build_report(
    config: Dict[str, Any],
    prerequisite: Dict[str, Any],
    campaign: Dict[str, Any],
    started_at: str,
    runtime_seconds: float,
) -> Dict[str, Any]:
    acceptance = dict(config.get("acceptance", {}))
    summary = dict(campaign.get("summary", {}))
    model = dict(summary.get("model_bot_summary", {}))
    placement_by_class = dict(summary.get("placement_summary_by_bot_class", {}))
    payout_by_class = dict(summary.get("payout_summary_by_bot_class", {}))
    bot_class_stats_table = build_bot_class_stats_table(placement_by_class, payout_by_class)
    random_summary = dict(placement_by_class.get("RandomBot", {}))
    equity_summary = dict(placement_by_class.get("AggressiveBot", {}))
    equity_payout = dict(payout_by_class.get("AggressiveBot", {}))
    model_avg = model.get("average_position")
    random_avg = random_summary.get("average_position")
    equity_avg = equity_summary.get("average_position")
    fallback_totals = dict(campaign.get("bot_fallback_summary", {}).get("totals", {}))
    action_mix = dict(campaign.get("action_mix_summary", {}))
    tournament_count = int(config.get("tournament_count", 0))
    stopped_rate = _rate(int(summary.get("stopped_max_hands_count", 0)), max(1, tournament_count))
    gates = {
        "phase28_prerequisite_gate_passed": bool(prerequisite.get("prerequisite_passed", False)),
        "campaign_runtime_gate_passed": not campaign.get("tournament_failures"),
        "completed_tournament_gate_passed": int(summary.get("completed_tournament_count", 0))
        >= int(acceptance.get("min_completed_tournaments", 1)),
        "model_entry_gate_passed": int(model.get("entries", 0) or 0) >= int(acceptance.get("min_model_entries", 1)),
        "stopped_max_hands_rate_gate_passed": stopped_rate <= float(acceptance.get("max_stopped_max_hands_rate", 0.2)),
        "model_inference_error_gate_passed": int(fallback_totals.get("inference_errors", 0))
        <= int(acceptance.get("max_model_inference_errors", 0)),
        "model_timeout_fallback_gate_passed": int(fallback_totals.get("timeouts", 0))
        <= int(acceptance.get("max_model_timeout_fallbacks", 0)),
        "model_illegal_action_gate_passed": int(fallback_totals.get("illegal_actions", 0))
        <= int(acceptance.get("max_model_illegal_actions", 0)),
        "random_baseline_available_gate_passed": random_avg is not None,
        "equity_baseline_available_gate_passed": equity_avg is not None,
        "beats_random_average_position_gate_passed": model_avg is not None
        and random_avg is not None
        and float(model_avg) <= float(random_avg) + float(acceptance.get("max_random_average_position_margin", 0.0)),
        "approaches_equity_average_position_gate_passed": model_avg is not None
        and equity_avg is not None
        and float(model_avg) <= float(equity_avg) + float(acceptance.get("max_equity_average_position_gap", 10.0)),
        "action_mix_gate_passed": int(action_mix.get("distinct_model_actions", 0))
        >= int(acceptance.get("min_distinct_model_actions", 1))
        and float(action_mix.get("top_model_action_rate", 0.0)) <= float(acceptance.get("max_single_model_action_rate", 1.0)),
    }
    failures = list(prerequisite.get("prerequisite_failures", []))
    failures.extend(str(failure) for failure in campaign.get("tournament_failures", []))
    for gate, passed in gates.items():
        if not passed:
            failures.append(f"{gate} failed")
    status = ACCEPTED if not failures else REJECTED
    comparison = {
        "model_average_position": model_avg,
        "random_average_position": random_avg,
        "equity_average_position": equity_avg,
        "model_minus_random_average_position": float(model_avg) - float(random_avg)
        if model_avg is not None and random_avg is not None
        else None,
        "model_minus_equity_average_position": float(model_avg) - float(equity_avg)
        if model_avg is not None and equity_avg is not None
        else None,
        "model_itm_rate": float(model.get("itm_rate", 0.0) or 0.0),
        "random_itm_rate": float(random_summary.get("itm_rate", 0.0) or 0.0),
        "equity_itm_rate": float(equity_summary.get("itm_rate", 0.0) or 0.0),
        "model_total_payout_pct": float(model.get("total_payout_pct", 0.0) or 0.0),
        "equity_total_payout_pct": float(equity_payout.get("payout_sum", 0.0) or 0.0),
    }
    return {
        "phase29_reduced_clone_engine_evaluation_id": config.get(
            "phase29_reduced_clone_engine_evaluation_id", "phase29_reduced_clone_engine_evaluation"
        ),
        "phase29_reduced_clone_engine_evaluation_status": status,
        "phase29_reduced_clone_engine_evaluation_failures": list(dict.fromkeys(failures)),
        "phase29_reduced_clone_engine_evaluation_report_path": str(
            config.get("reduced_clone_engine_evaluation_report_path", "")
        ),
        "phase29_reduced_clone_engine_evaluation_started_at": started_at,
        "phase29_reduced_clone_engine_evaluation_finished_at": _utc_now(),
        "source_phase28_training_report_path": prerequisite.get("source_phase28_training_report_path", ""),
        "source_phase28_training_status": prerequisite.get("source_phase28_training_status", ""),
        "source_collection_report_path": prerequisite.get("source_collection_report_path", ""),
        "source_dataset_path": prerequisite.get("source_dataset_path", ""),
        "candidate_checkpoint_path": prerequisite.get("candidate_checkpoint_path", ""),
        "lineup_summary": campaign.get("lineup_summary", {}),
        "tournament_count": tournament_count,
        "completed_tournament_count": int(summary.get("completed_tournament_count", 0)),
        "stopped_max_hands_rate": stopped_rate,
        "model_summary": model,
        "placement_summary_by_bot_class": placement_by_class,
        "payout_summary_by_bot_class": payout_by_class,
        "bot_class_stats_table": bot_class_stats_table,
        "comparison_summary": comparison,
        "action_mix_summary": action_mix,
        "bot_fallback_summary": campaign.get("bot_fallback_summary", {}),
        "model_equity_summary": campaign.get("model_equity_summary", {}),
        "artifact_paths": {
            "artifact_root": str(config.get("artifact_root", "")),
            "campaign_results_dir": str(config.get("campaign_results_dir", "")),
            "reduced_clone_engine_evaluation_report_path": str(
                config.get("reduced_clone_engine_evaluation_report_path", "")
            ),
            "event_log_paths": list(campaign.get("event_log_paths", [])),
        },
        "runtime_summary": {
            "runtime_seconds": runtime_seconds,
            "event_summary_count": len(campaign.get("event_summaries", [])),
            "sampled_event_log_count": len(campaign.get("event_log_paths", [])),
        },
        "phase29_gate_results": gates,
        "phase30_reduced_clone_decision_allowed": status == ACCEPTED,
        "next_phase_recommendation": "plan_phase30_reduced_clone_decision"
        if status == ACCEPTED
        else "inspect_phase29_reduced_clone_engine_evaluation",
        "config_path": config.get("config_path", ""),
    }


def run_phase29_reduced_clone_engine_evaluation(config: Dict[str, Any], config_path: str | Path | None = None) -> Dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    config = _resolve_paths(config)
    if config_path is not None:
        config["config_path"] = str(config_path)
    engine_root = Path(__file__).resolve().parents[1]
    artifact_root = Path(str(config["artifact_root"]))
    campaign_root = Path(str(config["campaign_results_dir"]))
    artifact_root.mkdir(parents=True, exist_ok=True)
    campaign_root.mkdir(parents=True, exist_ok=True)
    prerequisite = verify_prerequisite(config, engine_root)
    campaign: Dict[str, Any] = {}
    if prerequisite.get("prerequisite_passed"):
        try:
            campaign = run_campaign(config, prerequisite, engine_root)
        except Exception as exc:
            campaign = {"tournament_failures": [f"evaluation setup failed: {exc}"]}
    report = build_report(config, prerequisite, campaign, started_at, time.perf_counter() - started)
    _write_json(str(config["reduced_clone_engine_evaluation_report_path"]), report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase29_reduced_clone_engine_evaluation.json")
    args = parser.parse_args(argv)
    report = run_phase29_reduced_clone_engine_evaluation(load_config(args.config), config_path=args.config)
    print(
        json.dumps(
            {
                "phase29_reduced_clone_engine_evaluation_status": report[
                    "phase29_reduced_clone_engine_evaluation_status"
                ],
                "candidate_checkpoint_path": report["candidate_checkpoint_path"],
                "completed_tournament_count": report["completed_tournament_count"],
                "comparison_summary": report["comparison_summary"],
                "bot_class_stats_table": report["bot_class_stats_table"],
                "phase30_reduced_clone_decision_allowed": report["phase30_reduced_clone_decision_allowed"],
                "next_phase_recommendation": report["next_phase_recommendation"],
                "phase29_reduced_clone_engine_evaluation_report_path": report[
                    "phase29_reduced_clone_engine_evaluation_report_path"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["phase29_reduced_clone_engine_evaluation_status"] != ACCEPTED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
