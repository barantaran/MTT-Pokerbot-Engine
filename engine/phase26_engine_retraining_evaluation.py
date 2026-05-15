"""Phase 26 paired engine evaluation for a Phase 25 retrained checkpoint."""

from __future__ import annotations

import argparse
import glob
import json
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
from engine.phase23_larger_mtt_engine_simulation import summarize_results
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


def resolve_phase25_report(config: Dict[str, Any]) -> tuple[str, Dict[str, Any], str]:
    configured_path = str(config.get("phase25_report_path", "") or "")
    required_status = str(config.get("expected_phase25_status", ACCEPTED))
    if configured_path:
        report = _load_report(configured_path)
        if report is None:
            return configured_path, {}, "source Phase 25 report is missing or malformed"
        return configured_path, report, ""

    pattern = str(config.get("phase25_report_glob", "") or "")
    if not pattern:
        return "", {}, "phase25_report_path or phase25_report_glob is required"

    matches = sorted(glob.glob(pattern), key=lambda item: Path(item).stat().st_mtime, reverse=True)
    malformed_count = 0
    for path in matches:
        report = _load_report(path)
        if report is None:
            malformed_count += 1
            continue
        if report.get("phase25_engine_rollout_training_status") == required_status and bool(
            report.get("phase26_engine_evaluation_allowed", False)
        ):
            return path, report, ""
    if matches:
        suffix = f"; {malformed_count} malformed candidates skipped" if malformed_count else ""
        return matches[0], _load_report(matches[0]) or {}, f"no accepted Phase 25 report matched {pattern}{suffix}"
    return "", {}, f"no Phase 25 reports matched {pattern}"


def _resolve_checkpoint(raw_path: str, *, basemodel_root: Path, engine_root: Path) -> Path:
    return resolve_checkpoint_path(raw_path, basemodel_root=basemodel_root, engine_root=engine_root)


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(config)
    run_id = str(resolved.get("phase26_engine_retraining_evaluation_id", "phase26_engine_retraining_evaluation"))
    artifact_root_value = str(resolved.get("artifact_root", "") or "")
    if artifact_root_value:
        artifact_root = Path(artifact_root_value)
    else:
        artifact_root = Path(str(resolved.get("output_dir", Path("runs") / run_id))) / _run_stamp()
    resolved["artifact_root"] = str(artifact_root)
    resolved.setdefault("engine_retraining_evaluation_report_path", str(artifact_root / "engine_retraining_evaluation_report.json"))
    resolved.setdefault("campaign_results_dir", str(artifact_root / "campaigns"))
    return resolved


def verify_prerequisite(config: Dict[str, Any], engine_root: Path) -> Dict[str, Any]:
    source_path, source, load_failure = resolve_phase25_report(config)
    basemodel_root = Path(str(config.get("basemodel_root", "../poker-ai-basemodel"))).expanduser()
    if not basemodel_root.is_absolute():
        basemodel_root = (engine_root / basemodel_root).resolve()

    source_checkpoint_raw = str(config.get("source_checkpoint_path") or source.get("source_checkpoint_path") or "")
    candidate_checkpoint_raw = str(config.get("candidate_checkpoint_path") or source.get("candidate_checkpoint_path") or "")
    source_checkpoint = _resolve_checkpoint(source_checkpoint_raw, basemodel_root=basemodel_root, engine_root=engine_root) if source_checkpoint_raw else Path("")
    candidate_checkpoint = (
        _resolve_checkpoint(candidate_checkpoint_raw, basemodel_root=basemodel_root, engine_root=engine_root)
        if candidate_checkpoint_raw
        else Path("")
    )

    required_status = str(config.get("expected_phase25_status", ACCEPTED))
    require_allowed = bool(config.get("required_phase26_engine_evaluation_allowed", True))
    failures = []
    if load_failure:
        failures.append(load_failure)
    if source.get("phase25_engine_rollout_training_status") != required_status:
        failures.append(
            f"source Phase 25 status is {source.get('phase25_engine_rollout_training_status')!r}, "
            f"expected {required_status!r}"
        )
    if require_allowed and not bool(source.get("phase26_engine_evaluation_allowed", False)):
        failures.append("source Phase 25 did not allow Phase 26 engine evaluation")
    if not source_checkpoint_raw:
        failures.append("source Phase 25 report does not include source_checkpoint_path")
    if not candidate_checkpoint_raw:
        failures.append("source Phase 25 report does not include candidate_checkpoint_path")
    if source_checkpoint_raw and not source_checkpoint.is_file():
        failures.append(f"source checkpoint does not exist: {source_checkpoint}")
    if candidate_checkpoint_raw and not candidate_checkpoint.is_file():
        failures.append(f"candidate checkpoint does not exist: {candidate_checkpoint}")

    return {
        "source_phase25_report_path": source_path,
        "source_phase25_status": str(source.get("phase25_engine_rollout_training_status", "")),
        "source_phase25_failures": source.get("phase25_engine_rollout_training_failures", []),
        "source_phase26_engine_evaluation_allowed": bool(source.get("phase26_engine_evaluation_allowed", False)),
        "source_phase24_report_path": source.get("source_phase24_report_path", ""),
        "source_checkpoint_path": str(source_checkpoint) if source_checkpoint_raw else "",
        "source_checkpoint_path_from_report": source_checkpoint_raw,
        "candidate_checkpoint_path": str(candidate_checkpoint) if candidate_checkpoint_raw else "",
        "candidate_checkpoint_path_from_report": candidate_checkpoint_raw,
        "basemodel_root": str(basemodel_root),
        "prerequisite_failures": failures,
        "prerequisite_passed": not failures,
    }


def build_phase26_lineup(config: Dict[str, Any], checkpoint_path: str, basemodel_root: str, label: str) -> List[Any]:
    patched = dict(config)
    patched_lineup = dict(config.get("lineup", {}))
    model_count = int(patched_lineup.get("model", 1))
    patched_lineup["model"] = 0
    patched["lineup"] = patched_lineup

    bots: List[Any] = []
    for index in range(model_count):
        bots.append(
            BaselineModelEngineBot(
                checkpoint_path,
                basemodel_root=basemodel_root,
                name=f"Phase26{label.title()}ModelBot_{index + 1}",
                deterministic=bool(config.get("deterministic", True)),
                decision_timeout_ms=int(config.get("bot_decision_timeout_ms", 500)),
                equity_source=str(config.get("equity_source", "treys")),
                equity_fallback_source=config.get("equity_fallback_source", "constant"),
                equity_iterations=config.get("equity_iterations"),
                require_checkpoint=True,
            )
        )
    bots.extend(build_lineup(patched, checkpoint_path, basemodel_root))
    return bots


def should_write_full_event_log(config: Dict[str, Any], tournament_id: int) -> bool:
    event_logging = dict(config.get("event_logging", {}))
    if bool(event_logging.get("write_full_event_logs", False)):
        return True
    return tournament_id <= int(event_logging.get("write_full_event_logs_for_first_n", 0))


def summarize_phase26_events(
    tournament_id: int,
    events: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    model_name_prefix: str,
) -> Dict[str, Any]:
    type_counts: Dict[str, int] = {}
    action_counts: Dict[str, int] = {}
    model_action_counts: Dict[str, int] = {}
    for event in events:
        event_type = str(event.get("type", "unknown"))
        type_counts[event_type] = type_counts.get(event_type, 0) + 1
        if event_type == "action":
            action = str(event.get("action", "unknown"))
            action_counts[action] = action_counts.get(action, 0) + 1
            if str(event.get("player", "")).startswith(model_name_prefix):
                model_action_counts[action] = model_action_counts.get(action, 0) + 1
    return {
        "tournament_id": tournament_id,
        "event_count": len(events),
        "result_count": len(results),
        "winner": next((str(event.get("player", "")) for event in events if event.get("type") == "tournament_win"), ""),
        "stopped_max_hands": any(event.get("type") == "tournament_stopped_max_hands" for event in events),
        "hand_count": int(type_counts.get("hand_end", 0)),
        "knockout_count": int(type_counts.get("knockout", 0)),
        "level_up_count": int(type_counts.get("level_up", 0)),
        "table_broken_count": int(type_counts.get("table_broken", 0)),
        "event_type_counts": type_counts,
        "action_counts": action_counts,
        "model_action_counts": model_action_counts,
    }


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _action_mix_summary(event_summaries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    for summary in event_summaries:
        for action, count in dict(summary.get("model_action_counts", {})).items():
            counts[str(action)] = counts.get(str(action), 0) + int(count)
    total = sum(counts.values())
    distinct = sum(1 for count in counts.values() if count > 0)
    top_action = max(counts, key=counts.get) if counts else None
    top_count = int(counts.get(str(top_action), 0)) if top_action else 0
    return {
        "model_action_counts": counts,
        "model_action_total": total,
        "distinct_model_actions": distinct,
        "top_model_action": top_action,
        "top_model_action_rate": _rate(top_count, total),
    }


def run_campaign(
    config: Dict[str, Any],
    *,
    label: str,
    checkpoint_path: str,
    basemodel_root: str,
    campaign_root: Path,
) -> Dict[str, Any]:
    tournament_count = int(config.get("tournament_count", 1))
    seed = int(config.get("random_seed", 0))
    model_name_prefix = f"Phase26{label.title()}ModelBot_"
    bots = build_phase26_lineup(config, checkpoint_path, basemodel_root, label)
    model_bots = [bot for bot in bots if isinstance(bot, BaselineModelEngineBot)]
    fallback_before = [_fallback_counts(bot) for bot in model_bots]
    equity_before = [_equity_counts(bot) for bot in model_bots]
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
                event_summary = summarize_phase26_events(tournament_id, events, results, model_name_prefix)
                event_summaries.append(event_summary)
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
                tournament_failures.append(f"{label} tournament {tournament_id} failed: {exc}")

    fallback_after = [_fallback_counts(bot) for bot in model_bots]
    equity_after = [_equity_counts(bot) for bot in model_bots]
    model_fallback_deltas = [_subtract_counts(after, before) for before, after in zip(fallback_before, fallback_after)]
    model_equity_deltas = [_subtract_counts(after, before) for before, after in zip(equity_before, equity_after)]
    summary = summarize_results(result_rows)
    action_mix = _action_mix_summary(event_summaries)
    campaign = {
        "label": label,
        "checkpoint_path": checkpoint_path,
        "lineup_summary": lineup_summary(bots),
        "tournament_results": result_rows,
        "event_summaries": event_summaries,
        "event_log_paths": event_log_paths,
        "tournament_failures": tournament_failures,
        "summary": summary,
        "action_mix_summary": action_mix,
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
    _write_json(campaign_root / "tournament_results.json", result_rows)
    _write_json(campaign_root / "event_summaries.json", event_summaries)
    _write_json(campaign_root / "campaign_summary.json", {key: value for key, value in campaign.items() if key not in {"tournament_results", "event_summaries"}})
    return campaign


def compare_campaigns(source: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    source_model = dict(source.get("summary", {}).get("model_bot_summary", {}))
    candidate_model = dict(candidate.get("summary", {}).get("model_bot_summary", {}))
    source_avg = source_model.get("average_position")
    candidate_avg = candidate_model.get("average_position")
    return {
        "source_model_average_position": source_avg,
        "candidate_model_average_position": candidate_avg,
        "average_position_delta": float(candidate_avg) - float(source_avg)
        if source_avg is not None and candidate_avg is not None
        else None,
        "source_model_itm_rate": float(source_model.get("itm_rate", 0.0) or 0.0),
        "candidate_model_itm_rate": float(candidate_model.get("itm_rate", 0.0) or 0.0),
        "itm_rate_delta": float(candidate_model.get("itm_rate", 0.0) or 0.0)
        - float(source_model.get("itm_rate", 0.0) or 0.0),
        "source_model_win_rate": float(source_model.get("win_rate", 0.0) or 0.0),
        "candidate_model_win_rate": float(candidate_model.get("win_rate", 0.0) or 0.0),
        "win_rate_delta": float(candidate_model.get("win_rate", 0.0) or 0.0)
        - float(source_model.get("win_rate", 0.0) or 0.0),
        "source_model_total_payout_pct": float(source_model.get("total_payout_pct", 0.0) or 0.0),
        "candidate_model_total_payout_pct": float(candidate_model.get("total_payout_pct", 0.0) or 0.0),
        "total_payout_pct_delta": float(candidate_model.get("total_payout_pct", 0.0) or 0.0)
        - float(source_model.get("total_payout_pct", 0.0) or 0.0),
    }


def build_report(
    config: Dict[str, Any],
    prerequisite: Dict[str, Any],
    campaigns: Dict[str, Dict[str, Any]],
    started_at: str,
    runtime_seconds: float,
) -> Dict[str, Any]:
    acceptance = dict(config.get("acceptance", {}))
    source = dict(campaigns.get("source", {}))
    candidate = dict(campaigns.get("candidate", {}))
    comparison = compare_campaigns(source, candidate) if source and candidate else {}
    tournament_count = int(config.get("tournament_count", 0))
    source_summary = dict(source.get("summary", {}))
    candidate_summary = dict(candidate.get("summary", {}))
    source_fallbacks = dict(source.get("bot_fallback_summary", {}).get("totals", {}))
    candidate_fallbacks = dict(candidate.get("bot_fallback_summary", {}).get("totals", {}))
    candidate_action_mix = dict(candidate.get("action_mix_summary", {}))
    max_position_regression = float(acceptance.get("max_average_position_regression", 2.0))
    max_itm_regression = float(acceptance.get("max_itm_rate_regression", 0.2))
    max_payout_regression = float(acceptance.get("max_total_payout_pct_regression", 1.0))
    max_stopped_rate = float(acceptance.get("max_stopped_max_hands_rate", 0.2))
    candidate_stopped_rate = _rate(int(candidate_summary.get("stopped_max_hands_count", 0)), max(1, tournament_count))
    source_stopped_rate = _rate(int(source_summary.get("stopped_max_hands_count", 0)), max(1, tournament_count))
    average_position_delta = comparison.get("average_position_delta")
    gates = {
        "phase25_prerequisite_gate_passed": bool(prerequisite.get("prerequisite_passed", False)),
        "source_campaign_runtime_gate_passed": not source.get("tournament_failures"),
        "candidate_campaign_runtime_gate_passed": not candidate.get("tournament_failures"),
        "source_completed_tournament_gate_passed": int(source_summary.get("completed_tournament_count", 0))
        >= int(acceptance.get("min_completed_tournaments", 1)),
        "candidate_completed_tournament_gate_passed": int(candidate_summary.get("completed_tournament_count", 0))
        >= int(acceptance.get("min_completed_tournaments", 1)),
        "source_stopped_max_hands_rate_gate_passed": source_stopped_rate <= max_stopped_rate,
        "candidate_stopped_max_hands_rate_gate_passed": candidate_stopped_rate <= max_stopped_rate,
        "source_model_inference_error_gate_passed": int(source_fallbacks.get("inference_errors", 0))
        <= int(acceptance.get("max_model_inference_errors", 0)),
        "candidate_model_inference_error_gate_passed": int(candidate_fallbacks.get("inference_errors", 0))
        <= int(acceptance.get("max_model_inference_errors", 0)),
        "source_model_timeout_fallback_gate_passed": int(source_fallbacks.get("timeouts", 0))
        <= int(acceptance.get("max_model_timeout_fallbacks", 0)),
        "candidate_model_timeout_fallback_gate_passed": int(candidate_fallbacks.get("timeouts", 0))
        <= int(acceptance.get("max_model_timeout_fallbacks", 0)),
        "average_position_regression_gate_passed": average_position_delta is not None
        and float(average_position_delta) <= max_position_regression,
        "itm_rate_regression_gate_passed": float(comparison.get("itm_rate_delta", -1.0)) >= -max_itm_regression,
        "total_payout_regression_gate_passed": float(comparison.get("total_payout_pct_delta", -999.0))
        >= -max_payout_regression,
        "candidate_action_mix_gate_passed": int(candidate_action_mix.get("distinct_model_actions", 0))
        >= int(acceptance.get("min_candidate_distinct_model_actions", 1))
        and float(candidate_action_mix.get("top_model_action_rate", 0.0))
        <= float(acceptance.get("max_candidate_single_model_action_rate", 1.0)),
    }
    failures = list(prerequisite.get("prerequisite_failures", []))
    failures.extend(str(failure) for failure in source.get("tournament_failures", []))
    failures.extend(str(failure) for failure in candidate.get("tournament_failures", []))
    for gate, passed in gates.items():
        if not passed:
            failures.append(f"{gate} failed")

    status = ACCEPTED if not failures else REJECTED
    return {
        "phase26_engine_retraining_evaluation_id": config.get(
            "phase26_engine_retraining_evaluation_id", "phase26_engine_retraining_evaluation"
        ),
        "phase26_engine_retraining_evaluation_status": status,
        "phase26_engine_retraining_evaluation_failures": list(dict.fromkeys(failures)),
        "phase26_engine_retraining_evaluation_report_path": str(
            config.get("engine_retraining_evaluation_report_path", "")
        ),
        "phase26_engine_retraining_evaluation_started_at": started_at,
        "phase26_engine_retraining_evaluation_finished_at": _utc_now(),
        "source_phase25_report_path": prerequisite.get("source_phase25_report_path", ""),
        "source_phase25_status": prerequisite.get("source_phase25_status", ""),
        "source_phase25_failures": prerequisite.get("source_phase25_failures", []),
        "source_phase26_engine_evaluation_allowed": prerequisite.get(
            "source_phase26_engine_evaluation_allowed", False
        ),
        "source_phase24_report_path": prerequisite.get("source_phase24_report_path", ""),
        "source_checkpoint_path": prerequisite.get("source_checkpoint_path", ""),
        "candidate_checkpoint_path": prerequisite.get("candidate_checkpoint_path", ""),
        "lineup_summary": candidate.get("lineup_summary", source.get("lineup_summary", {})),
        "tournament_count": tournament_count,
        "source_completed_tournament_count": int(source_summary.get("completed_tournament_count", 0)),
        "candidate_completed_tournament_count": int(candidate_summary.get("completed_tournament_count", 0)),
        "source_stopped_max_hands_rate": source_stopped_rate,
        "candidate_stopped_max_hands_rate": candidate_stopped_rate,
        "source_model_summary": source_summary.get("model_bot_summary", {}),
        "candidate_model_summary": candidate_summary.get("model_bot_summary", {}),
        "comparison_summary": comparison,
        "source_action_mix_summary": source.get("action_mix_summary", {}),
        "candidate_action_mix_summary": candidate_action_mix,
        "source_bot_fallback_summary": source.get("bot_fallback_summary", {}),
        "candidate_bot_fallback_summary": candidate.get("bot_fallback_summary", {}),
        "source_model_equity_summary": source.get("model_equity_summary", {}),
        "candidate_model_equity_summary": candidate.get("model_equity_summary", {}),
        "artifact_paths": {
            "artifact_root": str(config.get("artifact_root", "")),
            "campaign_results_dir": str(config.get("campaign_results_dir", "")),
            "source_campaign_dir": str(Path(str(config.get("campaign_results_dir", ""))) / "source"),
            "candidate_campaign_dir": str(Path(str(config.get("campaign_results_dir", ""))) / "candidate"),
            "engine_retraining_evaluation_report_path": str(
                config.get("engine_retraining_evaluation_report_path", "")
            ),
        },
        "runtime_summary": {
            "runtime_seconds": runtime_seconds,
            "source_event_summary_count": len(source.get("event_summaries", [])),
            "candidate_event_summary_count": len(candidate.get("event_summaries", [])),
            "source_sampled_event_log_count": len(source.get("event_log_paths", [])),
            "candidate_sampled_event_log_count": len(candidate.get("event_log_paths", [])),
        },
        "phase26_gate_results": gates,
        "phase27_promotion_decision_allowed": status == ACCEPTED,
        "next_phase_recommendation": "plan_phase27_engine_checkpoint_promotion_decision"
        if status == ACCEPTED
        else "inspect_phase26_engine_retraining_evaluation",
        "config_path": config.get("config_path", ""),
    }


def run_phase26_engine_retraining_evaluation(config: Dict[str, Any], config_path: str | Path | None = None) -> Dict[str, Any]:
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
    campaigns: Dict[str, Dict[str, Any]] = {"source": {}, "candidate": {}}

    if prerequisite.get("prerequisite_passed"):
        try:
            campaigns["source"] = run_campaign(
                config,
                label="source",
                checkpoint_path=str(prerequisite["source_checkpoint_path"]),
                basemodel_root=str(prerequisite["basemodel_root"]),
                campaign_root=campaign_root / "source",
            )
            campaigns["candidate"] = run_campaign(
                config,
                label="candidate",
                checkpoint_path=str(prerequisite["candidate_checkpoint_path"]),
                basemodel_root=str(prerequisite["basemodel_root"]),
                campaign_root=campaign_root / "candidate",
            )
        except Exception as exc:
            campaigns["candidate"] = {"tournament_failures": [f"evaluation setup failed: {exc}"]}

    report = build_report(config, prerequisite, campaigns, started_at, time.perf_counter() - started)
    _write_json(str(config["engine_retraining_evaluation_report_path"]), report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase26_engine_retraining_evaluation.json")
    args = parser.parse_args(argv)

    report = run_phase26_engine_retraining_evaluation(load_config(args.config), config_path=args.config)
    print(
        json.dumps(
            {
                "phase26_engine_retraining_evaluation_status": report[
                    "phase26_engine_retraining_evaluation_status"
                ],
                "source_checkpoint_path": report["source_checkpoint_path"],
                "candidate_checkpoint_path": report["candidate_checkpoint_path"],
                "source_completed_tournament_count": report["source_completed_tournament_count"],
                "candidate_completed_tournament_count": report["candidate_completed_tournament_count"],
                "comparison_summary": report["comparison_summary"],
                "phase27_promotion_decision_allowed": report["phase27_promotion_decision_allowed"],
                "next_phase_recommendation": report["next_phase_recommendation"],
                "phase26_engine_retraining_evaluation_report_path": report[
                    "phase26_engine_retraining_evaluation_report_path"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["phase26_engine_retraining_evaluation_status"] != ACCEPTED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
