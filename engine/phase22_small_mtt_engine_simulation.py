"""Phase 22 small MTT engine simulation runner."""

from __future__ import annotations

import argparse
import glob
import json
import random
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from engine.baseline_model_bot import BaselineModelEngineBot, resolve_checkpoint_path
from engine.config import config as engine_config
from engine.tournament import Tournament
from players.aggressive_bot import AggressiveBot
from players.aggressive_no_equity_bot import AggressiveNoEquityBot
from players.call_bot import CallBot
from players.noisy_equity_bot import NoisyEquityBot
from players.range_policy_bot import RangePolicyBot
from players.random_bot import RandomBot
from players.tight_equity_bot import TightEquityBot


ACCEPTED = "accepted"
REJECTED = "rejected"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_config(path: str | Path) -> Dict[str, Any]:
    loaded = _read_json(path)
    loaded["config_path"] = str(path)
    return loaded


def _load_report(path: str | Path) -> Dict[str, Any] | None:
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def resolve_phase21_report(config: Dict[str, Any]) -> tuple[str, Dict[str, Any], str]:
    configured_path = str(config.get("phase21_report_path", "") or "")
    required_status = str(config.get("expected_phase21_status", ACCEPTED))
    if configured_path:
        report = _load_report(configured_path)
        if report is None:
            return configured_path, {}, "source Phase 21 report is missing or malformed"
        return configured_path, report, ""

    pattern = str(config.get("phase21_report_glob", "") or "")
    if not pattern:
        return "", {}, "phase21_report_path or phase21_report_glob is required"

    matches = sorted(glob.glob(pattern), key=lambda item: Path(item).stat().st_mtime, reverse=True)
    malformed_count = 0
    for path in matches:
        report = _load_report(path)
        if report is None:
            malformed_count += 1
            continue
        if report.get("phase21_engine_wiring_status") == required_status and bool(
            report.get("small_mtt_engine_simulation_allowed", False)
        ):
            return path, report, ""
    if matches:
        suffix = f"; {malformed_count} malformed candidates skipped" if malformed_count else ""
        return matches[0], _load_report(matches[0]) or {}, f"no accepted Phase 21 report matched {pattern}{suffix}"
    return "", {}, f"no Phase 21 reports matched {pattern}"


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(config)
    run_id = str(resolved.get("phase22_small_engine_simulation_id", "phase22_small_mtt_engine_simulation"))
    artifact_root_value = str(resolved.get("artifact_root", "") or "")
    if artifact_root_value:
        artifact_root = Path(artifact_root_value)
    else:
        artifact_root = Path(str(resolved.get("output_dir", Path("runs") / run_id))) / _run_stamp()
    resolved["artifact_root"] = str(artifact_root)
    resolved.setdefault("small_engine_simulation_report_path", str(artifact_root / "small_mtt_engine_simulation_report.json"))
    resolved.setdefault("tournament_results_path", str(artifact_root / "tournament_results.json"))
    return resolved


def verify_prerequisite(config: Dict[str, Any], engine_root: Path) -> Dict[str, Any]:
    source_path, source, load_failure = resolve_phase21_report(config)
    promoted = str(source.get("promoted_checkpoint_path") or "")
    basemodel_root = Path(str(config.get("basemodel_root", "../poker-ai-basemodel"))).expanduser()
    if not basemodel_root.is_absolute():
        basemodel_root = (engine_root / basemodel_root).resolve()
    checkpoint = resolve_checkpoint_path(promoted, basemodel_root=basemodel_root, engine_root=engine_root) if promoted else Path("")

    required_status = str(config.get("expected_phase21_status", ACCEPTED))
    require_allowed = bool(config.get("required_small_mtt_engine_simulation_allowed", True))
    failures = []
    if load_failure:
        failures.append(load_failure)
    if source.get("phase21_engine_wiring_status") != required_status:
        failures.append(
            f"source Phase 21 status is {source.get('phase21_engine_wiring_status')!r}, expected {required_status!r}"
        )
    if require_allowed and not bool(source.get("small_mtt_engine_simulation_allowed", False)):
        failures.append("source Phase 21 did not allow small MTT engine simulation")
    if not promoted:
        failures.append("source Phase 21 report does not include promoted_checkpoint_path")
    if promoted and not checkpoint.is_file():
        failures.append(f"promoted checkpoint does not exist: {checkpoint}")

    return {
        "source_phase21_report_path": source_path,
        "source_phase21_status": str(source.get("phase21_engine_wiring_status", "")),
        "source_phase21_failures": source.get("phase21_engine_wiring_failures", []),
        "source_small_mtt_engine_simulation_allowed": bool(source.get("small_mtt_engine_simulation_allowed", False)),
        "source_phase20_report_path": source.get("source_phase20_report_path", ""),
        "source_promoted_checkpoint_path": source.get("promoted_checkpoint_path", ""),
        "promoted_checkpoint_path": str(checkpoint) if promoted else "",
        "promoted_checkpoint_path_from_report": promoted,
        "basemodel_root": str(basemodel_root),
        "prerequisite_failures": failures,
        "prerequisite_passed": not failures,
    }


def _set_bot_name(bot: Any, name: str) -> Any:
    bot.name = name
    return bot


def _bool_config(config: Dict[str, Any], key: str, default: bool = False) -> bool:
    if key not in config:
        return default
    return bool(config.get(key))


def _range_enabled(config: Dict[str, Any], key: str, default: bool | None = None) -> bool:
    if default is None:
        default = bool(config.get("use_preflop_spot_range", False))
    return _bool_config(config, key, default)


def _range_policy_variants(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    lineup_config = dict(config.get("lineup", {}))
    variants = list(config.get("range_policy_variants", []))
    default_count = int(lineup_config.get("range_policy", 0) or 0)
    if default_count:
        variants.insert(0, {"variant": "balanced", "count": default_count})
    return [dict(variant) for variant in variants]


def build_lineup(config: Dict[str, Any], checkpoint_path: str, basemodel_root: str) -> List[Any]:
    lineup_config = dict(config.get("lineup", {}))
    bots: List[Any] = []

    for index in range(int(lineup_config.get("model", 1))):
        bots.append(
            BaselineModelEngineBot(
                checkpoint_path,
                basemodel_root=basemodel_root,
                name=f"Phase22ModelBot_{index + 1}",
                deterministic=bool(config.get("deterministic", True)),
                decision_timeout_ms=int(config.get("bot_decision_timeout_ms", 500)),
                equity_source=str(config.get("equity_source", "treys")),
                equity_fallback_source=config.get("equity_fallback_source", "constant"),
                equity_iterations=config.get("equity_iterations"),
                use_preflop_spot_range=_range_enabled(config, "model_use_preflop_spot_range"),
                require_checkpoint=True,
            )
        )
    for index in range(int(lineup_config.get("random", 0))):
        bots.append(_set_bot_name(RandomBot(), f"RandomBot_{index + 1}"))
    for index in range(int(lineup_config.get("equity_aggressive", 0))):
        bots.append(
            _set_bot_name(
                AggressiveBot(use_preflop_spot_range=_range_enabled(config, "equity_aggressive_use_preflop_spot_range", False)),
                f"EquityAggressiveBot_{index + 1}",
            )
        )
    for index in range(int(lineup_config.get("tight_equity", 0))):
        bots.append(
            _set_bot_name(
                TightEquityBot(use_preflop_spot_range=_range_enabled(config, "tight_equity_use_preflop_spot_range", False)),
                f"TightEquityBot_{index + 1}",
            )
        )
    for index in range(int(lineup_config.get("noisy_equity", 0))):
        bots.append(
            _set_bot_name(
                NoisyEquityBot(use_preflop_spot_range=_range_enabled(config, "noisy_equity_use_preflop_spot_range", False)),
                f"NoisyEquityBot_{index + 1}",
            )
        )
    for variant_config in _range_policy_variants(config):
        variant = str(variant_config.get("variant", "balanced") or "balanced")
        count = int(variant_config.get("count", 1) or 0)
        bot_kwargs = {
            key: value
            for key, value in variant_config.items()
            if key not in {"count", "name"} and value is not None
        }
        bot_kwargs.setdefault("variant", variant)
        bot_kwargs.setdefault(
            "use_preflop_spot_range",
            _range_enabled(config, "range_policy_use_preflop_spot_range", True),
        )
        for index in range(count):
            bots.append(_set_bot_name(RangePolicyBot(**bot_kwargs), f"RangePolicyBot_{variant}_{index + 1}"))
    for index in range(int(lineup_config.get("aggressive_no_equity", 0))):
        bots.append(_set_bot_name(AggressiveNoEquityBot(), f"AggressiveNoEquityBot_{index + 1}"))
    for index in range(int(lineup_config.get("call", 0))):
        bots.append(_set_bot_name(CallBot(), f"CallBot_{index + 1}"))

    return bots


def lineup_summary(bots: Iterable[Any]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    names = []
    for bot in bots:
        class_name = bot.__class__.__name__
        counts[class_name] = counts.get(class_name, 0) + 1
        names.append(getattr(bot, "name", class_name))
    return {"total_bots": len(names), "bot_class_counts": counts, "bot_names": names}


@contextmanager
def temporary_engine_config(overrides: Dict[str, Any]):
    sentinel = object()
    previous = {}
    for key, value in overrides.items():
        previous[key] = getattr(engine_config, key, sentinel)
        setattr(engine_config, key, value)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is sentinel:
                delattr(engine_config, key)
            else:
                setattr(engine_config, key, value)


def _engine_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    payouts_raw = dict(config.get("payouts", {"1": 1.0}))
    return {
        "bot_decision_timeout_ms": int(config.get("bot_decision_timeout_ms", 500)),
        "starting_stack": int(config.get("starting_stack", 1500)),
        "max_players_per_table": int(config.get("max_players_per_table", 9)),
        "hands_per_level": int(config.get("hands_per_level", 20)),
        "blinds_schedule": list(config.get("blinds_schedule", [{"small": 10, "big": 20}])),
        "payouts": {int(key): float(value) for key, value in payouts_raw.items()},
        "max_hands_per_tournament": int(config.get("max_hands_per_tournament", 250)),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy

        numpy.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _fallback_counts(bot: Any) -> Dict[str, int]:
    return dict(getattr(bot, "fallback_counts", {}))


def _equity_counts(bot: Any) -> Dict[str, int]:
    return dict(getattr(bot, "equity_counts", {}))


def _subtract_counts(after: Dict[str, int], before: Dict[str, int]) -> Dict[str, int]:
    keys = set(after) | set(before)
    return {key: int(after.get(key, 0)) - int(before.get(key, 0)) for key in sorted(keys)}


def summarize_results(results_by_tournament: List[Dict[str, Any]]) -> Dict[str, Any]:
    placement_summary: Dict[str, Dict[str, float]] = {}
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
            summary = placement_summary.setdefault(
                class_name,
                {"count": 0, "position_sum": 0.0, "payout_sum": 0.0, "wins": 0},
            )
            position = int(row.get("position", 0) or 0)
            payout = float(row.get("payout_pct", 0.0) or 0.0)
            summary["count"] += 1
            summary["position_sum"] += position
            summary["payout_sum"] += payout
            if position == 1:
                summary["wins"] += 1
            if class_name == "BaselineModelEngineBot":
                model_rows.append(dict(row, tournament_id=tournament.get("tournament_id")))

    for summary in placement_summary.values():
        count = max(1, int(summary["count"]))
        summary["average_position"] = summary["position_sum"] / count
        summary["average_payout_pct"] = summary["payout_sum"] / count

    model_positions = [int(row.get("position", 0) or 0) for row in model_rows]
    model_payouts = [float(row.get("payout_pct", 0.0) or 0.0) for row in model_rows]
    model_summary = {
        "entries": len(model_rows),
        "placements": model_rows,
        "best_position": min(model_positions) if model_positions else None,
        "average_position": sum(model_positions) / len(model_positions) if model_positions else None,
        "total_payout_pct": sum(model_payouts),
        "wins": sum(1 for position in model_positions if position == 1),
    }

    return {
        "completed_tournament_count": valid_tournaments,
        "stopped_max_hands_count": stopped_max_hands,
        "placement_summary_by_bot_class": placement_summary,
        "model_bot_summary": model_summary,
    }


def run_tournaments(config: Dict[str, Any], bots: List[Any], artifact_root: Path) -> Dict[str, Any]:
    tournament_count = int(config.get("tournament_count", 1))
    seed = int(config.get("random_seed", 0))
    result_rows = []
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
                event_log_path = artifact_root / f"tournament_{tournament_id}_events.json"
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
    model_fallback_deltas = [
        _subtract_counts(after, before) for before, after in zip(fallback_before, fallback_after)
    ]
    model_equity_deltas = [_subtract_counts(after, before) for before, after in zip(equity_before, equity_after)]

    return {
        "tournament_results": result_rows,
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


def _sum_count_dicts(rows: Iterable[Dict[str, int]]) -> Dict[str, int]:
    total: Dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            total[key] = total.get(key, 0) + int(value)
    return total


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
    fallback_totals = dict(simulation.get("bot_fallback_summary", {}).get("totals", {}))
    event_log_paths = list(simulation.get("event_log_paths", []))
    gates = {
        "source_phase21_accepted_gate_passed": prerequisite.get("source_phase21_status")
        == config.get("expected_phase21_status", ACCEPTED),
        "source_small_sim_allowed_gate_passed": bool(prerequisite.get("source_small_mtt_engine_simulation_allowed", False)),
        "promoted_checkpoint_exists_gate_passed": bool(prerequisite.get("promoted_checkpoint_path"))
        and Path(str(prerequisite.get("promoted_checkpoint_path"))).is_file(),
        "lineup_gate_passed": simulation.get("lineup_summary", {}).get("total_bots", 0) >= 2,
        "tournament_runtime_gate_passed": not simulation.get("tournament_failures"),
        "completed_tournament_gate_passed": summary["completed_tournament_count"]
        >= int(acceptance.get("min_completed_tournaments", 1)),
        "model_inference_error_gate_passed": int(fallback_totals.get("inference_errors", 0))
        <= int(acceptance.get("max_model_inference_errors", 0)),
        "model_illegal_action_gate_passed": int(fallback_totals.get("illegal_actions", 0))
        <= int(acceptance.get("max_model_illegal_actions", 0)),
        "model_timeout_fallback_gate_passed": int(fallback_totals.get("timeouts", 0))
        <= int(acceptance.get("max_model_timeout_fallbacks", 0)),
        "event_log_artifact_gate_passed": (not bool(acceptance.get("require_event_logs", True)))
        or bool(event_log_paths),
        "result_artifact_gate_passed": (not bool(acceptance.get("require_result_artifact", True)))
        or (bool(result_artifact_path) and Path(result_artifact_path).is_file()),
    }
    failures = list(prerequisite.get("prerequisite_failures", []))
    failures.extend(str(failure) for failure in simulation.get("tournament_failures", []))
    for gate, passed in gates.items():
        if not passed:
            failures.append(f"{gate} failed")

    status = ACCEPTED if not failures else REJECTED
    return {
        "phase22_small_engine_simulation_id": config.get(
            "phase22_small_engine_simulation_id", "phase22_small_mtt_engine_simulation"
        ),
        "phase22_small_engine_simulation_status": status,
        "phase22_small_engine_simulation_failures": failures,
        "phase22_small_engine_simulation_report_path": str(config.get("small_engine_simulation_report_path", "")),
        "phase22_small_engine_simulation_started_at": started_at,
        "phase22_small_engine_simulation_finished_at": _utc_now(),
        "source_phase21_report_path": prerequisite.get("source_phase21_report_path", ""),
        "source_phase21_status": prerequisite.get("source_phase21_status", ""),
        "source_phase21_failures": prerequisite.get("source_phase21_failures", []),
        "source_small_mtt_engine_simulation_allowed": prerequisite.get(
            "source_small_mtt_engine_simulation_allowed", False
        ),
        "source_phase20_report_path": prerequisite.get("source_phase20_report_path", ""),
        "promoted_checkpoint_path": prerequisite.get("promoted_checkpoint_path", ""),
        "lineup_summary": simulation.get("lineup_summary", {}),
        "tournament_count": int(config.get("tournament_count", 0)),
        "completed_tournament_count": summary["completed_tournament_count"],
        "stopped_max_hands_count": summary["stopped_max_hands_count"],
        "result_summary": {
            "tournament_result_count": len(simulation.get("tournament_results", [])),
            "tournament_failures": simulation.get("tournament_failures", []),
        },
        "placement_summary_by_bot_class": summary["placement_summary_by_bot_class"],
        "model_bot_summary": summary["model_bot_summary"],
        "runtime_summary": {
            "runtime_seconds": runtime_seconds,
            "event_log_count": len(event_log_paths),
        },
        "bot_fallback_summary": simulation.get("bot_fallback_summary", {}),
        "model_equity_summary": simulation.get("model_equity_summary", {}),
        "event_log_paths": event_log_paths,
        "result_artifact_path": result_artifact_path,
        "phase22_gate_results": gates,
        "larger_mtt_engine_simulation_allowed": status == ACCEPTED,
        "next_phase_recommendation": "launch_phase23_larger_mtt_engine_simulation"
        if status == ACCEPTED
        else "resolve_phase22_small_engine_simulation_failures",
        "config_path": config.get("config_path", ""),
    }


def run_phase22_small_mtt_engine_simulation(config: Dict[str, Any], config_path: str | Path | None = None) -> Dict[str, Any]:
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
        "event_log_paths": [],
        "tournament_failures": [],
        "bot_fallback_summary": {"model_bot_count": 0, "per_model_bot": [], "totals": {}},
        "model_equity_summary": {"per_model_bot": [], "totals": {}},
    }

    if prerequisite.get("prerequisite_passed"):
        try:
            bots = build_lineup(
                config,
                str(prerequisite["promoted_checkpoint_path"]),
                str(prerequisite["basemodel_root"]),
            )
            simulation["lineup_summary"] = lineup_summary(bots)
            simulation.update(run_tournaments(config, bots, artifact_root))
        except Exception as exc:
            simulation["tournament_failures"] = [f"simulation setup failed: {exc}"]

    _write_json(str(config["tournament_results_path"]), simulation.get("tournament_results", []))
    report = build_report(config, prerequisite, simulation, started_at, time.perf_counter() - started)
    _write_json(str(config["small_engine_simulation_report_path"]), report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase22_small_mtt_engine_simulation.json")
    args = parser.parse_args(argv)

    report = run_phase22_small_mtt_engine_simulation(load_config(args.config), config_path=args.config)
    print(
        json.dumps(
            {
                "phase22_small_engine_simulation_status": report["phase22_small_engine_simulation_status"],
                "promoted_checkpoint_path": report["promoted_checkpoint_path"],
                "larger_mtt_engine_simulation_allowed": report["larger_mtt_engine_simulation_allowed"],
                "next_phase_recommendation": report["next_phase_recommendation"],
                "phase22_small_engine_simulation_report_path": report[
                    "phase22_small_engine_simulation_report_path"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["phase22_small_engine_simulation_status"] != ACCEPTED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
