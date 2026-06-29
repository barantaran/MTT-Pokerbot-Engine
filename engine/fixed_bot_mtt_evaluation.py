from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from engine.bot_factory import BOT_REGISTRY, build_configurable_bots
from engine.evolutionary_reduced_mtt import (
    _engine_overrides,
    _seed_everything,
    _write_json,
    default_engine_config,
    load_json_config,
    merge_candidate_action_summaries,
    summarize_population_results,
    summarize_candidate_actions,
    temporary_engine_config,
    utc_stamp,
)
from engine.tournament import Tournament


def _build_bots(
    lineup: Dict[str, int],
    engine_config: Dict[str, Any],
    bot_specs: List[Dict[str, Any]] | None = None,
) -> tuple[List[Any], Dict[str, str]]:
    return build_configurable_bots(lineup, engine_config, extra_specs=bot_specs)


def _run_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    tournament_id = int(payload["tournament_id"])
    engine_config = dict(payload["engine_config"])
    try:
        _seed_everything(int(payload["seed"]))
        with temporary_engine_config(_engine_overrides(engine_config)):
            bots, name_to_population = _build_bots(
                dict(payload["lineup"]),
                engine_config,
                list(payload.get("lineup_variants", [])),
            )
            random.shuffle(bots)
            tournament = Tournament(bots, tournament_id=tournament_id)
            results, events = tournament.play()
            return {
                "tournament_id": tournament_id,
                "results": results,
                "events": events if bool(payload.get("write_events", False)) else [],
                "population_summary": summarize_population_results(results, name_to_population),
                "action_summary": summarize_candidate_actions(events, name_to_population),
                "stopped_max_hands": any(event.get("type") == "tournament_stopped_max_hands" for event in events),
                "event_count": len(events),
                "failure": "",
            }
    except Exception as exc:
        return {
            "tournament_id": tournament_id,
            "results": [],
            "events": [],
            "population_summary": {},
            "action_summary": {},
            "stopped_max_hands": False,
            "event_count": 0,
            "failure": str(exc),
        }


def run_fixed_bot_evaluation(config: Dict[str, Any], *, engine_root: Path) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(config.get("run_id", "fixed_bot_mtt_evaluation"))
    artifact_root = Path(str(config.get("artifact_root") or Path("runs") / run_id / utc_stamp()))
    artifact_root.mkdir(parents=True, exist_ok=True)
    mtt_count = int(config.get("mtt_count", 200))
    workers = max(1, min(int(config.get("workers", 8)), mtt_count))
    seed = int(config.get("random_seed", 72001))
    lineup = dict(config.get("lineup", {}))
    lineup_variants = list(config.get("lineup_variants", config.get("bot_lineup", [])) or [])
    engine_config = default_engine_config()
    engine_config.update(dict(config.get("engine", {})))

    all_results: List[Dict[str, Any]] = []
    tournament_summaries: List[Dict[str, Any]] = []
    action_summaries: List[Dict[str, Any]] = []
    failures: List[str] = []
    partial_summary_path = artifact_root / "partial_summary.json"
    started = time.perf_counter()
    write_events = bool(config.get("write_events", False))
    configured_populations = set(str(key) for key in lineup)
    for spec in lineup_variants:
        spec_type = str(spec.get("type") or spec.get("bot") or spec.get("bot_type") or spec.get("class") or "")
        configured_populations.add(str(spec.get("population") or BOT_REGISTRY[spec_type].population))

    def name_to_population_from_results() -> Dict[str, str]:
        prefixes = tuple(sorted((population + "_" for population in configured_populations), key=len, reverse=True))
        mapping = {}
        for row in all_results:
            name = str(row.get("name", ""))
            for prefix in prefixes:
                if name.startswith(prefix):
                    mapping[name] = prefix[:-1]
                    break
        return mapping

    def write_partial() -> None:
        _write_json(
            partial_summary_path,
            {
                "completed_tournament_count": len(tournament_summaries),
                "population_summary": summarize_population_results(all_results, name_to_population_from_results()),
                "population_action_summary": merge_candidate_action_summaries(action_summaries),
                "failures": failures,
            },
        )

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_id = {}
        for index in range(mtt_count):
            tournament_id = index + 1
            payload = {
                "tournament_id": tournament_id,
                "seed": seed + 100000 + index,
                "lineup": lineup,
                "lineup_variants": lineup_variants,
                "engine_config": engine_config,
                "write_events": write_events,
            }
            future_to_id[executor.submit(_run_worker, payload)] = tournament_id

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
                tournament_summaries.append(
                    {
                        "tournament_id": int(result.get("tournament_id", 0)),
                        "population_summary": dict(result.get("population_summary", {})),
                        "population_action_summary": dict(result.get("action_summary", {})),
                        "stopped_max_hands": bool(result.get("stopped_max_hands", False)),
                        "event_count": int(result.get("event_count", 0)),
                    }
                )
                if write_events:
                    event_path = artifact_root / "events" / f"tournament_{int(result.get('tournament_id', 0)):04d}_events.json"
                    _write_json(event_path, list(result.get("events", [])))
                action_summaries.append(dict(result.get("action_summary", {})))
            write_partial()

    tournament_summaries.sort(key=lambda row: int(row.get("tournament_id", 0)))
    population_summary = summarize_population_results(all_results, name_to_population_from_results())
    population_action_summary = merge_candidate_action_summaries(action_summaries)
    report = {
        "run_id": run_id,
        "mode": "fixed_bot_mtt_evaluation",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "lineup": lineup,
        "lineup_variants": lineup_variants,
        "mtt_count": mtt_count,
        "workers": workers,
        "random_seed": seed,
        "engine": engine_config,
        "population_summary": population_summary,
        "population_action_summary": population_action_summary,
        "simulation": {
            "population_summary": population_summary,
            "population_action_summary": population_action_summary,
            "tournaments": tournament_summaries,
            "failures": failures,
            "runtime_seconds": time.perf_counter() - started,
            "worker_count": workers,
        },
    }
    report_path = artifact_root / "fixed_bot_evaluation_report.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    args = parser.parse_args(argv)
    engine_root = Path(__file__).resolve().parents[1]
    report = run_fixed_bot_evaluation(load_json_config(args.config), engine_root=engine_root)
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "report_path": report["report_path"],
                "failures": report["simulation"]["failures"][:5],
                "runtime_seconds": report["simulation"]["runtime_seconds"],
                "population_summary": report["population_summary"],
                "population_action_summary": report["population_action_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
