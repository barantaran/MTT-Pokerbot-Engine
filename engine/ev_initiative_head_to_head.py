from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from engine.evolutionary_reduced_mtt import (
    _engine_overrides,
    _seed_everything,
    _write_json,
    default_engine_config,
    load_json_config,
    merge_candidate_action_summaries,
    summarize_candidate_actions,
    summarize_population_results,
    temporary_engine_config,
    utc_stamp,
)
from engine.tournament import Tournament
from players.ev_reaction_bot import EVInitiativeBot


def _load_candidate_params(report_path: Path, candidate_id: str) -> Dict[str, float]:
    with report_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    for generation_report in report.get("generation_reports", []):
        candidate_params = dict(generation_report.get("candidate_params", {}))
        if candidate_id in candidate_params:
            return {key: float(value) for key, value in dict(candidate_params[candidate_id]).items()}
    raise ValueError(f"candidate_id not found in evolution report: {candidate_id}")


def _build_bots(config: Dict[str, Any], engine_config: Dict[str, Any]) -> tuple[List[Any], Dict[str, str]]:
    default_report_path = Path(str(config["source_report"])) if config.get("source_report") else None
    use_ranges = bool(engine_config.get("fixed_bots_use_preflop_spot_range", False))
    bots: List[Any] = []
    name_to_population: Dict[str, str] = {}

    for side in ("left", "right"):
        side_config = dict(config[side])
        candidate_id = str(side_config["candidate_id"])
        label = str(side_config.get("label") or candidate_id)
        entries = int(side_config.get("entries", 25))
        report_path = Path(str(side_config.get("source_report") or default_report_path))
        params = _load_candidate_params(report_path, candidate_id)
        for index in range(entries):
            bot = EVInitiativeBot(use_preflop_spot_range=use_ranges, **params)
            bot.name = f"{label}_{index + 1:03d}"
            bots.append(bot)
            name_to_population[bot.name] = label
    return bots, name_to_population


def _run_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    tournament_id = int(payload["tournament_id"])
    engine_config = dict(payload["engine_config"])
    try:
        _seed_everything(int(payload["seed"]))
        with temporary_engine_config(_engine_overrides(engine_config)):
            bots, name_to_population = _build_bots(dict(payload["config"]), engine_config)
            random.shuffle(bots)
            tournament = Tournament(bots, tournament_id=tournament_id)
            results, events = tournament.play()
            return {
                "tournament_id": tournament_id,
                "results": results,
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
            "population_summary": {},
            "action_summary": {},
            "stopped_max_hands": False,
            "event_count": 0,
            "failure": str(exc),
        }


def run_ev_initiative_head_to_head(config: Dict[str, Any], *, engine_root: Path) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(config.get("run_id", "ev_initiative_head_to_head"))
    artifact_root = Path(str(config.get("artifact_root") or Path("runs") / run_id / utc_stamp()))
    artifact_root.mkdir(parents=True, exist_ok=True)
    mtt_count = int(config.get("mtt_count", 1000))
    workers = max(1, min(int(config.get("workers", 8)), mtt_count))
    seed = int(config.get("random_seed", 77001))
    engine_config = default_engine_config()
    engine_config.update(dict(config.get("engine", {})))

    all_results: List[Dict[str, Any]] = []
    action_summaries: List[Dict[str, Any]] = []
    tournament_summaries: List[Dict[str, Any]] = []
    failures: List[str] = []
    started = time.perf_counter()

    labels = [str(dict(config[side]).get("label") or dict(config[side])["candidate_id"]) for side in ("left", "right")]

    def name_to_population_from_results() -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for row in all_results:
            name = str(row.get("name", ""))
            for label in labels:
                if name.startswith(label + "_"):
                    mapping[name] = label
                    break
        return mapping

    def write_partial() -> None:
        _write_json(
            artifact_root / "partial_summary.json",
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
                "config": config,
                "engine_config": engine_config,
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
                action_summaries.append(dict(result.get("action_summary", {})))
                tournament_summaries.append(
                    {
                        "tournament_id": int(result.get("tournament_id", 0)),
                        "population_summary": dict(result.get("population_summary", {})),
                        "population_action_summary": dict(result.get("action_summary", {})),
                        "stopped_max_hands": bool(result.get("stopped_max_hands", False)),
                        "event_count": int(result.get("event_count", 0)),
                    }
                )
            write_partial()

    tournament_summaries.sort(key=lambda row: int(row.get("tournament_id", 0)))
    population_summary = summarize_population_results(all_results, name_to_population_from_results())
    population_action_summary = merge_candidate_action_summaries(action_summaries)
    report = {
        "run_id": run_id,
        "mode": "ev_initiative_head_to_head",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "source_report": str(config.get("source_report", "")),
        "left": dict(config["left"]),
        "right": dict(config["right"]),
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
    report_path = artifact_root / "ev_initiative_head_to_head_report.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    args = parser.parse_args(argv)
    engine_root = Path(__file__).resolve().parents[1]
    report = run_ev_initiative_head_to_head(load_json_config(args.config), engine_root=engine_root)
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
