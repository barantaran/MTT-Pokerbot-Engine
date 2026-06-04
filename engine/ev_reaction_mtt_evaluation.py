from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from engine.evolutionary_reduced_mtt import (
    Candidate,
    _engine_overrides,
    _seed_everything,
    attach_actions_to_ranking,
    attach_lineage_to_ranking,
    basemodel_root_from_engine,
    build_candidate_lineup,
    create_initial_candidates,
    default_engine_config,
    load_json_config,
    observation_contract,
    ranked_candidates,
    summarize_candidate_actions,
    summarize_candidate_results,
    temporary_engine_config,
    utc_stamp,
    _write_json,
)
from engine.tournament import Tournament
from players.ev_reaction_bot import EVInitiativeBot, EVReactionBot


def _ev_bots(count: int, *, use_preflop_spot_range: bool, strategy: str) -> tuple[List[Any], Dict[str, str]]:
    normalized_strategy = str(strategy or "reaction").lower()
    if normalized_strategy in {"initiative", "raise_initiative", "ev_initiative"}:
        bot_class = EVInitiativeBot
        population = "ev_initiative"
        name_prefix = "EVInitiativeBot"
    else:
        bot_class = EVReactionBot
        population = "ev_reaction"
        name_prefix = "EVReactionBot"

    bots = []
    name_to_population = {}
    for index in range(int(count)):
        bot = bot_class(use_preflop_spot_range=use_preflop_spot_range)
        bot.name = f"{name_prefix}_{index + 1:03d}"
        bots.append(bot)
        name_to_population[bot.name] = population
    return bots, name_to_population


def _run_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    tournament_id = int(payload["tournament_id"])
    engine_config = dict(payload["engine_config"])
    candidates = [
        Candidate(candidate_id=str(row["candidate_id"]), checkpoint_path=Path(str(row["checkpoint_path"])))
        for row in payload["candidates"]
    ]
    try:
        try:
            import torch

            torch.set_num_threads(1)
        except Exception:
            pass
        _seed_everything(int(payload["seed"]))
        with temporary_engine_config(_engine_overrides(engine_config)):
            model_bots, name_to_candidate = build_candidate_lineup(
                candidates,
                entries_per_candidate=int(payload.get("entries_per_candidate", 1)),
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
            name_to_population = {name: "random_weighted" for name in name_to_candidate}
            ev_bots, ev_names = _ev_bots(
                int(payload["ev_bot_count"]),
                use_preflop_spot_range=bool(engine_config.get("ev_use_preflop_spot_range", False)),
                strategy=str(payload.get("ev_bot_strategy", "reaction")),
            )
            name_to_population.update(ev_names)
            bots = [*model_bots, *ev_bots]
            random.shuffle(bots)
            tournament = Tournament(bots, tournament_id=tournament_id)
            results, events = tournament.play()
            population_summary = summarize_candidate_results(results, name_to_population)
            action_summary = summarize_candidate_actions(events, name_to_population)
            return {
                "tournament_id": tournament_id,
                "results": results,
                "population_summary": population_summary,
                "action_summary": action_summary,
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


def _merge_action_summaries(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    from engine.evolutionary_reduced_mtt import merge_candidate_action_summaries

    return merge_candidate_action_summaries(rows)


def _population_for_name(name: str) -> str:
    if name.startswith("EVReactionBot_"):
        return "ev_reaction"
    if name.startswith("EVInitiativeBot_"):
        return "ev_initiative"
    return "random_weighted"


def run_ev_reaction_evaluation(config: Dict[str, Any], *, engine_root: Path) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(config.get("run_id", "ev_reaction_mtt_evaluation"))
    artifact_root = Path(str(config.get("artifact_root") or Path("runs") / run_id / utc_stamp()))
    artifact_root.mkdir(parents=True, exist_ok=True)
    basemodel_root = basemodel_root_from_engine(engine_root, config.get("basemodel_root", "../poker-ai-basemodel"))
    observation_schema = str(config.get("observation_schema", "ev_reaction"))
    observation_size, observation_fields = observation_contract(basemodel_root, observation_schema)
    random_weighted_count = int(config.get("random_weighted_count", 50))
    ev_bot_count = int(config.get("ev_bot_count", 50))
    mtt_count = int(config.get("mtt_count", 1000))
    workers = max(1, min(int(config.get("workers", 8)), mtt_count))
    seed = int(config.get("random_seed", 69001))
    engine_config = default_engine_config()
    engine_config.update(dict(config.get("engine", {})))

    fixed_checkpoint = str(config.get("fixed_candidate_checkpoint") or "")
    fixed_candidate_id = str(config.get("fixed_candidate_id") or "random_weighted_champion")
    if fixed_checkpoint:
        candidates = [
            Candidate(candidate_id=fixed_candidate_id, checkpoint_path=Path(fixed_checkpoint).expanduser())
        ]
        entries_per_candidate = random_weighted_count
    else:
        candidates = create_initial_candidates(
            artifact_root=artifact_root,
            basemodel_root=basemodel_root,
            observation_size=observation_size,
            candidate_count=random_weighted_count,
            seed=seed,
        )
        entries_per_candidate = 1
    candidate_payload = [
        {"candidate_id": candidate.candidate_id, "checkpoint_path": str(candidate.checkpoint_path)}
        for candidate in candidates
    ]

    all_results: List[Dict[str, Any]] = []
    tournament_summaries: List[Dict[str, Any]] = []
    action_summaries: List[Dict[str, Any]] = []
    failures: List[str] = []
    started = time.perf_counter()
    partial_summary_path = artifact_root / "partial_summary.json"

    def write_partial() -> None:
        name_to_population = {
            str(row.get("name", "")): _population_for_name(str(row.get("name", "")))
            for row in all_results
        }
        _write_json(
            partial_summary_path,
            {
                "completed_tournament_count": len(tournament_summaries),
                "population_summary": summarize_candidate_results(all_results, name_to_population),
                "population_action_summary": _merge_action_summaries(action_summaries),
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
                "candidates": candidate_payload,
                "entries_per_candidate": entries_per_candidate,
                "ev_bot_count": ev_bot_count,
                "engine_config": engine_config,
                "basemodel_root": str(basemodel_root),
                "engine_root": str(engine_root),
                "observation_schema": observation_schema,
                "observation_size": observation_size,
                "deterministic": bool(config.get("deterministic", True)),
                "ev_bot_strategy": str(config.get("ev_bot_strategy", "reaction")),
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
                action_summaries.append(dict(result.get("action_summary", {})))
            write_partial()

    tournament_summaries.sort(key=lambda row: int(row.get("tournament_id", 0)))
    name_to_population = {
        str(row.get("name", "")): _population_for_name(str(row.get("name", "")))
        for row in all_results
    }
    population_summary = summarize_candidate_results(all_results, name_to_population)
    population_action_summary = _merge_action_summaries(action_summaries)
    ranking = attach_actions_to_ranking(
        attach_lineage_to_ranking(ranked_candidates(population_summary), []),
        population_action_summary,
    )
    report = {
        "run_id": run_id,
        "mode": "ev_reaction_vs_random_weighted",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "basemodel_root": str(basemodel_root),
        "observation_schema": observation_schema,
        "observation_size": observation_size,
        "observation_fields": observation_fields,
        "random_weighted_count": random_weighted_count,
        "ev_bot_count": ev_bot_count,
        "ev_bot_strategy": str(config.get("ev_bot_strategy", "reaction")),
        "mtt_count": mtt_count,
        "workers": workers,
        "random_seed": seed,
        "engine": engine_config,
        "random_weighted_checkpoints": {
            candidate.candidate_id: str(candidate.checkpoint_path) for candidate in candidates
        },
        "entries_per_candidate": entries_per_candidate,
        "ranking": ranking,
        "simulation": {
            "population_summary": population_summary,
            "population_action_summary": population_action_summary,
            "tournaments": tournament_summaries,
            "failures": failures,
            "runtime_seconds": time.perf_counter() - started,
            "worker_count": workers,
        },
    }
    report_path = artifact_root / "ev_reaction_evaluation_report.json"
    _write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    args = parser.parse_args(argv)
    engine_root = Path(__file__).resolve().parents[1]
    config = load_json_config(args.config)
    report = run_ev_reaction_evaluation(config, engine_root=engine_root)
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "report_path": report["report_path"],
                "failures": report["simulation"]["failures"][:5],
                "runtime_seconds": report["simulation"]["runtime_seconds"],
                "ranking": report["ranking"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
