from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from engine.evolutionary_reduced_mtt import (
    _engine_overrides,
    _seed_everything,
    _write_json,
    default_engine_config,
    load_json_config,
    merge_candidate_action_summaries,
    ranked_candidates,
    summarize_candidate_actions,
    summarize_candidate_results,
    summarize_population_results,
    temporary_engine_config,
    utc_stamp,
)
from engine.tournament import Tournament
from players.ev_reaction_bot import EVInitiativeBot
from players.tight_equity_bot import TightEquityBot
from players.tournament_equity_bot import TournamentEquityBot


PARAM_LIMITS = {
    "call_margin": (-0.02, 0.08),
    "sb_call_margin": (-0.02, 0.12),
    "bb_call_margin": (-0.06, 0.08),
    "open_threshold_shift": (-0.08, 0.08),
    "raise_threshold_shift": (-0.08, 0.10),
    "raise_size_scale": (0.70, 1.45),
}


def random_params(rng: random.Random) -> Dict[str, float]:
    return {name: rng.uniform(lo, hi) for name, (lo, hi) in PARAM_LIMITS.items()}


def clamp_param(name: str, value: float) -> float:
    lo, hi = PARAM_LIMITS[name]
    return max(lo, min(hi, value))


def mutate_params(params: Dict[str, float], rng: random.Random, sigma: float) -> Dict[str, float]:
    mutated = {}
    for name, (lo, hi) in PARAM_LIMITS.items():
        span = hi - lo
        value = float(params.get(name, (lo + hi) / 2.0)) + rng.gauss(0.0, sigma * span)
        mutated[name] = clamp_param(name, value)
    return mutated


def mutate_single_param(params: Dict[str, float], name: str, direction: int, step_fraction: float) -> Dict[str, float]:
    mutated = dict(params)
    lo, hi = PARAM_LIMITS[name]
    value = float(mutated.get(name, (lo + hi) / 2.0)) + (float(direction) * float(step_fraction) * (hi - lo))
    mutated[name] = clamp_param(name, value)
    return mutated


def score_summary(row: Dict[str, Any]) -> float:
    return float(row.get("total_payout_pct", 0.0) or 0.0)


def rank_candidate_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = [{"candidate_id": candidate_id, **dict(row)} for candidate_id, row in summary.items()]
    return sorted(
        rows,
        key=lambda row: (
            score_summary(row),
            float(row.get("top3", 0.0) or 0.0),
            float(row.get("wins", 0.0) or 0.0),
            float(row.get("final_table", 0.0) or 0.0),
        ),
        reverse=True,
    )


def build_bots(
    candidates: Dict[str, Dict[str, float]],
    *,
    entries_per_candidate: int,
    controls: Dict[str, int],
    use_ranges: bool,
) -> tuple[List[Any], Dict[str, str], Dict[str, str]]:
    bots: List[Any] = []
    name_to_candidate: Dict[str, str] = {}
    name_to_population: Dict[str, str] = {}
    for candidate_id, params in candidates.items():
        for entry_index in range(entries_per_candidate):
            bot = EVInitiativeBot(use_preflop_spot_range=use_ranges, **params)
            bot.name = f"{candidate_id}_entry_{entry_index + 1:03d}"
            bots.append(bot)
            name_to_candidate[bot.name] = candidate_id
            name_to_population[bot.name] = "ev_initiative_candidate"

    for index in range(int(controls.get("tight_equity", 0) or 0)):
        bot = TightEquityBot(use_preflop_spot_range=use_ranges)
        bot.name = f"tight_equity_{index + 1:03d}"
        bots.append(bot)
        name_to_population[bot.name] = "tight_equity"
    for index in range(int(controls.get("tournament_equity", 0) or 0)):
        bot = TournamentEquityBot(use_preflop_spot_range=use_ranges)
        bot.name = f"tournament_equity_{index + 1:03d}"
        bots.append(bot)
        name_to_population[bot.name] = "tournament_equity"

    return bots, name_to_candidate, name_to_population


def _run_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    tournament_id = int(payload["tournament_id"])
    engine_config = dict(payload["engine_config"])
    try:
        _seed_everything(int(payload["seed"]))
        with temporary_engine_config(_engine_overrides(engine_config)):
            bots, name_to_candidate, name_to_population = build_bots(
                dict(payload["candidates"]),
                entries_per_candidate=int(payload["entries_per_candidate"]),
                controls=dict(payload["controls"]),
                use_ranges=bool(engine_config.get("fixed_bots_use_preflop_spot_range", False)),
            )
            random.shuffle(bots)
            tournament = Tournament(bots, tournament_id=tournament_id)
            results, events = tournament.play()
            return {
                "tournament_id": tournament_id,
                "results": results,
                "candidate_summary": summarize_candidate_results(results, name_to_candidate),
                "candidate_action_summary": summarize_candidate_actions(events, name_to_candidate),
                "population_summary": summarize_population_results(results, name_to_population),
                "population_action_summary": summarize_candidate_actions(events, name_to_population),
                "failure": "",
            }
    except Exception as exc:
        return {"tournament_id": tournament_id, "results": [], "failure": str(exc)}


def run_generation(
    candidates: Dict[str, Dict[str, float]],
    *,
    generation: int,
    config: Dict[str, Any],
    artifact_root: Path,
    engine_config: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    mtt_count = int(config.get("mtt_count_per_generation", 4))
    workers = max(1, min(int(config.get("workers", 8)), mtt_count))
    entries_per_candidate = int(config.get("entries_per_candidate", 2))
    controls = dict(config.get("controls", {"tight_equity": 5, "tournament_equity": 5}))
    all_results: List[Dict[str, Any]] = []
    candidate_actions = []
    population_actions = []
    tournament_summaries = []
    failures = []
    started = time.perf_counter()

    def candidate_name_map_from_results() -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for row in all_results:
            name = str(row.get("name", ""))
            candidate_id = name.split("_entry_", 1)[0]
            if candidate_id in candidates:
                mapping[name] = candidate_id
        return mapping

    def write_partial() -> None:
        _write_json(
            artifact_root / "partial_summary.json",
            {
                "generation": generation,
                "completed_tournament_count": len(tournament_summaries),
                "candidate_summary": summarize_candidate_results(all_results, candidate_name_map_from_results()),
                "population_summary": summarize_population_results(
                    all_results,
                    {
                        str(row.get("name", "")): "ev_initiative_candidate"
                        if str(row.get("name", "")).startswith("evg")
                        else str(row.get("name", "")).rsplit("_", 1)[0]
                        for row in all_results
                    },
                ),
                "failures": failures,
            },
        )

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_id = {}
        for index in range(mtt_count):
            tournament_id = generation * 10000 + index + 1
            payload = {
                "tournament_id": tournament_id,
                "seed": seed + generation * 100000 + index,
                "candidates": candidates,
                "entries_per_candidate": entries_per_candidate,
                "controls": controls,
                "engine_config": engine_config,
            }
            future_to_id[executor.submit(_run_worker, payload)] = tournament_id

        for future in concurrent.futures.as_completed(future_to_id):
            tournament_id = future_to_id[future]
            result = future.result()
            failure = str(result.get("failure", "") or "")
            if failure:
                failures.append(f"tournament {tournament_id} failed: {failure}")
                write_partial()
                continue
            all_results.extend(list(result.get("results", [])))
            candidate_actions.append(dict(result.get("candidate_action_summary", {})))
            population_actions.append(dict(result.get("population_action_summary", {})))
            tournament_summaries.append(
                {
                    "tournament_id": tournament_id,
                    "candidate_summary": dict(result.get("candidate_summary", {})),
                    "population_summary": dict(result.get("population_summary", {})),
                }
            )
            write_partial()

    candidate_name_map = candidate_name_map_from_results()
    population_name_map = {
        str(row.get("name", "")): "ev_initiative_candidate"
        if str(row.get("name", "")).startswith("evg")
        else str(row.get("name", "")).rsplit("_", 1)[0]
        for row in all_results
    }
    candidate_summary = summarize_candidate_results(all_results, candidate_name_map)
    ranking = rank_candidate_summary(candidate_summary)
    return {
        "generation": generation,
        "candidate_params": candidates,
        "candidate_summary": candidate_summary,
        "candidate_action_summary": merge_candidate_action_summaries(candidate_actions),
        "population_summary": summarize_population_results(all_results, population_name_map),
        "population_action_summary": merge_candidate_action_summaries(population_actions),
        "ranking": ranking,
        "tournaments": sorted(tournament_summaries, key=lambda row: int(row.get("tournament_id", 0))),
        "failures": failures,
        "runtime_seconds": time.perf_counter() - started,
    }


def next_population(
    candidates: Dict[str, Dict[str, float]],
    ranking: List[Dict[str, Any]],
    *,
    rng: random.Random,
    generation: int,
    keep_count: int,
    sigma: float,
    population_size: int,
    mutation_mode: str = "gaussian_all_params",
    mutation_step_fraction: float = 0.10,
) -> Dict[str, Dict[str, float]]:
    ranked_ids = [str(row.get("candidate_id", "")) for row in ranking if str(row.get("candidate_id", "")) in candidates]
    survivors = ranked_ids[:keep_count]
    next_candidates = {candidate_id: dict(candidates[candidate_id]) for candidate_id in survivors}
    child_index = 1
    parent_ids = survivors or ranked_ids[:1]
    if not parent_ids:
        return {f"evg{generation + 1:02d}_{index + 1:03d}": random_params(rng) for index in range(population_size)}

    if mutation_mode == "paired_single_param":
        for parent_id in parent_ids:
            for param_name in PARAM_LIMITS:
                for direction in (1, -1):
                    if len(next_candidates) >= population_size:
                        return next_candidates
                    child_id = f"evg{generation + 1:02d}_{child_index:03d}"
                    child_index += 1
                    next_candidates[child_id] = mutate_single_param(
                        candidates[parent_id],
                        param_name,
                        direction,
                        mutation_step_fraction,
                    )

        while len(next_candidates) < population_size:
            child_id = f"evg{generation + 1:02d}_{child_index:03d}"
            child_index += 1
            next_candidates[child_id] = random_params(rng)
        return next_candidates

    while len(next_candidates) < population_size:
        parent_id = parent_ids[(len(next_candidates) - len(survivors)) % len(parent_ids)]
        child_id = f"evg{generation + 1:02d}_{child_index:03d}"
        child_index += 1
        next_candidates[child_id] = mutate_params(candidates[parent_id], rng, sigma=sigma)
    return next_candidates


def run_ev_initiative_evolution(config: Dict[str, Any], *, engine_root: Path) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(config.get("run_id", "ev_initiative_evolution"))
    artifact_root = Path(str(config.get("artifact_root") or Path("runs") / run_id / utc_stamp()))
    artifact_root.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("random_seed", 76001))
    rng = random.Random(seed)
    generations = int(config.get("generations", 10))
    population_size = int(config.get("population_size", 20))
    keep_count = int(config.get("keep_count", 8))
    sigma = float(config.get("mutation_sigma", 0.25))
    mutation_mode = str(config.get("mutation_mode", "gaussian_all_params"))
    mutation_step_fraction = float(config.get("mutation_step_fraction", 0.10))
    engine_config = default_engine_config()
    engine_config.update(dict(config.get("engine", {})))

    resumed_from_report = str(config.get("resume_from_report", "") or "")
    generation_start = 0
    if resumed_from_report:
        with Path(resumed_from_report).open("r", encoding="utf-8") as handle:
            source_report = json.load(handle)
        source_generations = list(source_report.get("generation_reports", []))
        if not source_generations:
            raise ValueError(f"resume_from_report has no generation_reports: {resumed_from_report}")
        source_last = dict(source_generations[-1])
        generation_start = int(source_last.get("generation", len(source_generations) - 1)) + 1
        candidates = next_population(
            {str(candidate_id): dict(params) for candidate_id, params in dict(source_last["candidate_params"]).items()},
            list(source_last.get("ranking", [])),
            rng=rng,
            generation=int(source_last.get("generation", generation_start - 1)),
            keep_count=keep_count,
            sigma=sigma,
            population_size=population_size,
            mutation_mode=mutation_mode,
            mutation_step_fraction=mutation_step_fraction,
        )
    else:
        candidates = {f"evg00_{index + 1:03d}": random_params(rng) for index in range(population_size)}
    generation_reports = []
    for generation in range(generation_start, generation_start + generations):
        generation_report = run_generation(
            candidates,
            generation=generation,
            config=config,
            artifact_root=artifact_root,
            engine_config=engine_config,
            seed=seed,
        )
        generation_reports.append(generation_report)
        _write_json(artifact_root / f"generation_{generation:02d}.json", generation_report)
        _write_json(
            artifact_root / "latest_summary.json",
            {
                "completed_generations": generation + 1,
                "latest_generation": generation_report,
                "best_per_generation": [
                    report["ranking"][0] if report.get("ranking") else {}
                    for report in generation_reports
                ],
            },
        )
        candidates = next_population(
            candidates,
            generation_report["ranking"],
            rng=rng,
            generation=generation,
            keep_count=keep_count,
            sigma=sigma,
            population_size=population_size,
            mutation_mode=mutation_mode,
            mutation_step_fraction=mutation_step_fraction,
        )

    report = {
        "run_id": run_id,
        "mode": "ev_initiative_evolution",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "artifact_root": str(artifact_root),
        "random_seed": seed,
        "generations": generations,
        "population_size": population_size,
        "keep_count": keep_count,
        "mutation_sigma": sigma,
        "mutation_mode": mutation_mode,
        "mutation_step_fraction": mutation_step_fraction,
        "resume_from_report": resumed_from_report,
        "generation_start": generation_start,
        "engine": engine_config,
        "generation_reports": generation_reports,
        "best_per_generation": [
            report["ranking"][0] if report.get("ranking") else {}
            for report in generation_reports
        ],
    }
    _write_json(artifact_root / "ev_initiative_evolution_report.json", report)
    report["report_path"] = str(artifact_root / "ev_initiative_evolution_report.json")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="")
    args = parser.parse_args(argv)
    engine_root = Path(__file__).resolve().parents[1]
    report = run_ev_initiative_evolution(load_json_config(args.config), engine_root=engine_root)
    print(
        json.dumps(
            {
                "mode": report["mode"],
                "report_path": report["report_path"],
                "best_per_generation": report["best_per_generation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
