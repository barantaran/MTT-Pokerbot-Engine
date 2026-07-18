from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

from engine.event_log import (
    ChunkLogWriter,
    atomic_write_json,
    completed_tournament_ids,
    load_tournament_result,
    tournament_events_dir,
    tournament_result_path,
)

from engine.bot_factory import BOT_REGISTRY, build_configurable_bots, population_for_spec
from engine.plugins import ENTRY_SEP as PLUGINS_ENTRY_SEP, ENV_VAR as PLUGINS_ENV_VAR, load_plugins
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


POPULATION_TABLE_COLUMNS = (
    ("#", "rank"),
    ("bot", "name"),
    ("entries", "entries"),
    ("payout", "total_payout_pct"),
    ("avg payout", "average_payout_pct"),
    ("ROI", "roi"),
    ("score", "score"),
    ("wins", "wins"),
    ("top3", "top3"),
    ("FT", "final_table"),
    ("ITM", "itm"),
    ("avg pos", "average_position"),
)


def _build_bots(
    lineup: Dict[str, int],
    engine_config: Dict[str, Any],
    bot_specs: List[Dict[str, Any]] | None = None,
) -> tuple[List[Any], Dict[str, str]]:
    return build_configurable_bots(lineup, engine_config, extra_specs=bot_specs)


def _load_bot_library(config: Mapping[str, Any], *, engine_root: Path) -> Dict[str, Dict[str, Any]]:
    paths: List[Path] = []
    for raw_path in config.get("bot_config_files", []) or []:
        paths.append(_config_path(raw_path, engine_root=engine_root))
    raw_dir = config.get("bot_config_dir", "")
    if raw_dir:
        paths.extend(sorted(_config_path(raw_dir, engine_root=engine_root).glob("*.json")))

    library: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        payload = load_json_config(path)
        specs = payload if isinstance(payload, list) else [payload]
        for raw_spec in specs:
            spec = dict(raw_spec)
            name = str(spec.get("name") or path.stem)
            if not name:
                raise ValueError(f"bot config has no name: {path}")
            if name in library:
                raise ValueError(f"duplicate bot config name {name!r}: {path}")
            spec.pop("count", None)
            library[name] = spec
    return library


def _config_path(path: str | Path, *, engine_root: Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = engine_root / resolved
    return resolved


def _resolve_plugin_entry(entry: str, *, engine_root: Path) -> str:
    """Anchor a plugin entry's directory to engine_root when relative.

    ``dir::module`` -> the ``dir`` is resolved via _config_path so it anchors the
    same way bot_config_dir does; ``module`` (no dir) is passed through unchanged.
    """
    entry = str(entry).strip()
    if "::" in entry:
        root, module = entry.split("::", 1)
        resolved_root = _config_path(root.strip(), engine_root=engine_root)
        return f"{resolved_root}::{module.strip()}"
    return entry


def _load_run_plugins(config: Mapping[str, Any], *, engine_root: Path) -> list[str]:
    """Load author-private plugins for this run (parent process) and re-export the
    merged entry set into ``MTT_PLUGINS`` so ProcessPool workers inherit it.

    Merges the ``MTT_PLUGINS`` env var with the config ``plugins`` list. Returns
    the merged, resolved entry list.
    """
    config_entries = [
        _resolve_plugin_entry(entry, engine_root=engine_root)
        for entry in (config.get("plugins", []) or [])
    ]
    merged = load_plugins(config_entries)
    os.environ[PLUGINS_ENV_VAR] = PLUGINS_ENTRY_SEP.join(merged)
    return merged


def _worker_init() -> None:
    """ProcessPool initializer: repopulate plugin registries in each worker.

    Reads MTT_PLUGINS from the inherited environment. Needed for spawn workers
    (in-memory registries are only inherited under fork); idempotent under fork.
    """
    load_plugins()


def _expand_named_lineup(
    named_lineup: List[Mapping[str, Any]],
    bot_library: Mapping[str, Mapping[str, Any]],
) -> tuple[Dict[str, int], List[Dict[str, Any]]]:
    lineup: Dict[str, int] = {}
    specs: List[Dict[str, Any]] = []
    for row in named_lineup:
        bot_name = str(row.get("bot") or row.get("name") or "")
        if not bot_name:
            raise ValueError("named lineup row requires a bot")
        count = int(row.get("count", 1) or 0)
        if count <= 0:
            continue
        if bot_name in bot_library:
            spec = dict(bot_library[bot_name])
            spec["count"] = count
            specs.append(spec)
            continue
        if bot_name not in BOT_REGISTRY:
            available = ", ".join(sorted(set(bot_library) | set(BOT_REGISTRY)))
            raise ValueError(f"unknown named lineup bot {bot_name!r}; available bots: {available}")
        lineup[bot_name] = lineup.get(bot_name, 0) + count
    return lineup, specs


def _resolve_tournament_seeds(config: Mapping[str, Any], mtt_count: int) -> List[int]:
    raw_seeds = config.get("tournament_seeds")
    if raw_seeds is not None:
        seeds = [int(seed) for seed in list(raw_seeds)]
        if len(seeds) != mtt_count:
            raise ValueError(f"tournament_seeds requires exactly {mtt_count} seeds")
    else:
        if "mtt_seed_start" in config:
            seed_start = int(config.get("mtt_seed_start", 72001))
            seeds = [seed_start + index for index in range(mtt_count)]
        else:
            seed_start = int(config.get("random_seed", 72001))
            seeds = [seed_start + 100000 + index for index in range(mtt_count)]

    if len(set(seeds)) != len(seeds):
        raise ValueError("tournament_seeds must be unique; one seed belongs to one MTT")
    return seeds


def _run_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    tournament_id = int(payload["tournament_id"])
    engine_config = dict(payload["engine_config"])
    artifact_root = payload.get("artifact_root")
    resume_log = bool(payload.get("resume_log", False)) and artifact_root is not None
    try:
        seed = int(payload["seed"])
        _seed_everything(seed)
        with temporary_engine_config(_engine_overrides(engine_config)):
            bots, name_to_population = _build_bots(
                dict(payload["lineup"]),
                engine_config,
                list(payload.get("lineup_variants", [])),
            )
            random.shuffle(bots)
            tournament = Tournament(bots, tournament_id=tournament_id)

            writer = None
            if resume_log:
                events_dir = tournament_events_dir(artifact_root, tournament_id)
                # Fresh run of this tournament: drop any stale partial log so
                # chunk numbering starts clean. (Replay-based resume, added
                # later, manages the existing log instead of wiping it.)
                if events_dir.exists():
                    shutil.rmtree(events_dir)
                writer = ChunkLogWriter(events_dir)
            try:
                results, events = tournament.play(event_sink=writer)
            finally:
                if writer is not None:
                    writer.close()

            result = {
                "tournament_id": tournament_id,
                "seed": seed,
                "results": results,
                "events": events if bool(payload.get("write_events", False)) else [],
                "population_summary": summarize_population_results(results, name_to_population),
                "action_summary": summarize_candidate_actions(events, name_to_population),
                "stopped_max_hands": any(event.get("type") == "tournament_stopped_max_hands" for event in events),
                "event_count": len(events),
                "failure": "",
            }
            if resume_log:
                # Completion checkpoint: full result minus bulky events (those
                # live in the chunk log). Its presence marks this tournament
                # done for the resume skip-set. Written only after writer.close()
                # so all chunks are guaranteed durable first.
                checkpoint = dict(result)
                checkpoint["events"] = []
                atomic_write_json(tournament_result_path(artifact_root, tournament_id), checkpoint)
            return result
    except Exception as exc:
        return {
            "tournament_id": tournament_id,
            "seed": int(payload.get("seed", 0) or 0),
            "results": [],
            "events": [],
            "population_summary": {},
            "action_summary": {},
            "stopped_max_hands": False,
            "event_count": 0,
            "failure": str(exc),
        }


def _population_summary_from_report(report: Mapping[str, Any]) -> Mapping[str, Any]:
    summary = report.get("population_summary")
    if isinstance(summary, Mapping):
        return summary
    simulation = report.get("simulation")
    if isinstance(simulation, Mapping):
        summary = simulation.get("population_summary")
        if isinstance(summary, Mapping):
            return summary
    return {}


def _summarize_merged_population(raw: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for name, row in raw.items():
        entries = int(row.get("entries", 0) or 0)
        wins = int(row.get("wins", 0) or 0)
        top3 = int(row.get("top3", 0) or 0)
        final_table = int(row.get("final_table", 0) or 0)
        itm = int(row.get("itm", 0) or 0)
        position_sum = float(row.get("position_sum", 0.0) or 0.0)
        total_payout_pct = float(row.get("total_payout_pct", 0.0) or 0.0)
        score = float(row.get("score", 0.0) or 0.0)
        summary[name] = {
            "entries": entries,
            "wins": wins,
            "win_rate": wins / entries if entries else 0.0,
            "top3": top3,
            "top3_rate": top3 / entries if entries else 0.0,
            "final_table": final_table,
            "final_table_rate": final_table / entries if entries else 0.0,
            "itm": itm,
            "itm_rate": itm / entries if entries else 0.0,
            "position_sum": position_sum,
            "average_position": position_sum / entries if entries else 0.0,
            "total_payout_pct": total_payout_pct,
            "average_payout_pct": total_payout_pct / entries if entries else 0.0,
            "score": score,
        }
    return population_summary_with_roi(summary)


def population_summary_with_roi(population_summary: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    total_entries = 0.0
    total_payout = 0.0
    for name, summary in population_summary.items():
        if not isinstance(summary, Mapping):
            continue
        row = dict(summary)
        total_entries += float(row.get("entries", 0.0) or 0.0)
        total_payout += float(row.get("total_payout_pct", 0.0) or 0.0)
        rows[str(name)] = row

    for row in rows.values():
        entries = float(row.get("entries", 0.0) or 0.0)
        payout = float(row.get("total_payout_pct", 0.0) or 0.0)
        expected_payout = total_payout * entries / total_entries if total_entries > 0.0 else 0.0
        row["roi"] = (payout / expected_payout) - 1.0 if expected_payout > 0.0 else 0.0
    return rows


def merge_population_summaries(reports: List[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    raw: Dict[str, Dict[str, Any]] = {}
    additive_fields = (
        "entries",
        "wins",
        "top3",
        "final_table",
        "itm",
        "position_sum",
        "total_payout_pct",
        "score",
    )
    for report in reports:
        for name, row in _population_summary_from_report(report).items():
            target = raw.setdefault(str(name), {field: 0.0 for field in additive_fields})
            if not isinstance(row, Mapping):
                continue
            for field in additive_fields:
                target[field] += float(row.get(field, 0.0) or 0.0)
    return _summarize_merged_population(raw)


def ranked_population_rows(population_summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, summary in population_summary.items():
        if isinstance(summary, Mapping):
            rows.append({"name": str(name), **dict(summary)})
    return sorted(
        rows,
        key=lambda row: (
            float(row.get("total_payout_pct", 0.0) or 0.0),
            float(row.get("score", 0.0) or 0.0),
            float(row.get("wins", 0.0) or 0.0),
            -float(row.get("average_position", 999999.0) or 999999.0),
            str(row.get("name", "")),
        ),
        reverse=True,
    )


def _format_table_value(key: str, value: Any) -> str:
    if key in {"name"}:
        return str(value)
    if key in {"total_payout_pct", "average_payout_pct"}:
        return f"{float(value or 0.0):.4f}"
    if key in {"roi"}:
        return f"{float(value or 0.0) * 100.0:+.1f}%"
    if key in {"win_rate", "top3_rate", "final_table_rate", "itm_rate"}:
        return f"{float(value or 0.0) * 100.0:.2f}"
    if key in {"average_position"}:
        return f"{float(value or 0.0):.1f}"
    if key in {"entries", "wins", "top3", "final_table", "itm"}:
        return str(int(value or 0))
    if key in {"score"}:
        return f"{float(value or 0.0):.0f}"
    return str(value)


def format_population_table(population_summary: Mapping[str, Any]) -> str:
    rows = ranked_population_rows(population_summary_with_roi(population_summary))
    headers = [header for header, _ in POPULATION_TABLE_COLUMNS]
    rendered_rows: List[List[str]] = []
    for rank, row in enumerate(rows, start=1):
        rendered_row: List[str] = []
        for _, key in POPULATION_TABLE_COLUMNS:
            if key == "rank":
                rendered_row.append(str(rank))
            else:
                rendered_row.append(_format_table_value(key, row.get(key, 0)))
        rendered_rows.append(rendered_row)

    align = ["---:"] + ["---"] + ["---:"] * (len(headers) - 2)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rendered_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _load_reports(paths: List[str], *, engine_root: Path) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for raw_path in paths:
        reports.append(load_json_config(_config_path(raw_path, engine_root=engine_root)))
    return reports


def _report_completion_line(reports: List[Mapping[str, Any]]) -> str:
    completed = 0
    failures = 0
    requested = 0
    for report in reports:
        requested += int(report.get("mtt_count", 0) or 0)
        simulation = report.get("simulation")
        if isinstance(simulation, Mapping):
            tournaments = simulation.get("tournaments", [])
            completed += len(tournaments) if isinstance(tournaments, list) else 0
            failure_rows = simulation.get("failures", [])
            failures += len(failure_rows) if isinstance(failure_rows, list) else 0
    return f"reports={len(reports)} configured_mtts={requested} completed_mtts={completed} recorded_failures={failures}"


def run_fixed_bot_evaluation(config: Dict[str, Any], *, engine_root: Path) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = str(config.get("run_id", "fixed_bot_mtt_evaluation"))
    artifact_root = Path(str(config.get("artifact_root") or Path("runs") / run_id / utc_stamp()))
    artifact_root.mkdir(parents=True, exist_ok=True)
    mtt_count = int(config.get("mtt_count", 200))
    workers = max(1, min(int(config.get("workers", 8)), mtt_count))
    tournament_seeds = _resolve_tournament_seeds(config, mtt_count)
    # Load author-private bot/tool plugins before any registry lookup (the bot
    # library / named-lineup expansion below resolves against BOT_REGISTRY).
    _load_run_plugins(config, engine_root=engine_root)
    lineup = dict(config.get("lineup", {}))
    lineup_variants = list(config.get("lineup_variants", config.get("bot_lineup", [])) or [])
    named_lineup = list(config.get("named_lineup", []) or [])
    bot_library = _load_bot_library(config, engine_root=engine_root)
    named_legacy_lineup, named_specs = _expand_named_lineup(named_lineup, bot_library)
    for bot_name, count in named_legacy_lineup.items():
        lineup[bot_name] = int(lineup.get(bot_name, 0)) + int(count)
    lineup_variants.extend(named_specs)
    engine_config = default_engine_config()
    engine_config.update(dict(config.get("engine", {})))

    all_results: List[Dict[str, Any]] = []
    tournament_summaries: List[Dict[str, Any]] = []
    action_summaries: List[Dict[str, Any]] = []
    failures: List[str] = []
    partial_summary_path = artifact_root / "partial_summary.json"
    started = time.perf_counter()
    last_completion_at = started
    progress_interval_seconds = max(1.0, float(config.get("progress_interval_seconds", 30.0) or 30.0))
    write_events = bool(config.get("write_events", False))
    resume_log = bool(config.get("resume_log", True))
    configured_populations = set(str(key) for key in lineup)
    use_ranges = bool(engine_config.get("fixed_bots_use_preflop_spot_range", False))
    for spec in lineup_variants:
        spec_type = str(spec.get("type") or spec.get("bot") or spec.get("bot_type") or spec.get("class") or "")
        explicit_name = str(spec.get("name") or "")
        if explicit_name:
            configured_populations.add(explicit_name)
        if spec.get("include_tool_set_in_name") or spec.get("name_tool_set"):
            configured_populations.add(population_for_spec(spec, use_ranges=use_ranges))
        else:
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

    def print_progress(*, heartbeat: bool = False) -> None:
        now = time.perf_counter()
        completed = len(tournament_summaries)
        prefix = "heartbeat" if heartbeat else "progress"
        print(
            f"{prefix}: completed_mtts={completed}/{mtt_count} "
            f"failures={len(failures)} "
            f"elapsed_s={now - started:.1f} "
            f"since_last_completion_s={now - last_completion_at:.1f} "
            f"partial={partial_summary_path}",
            flush=True,
        )

    def _ingest_result(result: Dict[str, Any]) -> None:
        all_results.extend(list(result.get("results", [])))
        tournament_summaries.append(
            {
                "tournament_id": int(result.get("tournament_id", 0)),
                "seed": int(result.get("seed", 0) or 0),
                "population_summary": dict(result.get("population_summary", {})),
                "population_action_summary": dict(result.get("action_summary", {})),
                "stopped_max_hands": bool(result.get("stopped_max_hands", False)),
                "event_count": int(result.get("event_count", 0)),
            }
        )
        if write_events and result.get("events"):
            event_path = artifact_root / "events" / f"tournament_{int(result.get('tournament_id', 0)):04d}_events.json"
            _write_json(event_path, list(result.get("events", [])))
        action_summaries.append(dict(result.get("action_summary", {})))

    # Resume skip-set: any tournament with a durable completion checkpoint is
    # loaded verbatim and not re-run. (Interrupted tournaments — a partial log
    # but no checkpoint — are re-run fresh here; replay-based continuation is
    # added by the --resume feature.)
    already_done = completed_tournament_ids(artifact_root) if resume_log else {}
    if already_done:
        for tournament_id in sorted(already_done):
            _ingest_result(load_tournament_result(already_done[tournament_id]))
        print(
            f"resume: loaded {len(already_done)} completed tournaments from {artifact_root}; "
            f"running remaining {mtt_count - len(already_done)}",
            flush=True,
        )

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init
    ) as executor:
        future_to_id = {}
        for index in range(mtt_count):
            tournament_id = index + 1
            if tournament_id in already_done:
                continue
            payload = {
                "tournament_id": tournament_id,
                "seed": tournament_seeds[index],
                "lineup": lineup,
                "lineup_variants": lineup_variants,
                "engine_config": engine_config,
                "write_events": write_events,
                "artifact_root": str(artifact_root),
                "resume_log": resume_log,
            }
            future_to_id[executor.submit(_run_worker, payload)] = tournament_id

        pending = set(future_to_id)
        while pending:
            done, pending = concurrent.futures.wait(
                pending,
                timeout=progress_interval_seconds,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                write_partial()
                print_progress(heartbeat=True)
                continue
            for future in done:
                tournament_id = future_to_id[future]
                try:
                    result = future.result()
                except Exception as exc:
                    failures.append(f"tournament {tournament_id} failed: {exc}")
                    write_partial()
                    print_progress()
                    continue
                failure = str(result.get("failure", "") or "")
                if failure:
                    failures.append(f"tournament {tournament_id} failed: {failure}")
                else:
                    _ingest_result(result)
                    last_completion_at = time.perf_counter()
                write_partial()
                print_progress()

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
        "named_lineup": named_lineup,
        "bot_config_dir": config.get("bot_config_dir", ""),
        "bot_config_files": list(config.get("bot_config_files", []) or []),
        "plugins": list(config.get("plugins", []) or []),
        "mtt_count": mtt_count,
        "workers": workers,
        "random_seed": config.get("random_seed", ""),
        "mtt_seed_start": config.get("mtt_seed_start", ""),
        "tournament_seeds": tournament_seeds,
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
    parser.add_argument(
        "--report",
        nargs="+",
        default=[],
        help="Print one stable complete population table from one or more fixed-bot reports.",
    )
    args = parser.parse_args(argv)
    engine_root = Path(__file__).resolve().parents[1]
    if args.report:
        reports = _load_reports(list(args.report), engine_root=engine_root)
        print(_report_completion_line(reports))
        print()
        print(format_population_table(merge_population_summaries(reports)))
        return

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
