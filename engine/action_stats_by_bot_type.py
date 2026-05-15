"""Action-count extraction grouped by bot type."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


PREFERRED_ACTION_ORDER = ("fold", "check", "call", "raise")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def display_bot_type(player_name: str, bot_class: str = "") -> str:
    if player_name.startswith(("Phase22ModelBot_", "Phase23ModelBot_")):
        return "ModelBot"
    if player_name.startswith("EquityAggressiveBot_"):
        return "EquityAggressiveBot"
    if player_name.startswith("AggressiveNoEquityBot_"):
        return "AggressiveNoEquityBot"
    if player_name.startswith("RandomBot_"):
        return "RandomBot"
    if player_name.startswith("CallBot_"):
        return "CallBot"
    return bot_class or infer_bot_type_from_name(player_name)


def infer_bot_type_from_name(player_name: str) -> str:
    for prefix in (
        "Phase22ModelBot",
        "Phase23ModelBot",
        "EquityAggressiveBot",
        "AggressiveNoEquityBot",
        "RandomBot",
        "CallBot",
        "AggressiveBot",
    ):
        if player_name == prefix or player_name.startswith(f"{prefix}_"):
            return "ModelBot" if prefix.startswith("Phase") else prefix
    return "Unknown"


def build_player_type_map(results: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    player_types = {}
    for row in results:
        name = str(row.get("name", ""))
        if not name:
            continue
        player_types[name] = display_bot_type(name, str(row.get("bot_class", "")))
    return player_types


def index_tournament_results(payload: Any) -> Dict[int, Dict[str, str]]:
    indexed: Dict[int, Dict[str, str]] = {}
    if not isinstance(payload, list):
        return indexed
    for index, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            continue
        tournament_id = int(row.get("tournament_id", index))
        indexed[tournament_id] = build_player_type_map(list(row.get("results", [])))
    return indexed


def tournament_id_from_event_path(path: str | Path) -> int | None:
    stem = Path(path).stem
    digits = "".join(char for char in stem if char.isdigit())
    return int(digits) if digits else None


def discover_event_logs(path: str | Path) -> List[Path]:
    source = Path(path)
    if source.is_file():
        return [source]
    event_dir = source / "events"
    if event_dir.is_dir():
        return sorted(event_dir.glob("tournament_*_events.json"))
    return sorted(source.glob("*events*.json"))


def load_run_player_maps(path: str | Path) -> Dict[int, Dict[str, str]]:
    source = Path(path)
    result_path = source / "tournament_results.json" if source.is_dir() else source.with_name("tournament_results.json")
    if result_path.is_file():
        return index_tournament_results(read_json(result_path))
    return {}


def summarize_event_log(
    path: str | Path,
    player_types: Dict[str, str] | None = None,
    tournament_id: int | None = None,
) -> Dict[str, Any]:
    payload = read_json(path)
    if isinstance(payload, dict):
        events = list(payload.get("events", []))
        if player_types is None:
            player_types = build_player_type_map(list(payload.get("results", [])))
        tournament_id = tournament_id or int(payload.get("simulation_id", 0) or 0) or tournament_id_from_event_path(path)
    else:
        events = list(payload)
        player_types = player_types or {}
        tournament_id = tournament_id or tournament_id_from_event_path(path)

    action_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    player_action_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    unknown_players = Counter()
    action_event_count = 0

    for event in events:
        if not isinstance(event, dict) or event.get("type") != "action":
            continue
        action_event_count += 1
        player = str(event.get("player", ""))
        action = str(event.get("action", "unknown"))
        bot_type = (player_types or {}).get(player)
        if not bot_type:
            bot_type = infer_bot_type_from_name(player)
        if bot_type == "Unknown":
            unknown_players[player] += 1
        action_counts[bot_type][action] += 1
        player_action_counts[player][action] += 1

    return {
        "event_log_path": str(path),
        "tournament_id": tournament_id,
        "action_event_count": action_event_count,
        "action_counts_by_bot_type": {bot: dict(counts) for bot, counts in sorted(action_counts.items())},
        "action_counts_by_player": {player: dict(counts) for player, counts in sorted(player_action_counts.items())},
        "unknown_player_action_counts": dict(unknown_players),
    }


def merge_summaries(summaries: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    merged_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    unknown_players = Counter()
    tournament_ids = []
    log_paths = []
    action_event_count = 0

    for summary in summaries:
        tournament_id = summary.get("tournament_id")
        if tournament_id is not None:
            tournament_ids.append(tournament_id)
        log_paths.append(summary.get("event_log_path", ""))
        action_event_count += int(summary.get("action_event_count", 0))
        for bot_type, counts in dict(summary.get("action_counts_by_bot_type", {})).items():
            merged_counts[str(bot_type)].update({str(action): int(count) for action, count in dict(counts).items()})
        unknown_players.update(
            {str(player): int(count) for player, count in dict(summary.get("unknown_player_action_counts", {})).items()}
        )

    return {
        "event_log_count": len(log_paths),
        "event_log_paths": log_paths,
        "tournament_ids": tournament_ids,
        "action_event_count": action_event_count,
        "action_counts_by_bot_type": {bot: dict(counts) for bot, counts in sorted(merged_counts.items())},
        "unknown_player_action_counts": dict(unknown_players),
    }


def summarize_path(path: str | Path) -> Dict[str, Any]:
    event_logs = discover_event_logs(path)
    player_maps = load_run_player_maps(path)
    summaries = []
    for event_log in event_logs:
        tournament_id = tournament_id_from_event_path(event_log)
        player_types = player_maps.get(tournament_id) if tournament_id is not None else None
        summaries.append(summarize_event_log(event_log, player_types, tournament_id))
    merged = merge_summaries(summaries)
    merged["source_path"] = str(path)
    return merged


def ordered_actions(counts_by_bot_type: Dict[str, Dict[str, int]]) -> List[str]:
    observed = {action for counts in counts_by_bot_type.values() for action in counts}
    ordered = [action for action in PREFERRED_ACTION_ORDER if action in observed]
    ordered.extend(sorted(observed - set(ordered)))
    return ordered


def format_table(summary: Dict[str, Any]) -> str:
    counts_by_bot_type = dict(summary.get("action_counts_by_bot_type", {}))
    actions = ordered_actions(counts_by_bot_type)
    headers = ["bot_type", "total", *actions]
    rows = []
    for bot_type, counts in counts_by_bot_type.items():
        total = sum(int(value) for value in counts.values())
        rows.append([bot_type, str(total), *[str(int(counts.get(action, 0))) for action in actions]])

    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    lines = [
        f"Source: {summary.get('source_path', '')}",
        f"Event logs analyzed: {summary.get('event_log_count', 0)}",
        f"Action events counted: {summary.get('action_event_count', 0)}",
        "",
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))
    unknowns = dict(summary.get("unknown_player_action_counts", {}))
    if unknowns:
        lines.extend(["", f"Unknown player mappings: {unknowns}"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Map tournament action counts to bot types.")
    parser.add_argument("path", help="Run directory, event log JSON, or old stats log JSON.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON instead of a table.")
    args = parser.parse_args()

    summary = summarize_path(args.path)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
