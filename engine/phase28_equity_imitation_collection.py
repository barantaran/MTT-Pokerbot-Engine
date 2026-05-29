"""Phase 28 collection of equity-bot supervised policy labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from engine.baseline_model_bot import add_basemodel_to_path, has_forbidden_state_keys, is_engine_action
from engine.phase22_small_mtt_engine_simulation import (
    ACCEPTED,
    REJECTED,
    _engine_overrides,
    _read_json,
    _run_stamp,
    _seed_everything,
    _utc_now,
    _write_json,
    build_lineup,
    lineup_summary,
    temporary_engine_config,
)
from engine.phase23_larger_mtt_engine_simulation import summarize_events, summarize_results
from engine.pokerstove_equity import estimate_equity, pot_odds
from engine.player_interface import Bot
from engine.tournament import Tournament
from players.aggressive_bot import AggressiveBot
from players.tight_equity_bot import TightEquityBot


SCHEMA_VERSION = 1


def load_config(path: str | Path) -> Dict[str, Any]:
    loaded = _read_json(path)
    loaded["config_path"] = str(path)
    return loaded


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(config)
    run_id = str(resolved.get("phase28_equity_imitation_collection_id", "phase28_equity_imitation_collection"))
    artifact_root_value = str(resolved.get("artifact_root", "") or "")
    if artifact_root_value:
        artifact_root = Path(artifact_root_value)
    else:
        artifact_root = Path(str(resolved.get("output_dir", Path("runs") / run_id))) / _run_stamp()
    dataset_dir = artifact_root / "dataset"
    resolved["artifact_root"] = str(artifact_root)
    resolved.setdefault("equity_imitation_collection_report_path", str(artifact_root / "equity_imitation_collection_report.json"))
    resolved.setdefault("tournament_results_path", str(artifact_root / "tournament_results.json"))
    resolved.setdefault("event_summaries_path", str(artifact_root / "event_summaries.json"))
    resolved.setdefault("dataset_path", str(dataset_dir / "equity_imitation_dataset.jsonl"))
    resolved.setdefault("manifest_path", str(dataset_dir / "manifest.json"))
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _street_from_board(board_cards: Iterable[Any]) -> str:
    board_len = len(list(board_cards or []))
    if board_len == 0:
        return "preflop"
    if board_len == 3:
        return "flop"
    if board_len == 4:
        return "turn"
    if board_len >= 5:
        return "river"
    return "unknown"


def _finite_floats(values: Iterable[Any]) -> list[float]:
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("reduced observation contains non-finite values")
    return converted


def _mask_as_ints(values: Iterable[Any]) -> list[int]:
    return [1 if bool(value) else 0 for value in values]


class EquityImitationCollector:
    def __init__(self, config: Dict[str, Any], *, basemodel_root: Path):
        self.schema_version = int(config.get("schema_version", SCHEMA_VERSION))
        self.dataset_path = Path(str(config["dataset_path"]))
        self.manifest_path = Path(str(config["manifest_path"]))
        self.basemodel_root = add_basemodel_to_path(basemodel_root)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_path.write_text("", encoding="utf-8")
        self.sample_count = 0
        self.validation_failures: list[str] = []
        self.privacy_violations: list[str] = []
        self.action_mapping_failures: list[str] = []
        self.action_counts = {str(index): 0 for index in range(9)}

    def record_decision(
        self,
        *,
        game_state: Dict[str, Any],
        action: Tuple[str, int | None],
        bot_type: str = "equity_aggressive",
        fallback_used: bool = False,
        fallback_reason: str | None = None,
    ) -> None:
        self.sample_count += 1
        sample_index = self.sample_count
        if has_forbidden_state_keys(game_state):
            self.privacy_violations.append(f"sample {sample_index} contains forbidden runtime state")

        try:
            from poker_ai.actions import ACTION_LABELS, legal_action_mask
            from poker_ai.equity_imitation import engine_action_to_action_id
            from poker_ai.reduced_observations import REDUCED_OBSERVATION_SIZE, encode_reduced_observation

            observation = _finite_floats(encode_reduced_observation(game_state))
            if len(observation) != REDUCED_OBSERVATION_SIZE:
                raise ValueError(f"reduced observation length is {len(observation)}, expected {REDUCED_OBSERVATION_SIZE}")
            legal_mask = _mask_as_ints(legal_action_mask(game_state))
            action_id = engine_action_to_action_id(action, game_state)
            if action_id < 0 or action_id >= len(ACTION_LABELS):
                raise ValueError(f"action id out of range: {action_id}")
            if not legal_mask[action_id]:
                raise ValueError(f"action id {action_id} is illegal under mask")
            action_name = ACTION_LABELS[action_id]
        except Exception as exc:
            self.validation_failures.append(f"sample {sample_index} invalid: {exc}")
            return

        if not is_engine_action(action):
            self.action_mapping_failures.append(f"sample {sample_index} invalid engine action: {action!r}")
            return

        blinds = game_state.get("blinds") if isinstance(game_state.get("blinds"), dict) else {}
        record = {
            "schema_version": self.schema_version,
            "phase": 28,
            "tournament_id": int(game_state.get("tournament_id", 0) or 0),
            "hand_id": int(game_state.get("hand_id", 0) or 0),
            "table_id": int(game_state.get("table_id", 0) or 0),
            "decision_index": sample_index,
            "player_name": str(game_state.get("player_id", "")),
            "bot_type": bot_type,
            "street": _street_from_board(game_state.get("board_cards", [])),
            "small_blind": int(blinds.get("small", 0) or 0),
            "big_blind": int(blinds.get("big", 0) or 0),
            "stack_before": int(game_state.get("stack_size", 0) or 0),
            "pot_before": int(game_state.get("pot_size", 0) or 0),
            "call_amount": int(game_state.get("call_amount", 0) or 0),
            "min_raise": int(game_state.get("min_raise", 0) or 0),
            "hero_equity": float(game_state.get("hero_equity", 0.5)),
            "pot_odds": float(game_state.get("pot_odds", 0.0)),
            "equity_source": str(game_state.get("hero_equity_source", "engine_estimate_equity")),
            "reduced_observation": observation,
            "legal_action_mask": legal_mask,
            "action_id": int(action_id),
            "action_name": action_name,
            "engine_action": action[0],
            "engine_amount": int(action[1] or 0),
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
        }
        with self.dataset_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.action_counts[str(action_id)] += 1

    def manifest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        manifest = {
            "schema_version": self.schema_version,
            "phase": 28,
            "dataset_path": str(self.dataset_path),
            "sample_count": self.sample_count,
            "valid_dataset_rows": _jsonl_count(self.dataset_path),
            "action_distribution": self.action_counts,
            "config_snapshot": config,
            "files": {
                "dataset": {
                    "path": str(self.dataset_path),
                    "exists": self.dataset_path.is_file(),
                    "size_bytes": self.dataset_path.stat().st_size if self.dataset_path.is_file() else 0,
                    "sha256": _file_sha256(self.dataset_path) if self.dataset_path.is_file() else "",
                    "line_count": _jsonl_count(self.dataset_path),
                }
            },
        }
        _write_json(self.manifest_path, manifest)
        return manifest

    def validation_summary(self) -> Dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "valid_dataset_rows": _jsonl_count(self.dataset_path),
            "validation_failures": self.validation_failures,
            "privacy_violations": self.privacy_violations,
            "action_mapping_failures": self.action_mapping_failures,
            "valid": not self.validation_failures and not self.privacy_violations and not self.action_mapping_failures,
        }


class CollectingTeacherBot(Bot):
    def __init__(
        self,
        *,
        teacher: Bot,
        collector: EquityImitationCollector,
        bot_type: str,
        name: str,
    ):
        super().__init__(name)
        self.teacher = teacher
        self.name = name
        self.collector = collector
        self.bot_type = bot_type

    def get_action(self, game_state: Dict[str, Any]):
        enriched = dict(game_state)
        try:
            equity = estimate_equity(
                hole_cards=enriched.get("hole_cards", []),
                board_cards=enriched.get("board_cards", []),
                active_players=int(enriched.get("active_players", 2) or 2),
            )
            enriched["hero_equity"] = float(equity)
            enriched["hero_equity_source"] = "pokerstove"
            enriched["pot_odds"] = pot_odds(int(enriched.get("call_amount", 0) or 0), int(enriched.get("pot_size", 0) or 0))
            action = self.teacher.get_action(enriched)
            self.collector.record_decision(game_state=enriched, action=action, bot_type=self.bot_type)
            return action
        except Exception as exc:
            self.collector.record_decision(
                game_state=enriched,
                action=("call", 0) if int(enriched.get("call_amount", 0) or 0) == 0 else ("fold", 0),
                bot_type=self.bot_type,
                fallback_used=True,
                fallback_reason=str(exc),
            )
            raise


def build_phase28_lineup(config: Dict[str, Any], collector: EquityImitationCollector) -> List[Any]:
    patched = dict(config)
    patched_lineup = dict(config.get("lineup", {}))
    equity_count = int(patched_lineup.get("equity_aggressive", 1))
    tight_count = int(patched_lineup.get("tight_equity", 0))
    patched_lineup["equity_aggressive"] = 0
    patched_lineup["tight_equity"] = 0
    patched["lineup"] = patched_lineup
    bots: List[Any] = []
    for index in range(tight_count):
        bots.append(
            CollectingTeacherBot(
                teacher=TightEquityBot(),
                collector=collector,
                bot_type="tight_equity",
                name=f"CollectingTightEquityBot_{index + 1}",
            )
        )
    for index in range(equity_count):
        bots.append(
            CollectingTeacherBot(
                teacher=AggressiveBot(),
                collector=collector,
                bot_type="equity_aggressive",
                name=f"CollectingEquityAggressiveBot_{index + 1}",
            )
        )
    bots.extend(build_lineup(patched, checkpoint_path="", basemodel_root=""))
    return bots


def run_tournaments(config: Dict[str, Any], bots: List[Any], collector: EquityImitationCollector) -> Dict[str, Any]:
    tournament_count = int(config.get("tournament_count", 1))
    seed = int(config.get("random_seed", 0))
    result_rows = []
    event_summaries = []
    tournament_failures = []
    with temporary_engine_config(_engine_overrides(config)):
        for index in range(tournament_count):
            tournament_id = index + 1
            _seed_everything(seed + index)
            try:
                tournament = Tournament(bots, tournament_id=tournament_id)
                results, events = tournament.play()
                stopped = any(event.get("type") == "tournament_stopped_max_hands" for event in events)
                event_summaries.append(summarize_events(tournament_id, events, results))
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
    return {
        "tournament_results": result_rows,
        "event_summaries": event_summaries,
        "tournament_failures": tournament_failures,
        "lineup_summary": lineup_summary(bots),
    }


def build_report(
    config: Dict[str, Any],
    simulation: Dict[str, Any],
    collector: EquityImitationCollector,
    manifest: Dict[str, Any],
    started_at: str,
    runtime_seconds: float,
) -> Dict[str, Any]:
    acceptance = dict(config.get("acceptance", {}))
    summary = summarize_results(list(simulation.get("tournament_results", [])))
    validation = collector.validation_summary()
    min_samples = int(acceptance.get("min_dataset_rows", 100))
    gates = {
        "tournament_runtime_gate_passed": not simulation.get("tournament_failures"),
        "min_dataset_rows_gate_passed": int(validation.get("valid_dataset_rows", 0)) >= min_samples,
        "dataset_artifact_gate_passed": Path(str(config.get("dataset_path", ""))).is_file(),
        "manifest_artifact_gate_passed": Path(str(config.get("manifest_path", ""))).is_file(),
        "collection_validation_gate_passed": validation["valid"],
    }
    failures = list(simulation.get("tournament_failures", []))
    failures.extend(validation["validation_failures"])
    failures.extend(validation["privacy_violations"])
    failures.extend(validation["action_mapping_failures"])
    for gate, passed in gates.items():
        if not passed:
            failures.append(f"{gate} failed")
    status = ACCEPTED if not failures else REJECTED
    return {
        "phase28_equity_imitation_collection_id": config.get(
            "phase28_equity_imitation_collection_id", "phase28_equity_imitation_collection"
        ),
        "phase28_equity_imitation_collection_status": status,
        "phase28_equity_imitation_collection_failures": list(dict.fromkeys(failures)),
        "phase28_equity_imitation_collection_started_at": started_at,
        "phase28_equity_imitation_collection_finished_at": _utc_now(),
        "phase28_equity_imitation_collection_runtime_seconds": runtime_seconds,
        "phase28_equity_imitation_collection_report_path": str(config.get("equity_imitation_collection_report_path", "")),
        "phase28_gate_results": gates,
        "tournament_summary": summary,
        "lineup_summary": simulation.get("lineup_summary", {}),
        "dataset_path": str(config.get("dataset_path", "")),
        "manifest_path": str(config.get("manifest_path", "")),
        "manifest": manifest,
        "collection_validation_summary": validation,
        "action_distribution": collector.action_counts,
        "phase28_equity_imitation_training_allowed": status == ACCEPTED,
        "next_phase_recommendation": "launch_phase28_equity_imitation_training"
        if status == ACCEPTED
        else "inspect_phase28_equity_imitation_collection",
    }


def run_equity_imitation_collection(config: Dict[str, Any], config_path: str | Path | None = None) -> Dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    config = _resolve_paths(dict(config))
    if config_path is not None:
        config["config_path"] = str(config_path)
    engine_root = Path(__file__).resolve().parents[1]
    basemodel_root = Path(str(config.get("basemodel_root", "../poker-ai-basemodel"))).expanduser()
    if not basemodel_root.is_absolute():
        basemodel_root = (engine_root / basemodel_root).resolve()
    Path(str(config["artifact_root"])).mkdir(parents=True, exist_ok=True)
    _write_json(Path(str(config["artifact_root"])) / "config.json", config)

    collector = EquityImitationCollector(config, basemodel_root=basemodel_root)
    bots = build_phase28_lineup(config, collector)
    simulation = run_tournaments(config, bots, collector)
    _write_json(config["tournament_results_path"], simulation.get("tournament_results", []))
    _write_json(config["event_summaries_path"], simulation.get("event_summaries", []))
    manifest = collector.manifest(config)
    report = build_report(config, simulation, collector, manifest, started_at, time.perf_counter() - started)
    _write_json(report["phase28_equity_imitation_collection_report_path"], report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase28_equity_imitation_collection.json")
    args = parser.parse_args(argv)
    print(json.dumps(run_equity_imitation_collection(load_config(args.config), config_path=args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
