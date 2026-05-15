"""Phase 24 engine rollout collection for retraining."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from engine.baseline_model_bot import (
    BaselineModelEngineBot,
    conservative_engine_action,
    has_forbidden_state_keys,
    is_engine_action,
    resolve_checkpoint_path,
)
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
from engine.phase23_larger_mtt_engine_simulation import summarize_events, summarize_results
from engine.tournament import Tournament


SCHEMA_VERSION = 1
ACTION_LABELS = (
    "Fold",
    "Check",
    "Call",
    "Bet_0_33",
    "Bet_0_5",
    "Bet_1",
    "Bet_1_5",
    "Bet_2",
    "Allin",
)


def load_config(path: str | Path) -> Dict[str, Any]:
    loaded = _read_json(path)
    loaded["config_path"] = str(path)
    return loaded


def _load_report(path: str | Path) -> Dict[str, Any] | None:
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def resolve_phase23_report(config: Dict[str, Any]) -> tuple[str, Dict[str, Any], str]:
    configured_path = str(config.get("phase23_report_path", "") or "")
    required_status = str(config.get("expected_phase23_status", ACCEPTED))
    if configured_path:
        report = _load_report(configured_path)
        if report is None:
            return configured_path, {}, "source Phase 23 report is missing or malformed"
        return configured_path, report, ""

    pattern = str(config.get("phase23_report_glob", "") or "")
    if not pattern:
        return "", {}, "phase23_report_path or phase23_report_glob is required"

    matches = sorted(glob.glob(pattern), key=lambda item: Path(item).stat().st_mtime, reverse=True)
    malformed_count = 0
    for path in matches:
        report = _load_report(path)
        if report is None:
            malformed_count += 1
            continue
        if report.get("phase23_larger_engine_simulation_status") == required_status and bool(
            report.get("engine_rollout_collection_allowed", False)
        ):
            return path, report, ""
    if matches:
        suffix = f"; {malformed_count} malformed candidates skipped" if malformed_count else ""
        return matches[0], _load_report(matches[0]) or {}, f"no accepted Phase 23 report matched {pattern}{suffix}"
    return "", {}, f"no Phase 23 reports matched {pattern}"


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    resolved = dict(config)
    run_id = str(resolved.get("phase24_engine_rollout_collection_id", "phase24_engine_rollout_collection"))
    artifact_root_value = str(resolved.get("artifact_root", "") or "")
    if artifact_root_value:
        artifact_root = Path(artifact_root_value)
    else:
        artifact_root = Path(str(resolved.get("output_dir", Path("runs") / run_id))) / _run_stamp()
    rollout_dir = artifact_root / "rollouts"
    resolved["artifact_root"] = str(artifact_root)
    resolved.setdefault("engine_rollout_collection_report_path", str(artifact_root / "engine_rollout_collection_report.json"))
    resolved.setdefault("tournament_results_path", str(artifact_root / "tournament_results.json"))
    resolved.setdefault("event_summaries_path", str(artifact_root / "event_summaries.json"))
    resolved.setdefault("event_log_dir", str(artifact_root / "events"))
    resolved.setdefault("rollout_dir", str(rollout_dir))
    resolved.setdefault("model_decisions_path", str(rollout_dir / "model_decisions.jsonl"))
    resolved.setdefault("terminal_outcomes_path", str(rollout_dir / "terminal_outcomes.jsonl"))
    resolved.setdefault("manifest_path", str(rollout_dir / "manifest.json"))
    return resolved


def verify_prerequisite(config: Dict[str, Any], engine_root: Path) -> Dict[str, Any]:
    source_path, source, load_failure = resolve_phase23_report(config)
    promoted = str(source.get("promoted_checkpoint_path") or "")
    basemodel_root = Path(str(config.get("basemodel_root", "../poker-ai-basemodel"))).expanduser()
    if not basemodel_root.is_absolute():
        basemodel_root = (engine_root / basemodel_root).resolve()
    checkpoint = resolve_checkpoint_path(promoted, basemodel_root=basemodel_root, engine_root=engine_root) if promoted else Path("")

    required_status = str(config.get("expected_phase23_status", ACCEPTED))
    require_allowed = bool(config.get("required_engine_rollout_collection_allowed", True))
    failures = []
    if load_failure:
        failures.append(load_failure)
    if source.get("phase23_larger_engine_simulation_status") != required_status:
        failures.append(
            f"source Phase 23 status is {source.get('phase23_larger_engine_simulation_status')!r}, "
            f"expected {required_status!r}"
        )
    if require_allowed and not bool(source.get("engine_rollout_collection_allowed", False)):
        failures.append("source Phase 23 did not allow engine rollout collection")
    if not promoted:
        failures.append("source Phase 23 report does not include promoted_checkpoint_path")
    if promoted and not checkpoint.is_file():
        failures.append(f"promoted checkpoint does not exist: {checkpoint}")

    return {
        "source_phase23_report_path": source_path,
        "source_phase23_status": str(source.get("phase23_larger_engine_simulation_status", "")),
        "source_phase23_failures": source.get("phase23_larger_engine_simulation_failures", []),
        "source_engine_rollout_collection_allowed": bool(source.get("engine_rollout_collection_allowed", False)),
        "source_phase22_report_path": source.get("source_phase22_report_path", ""),
        "source_phase22_status": source.get("source_phase22_status", ""),
        "source_phase23_lineup_summary": source.get("lineup_summary", {}),
        "source_phase23_bot_fallback_summary": source.get("bot_fallback_summary", {}),
        "source_promoted_checkpoint_path": source.get("promoted_checkpoint_path", ""),
        "promoted_checkpoint_path": str(checkpoint) if promoted else "",
        "promoted_checkpoint_path_from_report": promoted,
        "basemodel_root": str(basemodel_root),
        "prerequisite_failures": failures,
        "prerequisite_passed": not failures,
    }


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
    converted = []
    for value in values:
        item = float(value)
        if not math.isfinite(item):
            raise ValueError("observation contains a non-finite value")
        converted.append(item)
    return converted


def _mask_as_ints(values: Iterable[Any]) -> list[int]:
    return [1 if bool(value) else 0 for value in values]


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


class RolloutCollector:
    """Owns Phase 24 rollout JSONL artifact construction and validation."""

    def __init__(self, config: Dict[str, Any], prerequisite: Dict[str, Any]):
        rollout_config = dict(config.get("rollout_collection", {}))
        self.schema_version = int(rollout_config.get("schema_version", SCHEMA_VERSION))
        self.decision_path = Path(str(config["model_decisions_path"]))
        self.outcome_path = Path(str(config["terminal_outcomes_path"]))
        self.manifest_path = Path(str(config["manifest_path"]))
        self.source_phase23_report_path = str(prerequisite.get("source_phase23_report_path", ""))
        self.decision_path.parent.mkdir(parents=True, exist_ok=True)
        self.outcome_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.decision_count = 0
        self.terminal_outcome_count = 0
        self.validation_failures: list[str] = []
        self.privacy_violations: list[str] = []
        self.action_mapping_failures: list[str] = []
        self.reward_values: list[float] = []
        self.decision_path.write_text("", encoding="utf-8")
        self.outcome_path.write_text("", encoding="utf-8")

    def record_decision(
        self,
        *,
        game_state: Dict[str, Any],
        observation: Iterable[Any],
        legal_mask: Iterable[Any],
        selected_action_id: int,
        selected_action_logprob: Optional[float],
        value_estimate: Optional[float],
        engine_action: Tuple[str, Optional[int]],
        fallback_used: bool = False,
        fallback_reason: Optional[str] = None,
        inference_error: Optional[str] = None,
    ) -> None:
        decision_index = self.decision_count + 1
        if has_forbidden_state_keys(game_state):
            self.privacy_violations.append(f"decision {decision_index} contains forbidden runtime state")

        try:
            observation_values = _finite_floats(observation)
            if len(observation_values) != 62:
                raise ValueError(f"observation length is {len(observation_values)}, expected 62")
        except Exception as exc:
            observation_values = []
            self.validation_failures.append(f"decision {decision_index} invalid observation: {exc}")

        mask_values = _mask_as_ints(legal_mask)
        if len(mask_values) != 9:
            self.validation_failures.append(f"decision {decision_index} legal mask length is {len(mask_values)}, expected 9")
        if selected_action_id < 0 or selected_action_id >= 9:
            self.validation_failures.append(f"decision {decision_index} selected action id is out of range: {selected_action_id}")
        elif len(mask_values) == 9 and not mask_values[selected_action_id] and not fallback_used:
            self.validation_failures.append(f"decision {decision_index} selected action id is illegal under mask")
        if not is_engine_action(engine_action):
            self.action_mapping_failures.append(f"decision {decision_index} invalid engine action: {engine_action!r}")

        blinds = game_state.get("blinds") if isinstance(game_state.get("blinds"), dict) else {}
        action_name = ACTION_LABELS[selected_action_id] if 0 <= selected_action_id < len(ACTION_LABELS) else "Unknown"
        logprob = float(selected_action_logprob) if selected_action_logprob is not None else None
        probability = math.exp(logprob) if logprob is not None and math.isfinite(logprob) else None
        record = {
            "schema_version": self.schema_version,
            "phase": 24,
            "tournament_id": int(game_state.get("tournament_id", 0) or 0),
            "hand_id": int(game_state.get("hand_id", 0) or 0),
            "table_id": int(game_state.get("table_id", 0) or 0),
            "decision_index": decision_index,
            "player_name": str(game_state.get("player_id", "")),
            "player_seat": None,
            "street": _street_from_board(game_state.get("board_cards", [])),
            "small_blind": int(blinds.get("small", 0) or 0),
            "big_blind": int(blinds.get("big", 0) or 0),
            "stack_before": int(game_state.get("stack_size", 0) or 0),
            "pot_before": int(game_state.get("pot_size", 0) or 0),
            "call_amount": int(game_state.get("call_amount", 0) or 0),
            "min_raise": int(game_state.get("min_raise", 0) or 0),
            "observation": observation_values,
            "legal_action_mask": mask_values,
            "selected_action_id": int(selected_action_id),
            "selected_action_name": action_name,
            "selected_action_probability": probability,
            "selected_action_logprob": logprob,
            "value_estimate": float(value_estimate) if value_estimate is not None else None,
            "engine_action": engine_action[0],
            "engine_amount": int(engine_action[1] or 0),
            "fallback_used": bool(fallback_used),
            "fallback_reason": fallback_reason,
            "inference_error": inference_error,
            "terminal_payout_reward": None,
            "terminal_placement_reward": None,
            "combined_reward": None,
            "immediate_chip_delta": None,
        }
        with self.decision_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.decision_count += 1

    def record_terminal_outcomes(self, tournament_id: int, results: List[Dict[str, Any]]) -> None:
        field_size = len(results)
        for row in results:
            if row.get("bot_class") != "CollectingPhase24ModelBot":
                continue
            placement = int(row.get("position", 0) or 0)
            payout = float(row.get("payout_pct", 0.0) or 0.0)
            terminal_reward = 1.0 - ((placement - 1) / max(1, field_size - 1)) if placement > 0 else 0.0
            terminal_reward = (terminal_reward * 2.0) - 1.0
            record = {
                "schema_version": self.schema_version,
                "phase": 24,
                "tournament_id": int(tournament_id),
                "player_name": str(row.get("name", "")),
                "placement": placement,
                "field_size": field_size,
                "payout_pct": payout,
                "in_the_money": payout > 0.0,
                "final_stack": int(row.get("final_stack", 0) or 0),
                "bustout_hand_id": row.get("bustout_hand_id"),
                "terminal_payout_reward": payout,
                "terminal_placement_reward": terminal_reward,
                "terminal_reward": terminal_reward + payout,
                "combined_reward": terminal_reward + payout,
            }
            with self.outcome_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self.terminal_outcome_count += 1
            self.reward_values.append(float(record["combined_reward"]))

    def write_manifest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        manifest = {
            "schema_version": self.schema_version,
            "phase": 24,
            "source_phase23_report_path": self.source_phase23_report_path,
            "decision_count": self.decision_count,
            "terminal_outcome_count": self.terminal_outcome_count,
            "model_decisions_path": str(self.decision_path),
            "terminal_outcomes_path": str(self.outcome_path),
            "config_snapshot": config,
            "files": {},
        }
        for label, path in {"model_decisions": self.decision_path, "terminal_outcomes": self.outcome_path}.items():
            manifest["files"][label] = {
                "path": str(path),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
                "sha256": _file_sha256(path) if path.is_file() else "",
                "line_count": _jsonl_count(path),
            }
        _write_json(self.manifest_path, manifest)
        return manifest

    def validation_summary(self) -> Dict[str, Any]:
        return {
            "decision_count": self.decision_count,
            "terminal_outcome_count": self.terminal_outcome_count,
            "validation_failures": self.validation_failures,
            "privacy_violations": self.privacy_violations,
            "action_mapping_failures": self.action_mapping_failures,
            "valid": not self.validation_failures and not self.privacy_violations and not self.action_mapping_failures,
        }

    def reward_summary(self) -> Dict[str, Any]:
        if not self.reward_values:
            return {"count": 0, "average_combined_reward": None, "min_combined_reward": None, "max_combined_reward": None}
        return {
            "count": len(self.reward_values),
            "average_combined_reward": sum(self.reward_values) / len(self.reward_values),
            "min_combined_reward": min(self.reward_values),
            "max_combined_reward": max(self.reward_values),
        }


class CollectingPhase24ModelBot(BaselineModelEngineBot):
    """Phase 24 model bot that records model decisions before returning engine actions."""

    def __init__(self, *args, collector: RolloutCollector, **kwargs):
        super().__init__(*args, **kwargs)
        self.collector = collector

    def get_action(self, game_state: Dict[str, Any]):
        if has_forbidden_state_keys(game_state):
            return conservative_engine_action(game_state)
        if self._delegate is None or getattr(self._delegate, "model", None) is None:
            return conservative_engine_action(game_state)

        started = time.perf_counter()
        try:
            from poker_ai.actions import action_to_engine
            from poker_ai.policy import select_action

            model_state = self._delegate._state_with_equity(dict(game_state))
            decision = select_action(self._delegate.model, model_state, deterministic=self._delegate.deterministic)
            engine_action = action_to_engine(decision.action_id, model_state)
            fallback_used = False
            fallback_reason = None
            if self._delegate._timed_out(started):
                self._delegate.fallback_counts["timeouts"] += 1
                engine_action = conservative_engine_action(game_state)
                fallback_used = True
                fallback_reason = "timeout"
            elif (
                decision.action_id < 0
                or decision.action_id >= len(decision.legal_mask)
                or not decision.legal_mask[decision.action_id]
            ):
                self._delegate.fallback_counts["illegal_actions"] += 1
                engine_action = conservative_engine_action(game_state)
                fallback_used = True
                fallback_reason = "illegal_action"
            self.collector.record_decision(
                game_state=model_state,
                observation=decision.observation,
                legal_mask=decision.legal_mask,
                selected_action_id=decision.action_id,
                selected_action_logprob=decision.log_prob,
                value_estimate=decision.value,
                engine_action=engine_action,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
            )
            if not is_engine_action(engine_action):
                return conservative_engine_action(game_state)
            return engine_action
        except Exception as exc:
            self._delegate.fallback_counts["inference_errors"] += 1
            return conservative_engine_action(game_state)


def build_phase24_lineup(
    config: Dict[str, Any],
    checkpoint_path: str,
    basemodel_root: str,
    collector: RolloutCollector,
) -> List[Any]:
    patched = dict(config)
    patched_lineup = dict(config.get("lineup", {}))
    patched_lineup["model"] = 0
    patched["lineup"] = patched_lineup
    bots: List[Any] = []
    for index in range(int(config.get("lineup", {}).get("model", 1))):
        bots.append(
            CollectingPhase24ModelBot(
                checkpoint_path,
                basemodel_root=basemodel_root,
                name=f"Phase24ModelBot_{index + 1}",
                deterministic=bool(config.get("deterministic", True)),
                decision_timeout_ms=int(config.get("bot_decision_timeout_ms", 500)),
                equity_source=str(config.get("equity_source", "treys")),
                equity_fallback_source=config.get("equity_fallback_source", "constant"),
                equity_iterations=config.get("equity_iterations"),
                require_checkpoint=True,
                collector=collector,
            )
        )
    bots.extend(build_lineup(patched, checkpoint_path, basemodel_root))
    return bots


def should_write_full_event_log(config: Dict[str, Any], tournament_id: int) -> bool:
    event_logging = dict(config.get("event_logging", {}))
    if bool(event_logging.get("write_full_event_logs", False)):
        return True
    return tournament_id <= int(event_logging.get("write_full_event_logs_for_first_n", 0))


def _enrich_final_stacks(results: List[Dict[str, Any]], tournament: Tournament) -> List[Dict[str, Any]]:
    stacks = {player.name: int(player.stack) for player in getattr(tournament, "placements", [])}
    enriched = []
    for row in results:
        copy = dict(row)
        copy["final_stack"] = stacks.get(str(row.get("name", "")), 0)
        copy["bustout_hand_id"] = None
        enriched.append(copy)
    return enriched


def run_tournaments(config: Dict[str, Any], bots: List[Any], collector: RolloutCollector) -> Dict[str, Any]:
    tournament_count = int(config.get("tournament_count", 1))
    seed = int(config.get("random_seed", 0))
    result_rows = []
    event_summaries = []
    event_log_paths = []
    tournament_failures = []
    model_bots = [bot for bot in bots if isinstance(bot, CollectingPhase24ModelBot)]
    fallback_before = [_fallback_counts(bot) for bot in model_bots]
    equity_before = [_equity_counts(bot) for bot in model_bots]

    with temporary_engine_config(_engine_overrides(config)):
        for index in range(tournament_count):
            tournament_id = index + 1
            _seed_everything(seed + index)
            try:
                tournament = Tournament(bots, tournament_id=tournament_id)
                results, events = tournament.play()
                results = _enrich_final_stacks(results, tournament)
                stopped = any(event.get("type") == "tournament_stopped_max_hands" for event in events)
                event_summaries.append(summarize_events(tournament_id, events, results))
                if should_write_full_event_log(config, tournament_id):
                    event_log_path = Path(str(config["event_log_dir"])) / f"tournament_{tournament_id}_events.json"
                    _write_json(event_log_path, events)
                    event_log_paths.append(str(event_log_path))
                collector.record_terminal_outcomes(tournament_id, results)
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
    model_fallback_deltas = [_subtract_counts(after, before) for before, after in zip(fallback_before, fallback_after)]
    model_equity_deltas = [_subtract_counts(after, before) for before, after in zip(equity_before, equity_after)]
    return {
        "tournament_results": result_rows,
        "event_summaries": event_summaries,
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


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def build_report(
    config: Dict[str, Any],
    prerequisite: Dict[str, Any],
    simulation: Dict[str, Any],
    collector: RolloutCollector,
    manifest: Dict[str, Any],
    started_at: str,
    runtime_seconds: float,
) -> Dict[str, Any]:
    acceptance = dict(config.get("acceptance", {}))
    summary = summarize_results(list(simulation.get("tournament_results", [])))
    completed = int(summary["completed_tournament_count"])
    tournament_count = int(config.get("tournament_count", 0))
    stopped_rate = _rate(int(summary["stopped_max_hands_count"]), max(1, tournament_count))
    fallback_totals = dict(simulation.get("bot_fallback_summary", {}).get("totals", {}))
    validation = collector.validation_summary()
    decision_path = Path(str(config.get("model_decisions_path", "")))
    outcome_path = Path(str(config.get("terminal_outcomes_path", "")))
    manifest_path = Path(str(config.get("manifest_path", "")))
    gates = {
        "source_phase23_accepted_gate_passed": prerequisite.get("source_phase23_status")
        == config.get("expected_phase23_status", ACCEPTED),
        "source_engine_rollout_collection_allowed_gate_passed": bool(
            prerequisite.get("source_engine_rollout_collection_allowed", False)
        ),
        "promoted_checkpoint_exists_gate_passed": bool(prerequisite.get("promoted_checkpoint_path"))
        and Path(str(prerequisite.get("promoted_checkpoint_path"))).is_file(),
        "lineup_gate_passed": simulation.get("lineup_summary", {}).get("total_bots", 0)
        > int(config.get("max_players_per_table", 9)),
        "tournament_runtime_gate_passed": not simulation.get("tournament_failures"),
        "completed_tournament_gate_passed": completed >= int(acceptance.get("min_completed_tournaments", 1)),
        "stopped_max_hands_rate_gate_passed": stopped_rate <= float(acceptance.get("max_stopped_max_hands_rate", 1.0)),
        "min_model_decision_gate_passed": collector.decision_count >= int(acceptance.get("min_model_decisions", 1)),
        "model_inference_error_gate_passed": int(fallback_totals.get("inference_errors", 0))
        <= int(acceptance.get("max_model_inference_errors", 0)),
        "model_illegal_action_gate_passed": int(fallback_totals.get("illegal_actions", 0))
        <= int(acceptance.get("max_model_illegal_actions", 0)),
        "model_timeout_fallback_gate_passed": int(fallback_totals.get("timeouts", 0))
        <= int(acceptance.get("max_model_timeout_fallbacks", 0)),
        "rollout_artifact_gate_passed": (not bool(acceptance.get("require_rollout_artifact", True)))
        or (decision_path.is_file() and decision_path.stat().st_size > 0),
        "terminal_outcome_artifact_gate_passed": (not bool(acceptance.get("require_terminal_outcomes", True)))
        or (outcome_path.is_file() and collector.terminal_outcome_count > 0),
        "manifest_artifact_gate_passed": (not bool(acceptance.get("require_manifest", True))) or manifest_path.is_file(),
        "privacy_gate_passed": (not bool(acceptance.get("require_no_privacy_violations", True)))
        or not validation["privacy_violations"],
        "rollout_validation_gate_passed": validation["valid"],
    }
    failures = list(prerequisite.get("prerequisite_failures", []))
    failures.extend(str(failure) for failure in simulation.get("tournament_failures", []))
    failures.extend(validation["validation_failures"])
    failures.extend(validation["privacy_violations"])
    failures.extend(validation["action_mapping_failures"])
    for gate, passed in gates.items():
        if not passed:
            failures.append(f"{gate} failed")

    status = ACCEPTED if not failures else REJECTED
    artifact_paths = {
        "artifact_root": str(config.get("artifact_root", "")),
        "engine_rollout_collection_report_path": str(config.get("engine_rollout_collection_report_path", "")),
        "tournament_results_path": str(config.get("tournament_results_path", "")),
        "event_summaries_path": str(config.get("event_summaries_path", "")),
        "model_decisions_path": str(decision_path),
        "terminal_outcomes_path": str(outcome_path),
        "manifest_path": str(manifest_path),
        "event_log_paths": list(simulation.get("event_log_paths", [])),
    }
    return {
        "phase24_engine_rollout_collection_id": config.get(
            "phase24_engine_rollout_collection_id", "phase24_engine_rollout_collection"
        ),
        "phase24_engine_rollout_collection_status": status,
        "phase24_engine_rollout_collection_failures": failures,
        "phase24_engine_rollout_collection_report_path": str(config.get("engine_rollout_collection_report_path", "")),
        "phase24_engine_rollout_collection_started_at": started_at,
        "phase24_engine_rollout_collection_finished_at": _utc_now(),
        "source_phase23_report_path": prerequisite.get("source_phase23_report_path", ""),
        "source_phase23_status": prerequisite.get("source_phase23_status", ""),
        "source_phase23_failures": prerequisite.get("source_phase23_failures", []),
        "source_engine_rollout_collection_allowed": prerequisite.get("source_engine_rollout_collection_allowed", False),
        "source_phase22_report_path": prerequisite.get("source_phase22_report_path", ""),
        "source_phase22_status": prerequisite.get("source_phase22_status", ""),
        "promoted_checkpoint_path": prerequisite.get("promoted_checkpoint_path", ""),
        "lineup_summary": simulation.get("lineup_summary", {}),
        "tournament_count": tournament_count,
        "completed_tournament_count": completed,
        "stopped_max_hands_count": summary["stopped_max_hands_count"],
        "stopped_max_hands_rate": stopped_rate,
        "model_decision_count": collector.decision_count,
        "model_terminal_outcome_count": collector.terminal_outcome_count,
        "rollout_schema_version": collector.schema_version,
        "rollout_artifact_paths": artifact_paths,
        "rollout_validation_summary": validation,
        "observation_validation_summary": {
            "expected_length": 62,
            "failures": [item for item in validation["validation_failures"] if "observation" in item],
        },
        "legal_mask_validation_summary": {
            "expected_length": 9,
            "failures": [item for item in validation["validation_failures"] if "mask" in item],
        },
        "action_mapping_validation_summary": {"failures": validation["action_mapping_failures"]},
        "privacy_validation_summary": {"violations": validation["privacy_violations"]},
        "reward_summary": collector.reward_summary(),
        "runtime_summary": {
            "runtime_seconds": runtime_seconds,
            "event_summary_count": len(simulation.get("event_summaries", [])),
            "sampled_event_log_count": len(simulation.get("event_log_paths", [])),
            "tournament_failures": simulation.get("tournament_failures", []),
        },
        "bot_fallback_summary": simulation.get("bot_fallback_summary", {}),
        "model_equity_summary": simulation.get("model_equity_summary", {}),
        "artifact_integrity_summary": manifest,
        "phase24_gate_results": gates,
        "phase25_engine_training_allowed": status == ACCEPTED,
        "next_phase_recommendation": "launch_phase25_engine_rollout_training"
        if status == ACCEPTED
        else "resolve_phase24_rollout_collection_failures",
        "config_path": config.get("config_path", ""),
    }


def run_phase24_engine_rollout_collection(config: Dict[str, Any], config_path: str | Path | None = None) -> Dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    config = _resolve_paths(config)
    if config_path is not None:
        config["config_path"] = str(config_path)
    engine_root = Path(__file__).resolve().parents[1]
    artifact_root = Path(str(config["artifact_root"]))
    artifact_root.mkdir(parents=True, exist_ok=True)
    prerequisite = verify_prerequisite(config, engine_root)
    collector = RolloutCollector(config, prerequisite)
    simulation: Dict[str, Any] = {
        "lineup_summary": {},
        "tournament_results": [],
        "event_summaries": [],
        "event_log_paths": [],
        "tournament_failures": [],
        "bot_fallback_summary": {"model_bot_count": 0, "per_model_bot": [], "totals": {}},
        "model_equity_summary": {"per_model_bot": [], "totals": {}},
    }

    if prerequisite.get("prerequisite_passed"):
        try:
            bots = build_phase24_lineup(
                config,
                str(prerequisite["promoted_checkpoint_path"]),
                str(prerequisite["basemodel_root"]),
                collector,
            )
            simulation["lineup_summary"] = lineup_summary(bots)
            simulation.update(run_tournaments(config, bots, collector))
        except Exception as exc:
            simulation["tournament_failures"] = [f"simulation setup failed: {exc}"]

    _write_json(str(config["tournament_results_path"]), simulation.get("tournament_results", []))
    _write_json(str(config["event_summaries_path"]), simulation.get("event_summaries", []))
    manifest = collector.write_manifest(config)
    report = build_report(config, prerequisite, simulation, collector, manifest, started_at, time.perf_counter() - started)
    _write_json(str(config["engine_rollout_collection_report_path"]), report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase24_engine_rollout_collection.json")
    args = parser.parse_args(argv)

    report = run_phase24_engine_rollout_collection(load_config(args.config), config_path=args.config)
    print(
        json.dumps(
            {
                "phase24_engine_rollout_collection_status": report["phase24_engine_rollout_collection_status"],
                "promoted_checkpoint_path": report["promoted_checkpoint_path"],
                "completed_tournament_count": report["completed_tournament_count"],
                "model_decision_count": report["model_decision_count"],
                "model_terminal_outcome_count": report["model_terminal_outcome_count"],
                "phase25_engine_training_allowed": report["phase25_engine_training_allowed"],
                "next_phase_recommendation": report["next_phase_recommendation"],
                "phase24_engine_rollout_collection_report_path": report[
                    "phase24_engine_rollout_collection_report_path"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["phase24_engine_rollout_collection_status"] != ACCEPTED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
