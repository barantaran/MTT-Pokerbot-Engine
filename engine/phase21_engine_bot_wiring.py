"""Phase 21 engine bot wiring smoke test."""

from __future__ import annotations

import argparse
import glob
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from engine.baseline_model_bot import (
    BaselineModelEngineBot,
    add_basemodel_to_path,
    is_engine_action,
    resolve_checkpoint_path,
    validate_visible_state,
)


ACCEPTED = "accepted"
REJECTED = "rejected"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


def _read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_config(path: str | Path) -> Dict[str, Any]:
    config = _read_json(path)
    config["config_path"] = str(path)
    return config


def _load_report(path: str | Path) -> Dict[str, Any] | None:
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def resolve_phase20_report(config: Dict[str, Any]) -> tuple[str, Dict[str, Any], str]:
    configured_path = str(config.get("source_phase20_report_path", "") or "")
    required_status = str(config.get("required_phase20_status", ACCEPTED))
    if configured_path:
        report = _load_report(configured_path)
        if report is None:
            return configured_path, {}, "source Phase 20 report is missing or malformed"
        return configured_path, report, ""

    pattern = str(config.get("source_phase20_report_glob", "") or "")
    if not pattern:
        return "", {}, "source Phase 20 report path or glob is required"

    matches = sorted(glob.glob(pattern), key=lambda item: Path(item).stat().st_mtime, reverse=True)
    malformed_count = 0
    for path in matches:
        report = _load_report(path)
        if report is None:
            malformed_count += 1
            continue
        if report.get("phase20_promotion_status") == required_status and bool(
            report.get("mtt_engine_bot_simulation_allowed", False)
        ):
            return path, report, ""
    if matches:
        suffix = f"; {malformed_count} malformed candidates skipped" if malformed_count else ""
        return matches[0], _load_report(matches[0]) or {}, f"no accepted Phase 20 report matched {pattern}{suffix}"
    return "", {}, f"no Phase 20 reports matched {pattern}"


def _resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(config)
    run_id = str(config.get("phase21_engine_wiring_id", "phase21_engine_bot_wiring_smoke_test"))
    variables = {"phase21_engine_wiring_id": run_id, "run_stamp": _run_stamp()}
    if config.get("artifact_root_template"):
        config["artifact_root"] = str(config["artifact_root_template"]).format(**variables)
    artifact_root = Path(str(config.get("artifact_root", Path("runs") / run_id)))
    variables["artifact_root"] = str(artifact_root)
    if config.get("engine_wiring_report_path_template"):
        config["engine_wiring_report_path"] = str(config["engine_wiring_report_path_template"]).format(**variables)
    else:
        config.setdefault("engine_wiring_report_path", str(artifact_root / "engine_wiring_report.json"))
    config["artifact_root"] = str(artifact_root)
    return config


def _verify_prerequisite(config: Dict[str, Any], engine_root: Path) -> Dict[str, Any]:
    source_path, source, load_failure = resolve_phase20_report(config)
    promoted = str(source.get("promoted_checkpoint_path") or source.get("candidate_checkpoint_path") or "")
    basemodel_root = Path(str(config.get("basemodel_root", "../poker-ai-basemodel"))).expanduser()
    if not basemodel_root.is_absolute():
        basemodel_root = (engine_root / basemodel_root).resolve()
    checkpoint = resolve_checkpoint_path(promoted, basemodel_root=basemodel_root, engine_root=engine_root) if promoted else Path("")
    required_status = str(config.get("required_phase20_status", ACCEPTED))
    required_allowed = bool(config.get("required_mtt_engine_bot_simulation_allowed", True))
    failures = []
    if load_failure:
        failures.append(load_failure)
    if source.get("phase20_promotion_status") != required_status:
        failures.append(f"source Phase 20 status is {source.get('phase20_promotion_status')!r}, expected {required_status!r}")
    if required_allowed and not bool(source.get("mtt_engine_bot_simulation_allowed", False)):
        failures.append("source Phase 20 did not allow MTT engine bot simulation")
    if not promoted:
        failures.append("source Phase 20 report does not include promoted_checkpoint_path")
    if promoted and not checkpoint.is_file():
        failures.append(f"promoted checkpoint does not exist: {checkpoint}")

    return {
        "source_phase20_report_path": source_path,
        "source_phase20_status": str(source.get("phase20_promotion_status", "")),
        "source_phase20_failures": source.get("phase20_promotion_failures", []),
        "source_mtt_engine_bot_simulation_allowed": bool(source.get("mtt_engine_bot_simulation_allowed", False)),
        "source_candidate_checkpoint_path": source.get("candidate_checkpoint_path"),
        "source_promoted_checkpoint_path": source.get("promoted_checkpoint_path"),
        "source_score_delta": source.get("score_delta"),
        "source_roi_delta": source.get("roi_delta"),
        "promoted_checkpoint_path": str(checkpoint) if promoted else "",
        "promoted_checkpoint_path_from_report": promoted,
        "basemodel_root": str(basemodel_root),
        "prerequisite_failures": failures,
        "prerequisite_passed": not failures,
    }


def _probe_observation_and_mask(config: Dict[str, Any], states: List[Dict[str, Any]], basemodel_root: str) -> Dict[str, Any]:
    add_basemodel_to_path(basemodel_root)
    from poker_ai.actions import legal_action_mask
    from poker_ai.observations import OBSERVATION_SIZE, encode_observation

    failures = []
    rows = []
    for item in states:
        name = str(item.get("name", "state"))
        state = dict(item.get("game_state", {}))
        try:
            observation = encode_observation(state)
            mask = legal_action_mask(state)
            finite = all(isinstance(value, float) and value == value and value not in (float("inf"), float("-inf")) for value in observation)
            row = {
                "name": name,
                "observation_size": len(observation),
                "observation_finite": finite,
                "legal_mask_size": len(mask),
                "legal_action_count": sum(1 for value in mask if value),
            }
            if len(observation) != OBSERVATION_SIZE:
                failures.append(f"{name} observation size {len(observation)} != {OBSERVATION_SIZE}")
            if not finite:
                failures.append(f"{name} observation contains non-finite values")
            if len(mask) != 9 or not any(mask):
                failures.append(f"{name} legal action mask is invalid")
            rows.append(row)
        except Exception as exc:
            failures.append(f"{name} observation/mask probe failed: {exc}")
    return {"passed": not failures, "failures": failures, "rows": rows}


def _probe_privacy(states: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures = []
    rows = []
    for item in states:
        name = str(item.get("name", "state"))
        ok, state_failures = validate_visible_state(dict(item.get("game_state", {})))
        rows.append({"name": name, "passed": ok, "failures": state_failures})
        failures.extend(f"{name}: {failure}" for failure in state_failures)
    injected = dict(states[0].get("game_state", {})) if states else {}
    injected["opponent_hole_cards"] = [[1, 2]]
    injected_ok, _ = validate_visible_state(injected)
    if injected_ok:
        failures.append("privacy probe did not reject injected opponent_hole_cards")
    return {"passed": not failures, "failures": failures, "rows": rows, "injected_hidden_cards_rejected": not injected_ok}


def _probe_actions(bot: BaselineModelEngineBot, states: List[Dict[str, Any]]) -> Dict[str, Any]:
    failures = []
    outputs = []
    started = time.perf_counter()
    for item in states:
        name = str(item.get("name", "state"))
        state = dict(item.get("game_state", {}))
        before = time.perf_counter()
        try:
            action = bot.get_action(state)
            elapsed_ms = (time.perf_counter() - before) * 1000.0
            valid = is_engine_action(action)
            if not valid:
                failures.append(f"{name} returned invalid action {action!r}")
            if state.get("call_amount", 0) == 0 and action[0] == "call" and action[1] != 0:
                failures.append(f"{name} check path returned non-zero call amount")
            outputs.append({"name": name, "action": list(action), "elapsed_ms": elapsed_ms, "valid": valid})
        except Exception as exc:
            failures.append(f"{name} action probe raised: {exc}")
    return {
        "passed": not failures,
        "failures": failures,
        "outputs": outputs,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def _probe_fallback(config: Dict[str, Any], basemodel_root: str) -> Dict[str, Any]:
    fallback_bot = BaselineModelEngineBot(
        None,
        basemodel_root=basemodel_root,
        name="Phase21FallbackProbeBot",
        require_checkpoint=False,
        decision_timeout_ms=int(config.get("decision_timeout_ms", 500)),
        equity_source=str(config.get("equity_source", "treys")),
        equity_fallback_source=config.get("equity_fallback_source", "constant"),
        equity_iterations=config.get("equity_iterations"),
    )
    states = [
        {"call_amount": 0, "stack_size": 500},
        {"call_amount": 50, "stack_size": 500},
        {"call_amount": 50, "stack_size": 0},
        {"call_amount": 0, "stack_size": 500, "opponent_hole_cards": [[1, 2]]},
    ]
    outputs = [fallback_bot.get_action(state) for state in states]
    failures = [f"invalid fallback action {action!r}" for action in outputs if not is_engine_action(action)]
    return {
        "passed": not failures,
        "failures": failures,
        "outputs": [list(action) for action in outputs],
        "fallback_counts": fallback_bot.fallback_counts,
    }


def _probe_timeout(config: Dict[str, Any], basemodel_root: str) -> Dict[str, Any]:
    add_basemodel_to_path(basemodel_root)

    class SlowModel:
        def eval(self):
            return self

        def __call__(self, observations):
            import torch

            time.sleep(0.01)
            return torch.zeros((1, 9), dtype=torch.float32), torch.zeros((1,), dtype=torch.float32)

    bot = BaselineModelEngineBot(
        None,
        basemodel_root=basemodel_root,
        name="Phase21TimeoutProbeBot",
        model=SlowModel(),
        require_checkpoint=False,
        decision_timeout_ms=1,
        equity_source="constant",
        equity_fallback_source=None,
    )
    action = bot.get_action({"call_amount": 0, "stack_size": 500, "pot_size": 100, "min_raise": 20, "active_players": 2})
    passed = is_engine_action(action) and action == ("call", 0) and bot.fallback_counts.get("timeouts", 0) >= 1
    return {"passed": passed, "action": list(action), "fallback_counts": bot.fallback_counts}


def build_report(config: Dict[str, Any], prerequisite: Dict[str, Any], probes: Dict[str, Any], started_at: str) -> Dict[str, Any]:
    finished_at = _utc_now()
    gates = {
        "source_accepted_gate_passed": prerequisite.get("source_phase20_status") == config.get("required_phase20_status", ACCEPTED),
        "source_engine_sim_allowed_gate_passed": bool(prerequisite.get("source_mtt_engine_bot_simulation_allowed", False)),
        "promoted_checkpoint_exists_gate_passed": bool(prerequisite.get("promoted_checkpoint_path"))
        and Path(str(prerequisite.get("promoted_checkpoint_path"))).is_file(),
        "basemodel_import_gate_passed": probes.get("basemodel_import", {}).get("passed", False),
        "engine_adapter_import_gate_passed": True,
        "checkpoint_load_gate_passed": probes.get("checkpoint_load", {}).get("passed", False),
        "observation_size_gate_passed": probes.get("observation", {}).get("passed", False),
        "observation_finite_gate_passed": probes.get("observation", {}).get("passed", False),
        "legal_action_mask_gate_passed": probes.get("observation", {}).get("passed", False),
        "engine_action_shape_gate_passed": probes.get("actions", {}).get("passed", False),
        "engine_action_legal_gate_passed": probes.get("actions", {}).get("passed", False),
        "check_call_contract_gate_passed": probes.get("actions", {}).get("passed", False),
        "fallback_action_gate_passed": probes.get("fallback", {}).get("passed", False),
        "timeout_fallback_gate_passed": probes.get("timeout", {}).get("passed", False),
        "privacy_gate_passed": probes.get("privacy", {}).get("passed", False),
        "runtime_gate_passed": probes.get("runtime", {}).get("passed", False),
        "artifact_report_write_gate_passed": True,
    }
    failures = list(prerequisite.get("prerequisite_failures", []))
    for name, probe in probes.items():
        if isinstance(probe, dict):
            failures.extend(str(failure) for failure in probe.get("failures", []))
    for gate, passed in gates.items():
        if not passed:
            failures.append(f"{gate} failed")
    status = ACCEPTED if not failures else REJECTED
    return {
        "phase21_engine_wiring_id": config.get("phase21_engine_wiring_id", "phase21_engine_bot_wiring_smoke_test"),
        "phase21_engine_wiring_status": status,
        "phase21_engine_wiring_failures": failures,
        "phase21_engine_wiring_report_path": str(config.get("engine_wiring_report_path", "")),
        "phase21_engine_wiring_started_at": started_at,
        "phase21_engine_wiring_finished_at": finished_at,
        "source_phase20_report_path": prerequisite.get("source_phase20_report_path", ""),
        "source_phase20_status": prerequisite.get("source_phase20_status", ""),
        "source_phase20_failures": prerequisite.get("source_phase20_failures", []),
        "source_mtt_engine_bot_simulation_allowed": prerequisite.get("source_mtt_engine_bot_simulation_allowed", False),
        "source_candidate_checkpoint_path": prerequisite.get("source_candidate_checkpoint_path"),
        "source_promoted_checkpoint_path": prerequisite.get("source_promoted_checkpoint_path"),
        "source_score_delta": prerequisite.get("source_score_delta"),
        "source_roi_delta": prerequisite.get("source_roi_delta"),
        "promoted_checkpoint_path": prerequisite.get("promoted_checkpoint_path", ""),
        "engine_adapter_path": "engine/baseline_model_bot.py",
        "basemodel_import_path": prerequisite.get("basemodel_root", ""),
        "checkpoint_load_status": "loaded" if probes.get("checkpoint_load", {}).get("passed", False) else "failed",
        "checkpoint_load_error": probes.get("checkpoint_load", {}).get("error", ""),
        "smoke_state_count": len(config.get("smoke_states", [])),
        "smoke_action_outputs": probes.get("actions", {}).get("outputs", []),
        "fallback_action_outputs": probes.get("fallback", {}).get("outputs", []),
        "timeout_probe_result": probes.get("timeout", {}),
        "privacy_probe_result": probes.get("privacy", {}),
        "observation_probe_result": probes.get("observation", {}),
        "action_contract_probe_result": probes.get("actions", {}),
        "runtime_summary": probes.get("runtime", {}),
        "artifact_integrity_summary": {
            "source_report_exists": bool(prerequisite.get("source_phase20_report_path"))
            and Path(str(prerequisite.get("source_phase20_report_path"))).is_file(),
            "promoted_checkpoint_exists": bool(prerequisite.get("promoted_checkpoint_path"))
            and Path(str(prerequisite.get("promoted_checkpoint_path"))).is_file(),
            "report_path": str(config.get("engine_wiring_report_path", "")),
        },
        "phase21_gate_results": gates,
        "small_mtt_engine_simulation_allowed": status == ACCEPTED,
        "next_phase_recommendation": "launch_phase22_small_mtt_engine_simulation"
        if status == ACCEPTED
        else "inspect_phase21_engine_bot_wiring_smoke_test",
        "config_path": config.get("config_path", ""),
    }


def run_phase21_engine_bot_wiring(config: Dict[str, Any], config_path: str | Path | None = None) -> Dict[str, Any]:
    started_at = _utc_now()
    started = time.perf_counter()
    config = _resolve_paths(config)
    if config_path is not None:
        config["config_path"] = str(config_path)
    engine_root = Path(__file__).resolve().parents[1]
    prerequisite = _verify_prerequisite(config, engine_root)
    probes: Dict[str, Any] = {
        "basemodel_import": {"passed": False, "failures": []},
        "checkpoint_load": {"passed": False, "failures": [], "error": ""},
    }

    if prerequisite.get("basemodel_root"):
        try:
            add_basemodel_to_path(str(prerequisite["basemodel_root"]))
            import poker_ai.bot  # noqa: F401

            probes["basemodel_import"] = {"passed": True, "failures": []}
        except Exception as exc:
            probes["basemodel_import"] = {"passed": False, "failures": [f"basemodel import failed: {exc}"]}

    bot = None
    if prerequisite.get("prerequisite_passed") and probes["basemodel_import"]["passed"]:
        try:
            bot = BaselineModelEngineBot(
                str(prerequisite["promoted_checkpoint_path"]),
                basemodel_root=str(prerequisite["basemodel_root"]),
                name=str(config.get("adapter_name", "Phase21PromotedMTTBot")),
                deterministic=bool(config.get("deterministic", True)),
                decision_timeout_ms=int(config.get("decision_timeout_ms", 500)),
                equity_source=str(config.get("equity_source", "treys")),
                equity_fallback_source=config.get("equity_fallback_source", "constant"),
                equity_iterations=config.get("equity_iterations"),
                require_checkpoint=True,
            )
            probes["checkpoint_load"] = {"passed": True, "failures": [], "error": ""}
        except Exception as exc:
            probes["checkpoint_load"] = {
                "passed": False,
                "failures": [f"checkpoint load failed: {exc}"],
                "error": str(exc),
            }

    states = list(config.get("smoke_states", []))
    probes["privacy"] = _probe_privacy(states)
    if probes["basemodel_import"]["passed"]:
        probes["observation"] = _probe_observation_and_mask(config, states, str(prerequisite.get("basemodel_root", "")))
        probes["fallback"] = _probe_fallback(config, str(prerequisite.get("basemodel_root", "")))
        probes["timeout"] = _probe_timeout(config, str(prerequisite.get("basemodel_root", "")))
    else:
        probes["observation"] = {"passed": False, "failures": ["basemodel import failed"], "rows": []}
        probes["fallback"] = {"passed": False, "failures": ["basemodel import failed"], "outputs": []}
        probes["timeout"] = {"passed": False, "failures": ["basemodel import failed"]}
    probes["actions"] = _probe_actions(bot, states) if bot is not None else {"passed": False, "failures": ["checkpoint-backed bot was not loaded"], "outputs": []}

    elapsed = time.perf_counter() - started
    probes["runtime"] = {
        "passed": elapsed <= float(config.get("max_runtime_seconds", 30.0)),
        "runtime_seconds": elapsed,
        "max_runtime_seconds": float(config.get("max_runtime_seconds", 30.0)),
        "bot_fallback_counts": bot.fallback_counts if bot is not None else {},
        "bot_equity_counts": bot.equity_counts if bot is not None else {},
    }
    if not probes["runtime"]["passed"]:
        probes["runtime"]["failures"] = [f"runtime {elapsed:.3f}s exceeded budget"]

    report = build_report(config, prerequisite, probes, started_at)
    _write_json(str(config["engine_wiring_report_path"]), report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase21_engine_bot_wiring_smoke_test.json")
    args = parser.parse_args(argv)

    report = run_phase21_engine_bot_wiring(load_config(args.config), config_path=args.config)
    print(
        json.dumps(
            {
                "phase21_engine_wiring_status": report["phase21_engine_wiring_status"],
                "promoted_checkpoint_path": report["promoted_checkpoint_path"],
                "small_mtt_engine_simulation_allowed": report["small_mtt_engine_simulation_allowed"],
                "next_phase_recommendation": report["next_phase_recommendation"],
                "phase21_engine_wiring_report_path": report["phase21_engine_wiring_report_path"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if report["phase21_engine_wiring_status"] != ACCEPTED:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
