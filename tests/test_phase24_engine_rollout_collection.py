from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.phase24_engine_rollout_collection import (
    RolloutCollector,
    build_phase24_lineup,
    build_report,
    resolve_phase23_report,
    run_phase24_engine_rollout_collection,
)


class DummyModel:
    def eval(self):
        return self


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def collector_config(root: Path) -> dict:
    return {
        "artifact_root": str(root),
        "model_decisions_path": str(root / "rollouts" / "model_decisions.jsonl"),
        "terminal_outcomes_path": str(root / "rollouts" / "terminal_outcomes.jsonl"),
        "manifest_path": str(root / "rollouts" / "manifest.json"),
        "rollout_collection": {"schema_version": 1},
    }


def valid_state() -> dict:
    return {
        "hole_cards": [],
        "board_cards": [],
        "pot_size": 120,
        "stack_size": 1500,
        "call_amount": 20,
        "min_raise": 40,
        "blinds": {"small": 10, "big": 20},
        "active_players": 9,
        "player_id": "Phase24ModelBot_1",
        "table_id": 1,
        "hand_id": 7,
        "tournament_id": 3,
        "position": "BTN",
    }


class Phase24EngineRolloutCollectionTests(unittest.TestCase):
    def test_resolve_phase23_report_selects_latest_accepted_allowed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rejected = write_json(
                root / "old" / "larger_mtt_engine_simulation_report.json",
                {"phase23_larger_engine_simulation_status": "rejected", "engine_rollout_collection_allowed": False},
            )
            accepted = write_json(
                root / "latest" / "larger_mtt_engine_simulation_report.json",
                {"phase23_larger_engine_simulation_status": "accepted", "engine_rollout_collection_allowed": True},
            )
            rejected.touch()
            accepted.touch()

            path, report, failure = resolve_phase23_report(
                {"phase23_report_glob": str(root / "*" / "larger_mtt_engine_simulation_report.json")}
            )

        self.assertEqual(path, str(accepted))
        self.assertEqual(report["phase23_larger_engine_simulation_status"], "accepted")
        self.assertEqual(failure, "")

    def test_resolve_phase23_report_rejects_disallowed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "larger_mtt_engine_simulation_report.json",
                {"phase23_larger_engine_simulation_status": "accepted", "engine_rollout_collection_allowed": False},
            )

            _path, _report, failure = resolve_phase23_report(
                {"phase23_report_glob": str(root / "larger_mtt_engine_simulation_report.json")}
            )

        self.assertIn("no accepted Phase 23 report matched", failure)

    def test_rollout_collector_writes_valid_decision_and_terminal_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = collector_config(root)
            collector = RolloutCollector(config, {"source_phase23_report_path": "phase23.json"})
            collector.record_decision(
                game_state=valid_state(),
                observation=[0.0] * 62,
                legal_mask=[True] * 9,
                selected_action_id=5,
                selected_action_logprob=-1.5,
                value_estimate=0.25,
                engine_action=("raise", 100),
            )
            collector.record_terminal_outcomes(
                3,
                [
                    {
                        "position": 2,
                        "name": "Phase24ModelBot_1",
                        "bot_class": "CollectingPhase24ModelBot",
                        "payout_pct": 0.25,
                        "final_stack": 0,
                    }
                ],
            )
            manifest = collector.write_manifest(config)

            decision = json.loads((root / "rollouts" / "model_decisions.jsonl").read_text(encoding="utf-8").strip())
            outcome = json.loads((root / "rollouts" / "terminal_outcomes.jsonl").read_text(encoding="utf-8").strip())

        self.assertEqual(decision["schema_version"], 1)
        self.assertEqual(len(decision["observation"]), 62)
        self.assertEqual(len(decision["legal_action_mask"]), 9)
        self.assertEqual(decision["selected_action_name"], "Bet_1")
        self.assertEqual(outcome["placement"], 2)
        self.assertEqual(manifest["decision_count"], 1)
        self.assertTrue(manifest["files"]["model_decisions"]["exists"])

    def test_rollout_collector_rejects_bad_observation_mask_and_privacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = collector_config(root)
            collector = RolloutCollector(config, {})
            state = dict(valid_state())
            state["opponent_hole_cards"] = ["As", "Ad"]
            collector.record_decision(
                game_state=state,
                observation=[0.0] * 61,
                legal_mask=[False] * 9,
                selected_action_id=5,
                selected_action_logprob=None,
                value_estimate=None,
                engine_action=("raise", 100),
            )
            summary = collector.validation_summary()

        self.assertFalse(summary["valid"])
        self.assertTrue(summary["privacy_violations"])
        self.assertIn("invalid observation", " ".join(summary["validation_failures"]))
        self.assertIn("illegal under mask", " ".join(summary["validation_failures"]))

    @patch("engine.baseline_model_bot.load_checkpoint_model", return_value=DummyModel())
    def test_build_phase24_lineup_uses_collecting_model_bots_and_mixed_opponents(self, _load_model):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "latest.pt"
            checkpoint.write_bytes(b"checkpoint")
            collector = RolloutCollector(collector_config(root / "artifacts"), {})
            bots = build_phase24_lineup(
                {
                    "lineup": {
                        "model": 2,
                        "random": 1,
                        "equity_aggressive": 1,
                        "aggressive_no_equity": 1,
                        "call": 1,
                    },
                    "bot_decision_timeout_ms": 500,
                    "equity_source": "constant",
                },
                str(checkpoint),
                str(Path(__file__).resolve().parents[2] / "poker-ai-basemodel"),
                collector,
            )

        self.assertEqual(len(bots), 6)
        self.assertEqual([bot.name for bot in bots[:2]], ["Phase24ModelBot_1", "Phase24ModelBot_2"])
        self.assertEqual(bots[0].__class__.__name__, "CollectingPhase24ModelBot")
        self.assertEqual(bots[2].__class__.__name__, "RandomBot")

    def test_build_report_accepts_valid_rollout_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = collector_config(root)
            config.update(
                {
                    "expected_phase23_status": "accepted",
                    "tournament_count": 1,
                    "max_players_per_table": 2,
                    "engine_rollout_collection_report_path": str(root / "report.json"),
                    "tournament_results_path": str(root / "tournament_results.json"),
                    "event_summaries_path": str(root / "event_summaries.json"),
                    "acceptance": {
                        "min_completed_tournaments": 1,
                        "min_model_decisions": 1,
                        "max_stopped_max_hands_rate": 0.1,
                        "max_model_timeout_fallbacks": 0,
                    },
                }
            )
            checkpoint = root / "latest.pt"
            checkpoint.write_bytes(b"checkpoint")
            collector = RolloutCollector(config, {})
            collector.record_decision(
                game_state=valid_state(),
                observation=[0.0] * 62,
                legal_mask=[True] * 9,
                selected_action_id=1,
                selected_action_logprob=0.0,
                value_estimate=0.0,
                engine_action=("call", 0),
            )
            collector.record_terminal_outcomes(
                1,
                [
                    {
                        "position": 1,
                        "name": "Phase24ModelBot_1",
                        "bot_class": "CollectingPhase24ModelBot",
                        "payout_pct": 1.0,
                        "final_stack": 3000,
                    }
                ],
            )
            manifest = collector.write_manifest(config)
            report = build_report(
                config,
                {
                    "source_phase23_status": "accepted",
                    "source_engine_rollout_collection_allowed": True,
                    "promoted_checkpoint_path": str(checkpoint),
                    "prerequisite_failures": [],
                },
                {
                    "lineup_summary": {"total_bots": 3},
                    "tournament_results": [
                        {
                            "tournament_id": 1,
                            "stopped_max_hands": False,
                            "results": [
                                {
                                    "position": 1,
                                    "name": "Phase24ModelBot_1",
                                    "bot_class": "CollectingPhase24ModelBot",
                                    "payout_pct": 1.0,
                                }
                            ],
                        }
                    ],
                    "event_summaries": [],
                    "event_log_paths": [],
                    "tournament_failures": [],
                    "bot_fallback_summary": {"totals": {"timeouts": 0}},
                    "model_equity_summary": {"totals": {}},
                },
                collector,
                manifest,
                "2026-05-15T00:00:00+00:00",
                0.1,
            )

        self.assertEqual(report["phase24_engine_rollout_collection_status"], "accepted")
        self.assertTrue(report["phase25_engine_training_allowed"])

    def test_run_phase24_rejects_missing_checkpoint_before_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase23 = write_json(
                root / "phase23" / "larger_mtt_engine_simulation_report.json",
                {
                    "phase23_larger_engine_simulation_status": "accepted",
                    "engine_rollout_collection_allowed": True,
                    "promoted_checkpoint_path": "runs/missing/latest.pt",
                },
            )
            config = {
                "phase23_report_path": str(phase23),
                "basemodel_root": str(root / "basemodel"),
                "artifact_root": str(root / "phase24"),
                "engine_rollout_collection_report_path": str(root / "phase24" / "report.json"),
                "tournament_results_path": str(root / "phase24" / "tournament_results.json"),
                "event_summaries_path": str(root / "phase24" / "event_summaries.json"),
                "tournament_count": 1,
            }

            report = run_phase24_engine_rollout_collection(config)

            self.assertEqual(report["phase24_engine_rollout_collection_status"], "rejected")
            self.assertIn(
                "promoted checkpoint does not exist",
                " ".join(report["phase24_engine_rollout_collection_failures"]),
            )
            self.assertTrue((root / "phase24" / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
