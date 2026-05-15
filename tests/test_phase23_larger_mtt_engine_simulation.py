from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.phase23_larger_mtt_engine_simulation import (
    build_phase23_lineup,
    build_report,
    resolve_phase22_report,
    run_phase23_larger_mtt_engine_simulation,
    should_write_full_event_log,
    summarize_events,
    summarize_results,
)


class DummyModel:
    def eval(self):
        return self


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class Phase23LargerMttEngineSimulationTests(unittest.TestCase):
    def test_resolve_phase22_report_selects_latest_accepted_allowed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rejected = write_json(
                root / "old" / "small_mtt_engine_simulation_report.json",
                {"phase22_small_engine_simulation_status": "rejected", "larger_mtt_engine_simulation_allowed": False},
            )
            accepted = write_json(
                root / "latest" / "small_mtt_engine_simulation_report.json",
                {"phase22_small_engine_simulation_status": "accepted", "larger_mtt_engine_simulation_allowed": True},
            )
            rejected.touch()
            accepted.touch()

            path, report, failure = resolve_phase22_report(
                {"phase22_report_glob": str(root / "*" / "small_mtt_engine_simulation_report.json")}
            )

        self.assertEqual(path, str(accepted))
        self.assertEqual(report["phase22_small_engine_simulation_status"], "accepted")
        self.assertEqual(failure, "")

    def test_resolve_phase22_report_rejects_disallowed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "small_mtt_engine_simulation_report.json",
                {"phase22_small_engine_simulation_status": "accepted", "larger_mtt_engine_simulation_allowed": False},
            )

            _path, _report, failure = resolve_phase22_report(
                {"phase22_report_glob": str(root / "small_mtt_engine_simulation_report.json")}
            )

        self.assertIn("no accepted Phase 22 report matched", failure)

    @patch("engine.baseline_model_bot.load_checkpoint_model", return_value=DummyModel())
    def test_build_phase23_lineup_renames_model_bots_and_keeps_opponents(self, _load_model):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "latest.pt"
            checkpoint.write_bytes(b"checkpoint")
            config = {
                "lineup": {
                    "model": 2,
                    "random": 1,
                    "equity_aggressive": 1,
                    "aggressive_no_equity": 1,
                    "call": 1,
                },
                "bot_decision_timeout_ms": 500,
                "equity_source": "constant",
            }

            bots = build_phase23_lineup(
                config,
                str(checkpoint),
                str(Path(__file__).resolve().parents[2] / "poker-ai-basemodel"),
            )

        self.assertEqual(len(bots), 6)
        self.assertEqual([bot.name for bot in bots[:2]], ["Phase23ModelBot_1", "Phase23ModelBot_2"])
        self.assertEqual(bots[2].__class__.__name__, "RandomBot")

    def test_event_logging_sampling_respects_full_log_and_first_n(self):
        self.assertTrue(should_write_full_event_log({"event_logging": {"write_full_event_logs": True}}, 99))
        self.assertTrue(
            should_write_full_event_log({"event_logging": {"write_full_event_logs_for_first_n": 2}}, 2)
        )
        self.assertFalse(
            should_write_full_event_log({"event_logging": {"write_full_event_logs_for_first_n": 2}}, 3)
        )

    def test_summarize_events_counts_actions_and_model_actions(self):
        summary = summarize_events(
            1,
            [
                {"type": "action", "player": "Phase23ModelBot_1", "action": "raise"},
                {"type": "action", "player": "RandomBot_1", "action": "call"},
                {"type": "knockout", "player": "RandomBot_1"},
                {"type": "tournament_win", "player": "Phase23ModelBot_1"},
            ],
            [{"position": 1, "name": "Phase23ModelBot_1"}],
        )

        self.assertEqual(summary["event_count"], 4)
        self.assertEqual(summary["knockout_count"], 1)
        self.assertEqual(summary["action_counts"]["raise"], 1)
        self.assertEqual(summary["model_action_counts"]["raise"], 1)
        self.assertEqual(summary["winner"], "Phase23ModelBot_1")

    def test_summarize_results_tracks_model_distribution_and_rates(self):
        summary = summarize_results(
            [
                {
                    "tournament_id": 1,
                    "stopped_max_hands": False,
                    "results": [
                        {
                            "position": 1,
                            "name": "Phase23ModelBot_1",
                            "bot_class": "BaselineModelEngineBot",
                            "payout_pct": 0.5,
                        },
                        {
                            "position": 4,
                            "name": "Phase23ModelBot_2",
                            "bot_class": "BaselineModelEngineBot",
                            "payout_pct": 0.0,
                        },
                        {"position": 2, "name": "RandomBot_1", "bot_class": "RandomBot", "payout_pct": 0.3},
                    ],
                }
            ]
        )

        self.assertEqual(summary["completed_tournament_count"], 1)
        self.assertEqual(summary["model_bot_summary"]["entries"], 2)
        self.assertEqual(summary["model_bot_summary"]["median_position"], 2.5)
        self.assertEqual(summary["model_bot_summary"]["top_3_rate"], 0.5)
        self.assertEqual(summary["payout_summary_by_bot_class"]["BaselineModelEngineBot"]["payout_sum"], 0.5)

    def test_build_report_rejects_excess_max_hand_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "tournament_results.json"
            event_summary_path = root / "event_summaries.json"
            result_path.write_text("[]", encoding="utf-8")
            event_summary_path.write_text("[]", encoding="utf-8")
            checkpoint = root / "latest.pt"
            checkpoint.write_bytes(b"checkpoint")
            report = build_report(
                {
                    "expected_phase22_status": "accepted",
                    "tournament_count": 2,
                    "max_players_per_table": 2,
                    "tournament_results_path": str(result_path),
                    "event_summaries_path": str(event_summary_path),
                    "acceptance": {
                        "min_completed_tournaments": 2,
                        "max_stopped_max_hands_rate": 0.1,
                        "max_model_timeout_fallbacks": 0,
                    },
                    "decision_thresholds": {"min_model_entries_for_decision": 1},
                },
                {
                    "source_phase22_status": "accepted",
                    "source_larger_mtt_engine_simulation_allowed": True,
                    "promoted_checkpoint_path": str(checkpoint),
                    "prerequisite_failures": [],
                },
                {
                    "lineup_summary": {"total_bots": 3},
                    "tournament_results": [
                        {"tournament_id": 1, "stopped_max_hands": True, "results": [{"position": 1, "bot_class": "RandomBot"}]},
                        {"tournament_id": 2, "stopped_max_hands": False, "results": [{"position": 1, "bot_class": "RandomBot"}]},
                    ],
                    "event_summaries": [],
                    "event_log_paths": [],
                    "tournament_failures": [],
                    "bot_fallback_summary": {"totals": {"timeouts": 0}},
                    "model_equity_summary": {"totals": {}},
                },
                "2026-05-15T00:00:00+00:00",
                0.1,
            )

        self.assertEqual(report["phase23_larger_engine_simulation_status"], "rejected")
        self.assertFalse(report["engine_rollout_collection_allowed"])
        self.assertIn(
            "stopped_max_hands_rate_gate_passed failed",
            report["phase23_larger_engine_simulation_failures"],
        )

    def test_run_phase23_rejects_missing_checkpoint_before_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase22 = write_json(
                root / "phase22" / "small_mtt_engine_simulation_report.json",
                {
                    "phase22_small_engine_simulation_status": "accepted",
                    "larger_mtt_engine_simulation_allowed": True,
                    "promoted_checkpoint_path": "runs/missing/latest.pt",
                },
            )
            config = {
                "phase22_report_path": str(phase22),
                "basemodel_root": str(root / "basemodel"),
                "artifact_root": str(root / "phase23"),
                "larger_engine_simulation_report_path": str(root / "phase23" / "report.json"),
                "tournament_results_path": str(root / "phase23" / "tournament_results.json"),
                "event_summaries_path": str(root / "phase23" / "event_summaries.json"),
                "tournament_count": 1,
            }

            report = run_phase23_larger_mtt_engine_simulation(config)

            self.assertEqual(report["phase23_larger_engine_simulation_status"], "rejected")
            self.assertIn(
                "promoted checkpoint does not exist",
                " ".join(report["phase23_larger_engine_simulation_failures"]),
            )
            self.assertTrue((root / "phase23" / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
