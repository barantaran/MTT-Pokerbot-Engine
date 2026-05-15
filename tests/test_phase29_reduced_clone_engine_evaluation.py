from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.phase29_reduced_clone_engine_evaluation import (
    build_bot_class_stats_table,
    build_report,
    resolve_phase28_training_report,
    run_phase29_reduced_clone_engine_evaluation,
    summarize_reduced_results,
)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class Phase29ReducedCloneEngineEvaluationTests(unittest.TestCase):
    def test_resolve_phase28_training_report_selects_latest_accepted_allowed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "old" / "equity_imitation_training_report.json",
                {"phase28_equity_imitation_training_status": "rejected", "phase29_reduced_clone_evaluation_allowed": False},
            )
            accepted = write_json(
                root / "latest" / "equity_imitation_training_report.json",
                {"phase28_equity_imitation_training_status": "accepted", "phase29_reduced_clone_evaluation_allowed": True},
            )
            accepted.touch()

            path, report, failure = resolve_phase28_training_report(
                {"phase28_training_report_glob": str(root / "*" / "equity_imitation_training_report.json")}
            )

        self.assertEqual(path, str(accepted))
        self.assertEqual(report["phase28_equity_imitation_training_status"], "accepted")
        self.assertEqual(failure, "")

    def test_summarize_reduced_results_counts_reduced_model_as_model(self):
        summary = summarize_reduced_results(
            [
                {
                    "tournament_id": 1,
                    "results": [
                        {"position": 1, "bot_class": "ReducedModelEngineBot", "payout_pct": 0.5},
                        {"position": 2, "bot_class": "RandomBot", "payout_pct": 0.0},
                    ],
                }
            ]
        )

        self.assertEqual(summary["model_bot_summary"]["entries"], 1)
        self.assertEqual(summary["model_bot_summary"]["average_position"], 1.0)

    def test_build_bot_class_stats_table_includes_every_bot_type(self):
        table = build_bot_class_stats_table(
            {
                "ReducedModelEngineBot": {"count": 3, "average_position": 9.0, "itm_rate": 0.5, "win_rate": 0.1, "top_3_rate": 0.2},
                "AggressiveBot": {"count": 3, "average_position": 10.0, "itm_rate": 0.4, "win_rate": 0.0, "top_3_rate": 0.3},
                "AggressiveNoEquityBot": {"count": 3, "average_position": 15.0, "itm_rate": 0.1, "win_rate": 0.0, "top_3_rate": 0.0},
                "RandomBot": {"count": 3, "average_position": 20.0, "itm_rate": 0.0, "win_rate": 0.0, "top_3_rate": 0.0},
                "CallBot": {"count": 3, "average_position": 25.0, "itm_rate": 0.0, "win_rate": 0.0, "top_3_rate": 0.0},
            },
            {
                "ReducedModelEngineBot": {"payout_sum": 0.3, "average_payout_pct": 0.1},
                "AggressiveBot": {"payout_sum": 0.2, "average_payout_pct": 0.0666666667},
            },
        )

        self.assertEqual(
            [row["bot_class"] for row in table],
            ["AggressiveBot", "AggressiveNoEquityBot", "CallBot", "RandomBot", "ReducedModelEngineBot"],
        )
        reduced = next(row for row in table if row["bot_class"] == "ReducedModelEngineBot")
        self.assertEqual(reduced["entries"], 3)
        self.assertEqual(reduced["total_payout_pct"], 0.3)

    def test_build_report_accepts_clone_that_beats_random_and_approaches_equity(self):
        campaign = {
            "tournament_failures": [],
            "lineup_summary": {"total_bots": 6},
            "event_summaries": [],
            "event_log_paths": [],
            "summary": {
                "completed_tournament_count": 2,
                "stopped_max_hands_count": 0,
                "model_bot_summary": {"entries": 2, "average_position": 12.0, "itm_rate": 0.1, "total_payout_pct": 0.2},
                "placement_summary_by_bot_class": {
                    "ReducedModelEngineBot": {"average_position": 12.0, "itm_rate": 0.1},
                    "RandomBot": {"average_position": 20.0, "itm_rate": 0.0},
                    "AggressiveBot": {"average_position": 10.0, "itm_rate": 0.2},
                    "CallBot": {"average_position": 24.0, "itm_rate": 0.0},
                },
                "payout_summary_by_bot_class": {"AggressiveBot": {"payout_sum": 0.4}},
            },
            "action_mix_summary": {"distinct_model_actions": 2, "top_model_action_rate": 0.8},
            "bot_fallback_summary": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
            "model_equity_summary": {"totals": {}},
        }

        report = build_report(
            {
                "tournament_count": 2,
                "reduced_clone_engine_evaluation_report_path": "report.json",
                "acceptance": {
                    "min_completed_tournaments": 2,
                    "min_model_entries": 2,
                    "max_equity_average_position_gap": 3.0,
                },
            },
            {"prerequisite_passed": True, "prerequisite_failures": []},
            campaign,
            "2026-05-15T00:00:00+00:00",
            0.1,
        )

        self.assertEqual(report["phase29_reduced_clone_engine_evaluation_status"], "accepted")
        self.assertTrue(report["phase30_reduced_clone_decision_allowed"])
        self.assertEqual(
            [row["bot_class"] for row in report["bot_class_stats_table"]],
            ["AggressiveBot", "CallBot", "RandomBot", "ReducedModelEngineBot"],
        )

    def test_run_phase29_rejects_missing_checkpoint_before_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase28 = write_json(
                root / "phase28" / "equity_imitation_training_report.json",
                {
                    "phase28_equity_imitation_training_status": "accepted",
                    "phase29_reduced_clone_evaluation_allowed": True,
                    "candidate_checkpoint_path": str(root / "missing.pt"),
                },
            )

            report = run_phase29_reduced_clone_engine_evaluation(
                {
                    "phase28_training_report_path": str(phase28),
                    "artifact_root": str(root / "phase29"),
                    "reduced_clone_engine_evaluation_report_path": str(root / "phase29" / "report.json"),
                    "tournament_count": 1,
                }
            )

            self.assertEqual(report["phase29_reduced_clone_engine_evaluation_status"], "rejected")
            self.assertIn("candidate checkpoint does not exist", " ".join(report["phase29_reduced_clone_engine_evaluation_failures"]))
            self.assertTrue((root / "phase29" / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
