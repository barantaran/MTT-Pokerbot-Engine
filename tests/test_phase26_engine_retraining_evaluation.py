from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.phase26_engine_retraining_evaluation import (
    build_phase26_lineup,
    build_report,
    compare_campaigns,
    resolve_phase25_report,
    run_phase26_engine_retraining_evaluation,
)


class DummyModel:
    def eval(self):
        return self


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def campaign(avg_position: float, itm_rate: float, total_payout: float, *, completed: int = 2) -> dict:
    return {
        "lineup_summary": {"total_bots": 4},
        "tournament_failures": [],
        "event_summaries": [],
        "event_log_paths": [],
        "summary": {
            "completed_tournament_count": completed,
            "stopped_max_hands_count": 0,
            "model_bot_summary": {
                "entries": completed,
                "average_position": avg_position,
                "itm_rate": itm_rate,
                "win_rate": 0.0,
                "total_payout_pct": total_payout,
            },
        },
        "action_mix_summary": {
            "model_action_counts": {"call": 3, "raise": 1},
            "model_action_total": 4,
            "distinct_model_actions": 2,
            "top_model_action": "call",
            "top_model_action_rate": 0.75,
        },
        "bot_fallback_summary": {"totals": {"inference_errors": 0, "timeouts": 0}},
        "model_equity_summary": {"totals": {}},
    }


class Phase26EngineRetrainingEvaluationTests(unittest.TestCase):
    def test_resolve_phase25_report_selects_latest_accepted_allowed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rejected = write_json(
                root / "old" / "engine_rollout_training_report.json",
                {"phase25_engine_rollout_training_status": "rejected", "phase26_engine_evaluation_allowed": False},
            )
            accepted = write_json(
                root / "latest" / "engine_rollout_training_report.json",
                {"phase25_engine_rollout_training_status": "accepted", "phase26_engine_evaluation_allowed": True},
            )
            rejected.touch()
            accepted.touch()

            path, report, failure = resolve_phase25_report(
                {"phase25_report_glob": str(root / "*" / "engine_rollout_training_report.json")}
            )

        self.assertEqual(path, str(accepted))
        self.assertEqual(report["phase25_engine_rollout_training_status"], "accepted")
        self.assertEqual(failure, "")

    def test_resolve_phase25_report_rejects_disallowed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "engine_rollout_training_report.json",
                {"phase25_engine_rollout_training_status": "accepted", "phase26_engine_evaluation_allowed": False},
            )

            _path, _report, failure = resolve_phase25_report(
                {"phase25_report_glob": str(root / "engine_rollout_training_report.json")}
            )

        self.assertIn("no accepted Phase 25 report matched", failure)

    @patch("engine.baseline_model_bot.load_checkpoint_model", return_value=DummyModel())
    def test_build_phase26_lineup_uses_named_model_bots_and_mixed_opponents(self, _load_model):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "latest.pt"
            checkpoint.write_bytes(b"checkpoint")
            bots = build_phase26_lineup(
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
                "candidate",
            )

        self.assertEqual(len(bots), 6)
        self.assertEqual([bot.name for bot in bots[:2]], ["Phase26CandidateModelBot_1", "Phase26CandidateModelBot_2"])
        self.assertEqual(bots[2].__class__.__name__, "RandomBot")

    def test_compare_campaigns_reports_candidate_deltas(self):
        comparison = compare_campaigns(campaign(10.0, 0.1, 0.2), campaign(9.0, 0.2, 0.5))

        self.assertEqual(comparison["average_position_delta"], -1.0)
        self.assertEqual(comparison["itm_rate_delta"], 0.1)
        self.assertEqual(comparison["total_payout_pct_delta"], 0.3)

    def test_build_report_accepts_non_regressing_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_report(
                {
                    "artifact_root": str(root),
                    "campaign_results_dir": str(root / "campaigns"),
                    "engine_retraining_evaluation_report_path": str(root / "report.json"),
                    "tournament_count": 2,
                    "acceptance": {
                        "min_completed_tournaments": 2,
                        "max_average_position_regression": 1.0,
                        "max_itm_rate_regression": 0.1,
                        "max_total_payout_pct_regression": 0.25,
                    },
                },
                {
                    "prerequisite_passed": True,
                    "source_phase25_status": "accepted",
                    "source_phase26_engine_evaluation_allowed": True,
                    "source_checkpoint_path": str(root / "source.pt"),
                    "candidate_checkpoint_path": str(root / "candidate.pt"),
                    "prerequisite_failures": [],
                },
                {"source": campaign(10.0, 0.1, 0.2), "candidate": campaign(9.5, 0.1, 0.2)},
                "2026-05-15T00:00:00+00:00",
                0.1,
            )

        self.assertEqual(report["phase26_engine_retraining_evaluation_status"], "accepted")
        self.assertTrue(report["phase27_promotion_decision_allowed"])

    def test_run_phase26_rejects_missing_candidate_checkpoint_before_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_checkpoint = root / "source.pt"
            source_checkpoint.write_bytes(b"source")
            phase25 = write_json(
                root / "phase25" / "engine_rollout_training_report.json",
                {
                    "phase25_engine_rollout_training_status": "accepted",
                    "phase26_engine_evaluation_allowed": True,
                    "source_checkpoint_path": str(source_checkpoint),
                    "candidate_checkpoint_path": str(root / "missing.pt"),
                },
            )
            report = run_phase26_engine_retraining_evaluation(
                {
                    "phase25_report_path": str(phase25),
                    "artifact_root": str(root / "phase26"),
                    "engine_retraining_evaluation_report_path": str(root / "phase26" / "report.json"),
                    "tournament_count": 1,
                }
            )

            self.assertEqual(report["phase26_engine_retraining_evaluation_status"], "rejected")
            self.assertIn(
                "candidate checkpoint does not exist",
                " ".join(report["phase26_engine_retraining_evaluation_failures"]),
            )
            self.assertTrue((root / "phase26" / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
