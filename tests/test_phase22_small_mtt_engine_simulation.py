from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.config import config as engine_config
from engine.phase22_small_mtt_engine_simulation import (
    build_lineup,
    build_report,
    resolve_phase21_report,
    run_phase22_small_mtt_engine_simulation,
    summarize_results,
    temporary_engine_config,
)


class DummyModel:
    def eval(self):
        return self


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class Phase22SmallMttEngineSimulationTests(unittest.TestCase):
    def test_resolve_phase21_report_selects_latest_accepted_allowed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rejected = write_json(
                root / "old" / "engine_wiring_report.json",
                {"phase21_engine_wiring_status": "rejected", "small_mtt_engine_simulation_allowed": False},
            )
            accepted = write_json(
                root / "latest" / "engine_wiring_report.json",
                {"phase21_engine_wiring_status": "accepted", "small_mtt_engine_simulation_allowed": True},
            )
            rejected.touch()
            accepted.touch()

            path, report, failure = resolve_phase21_report(
                {"phase21_report_glob": str(root / "*" / "engine_wiring_report.json")}
            )

        self.assertEqual(path, str(accepted))
        self.assertEqual(report["phase21_engine_wiring_status"], "accepted")
        self.assertEqual(failure, "")

    def test_resolve_phase21_report_rejects_disallowed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "engine_wiring_report.json",
                {"phase21_engine_wiring_status": "accepted", "small_mtt_engine_simulation_allowed": False},
            )

            _path, _report, failure = resolve_phase21_report(
                {"phase21_report_glob": str(root / "engine_wiring_report.json")}
            )

        self.assertIn("no accepted Phase 21 report matched", failure)

    @patch("engine.baseline_model_bot.load_checkpoint_model", return_value=DummyModel())
    def test_build_lineup_includes_model_random_equity_and_fixed_bots(self, _load_model):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "latest.pt"
            checkpoint.write_bytes(b"checkpoint")
            config = {
                "lineup": {
                    "model": 1,
                    "random": 2,
                    "equity_aggressive": 1,
                    "tight_equity": 1,
                    "noisy_equity": 1,
                    "range_policy": 1,
                    "aggressive_no_equity": 1,
                    "call": 1,
                },
                "range_policy_variants": [
                    {"variant": "tight", "count": 1, "call_margin": 0.05, "raise_threshold": 0.62}
                ],
                "bot_decision_timeout_ms": 500,
                "equity_source": "constant",
                "model_use_preflop_spot_range": True,
            }

            bots = build_lineup(config, str(checkpoint), str(Path(__file__).resolve().parents[2] / "poker-ai-basemodel"))

        self.assertEqual(len(bots), 10)
        self.assertEqual(
            [bot.__class__.__name__ for bot in bots],
            [
                "BaselineModelEngineBot",
                "RandomBot",
                "RandomBot",
                "AggressiveBot",
                "TightEquityBot",
                "NoisyEquityBot",
                "RangePolicyBot",
                "RangePolicyBot",
                "AggressiveNoEquityBot",
                "CallBot",
            ],
        )
        self.assertTrue(bots[0]._delegate.equity_config.use_preflop_spot_range)
        self.assertFalse(bots[3].use_preflop_spot_range)
        self.assertFalse(bots[4].use_preflop_spot_range)
        self.assertFalse(bots[5].use_preflop_spot_range)
        self.assertTrue(bots[6].use_preflop_spot_range)
        self.assertEqual(bots[6].variant, "balanced")
        self.assertEqual(bots[7].variant, "tight")
        self.assertAlmostEqual(bots[7].call_margin, 0.05)

    def test_temporary_engine_config_restores_existing_and_removes_new_values(self):
        original_stack = engine_config.starting_stack
        self.assertFalse(hasattr(engine_config, "_phase22_test_value"))

        with temporary_engine_config({"starting_stack": 777, "_phase22_test_value": "x"}):
            self.assertEqual(engine_config.starting_stack, 777)
            self.assertEqual(engine_config._phase22_test_value, "x")

        self.assertEqual(engine_config.starting_stack, original_stack)
        self.assertFalse(hasattr(engine_config, "_phase22_test_value"))

    def test_summarize_results_tracks_model_placements_and_max_hand_stops(self):
        summary = summarize_results(
            [
                {
                    "tournament_id": 1,
                    "stopped_max_hands": True,
                    "results": [
                        {
                            "position": 1,
                            "name": "Phase22ModelBot_1",
                            "bot_class": "BaselineModelEngineBot",
                            "payout_pct": 0.5,
                        },
                        {"position": 2, "name": "RandomBot_1", "bot_class": "RandomBot", "payout_pct": 0.3},
                    ],
                }
            ]
        )

        self.assertEqual(summary["completed_tournament_count"], 1)
        self.assertEqual(summary["stopped_max_hands_count"], 1)
        self.assertEqual(summary["model_bot_summary"]["best_position"], 1)
        self.assertEqual(summary["placement_summary_by_bot_class"]["RandomBot"]["count"], 1)

    def test_build_report_rejects_model_timeout_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "tournament_results.json"
            result_path.write_text("[]", encoding="utf-8")
            checkpoint = Path(tmp) / "latest.pt"
            checkpoint.write_bytes(b"checkpoint")
            report = build_report(
                {
                    "expected_phase21_status": "accepted",
                    "tournament_count": 1,
                    "tournament_results_path": str(result_path),
                    "acceptance": {"min_completed_tournaments": 1, "max_model_timeout_fallbacks": 0},
                },
                {
                    "source_phase21_status": "accepted",
                    "source_small_mtt_engine_simulation_allowed": True,
                    "promoted_checkpoint_path": str(checkpoint),
                    "prerequisite_failures": [],
                },
                {
                    "lineup_summary": {"total_bots": 2},
                    "tournament_results": [{"tournament_id": 1, "results": [{"position": 1, "bot_class": "RandomBot"}]}],
                    "event_log_paths": [str(Path(tmp) / "events.json")],
                    "tournament_failures": [],
                    "bot_fallback_summary": {"totals": {"timeouts": 1}},
                    "model_equity_summary": {"totals": {}},
                },
                "2026-05-15T00:00:00+00:00",
                0.1,
            )

        self.assertEqual(report["phase22_small_engine_simulation_status"], "rejected")
        self.assertFalse(report["larger_mtt_engine_simulation_allowed"])
        self.assertIn("model_timeout_fallback_gate_passed failed", report["phase22_small_engine_simulation_failures"])

    def test_run_phase22_rejects_missing_checkpoint_before_simulation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase21 = write_json(
                root / "phase21" / "engine_wiring_report.json",
                {
                    "phase21_engine_wiring_status": "accepted",
                    "small_mtt_engine_simulation_allowed": True,
                    "promoted_checkpoint_path": "runs/missing/latest.pt",
                },
            )
            config = {
                "phase21_report_path": str(phase21),
                "basemodel_root": str(root / "basemodel"),
                "artifact_root": str(root / "phase22"),
                "small_engine_simulation_report_path": str(root / "phase22" / "report.json"),
                "tournament_results_path": str(root / "phase22" / "tournament_results.json"),
                "tournament_count": 1,
            }

            report = run_phase22_small_mtt_engine_simulation(config)

            self.assertEqual(report["phase22_small_engine_simulation_status"], "rejected")
            self.assertIn(
                "promoted checkpoint does not exist",
                " ".join(report["phase22_small_engine_simulation_failures"]),
            )
            self.assertTrue((root / "phase22" / "report.json").is_file())


if __name__ == "__main__":
    unittest.main()
