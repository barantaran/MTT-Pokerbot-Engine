from __future__ import annotations

import unittest
from pathlib import Path

from engine.phase32_reduced_v1_v2_engine_evaluation import build_report
from engine.phase22_small_mtt_engine_simulation import _read_json


class Phase32ReducedV1V2EngineEvaluationTests(unittest.TestCase):
    def test_build_report_accepts_clean_v2_that_beats_random(self):
        report = build_report(
            {
                "tournament_count": 2,
                "reduced_v1_v2_engine_evaluation_report_path": "report.json",
                "acceptance": {"min_completed_tournaments": 2, "min_v1_entries": 2, "min_v2_entries": 2},
            },
            {"prerequisite_passed": True, "prerequisite_failures": [], "v1": {}, "v2": {}},
            {
                "tournament_failures": [],
                "lineup_summary": {"total_bots": 6},
                "event_summaries": [],
                "event_log_paths": [],
                "summary": {
                    "completed_tournament_count": 2,
                    "stopped_max_hands_count": 0,
                    "groups": {
                        "reduced_v1": {"entries": 2, "average_position": 10.0, "itm_rate": 0.1, "total_payout_pct": 0.2},
                        "reduced_v2": {"entries": 2, "average_position": 8.0, "itm_rate": 0.2, "total_payout_pct": 0.5},
                        "random": {"entries": 2, "average_position": 20.0},
                    },
                },
                "action_mix_summary": {},
                "bot_fallback_summary": {
                    "reduced_v1": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v2": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                },
                "model_equity_summary": {
                    "reduced_v1": {"totals": {"fallbacks": 0}},
                    "reduced_v2": {"totals": {"fallbacks": 0}},
                },
            },
            "2026-05-29T00:00:00+00:00",
            0.1,
        )

        self.assertEqual(report["phase32_reduced_v1_v2_engine_evaluation_status"], "accepted")
        self.assertEqual(report["comparison_summary"]["v2_minus_v1_average_position"], -2.0)
        self.assertEqual(report["comparison_summary"]["v2_minus_v1_total_payout_pct"], 0.3)

    def test_phase33_real_mtt_config_keeps_aggressive_and_tight_opponents(self):
        engine_root = Path(__file__).resolve().parents[1]
        config = _read_json(engine_root / "configs" / "phase33_real_tournament_reduced_v1_v2_engine_evaluation.json")

        self.assertEqual(config["lineup"]["equity_aggressive"], 6)
        self.assertEqual(config["lineup"]["tight_equity"], 6)
        self.assertNotIn("ante", config["blinds_schedule"][0])

    def test_build_report_accepts_no_random_v3_field_when_random_gate_disabled(self):
        report = build_report(
            {
                "tournament_count": 2,
                "reduced_v1_v2_engine_evaluation_report_path": "report.json",
                "acceptance": {
                    "min_completed_tournaments": 2,
                    "min_v1_entries": 2,
                    "min_v2_entries": 2,
                    "min_v3_entries": 2,
                    "require_v2_beats_random": False,
                },
            },
            {"prerequisite_passed": True, "prerequisite_failures": [], "v1": {}, "v2": {}, "v3": {}},
            {
                "tournament_failures": [],
                "lineup_summary": {"total_bots": 10},
                "event_summaries": [],
                "event_log_paths": [],
                "summary": {
                    "completed_tournament_count": 2,
                    "stopped_max_hands_count": 0,
                    "groups": {
                        "reduced_v1": {"entries": 2, "average_position": 11.0, "itm_rate": 0.1, "total_payout_pct": 0.2},
                        "reduced_v2": {"entries": 2, "average_position": 9.0, "itm_rate": 0.2, "total_payout_pct": 0.5},
                        "reduced_v3": {"entries": 2, "average_position": 8.0, "itm_rate": 0.3, "total_payout_pct": 0.7},
                        "tight_equity": {"entries": 2, "average_position": 7.0},
                        "equity_aggressive": {"entries": 2, "average_position": 10.0},
                    },
                },
                "action_mix_summary": {},
                "bot_fallback_summary": {
                    "reduced_v1": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v2": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v3": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                },
                "model_equity_summary": {
                    "reduced_v1": {"totals": {"fallbacks": 0}},
                    "reduced_v2": {"totals": {"fallbacks": 0}},
                    "reduced_v3": {"totals": {"fallbacks": 0}},
                },
            },
            "2026-05-29T00:00:00+00:00",
            0.1,
        )

        self.assertEqual(report["phase32_reduced_v1_v2_engine_evaluation_status"], "accepted")
        self.assertTrue(report["phase32_gate_results"]["v2_beats_random_gate_passed"])
        self.assertEqual(report["comparison_summary"]["v3_minus_v2_average_position"], -1.0)
        self.assertAlmostEqual(report["comparison_summary"]["v3_minus_v2_total_payout_pct"], 0.2)

    def test_phase34_config_uses_equal_no_random_lineup(self):
        engine_root = Path(__file__).resolve().parents[1]
        config = _read_json(engine_root / "configs" / "phase34_equal_no_random_reduced_v1_v2_v3_engine_evaluation.json")

        self.assertNotIn("random", config["lineup"])
        self.assertEqual(config["lineup"]["v1_model"], 6)
        self.assertEqual(config["lineup"]["v2_model"], 6)
        self.assertEqual(config["lineup"]["v3_model"], 6)
        self.assertEqual(config["lineup"]["equity_aggressive"], 6)
        self.assertEqual(config["lineup"]["tight_equity"], 6)
        self.assertEqual(config["v1_reduced_observation_size"], 8)
        self.assertEqual(config["v2_reduced_observation_size"], 8)
        self.assertEqual(config["v3_reduced_observation_size"], 9)
        self.assertEqual(config["v3_reduced_observation_schema"], "reduced_v3")
        self.assertFalse(config["acceptance"]["require_v2_beats_random"])


if __name__ == "__main__":
    unittest.main()
