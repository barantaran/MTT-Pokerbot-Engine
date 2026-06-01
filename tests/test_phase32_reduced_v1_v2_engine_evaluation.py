from __future__ import annotations

import unittest
from pathlib import Path

from engine.phase32_reduced_v1_v2_engine_evaluation import _summarize_results, build_report
from engine.phase22_small_mtt_engine_simulation import _read_json


class Phase32ReducedV1V2EngineEvaluationTests(unittest.TestCase):
    def test_summary_reports_final_table_rate_and_average_payout(self):
        summary = _summarize_results(
            [
                {
                    "results": [
                        {"position": 1, "bot_class": "ReducedV4ModelBot", "name": "ReducedV4ModelBot_1", "payout_pct": 0.5},
                        {"position": 9, "bot_class": "ReducedV4ModelBot", "name": "ReducedV4ModelBot_2", "payout_pct": 0.03},
                        {"position": 10, "bot_class": "ReducedV5ModelBot", "name": "ReducedV5ModelBot_1", "payout_pct": 0.0},
                    ]
                }
            ],
            final_table_size=9,
        )

        v4 = summary["groups"]["reduced_v4"]
        v5 = summary["groups"]["reduced_v5"]
        self.assertEqual(v4["final_table"], 2)
        self.assertEqual(v4["final_table_rate"], 1.0)
        self.assertAlmostEqual(v4["average_payout_pct"], 0.265)
        self.assertEqual(v5["final_table"], 0)
        self.assertEqual(v5["final_table_rate"], 0.0)

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

    def test_build_report_accepts_no_random_v4_field_when_random_gate_disabled(self):
        report = build_report(
            {
                "tournament_count": 2,
                "reduced_v1_v2_engine_evaluation_report_path": "report.json",
                "acceptance": {
                    "min_completed_tournaments": 2,
                    "min_v1_entries": 2,
                    "min_v2_entries": 2,
                    "min_v3_entries": 2,
                    "min_v4_entries": 2,
                    "require_v2_beats_random": False,
                },
            },
            {"prerequisite_passed": True, "prerequisite_failures": [], "v1": {}, "v2": {}, "v3": {}, "v4": {}},
            {
                "tournament_failures": [],
                "lineup_summary": {"total_bots": 12},
                "event_summaries": [],
                "event_log_paths": [],
                "summary": {
                    "completed_tournament_count": 2,
                    "stopped_max_hands_count": 0,
                    "groups": {
                        "reduced_v1": {"entries": 2, "average_position": 11.0, "itm_rate": 0.1, "total_payout_pct": 0.2},
                        "reduced_v2": {"entries": 2, "average_position": 9.0, "itm_rate": 0.2, "total_payout_pct": 0.5},
                        "reduced_v3": {"entries": 2, "average_position": 8.0, "itm_rate": 0.3, "total_payout_pct": 0.7},
                        "reduced_v4": {"entries": 2, "average_position": 7.5, "itm_rate": 0.4, "total_payout_pct": 0.9},
                        "tight_equity": {"entries": 2, "average_position": 7.0},
                        "equity_aggressive": {"entries": 2, "average_position": 10.0},
                    },
                },
                "action_mix_summary": {},
                "bot_fallback_summary": {
                    "reduced_v1": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v2": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v3": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v4": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                },
                "model_equity_summary": {
                    "reduced_v1": {"totals": {"fallbacks": 0}},
                    "reduced_v2": {"totals": {"fallbacks": 0}},
                    "reduced_v3": {"totals": {"fallbacks": 0}},
                    "reduced_v4": {"totals": {"fallbacks": 0}},
                },
            },
            "2026-05-29T00:00:00+00:00",
            0.1,
        )

        self.assertEqual(report["phase32_reduced_v1_v2_engine_evaluation_status"], "accepted")
        self.assertTrue(report["phase32_gate_results"]["v4_entry_gate_passed"])
        self.assertEqual(report["comparison_summary"]["v4_minus_v3_average_position"], -0.5)
        self.assertAlmostEqual(report["comparison_summary"]["v4_minus_v3_total_payout_pct"], 0.2)

    def test_build_report_accepts_no_random_v5_field_when_random_gate_disabled(self):
        report = build_report(
            {
                "tournament_count": 2,
                "reduced_v1_v2_engine_evaluation_report_path": "report.json",
                "acceptance": {
                    "min_completed_tournaments": 2,
                    "min_v1_entries": 2,
                    "min_v2_entries": 2,
                    "min_v4_entries": 2,
                    "min_v5_entries": 2,
                    "require_v2_beats_random": False,
                },
            },
            {"prerequisite_passed": True, "prerequisite_failures": [], "v1": {}, "v2": {}, "v4": {}, "v5": {}},
            {
                "tournament_failures": [],
                "lineup_summary": {"total_bots": 12},
                "event_summaries": [],
                "event_log_paths": [],
                "summary": {
                    "completed_tournament_count": 2,
                    "stopped_max_hands_count": 0,
                    "groups": {
                        "reduced_v1": {"entries": 2, "average_position": 11.0, "itm_rate": 0.1, "total_payout_pct": 0.2},
                        "reduced_v2": {"entries": 2, "average_position": 9.0, "itm_rate": 0.2, "total_payout_pct": 0.5},
                        "reduced_v4": {"entries": 2, "average_position": 7.5, "itm_rate": 0.4, "total_payout_pct": 0.9},
                        "reduced_v5": {"entries": 2, "average_position": 7.0, "itm_rate": 0.5, "total_payout_pct": 1.1},
                        "tight_equity": {"entries": 2, "average_position": 7.0},
                        "equity_aggressive": {"entries": 2, "average_position": 10.0},
                    },
                },
                "action_mix_summary": {},
                "bot_fallback_summary": {
                    "reduced_v1": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v2": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v4": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v5": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                },
                "model_equity_summary": {
                    "reduced_v1": {"totals": {"fallbacks": 0}},
                    "reduced_v2": {"totals": {"fallbacks": 0}},
                    "reduced_v4": {"totals": {"fallbacks": 0}},
                    "reduced_v5": {"totals": {"fallbacks": 0}},
                },
            },
            "2026-05-29T00:00:00+00:00",
            0.1,
        )

        self.assertEqual(report["phase32_reduced_v1_v2_engine_evaluation_status"], "accepted")
        self.assertTrue(report["phase32_gate_results"]["v5_entry_gate_passed"])
        self.assertEqual(report["comparison_summary"]["v5_minus_v4_average_position"], -0.5)
        self.assertAlmostEqual(report["comparison_summary"]["v5_minus_v4_total_payout_pct"], 0.2)

    def test_build_report_accepts_no_random_v6_field_when_random_gate_disabled(self):
        report = build_report(
            {
                "tournament_count": 2,
                "reduced_v1_v2_engine_evaluation_report_path": "report.json",
                "acceptance": {
                    "min_completed_tournaments": 2,
                    "min_v1_entries": 2,
                    "min_v2_entries": 2,
                    "min_v5_entries": 2,
                    "min_v6_entries": 2,
                    "require_v2_beats_random": False,
                },
            },
            {"prerequisite_passed": True, "prerequisite_failures": [], "v1": {}, "v2": {}, "v5": {}, "v6": {}},
            {
                "tournament_failures": [],
                "lineup_summary": {"total_bots": 12},
                "event_summaries": [],
                "event_log_paths": [],
                "summary": {
                    "completed_tournament_count": 2,
                    "stopped_max_hands_count": 0,
                    "groups": {
                        "reduced_v1": {"entries": 2, "average_position": 11.0, "itm_rate": 0.1, "total_payout_pct": 0.2},
                        "reduced_v2": {"entries": 2, "average_position": 9.0, "itm_rate": 0.2, "total_payout_pct": 0.5},
                        "reduced_v5": {"entries": 2, "average_position": 7.0, "itm_rate": 0.5, "total_payout_pct": 1.1},
                        "reduced_v6": {"entries": 2, "average_position": 6.5, "itm_rate": 0.5, "total_payout_pct": 1.3},
                        "tight_equity": {"entries": 2, "average_position": 7.0},
                        "equity_aggressive": {"entries": 2, "average_position": 10.0},
                    },
                },
                "action_mix_summary": {},
                "bot_fallback_summary": {
                    "reduced_v1": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v2": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v5": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v6": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                },
                "model_equity_summary": {
                    "reduced_v1": {"totals": {"fallbacks": 0}},
                    "reduced_v2": {"totals": {"fallbacks": 0}},
                    "reduced_v5": {"totals": {"fallbacks": 0}},
                    "reduced_v6": {"totals": {"fallbacks": 0}},
                },
            },
            "2026-05-29T00:00:00+00:00",
            0.1,
        )

        self.assertEqual(report["phase32_reduced_v1_v2_engine_evaluation_status"], "accepted")
        self.assertTrue(report["phase32_gate_results"]["v6_entry_gate_passed"])
        self.assertEqual(report["comparison_summary"]["v6_minus_v5_average_position"], -0.5)
        self.assertAlmostEqual(report["comparison_summary"]["v6_minus_v5_total_payout_pct"], 0.2)

    def test_build_report_accepts_no_random_v7_field_when_random_gate_disabled(self):
        report = build_report(
            {
                "tournament_count": 2,
                "reduced_v1_v2_engine_evaluation_report_path": "report.json",
                "acceptance": {
                    "min_completed_tournaments": 2,
                    "min_v1_entries": 2,
                    "min_v2_entries": 2,
                    "min_v6_entries": 2,
                    "min_v7_entries": 2,
                    "require_v2_beats_random": False,
                },
            },
            {"prerequisite_passed": True, "prerequisite_failures": [], "v1": {}, "v2": {}, "v6": {}, "v7": {}},
            {
                "tournament_failures": [],
                "lineup_summary": {"total_bots": 12},
                "event_summaries": [],
                "event_log_paths": [],
                "summary": {
                    "completed_tournament_count": 2,
                    "stopped_max_hands_count": 0,
                    "groups": {
                        "reduced_v1": {"entries": 2, "average_position": 11.0, "itm_rate": 0.1, "total_payout_pct": 0.2},
                        "reduced_v2": {"entries": 2, "average_position": 9.0, "itm_rate": 0.2, "total_payout_pct": 0.5},
                        "reduced_v6": {"entries": 2, "average_position": 6.5, "itm_rate": 0.5, "total_payout_pct": 1.3},
                        "reduced_v7": {"entries": 2, "average_position": 6.0, "itm_rate": 0.6, "total_payout_pct": 1.6},
                        "tight_equity": {"entries": 2, "average_position": 7.0},
                        "equity_aggressive": {"entries": 2, "average_position": 10.0},
                    },
                },
                "action_mix_summary": {},
                "bot_fallback_summary": {
                    "reduced_v1": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v2": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v6": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v7": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                },
                "model_equity_summary": {
                    "reduced_v1": {"totals": {"fallbacks": 0}},
                    "reduced_v2": {"totals": {"fallbacks": 0}},
                    "reduced_v6": {"totals": {"fallbacks": 0}},
                    "reduced_v7": {"totals": {"fallbacks": 0}},
                },
            },
            "2026-05-29T00:00:00+00:00",
            0.1,
        )

        self.assertEqual(report["phase32_reduced_v1_v2_engine_evaluation_status"], "accepted")
        self.assertTrue(report["phase32_gate_results"]["v7_entry_gate_passed"])
        self.assertEqual(report["comparison_summary"]["v7_minus_v6_average_position"], -0.5)
        self.assertAlmostEqual(report["comparison_summary"]["v7_minus_v6_total_payout_pct"], 0.3)

    def test_build_report_accepts_no_random_v8_field_when_random_gate_disabled(self):
        report = build_report(
            {
                "tournament_count": 2,
                "reduced_v1_v2_engine_evaluation_report_path": "report.json",
                "acceptance": {
                    "min_completed_tournaments": 2,
                    "min_v1_entries": 2,
                    "min_v2_entries": 2,
                    "min_v6_entries": 2,
                    "min_v8_entries": 2,
                    "require_v2_beats_random": False,
                },
            },
            {"prerequisite_passed": True, "prerequisite_failures": [], "v1": {}, "v2": {}, "v6": {}, "v8": {}},
            {
                "tournament_failures": [],
                "lineup_summary": {"total_bots": 12},
                "event_summaries": [],
                "event_log_paths": [],
                "summary": {
                    "completed_tournament_count": 2,
                    "stopped_max_hands_count": 0,
                    "groups": {
                        "reduced_v1": {"entries": 2, "average_position": 11.0, "itm_rate": 0.1, "total_payout_pct": 0.2},
                        "reduced_v2": {"entries": 2, "average_position": 9.0, "itm_rate": 0.2, "total_payout_pct": 0.5},
                        "reduced_v6": {"entries": 2, "average_position": 6.5, "itm_rate": 0.5, "total_payout_pct": 1.3},
                        "reduced_v8": {"entries": 2, "average_position": 5.5, "itm_rate": 0.6, "total_payout_pct": 1.8},
                        "tight_equity": {"entries": 2, "average_position": 7.0},
                        "equity_aggressive": {"entries": 2, "average_position": 10.0},
                    },
                },
                "action_mix_summary": {},
                "bot_fallback_summary": {
                    "reduced_v1": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v2": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v6": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                    "reduced_v8": {"totals": {"inference_errors": 0, "timeouts": 0, "illegal_actions": 0}},
                },
                "model_equity_summary": {
                    "reduced_v1": {"totals": {"fallbacks": 0}},
                    "reduced_v2": {"totals": {"fallbacks": 0}},
                    "reduced_v6": {"totals": {"fallbacks": 0}},
                    "reduced_v8": {"totals": {"fallbacks": 0}},
                },
            },
            "2026-05-29T00:00:00+00:00",
            0.1,
        )

        self.assertEqual(report["phase32_reduced_v1_v2_engine_evaluation_status"], "accepted")
        self.assertTrue(report["phase32_gate_results"]["v8_entry_gate_passed"])
        self.assertEqual(report["comparison_summary"]["v8_minus_v6_average_position"], -1.0)
        self.assertAlmostEqual(report["comparison_summary"]["v8_minus_v6_total_payout_pct"], 0.5)

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

    def test_phase35_config_uses_v4_tournament_stage_schema(self):
        engine_root = Path(__file__).resolve().parents[1]
        config = _read_json(engine_root / "configs" / "phase35_equal_no_random_reduced_v2_v3_v4_engine_evaluation.json")

        self.assertNotIn("random", config["lineup"])
        self.assertEqual(config["lineup"]["v2_model"], 6)
        self.assertEqual(config["lineup"]["v3_model"], 6)
        self.assertEqual(config["lineup"]["v4_model"], 6)
        self.assertEqual(config["v4_reduced_observation_size"], 10)
        self.assertEqual(config["v4_reduced_observation_schema"], "reduced_v4")
        self.assertNotIn("ante", config["blinds_schedule"][0])

    def test_phase36_config_uses_v5_itm_distance_schema(self):
        engine_root = Path(__file__).resolve().parents[1]
        config = _read_json(engine_root / "configs" / "phase36_equal_no_random_reduced_v4_v5_engine_evaluation.json")

        self.assertNotIn("random", config["lineup"])
        self.assertEqual(config["lineup"]["v4_model"], 6)
        self.assertEqual(config["lineup"]["v5_model"], 6)
        self.assertEqual(config["v5_reduced_observation_size"], 11)
        self.assertEqual(config["v5_reduced_observation_schema"], "reduced_v5")
        self.assertNotIn("ante", config["blinds_schedule"][0])

    def test_phase39_config_uses_v8_preflop_spot_schema(self):
        engine_root = Path(__file__).resolve().parents[1]
        config = _read_json(engine_root / "configs" / "phase39_equal_no_random_reduced_v6_v8_engine_evaluation.json")

        self.assertNotIn("random", config["lineup"])
        self.assertEqual(config["lineup"]["v6_model"], 33)
        self.assertEqual(config["lineup"]["v8_model"], 33)
        self.assertEqual(config["lineup"]["tight_equity"], 33)
        self.assertEqual(config["v8_reduced_observation_size"], 11)
        self.assertEqual(config["v8_reduced_observation_schema"], "reduced_v6")
        self.assertNotIn("ante", config["blinds_schedule"][0])


if __name__ == "__main__":
    unittest.main()
