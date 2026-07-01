from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.fixed_bot_mtt_evaluation import (
    format_population_table,
    merge_population_summaries,
    population_summary_with_roi,
    _expand_named_lineup,
    _load_bot_library,
    _report_completion_line,
    _resolve_tournament_seeds,
)
from engine.evolutionary_reduced_mtt import merge_candidate_action_summaries, summarize_candidate_actions


class FixedBotMttEvaluationTests(unittest.TestCase):
    def test_load_bot_library_uses_named_bot_files(self):
        with tempfile.TemporaryDirectory() as root:
            engine_root = Path(root)
            bot_dir = engine_root / "bot_configs"
            bot_dir.mkdir()
            (bot_dir / "champ.json").write_text(
                json.dumps(
                    {
                        "name": "champ",
                        "type": "configured_tournament_equity",
                        "population": "configured_tournament_equity",
                        "include_tool_set_in_name": True,
                        "params": {"tools": [{"type": "icm_pressure"}]},
                    }
                ),
                encoding="utf-8",
            )

            library = _load_bot_library({"bot_config_dir": "bot_configs"}, engine_root=engine_root)

        self.assertEqual(sorted(library), ["champ"])
        self.assertEqual(library["champ"]["type"], "configured_tournament_equity")
        self.assertNotIn("count", library["champ"])

    def test_expand_named_lineup_splits_legacy_and_configured_bots(self):
        legacy_lineup, specs = _expand_named_lineup(
            [
                {"bot": "tight_equity", "count": 2},
                {"bot": "champ", "count": 3},
            ],
            {
                "champ": {
                    "name": "champ",
                    "type": "configured_tournament_equity",
                    "params": {"tools": [{"type": "icm_pressure"}]},
                }
            },
        )

        self.assertEqual(legacy_lineup, {"tight_equity": 2})
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0]["name"], "champ")
        self.assertEqual(specs[0]["count"], 3)

    def test_expand_named_lineup_rejects_unknown_bots(self):
        with self.assertRaisesRegex(ValueError, "unknown named lineup bot"):
            _expand_named_lineup([{"bot": "missing_bot", "count": 1}], {})

    def test_resolve_tournament_seeds_uses_one_seed_per_mtt(self):
        self.assertEqual(
            _resolve_tournament_seeds({"tournament_seeds": [11, 12, 13]}, 3),
            [11, 12, 13],
        )
        self.assertEqual(
            _resolve_tournament_seeds({"mtt_seed_start": 100}, 3),
            [100, 101, 102],
        )

    def test_resolve_tournament_seeds_rejects_reuse(self):
        with self.assertRaisesRegex(ValueError, "one seed belongs to one MTT"):
            _resolve_tournament_seeds({"tournament_seeds": [11, 11]}, 2)

    def test_resolve_tournament_seeds_rejects_wrong_count(self):
        with self.assertRaisesRegex(ValueError, "exactly 3 seeds"):
            _resolve_tournament_seeds({"tournament_seeds": [11, 12]}, 3)

    def test_format_population_table_uses_stable_complete_columns(self):
        table = format_population_table(
            {
                "small": {
                    "entries": 2,
                    "wins": 0,
                    "win_rate": 0.0,
                    "top3": 1,
                    "top3_rate": 0.5,
                    "final_table": 1,
                    "final_table_rate": 0.5,
                    "itm": 1,
                    "itm_rate": 0.5,
                    "average_position": 8.5,
                    "total_payout_pct": 0.25,
                    "average_payout_pct": 0.125,
                    "score": 4.0,
                },
                "leader": {
                    "entries": 2,
                    "wins": 1,
                    "win_rate": 0.5,
                    "top3": 1,
                    "top3_rate": 0.5,
                    "final_table": 2,
                    "final_table_rate": 1.0,
                    "itm": 2,
                    "itm_rate": 1.0,
                    "average_position": 2.0,
                    "total_payout_pct": 0.5,
                    "average_payout_pct": 0.25,
                    "score": 13.0,
                },
            }
        )

        lines = table.splitlines()
        self.assertEqual(
            lines[0],
            "| # | bot | entries | payout | avg payout | ROI | score | wins | top3 | FT | ITM | avg pos |",
        )
        self.assertIn("| 1 | leader | 2 | 0.5000 | 0.2500 | +33.3% | 13 | 1 | 1 | 2 | 2 | 2.0 |", lines)
        self.assertIn("| 2 | small | 2 | 0.2500 | 0.1250 | -33.3% | 4 | 0 | 1 | 1 | 1 | 8.5 |", lines)
        self.assertEqual(len(lines), 4)

    def test_population_summary_with_roi_normalizes_by_entry_share(self):
        summary = population_summary_with_roi(
            {
                "bot_a": {"entries": 3, "total_payout_pct": 6.0},
                "bot_b": {"entries": 1, "total_payout_pct": 2.0},
            }
        )

        self.assertAlmostEqual(summary["bot_a"]["roi"], 0.0)
        self.assertAlmostEqual(summary["bot_b"]["roi"], 0.0)

    def test_merge_population_summaries_recomputes_rates(self):
        reports = [
            {
                "population_summary": {
                    "bot_a": {
                        "entries": 2,
                        "wins": 1,
                        "top3": 1,
                        "final_table": 1,
                        "itm": 1,
                        "position_sum": 6.0,
                        "total_payout_pct": 0.5,
                        "score": 12.0,
                    }
                }
            },
            {
                "population_summary": {
                    "bot_a": {
                        "entries": 2,
                        "wins": 0,
                        "top3": 1,
                        "final_table": 2,
                        "itm": 2,
                        "position_sum": 10.0,
                        "total_payout_pct": 0.25,
                        "score": 5.0,
                    },
                    "bot_b": {
                        "entries": 1,
                        "wins": 1,
                        "top3": 1,
                        "final_table": 1,
                        "itm": 1,
                        "position_sum": 1.0,
                        "total_payout_pct": 0.4,
                        "score": 12.0,
                    },
                }
            },
        ]

        merged = merge_population_summaries(reports)

        self.assertEqual(merged["bot_a"]["entries"], 4)
        self.assertEqual(merged["bot_a"]["wins"], 1)
        self.assertAlmostEqual(merged["bot_a"]["win_rate"], 0.25)
        self.assertAlmostEqual(merged["bot_a"]["average_position"], 4.0)
        self.assertAlmostEqual(merged["bot_a"]["average_payout_pct"], 0.1875)
        self.assertEqual(merged["bot_b"]["entries"], 1)

    def test_report_completion_line_counts_reports_tournaments_and_failures(self):
        line = _report_completion_line(
            [
                {"mtt_count": 3, "simulation": {"tournaments": [{}, {}], "failures": ["x"]}},
                {"mtt_count": 2, "simulation": {"tournaments": [{}], "failures": []}},
            ]
        )

        self.assertEqual(line, "reports=2 configured_mtts=5 completed_mtts=3 recorded_failures=1")

    def test_summarize_candidate_actions_tracks_tool_raise_fold_response(self):
        summary = summarize_candidate_actions(
            [
                {
                    "type": "action",
                    "tournament_id": 1,
                    "table_id": 1,
                    "hand_id": 1,
                    "player": "hero_entry_0",
                    "action": "raise",
                    "amount": 900,
                    "street": "river",
                    "call_amount": 0,
                    "pot_size": 1000,
                    "tool_event": {"tool": "bluff_pressure", "decision": "force_raise"},
                },
                {
                    "type": "action",
                    "tournament_id": 1,
                    "table_id": 1,
                    "hand_id": 1,
                    "player": "villain_entry_0",
                    "action": "fold",
                    "amount": 0,
                    "street": "river",
                    "call_amount": 900,
                    "pot_size": 1900,
                },
            ],
            {
                "hero_entry_0": "hero",
                "villain_entry_0": "villain",
            },
        )

        counts = summary["hero"]["tool_response_counts"]["bluff_pressure"]
        amounts = summary["hero"]["tool_response_amounts"]["bluff_pressure"]
        rates = summary["hero"]["tool_response_rates"]["bluff_pressure"]
        self.assertEqual(counts["raises"], 1)
        self.assertEqual(counts["opponent_responses"], 1)
        self.assertEqual(counts["opponents_folded"], 1)
        self.assertEqual(counts["won_immediately"], 1)
        self.assertEqual(amounts["risk_amount_total"], 900)
        self.assertEqual(amounts["pot_before_total"], 1000)
        self.assertEqual(amounts["won_immediately_pot_total"], 1000)
        self.assertAlmostEqual(rates["opponent_fold_rate"], 1.0)
        self.assertAlmostEqual(rates["won_immediately_rate"], 1.0)
        self.assertAlmostEqual(rates["immediate_chip_delta_per_raise_estimate"], 1000.0)

    def test_summarize_candidate_actions_tracks_tool_raise_called_response(self):
        summary = summarize_candidate_actions(
            [
                {
                    "type": "action",
                    "tournament_id": 1,
                    "table_id": 1,
                    "hand_id": 2,
                    "player": "hero_entry_0",
                    "action": "raise",
                    "amount": 750,
                    "street": "river",
                    "call_amount": 0,
                    "pot_size": 1000,
                    "tool_event": {"tool": "bluff_pressure", "decision": "force_raise"},
                },
                {
                    "type": "action",
                    "tournament_id": 1,
                    "table_id": 1,
                    "hand_id": 2,
                    "player": "villain_entry_0",
                    "action": "call",
                    "amount": 750,
                    "street": "river",
                    "call_amount": 750,
                    "pot_size": 1750,
                },
            ],
            {
                "hero_entry_0": "hero",
                "villain_entry_0": "villain",
            },
        )

        counts = summary["hero"]["tool_response_counts"]["bluff_pressure"]
        amounts = summary["hero"]["tool_response_amounts"]["bluff_pressure"]
        rates = summary["hero"]["tool_response_rates"]["bluff_pressure"]
        self.assertEqual(counts["raises"], 1)
        self.assertEqual(counts["opponents_called"], 1)
        self.assertEqual(counts["called_or_raised"], 1)
        self.assertEqual(amounts["risk_amount_total"], 750)
        self.assertEqual(amounts["called_risk_amount_total"], 750)
        self.assertEqual(amounts["opponent_call_amount_total"], 750)
        self.assertAlmostEqual(rates["opponent_call_rate"], 1.0)
        self.assertAlmostEqual(rates["called_or_raised_rate"], 1.0)
        self.assertAlmostEqual(rates["immediate_chip_delta_per_raise_estimate"], -750.0)

    def test_merge_candidate_action_summaries_merges_tool_response_counts(self):
        first = {
            "hero": {
                "action_total": 1,
                "tool_response_counts": {
                    "bluff_pressure": {
                        "raises": 1,
                        "opponent_responses": 1,
                        "opponents_folded": 1,
                    }
                },
                "tool_response_amounts": {
                    "bluff_pressure": {
                        "risk_amount_total": 900,
                        "pot_before_total": 1000,
                        "won_immediately_pot_total": 1000,
                    }
                },
            }
        }
        second = {
            "hero": {
                "action_total": 1,
                "tool_response_counts": {
                    "bluff_pressure": {
                        "raises": 1,
                        "opponent_responses": 1,
                        "opponents_called": 1,
                    }
                },
                "tool_response_amounts": {
                    "bluff_pressure": {
                        "risk_amount_total": 750,
                        "pot_before_total": 1000,
                        "called_risk_amount_total": 750,
                    }
                },
            }
        }

        merged = merge_candidate_action_summaries([first, second])

        counts = merged["hero"]["tool_response_counts"]["bluff_pressure"]
        amounts = merged["hero"]["tool_response_amounts"]["bluff_pressure"]
        rates = merged["hero"]["tool_response_rates"]["bluff_pressure"]
        self.assertEqual(counts["raises"], 2)
        self.assertEqual(counts["opponent_responses"], 2)
        self.assertEqual(counts["opponents_folded"], 1)
        self.assertEqual(counts["opponents_called"], 1)
        self.assertEqual(amounts["risk_amount_total"], 1650)
        self.assertEqual(amounts["won_immediately_pot_total"], 1000)
        self.assertEqual(amounts["called_risk_amount_total"], 750)
        self.assertAlmostEqual(rates["opponent_fold_rate"], 0.5)
        self.assertAlmostEqual(rates["immediate_chip_delta_estimate"], 250.0)

    def test_summarize_candidate_actions_tracks_river_betting_range_bluff_share(self):
        summary = summarize_candidate_actions(
            [
                {
                    "type": "action",
                    "tournament_id": 1,
                    "table_id": 1,
                    "hand_id": 1,
                    "player": "hero_entry_0",
                    "action": "raise",
                    "amount": 900,
                    "street": "river",
                    "call_amount": 0,
                    "pot_size": 1000,
                    "tool_event": {"tool": "bluff_pressure", "decision": "force_raise"},
                },
                {
                    "type": "action",
                    "tournament_id": 1,
                    "table_id": 1,
                    "hand_id": 2,
                    "player": "hero_entry_0",
                    "action": "raise",
                    "amount": 500,
                    "street": "river",
                    "call_amount": 0,
                    "pot_size": 1000,
                },
                {
                    "type": "action",
                    "tournament_id": 1,
                    "table_id": 1,
                    "hand_id": 3,
                    "player": "hero_entry_0",
                    "action": "raise",
                    "amount": 700,
                    "street": "river",
                    "call_amount": 300,
                    "pot_size": 1000,
                    "tool_event": {"tool": "bluff_pressure", "decision": "force_raise"},
                },
            ],
            {"hero_entry_0": "hero"},
        )

        counts = summary["hero"]["river_betting_range_counts"]
        rates = summary["hero"]["river_betting_range_rates"]
        self.assertEqual(counts["no_facing_raise_total"], 2)
        self.assertEqual(counts["no_facing_bluff_raise"], 1)
        self.assertEqual(counts["no_facing_non_bluff_raise"], 1)
        self.assertEqual(counts["size_buckets"]["le_100"]["no_facing_bluff_raise"], 1)
        self.assertEqual(counts["size_buckets"]["le_50"]["no_facing_non_bluff_raise"], 1)
        self.assertAlmostEqual(rates["no_facing_bluff_share"], 0.5)
        self.assertAlmostEqual(rates["size_buckets"]["le_100"]["no_facing_bluff_share"], 1.0)

    def test_merge_candidate_action_summaries_merges_river_betting_range_counts(self):
        merged = merge_candidate_action_summaries(
            [
                {
                    "hero": {
                        "action_total": 1,
                        "river_betting_range_counts": {
                            "no_facing_raise_total": 2,
                            "no_facing_bluff_raise": 1,
                            "no_facing_non_bluff_raise": 1,
                            "size_buckets": {
                                "le_100": {
                                    "no_facing_raise_total": 1,
                                    "no_facing_bluff_raise": 1,
                                    "no_facing_non_bluff_raise": 0,
                                }
                            },
                        },
                    }
                },
                {
                    "hero": {
                        "action_total": 1,
                        "river_betting_range_counts": {
                            "no_facing_raise_total": 1,
                            "no_facing_bluff_raise": 1,
                            "no_facing_non_bluff_raise": 0,
                            "size_buckets": {
                                "le_100": {
                                    "no_facing_raise_total": 1,
                                    "no_facing_bluff_raise": 1,
                                    "no_facing_non_bluff_raise": 0,
                                }
                            },
                        },
                    }
                },
            ]
        )

        counts = merged["hero"]["river_betting_range_counts"]
        rates = merged["hero"]["river_betting_range_rates"]
        self.assertEqual(counts["no_facing_raise_total"], 3)
        self.assertEqual(counts["no_facing_bluff_raise"], 2)
        self.assertEqual(counts["no_facing_non_bluff_raise"], 1)
        self.assertEqual(counts["size_buckets"]["le_100"]["no_facing_raise_total"], 2)
        self.assertAlmostEqual(rates["no_facing_bluff_share"], 2 / 3)
        self.assertAlmostEqual(rates["size_buckets"]["le_100"]["no_facing_bluff_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
