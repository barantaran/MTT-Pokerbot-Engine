from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.action_stats_by_bot_type import format_table, summarize_event_log, summarize_path


def write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class ActionStatsByBotTypeTests(unittest.TestCase):
    def test_summarize_event_log_maps_actions_from_embedded_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = write_json(
                Path(tmp) / "sim_1.json",
                {
                    "simulation_id": 1,
                    "results": [
                        {"name": "Phase23ModelBot_1", "bot_class": "BaselineModelEngineBot"},
                        {"name": "RandomBot_1", "bot_class": "RandomBot"},
                    ],
                    "events": [
                        {"type": "action", "player": "Phase23ModelBot_1", "action": "raise"},
                        {"type": "action", "player": "Phase23ModelBot_1", "action": "call"},
                        {"type": "action", "player": "RandomBot_1", "action": "fold"},
                        {"type": "knockout", "player": "RandomBot_1"},
                    ],
                },
            )

            summary = summarize_event_log(log_path)

        self.assertEqual(summary["action_event_count"], 3)
        self.assertEqual(summary["action_counts_by_bot_type"]["ModelBot"], {"raise": 1, "call": 1})
        self.assertEqual(summary["action_counts_by_bot_type"]["RandomBot"], {"fold": 1})

    def test_summarize_path_joins_phase_run_events_with_tournament_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "tournament_results.json",
                [
                    {
                        "tournament_id": 1,
                        "results": [
                            {"name": "EquityAggressiveBot_1", "bot_class": "AggressiveBot"},
                            {"name": "AggressiveNoEquityBot_1", "bot_class": "AggressiveNoEquityBot"},
                            {"name": "TightEquityBot_1", "bot_class": "TightEquityBot"},
                            {"name": "CallBot_1", "bot_class": "CallBot"},
                        ],
                    }
                ],
            )
            write_json(
                root / "events" / "tournament_1_events.json",
                [
                    {"type": "action", "player": "EquityAggressiveBot_1", "action": "raise"},
                    {"type": "action", "player": "AggressiveNoEquityBot_1", "action": "fold"},
                    {"type": "action", "player": "TightEquityBot_1", "action": "call"},
                    {"type": "action", "player": "CallBot_1", "action": "call"},
                    {"type": "action", "player": "CallBot_1", "action": "check"},
                ],
            )

            summary = summarize_path(root)

        self.assertEqual(summary["event_log_count"], 1)
        self.assertEqual(summary["action_event_count"], 5)
        self.assertEqual(summary["action_counts_by_bot_type"]["EquityAggressiveBot"], {"raise": 1})
        self.assertEqual(summary["action_counts_by_bot_type"]["AggressiveNoEquityBot"], {"fold": 1})
        self.assertEqual(summary["action_counts_by_bot_type"]["TightEquityBot"], {"call": 1})
        self.assertEqual(summary["action_counts_by_bot_type"]["CallBot"], {"call": 1, "check": 1})

    def test_format_table_includes_bot_type_totals_and_actions(self):
        table = format_table(
            {
                "source_path": "run",
                "event_log_count": 1,
                "action_event_count": 2,
                "action_counts_by_bot_type": {"RandomBot": {"fold": 1, "raise": 1}},
            }
        )

        self.assertIn("RandomBot", table)
        self.assertIn("fold", table)
        self.assertIn("raise", table)


if __name__ == "__main__":
    unittest.main()
