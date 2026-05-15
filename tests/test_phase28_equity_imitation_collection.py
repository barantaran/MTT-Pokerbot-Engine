import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from engine.phase28_equity_imitation_collection import EquityImitationCollector


class Phase28EquityImitationCollectionTests(unittest.TestCase):
    def test_collector_writes_reduced_imitation_row(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            collector = EquityImitationCollector(
                {
                    "dataset_path": str(root / "dataset.jsonl"),
                    "manifest_path": str(root / "manifest.json"),
                },
                basemodel_root=Path(__file__).resolve().parents[2] / "poker-ai-basemodel",
            )
            state = {
                "hero_equity": 0.7,
                "pot_odds": 0.25,
                "pot_size": 100,
                "stack_size": 1000,
                "call_amount": 0,
                "min_raise": 20,
                "blinds": {"small": 10, "big": 20},
                "active_players": 4,
                "board_cards": [],
                "position": "BTN",
                "player_id": "EquityBot_1",
                "table_id": 1,
                "hand_id": 1,
                "tournament_id": 1,
            }

            collector.record_decision(game_state=state, action=("raise", 100))
            rows = [json.loads(line) for line in Path(collector.dataset_path).read_text(encoding="utf-8").splitlines()]

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["phase"], 28)
            self.assertEqual(len(rows[0]["reduced_observation"]), 8)
            self.assertEqual(len(rows[0]["legal_action_mask"]), 9)
            self.assertFalse(rows[0]["fallback_used"])
            self.assertTrue(collector.validation_summary()["valid"])


if __name__ == "__main__":
    unittest.main()
