import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from engine.phase28_equity_imitation_collection import CollectingTeacherBot, EquityImitationCollector, build_phase28_lineup


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

    def test_collector_can_write_reduced_v3_imitation_row(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            collector = EquityImitationCollector(
                {
                    "dataset_path": str(root / "dataset.jsonl"),
                    "manifest_path": str(root / "manifest.json"),
                    "reduced_observation_schema": "reduced_v3",
                },
                basemodel_root=Path(__file__).resolve().parents[2] / "poker-ai-basemodel",
            )
            state = {
                "hero_equity": 0.7,
                "pot_odds": 0.25,
                "pot_size": 100,
                "stack_size": 1000,
                "avg_table_stack": 500,
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
            manifest = collector.manifest({})
            rows = [json.loads(line) for line in Path(collector.dataset_path).read_text(encoding="utf-8").splitlines()]

            self.assertEqual(len(rows[0]["reduced_observation"]), 9)
            self.assertEqual(rows[0]["reduced_observation_schema"], "reduced_v3")
            self.assertEqual(manifest["observation_size"], 9)
            self.assertEqual(manifest["observation_fields"][-1], "stack_over_avg_table_stack")

    def test_collector_can_write_reduced_v4_imitation_row(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            collector = EquityImitationCollector(
                {
                    "dataset_path": str(root / "dataset.jsonl"),
                    "manifest_path": str(root / "manifest.json"),
                    "reduced_observation_schema": "reduced_v4",
                },
                basemodel_root=Path(__file__).resolve().parents[2] / "poker-ai-basemodel",
            )
            state = {
                "hero_equity": 0.7,
                "pot_odds": 0.25,
                "pot_size": 100,
                "stack_size": 1000,
                "avg_table_stack": 500,
                "players_left": 12,
                "starting_field": 30,
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
            manifest = collector.manifest({})
            rows = [json.loads(line) for line in Path(collector.dataset_path).read_text(encoding="utf-8").splitlines()]

            self.assertEqual(len(rows[0]["reduced_observation"]), 10)
            self.assertEqual(rows[0]["reduced_observation_schema"], "reduced_v4")
            self.assertEqual(rows[0]["reduced_observation"][9], 0.4)
            self.assertEqual(manifest["observation_size"], 10)
            self.assertEqual(manifest["observation_fields"][-1], "field_pct_remaining")

    def test_collector_can_write_reduced_v5_imitation_row(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            collector = EquityImitationCollector(
                {
                    "dataset_path": str(root / "dataset.jsonl"),
                    "manifest_path": str(root / "manifest.json"),
                    "reduced_observation_schema": "reduced_v5",
                },
                basemodel_root=Path(__file__).resolve().parents[2] / "poker-ai-basemodel",
            )
            state = {
                "hero_equity": 0.7,
                "pot_odds": 0.25,
                "pot_size": 100,
                "stack_size": 1000,
                "avg_table_stack": 500,
                "players_left": 12,
                "starting_field": 30,
                "paid_places": 7,
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
            manifest = collector.manifest({})
            rows = [json.loads(line) for line in Path(collector.dataset_path).read_text(encoding="utf-8").splitlines()]

            self.assertEqual(len(rows[0]["reduced_observation"]), 11)
            self.assertEqual(rows[0]["reduced_observation_schema"], "reduced_v5")
            self.assertAlmostEqual(rows[0]["reduced_observation"][10], 5.0 / 23.0)
            self.assertEqual(manifest["observation_size"], 11)
            self.assertEqual(manifest["observation_fields"][-1], "itm_distance")

    def test_phase31_teacher_lineup_collects_tight_and_aggressive_but_not_model(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            collector = EquityImitationCollector(
                {
                    "dataset_path": str(root / "dataset.jsonl"),
                    "manifest_path": str(root / "manifest.json"),
                },
                basemodel_root=Path(__file__).resolve().parents[2] / "poker-ai-basemodel",
            )

            bots = build_phase28_lineup(
                {
                    "lineup": {
                        "model": 0,
                        "tight_equity": 2,
                        "equity_aggressive": 1,
                        "random": 1,
                    }
                },
                collector,
            )

            collecting_bots = [bot for bot in bots if isinstance(bot, CollectingTeacherBot)]
            self.assertEqual([bot.bot_type for bot in collecting_bots], ["tight_equity", "tight_equity", "equity_aggressive"])
            self.assertFalse(any(bot.bot_type == "model" for bot in collecting_bots))


if __name__ == "__main__":
    unittest.main()
