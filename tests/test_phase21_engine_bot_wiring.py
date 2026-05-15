from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from engine.baseline_model_bot import BaselineModelEngineBot, is_engine_action, validate_visible_state
from engine.phase21_engine_bot_wiring import resolve_phase20_report, run_phase21_engine_bot_wiring


class DummyModel:
    def eval(self):
        return self


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def base_state(**overrides):
    state = {
        "hole_cards": [],
        "board_cards": [],
        "pot_size": 100,
        "stack_size": 1000,
        "call_amount": 0,
        "min_raise": 20,
        "blinds": {"small": 10, "big": 20},
        "active_players": 4,
        "position": "BTN",
    }
    state.update(overrides)
    return state


class Phase21EngineBotWiringTests(unittest.TestCase):
    def test_resolve_phase20_report_selects_latest_accepted_allowed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = write_json(
                root / "old" / "promotion_evaluation_report.json",
                {"phase20_promotion_status": "rejected", "mtt_engine_bot_simulation_allowed": False},
            )
            latest = write_json(
                root / "latest" / "promotion_evaluation_report.json",
                {"phase20_promotion_status": "accepted", "mtt_engine_bot_simulation_allowed": True},
            )
            old.touch()
            latest.touch()

            path, report, failure = resolve_phase20_report(
                {"source_phase20_report_glob": str(root / "*" / "promotion_evaluation_report.json")}
            )

        self.assertEqual(path, str(latest))
        self.assertEqual(report["phase20_promotion_status"], "accepted")
        self.assertEqual(failure, "")

    def test_resolve_phase20_report_rejects_disallowed_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(
                root / "promotion_evaluation_report.json",
                {"phase20_promotion_status": "accepted", "mtt_engine_bot_simulation_allowed": False},
            )

            _path, _report, failure = resolve_phase20_report(
                {"source_phase20_report_glob": str(root / "promotion_evaluation_report.json")}
            )

        self.assertIn("no accepted Phase 20 report matched", failure)

    def test_visible_state_rejects_hidden_card_fields(self):
        ok, failures = validate_visible_state(base_state(opponent_hole_cards=[[1, 2]]))

        self.assertFalse(ok)
        self.assertTrue(failures)

    def test_adapter_returns_engine_action_with_injected_model(self):
        bot = BaselineModelEngineBot(
            None,
            basemodel_root=Path(__file__).resolve().parents[2] / "poker-ai-basemodel",
            model=DummyModel(),
            require_checkpoint=False,
            equity_source="constant",
        )

        action = bot.get_action(base_state())

        self.assertTrue(is_engine_action(action))
        self.assertEqual(action, ("call", 0))

    def test_adapter_hidden_card_state_uses_conservative_fallback(self):
        bot = BaselineModelEngineBot(
            None,
            basemodel_root=Path(__file__).resolve().parents[2] / "poker-ai-basemodel",
            model=DummyModel(),
            require_checkpoint=False,
            equity_source="constant",
        )

        action = bot.get_action(base_state(call_amount=20, opponent_hole_cards=[[1, 2]]))

        self.assertEqual(action, ("call", 0))

    @patch("engine.baseline_model_bot.load_checkpoint_model", return_value=DummyModel())
    def test_run_phase21_accepts_valid_prerequisites_and_probes(self, _load_model):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "basemodel" / "runs" / "phase19" / "latest.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            phase20 = write_json(
                root / "phase20" / "promotion_evaluation_report.json",
                {
                    "phase20_promotion_status": "accepted",
                    "phase20_promotion_failures": [],
                    "mtt_engine_bot_simulation_allowed": True,
                    "candidate_checkpoint_path": "runs/phase19/latest.pt",
                    "promoted_checkpoint_path": "runs/phase19/latest.pt",
                    "score_delta": 0.0,
                    "roi_delta": 0.0,
                },
            )
            config = {
                "source_phase20_report_path": str(phase20),
                "basemodel_root": str(root / "basemodel"),
                "artifact_root": str(root / "phase21"),
                "engine_wiring_report_path": str(root / "phase21" / "engine_wiring_report.json"),
                "equity_source": "constant",
                "decision_timeout_ms": 500,
                "smoke_states": [{"name": "check", "game_state": base_state()}],
            }

            report = run_phase21_engine_bot_wiring(config)

        self.assertEqual(report["phase21_engine_wiring_status"], "accepted")
        self.assertTrue(report["small_mtt_engine_simulation_allowed"])
        self.assertEqual(report["next_phase_recommendation"], "launch_phase22_small_mtt_engine_simulation")

    def test_run_phase21_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            phase20 = write_json(
                root / "phase20" / "promotion_evaluation_report.json",
                {
                    "phase20_promotion_status": "accepted",
                    "phase20_promotion_failures": [],
                    "mtt_engine_bot_simulation_allowed": True,
                    "promoted_checkpoint_path": "runs/missing/latest.pt",
                },
            )
            config = {
                "source_phase20_report_path": str(phase20),
                "basemodel_root": str(root / "basemodel"),
                "artifact_root": str(root / "phase21"),
                "engine_wiring_report_path": str(root / "phase21" / "engine_wiring_report.json"),
                "smoke_states": [{"name": "check", "game_state": base_state()}],
            }

            report = run_phase21_engine_bot_wiring(config)

        self.assertEqual(report["phase21_engine_wiring_status"], "rejected")
        self.assertFalse(report["small_mtt_engine_simulation_allowed"])
        self.assertIn("promoted checkpoint does not exist", " ".join(report["phase21_engine_wiring_failures"]))


if __name__ == "__main__":
    unittest.main()
