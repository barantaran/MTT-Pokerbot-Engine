from __future__ import annotations

import unittest
from unittest.mock import patch

from players.tight_equity_bot import TightEquityBot


def state(**overrides):
    base = {
        "hole_cards": [0, 1],
        "board_cards": [],
        "pot_size": 100,
        "stack_size": 1000,
        "call_amount": 0,
        "min_raise": 20,
        "active_players": 3,
        "blinds": {"small": 10, "big": 20},
    }
    base.update(overrides)
    return base


class TightEquityBotTests(unittest.TestCase):
    def test_checks_when_no_call_amount_and_equity_is_not_strong(self):
        bot = TightEquityBot()

        with patch("players.tight_equity_bot.estimate_equity", return_value=0.35):
            action = bot.get_action(state())

        self.assertEqual(action, ("call", 0))

    def test_uses_provided_hero_equity_without_recomputing(self):
        bot = TightEquityBot()

        with patch("players.tight_equity_bot.estimate_equity", side_effect=AssertionError("should not compute")):
            action = bot.get_action(state(hero_equity=0.70))

        self.assertEqual(action, ("raise", 50))

    def test_folds_when_call_price_exceeds_equity_margin(self):
        bot = TightEquityBot()

        with patch("players.tight_equity_bot.estimate_equity", return_value=0.25):
            action = bot.get_action(state(pot_size=100, call_amount=80))

        self.assertEqual(action, ("fold", 0))

    def test_raises_only_with_strong_equity(self):
        bot = TightEquityBot()

        with patch("players.tight_equity_bot.estimate_equity", return_value=0.70):
            action = bot.get_action(state())

        self.assertEqual(action, ("raise", 50))

    def test_short_stack_strong_equity_jams(self):
        bot = TightEquityBot()

        with patch("players.tight_equity_bot.estimate_equity", return_value=0.70):
            action = bot.get_action(state(stack_size=120, call_amount=20, min_raise=20))

        self.assertEqual(action, ("raise", 100))

    def test_call_margin_shift_tightens_marginal_calls(self):
        bot = TightEquityBot(call_margin_shift=0.08)

        action = bot.get_action(state(pot_size=100, call_amount=40, hero_equity=0.37))

        self.assertEqual(action, ("fold", 0))

    def test_raise_threshold_shift_tightens_value_raises(self):
        bot = TightEquityBot(raise_threshold_shift=0.10)

        action = bot.get_action(state(hero_equity=0.70))

        self.assertEqual(action, ("call", 0))

    def test_preflop_vpip_gate_can_fold_marginal_free_position(self):
        bot = TightEquityBot(preflop_vpip_gate_shift=0.08)

        action = bot.get_action(state(hero_equity=0.25, position="HJ"))

        self.assertEqual(action, ("fold", 0))

    def test_preflop_vpip_gate_keeps_big_blind_check_available(self):
        bot = TightEquityBot(preflop_vpip_gate_shift=0.20)

        action = bot.get_action(state(hero_equity=0.10, position="BB", call_amount=0))

        self.assertEqual(action, ("call", 0))


if __name__ == "__main__":
    unittest.main()
