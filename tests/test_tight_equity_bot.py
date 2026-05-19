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


if __name__ == "__main__":
    unittest.main()
