from __future__ import annotations

import unittest
from unittest.mock import patch

from players.aggressive_bot import AggressiveBot


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


class AggressiveBotTests(unittest.TestCase):
    def test_uses_provided_hero_equity_without_recomputing(self):
        bot = AggressiveBot()

        with patch("players.aggressive_bot.estimate_equity", side_effect=AssertionError("should not compute")), patch(
            "players.aggressive_bot.random.random", return_value=0.99
        ):
            action = bot.get_action(state(hero_equity=0.44))

        self.assertEqual(action, ("raise", 70))


if __name__ == "__main__":
    unittest.main()
