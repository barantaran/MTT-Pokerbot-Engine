from __future__ import annotations

import unittest
from unittest.mock import patch

from players.noisy_equity_bot import NoisyEquityBot


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


class NoisyEquityBotTests(unittest.TestCase):
    def test_returns_tight_action_without_noise(self):
        bot = NoisyEquityBot()

        with patch("players.tight_equity_bot.estimate_equity", return_value=0.70), patch(
            "players.noisy_equity_bot.random.random", return_value=0.99
        ):
            action = bot.get_action(state())

        self.assertEqual(action, ("raise", 50))

    def test_adjacent_noise_turns_fold_into_loose_call(self):
        bot = NoisyEquityBot(adjacent_noise_rate=0.15, mistake_rate=0.05)

        with patch("players.tight_equity_bot.estimate_equity", return_value=0.10), patch(
            "players.noisy_equity_bot.random.random", return_value=0.10
        ):
            action = bot.get_action(state(call_amount=80))

        self.assertEqual(action, ("call", 0))

    def test_bigger_mistake_can_turn_call_into_raise(self):
        bot = NoisyEquityBot(adjacent_noise_rate=0.15, mistake_rate=0.05)

        with patch("players.tight_equity_bot.estimate_equity", return_value=0.35), patch(
            "players.noisy_equity_bot.random.random", return_value=0.01
        ):
            action = bot.get_action(state())

        self.assertEqual(action, ("raise", 125))


if __name__ == "__main__":
    unittest.main()
