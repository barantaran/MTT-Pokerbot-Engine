from __future__ import annotations

import unittest
from unittest.mock import patch

from players.tournament_equity_bot import TournamentEquityBot


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
        "players_left": 50,
        "starting_field": 200,
        "paid_places": 30,
        "next_prize_gain_pct": 0.0,
        "preflop_spot_type": "unknown",
        "position": "HJ",
    }
    base.update(overrides)
    return base


class TournamentEquityBotTests(unittest.TestCase):
    def test_uses_range_aware_equity_by_default(self):
        bot = TournamentEquityBot()

        with patch("players.tournament_equity_bot.estimate_equity", return_value=0.50) as equity:
            action = bot.get_action(state(call_amount=40, preflop_spot_type="three_bet"))

        self.assertEqual(action, ("call", 0))
        self.assertTrue(equity.call_args.kwargs["use_preflop_spot_range"])
        self.assertEqual(equity.call_args.kwargs["preflop_spot_type"], "three_bet")

    def test_folds_bad_priced_call(self):
        bot = TournamentEquityBot()

        action = bot.get_action(state(call_amount=80, hero_equity=0.20))

        self.assertEqual(action, ("fold", 0))

    def test_deep_stack_value_raise_is_not_all_in(self):
        bot = TournamentEquityBot()

        action = bot.get_action(state(hero_equity=0.80, stack_size=3000, call_amount=0))

        self.assertEqual(action, ("raise", 46))

    def test_button_opens_wider_than_early_position(self):
        bot = TournamentEquityBot()

        button_action = bot.get_action(state(hero_equity=0.51, position="BTN"))
        early_action = bot.get_action(state(hero_equity=0.51, position="UTG"))

        self.assertEqual(button_action, ("raise", 44))
        self.assertEqual(early_action, ("call", 0))

    def test_early_position_uses_larger_open_size(self):
        bot = TournamentEquityBot()

        action = bot.get_action(state(hero_equity=0.70, position="UTG"))

        self.assertEqual(action, ("raise", 50))

    def test_short_stack_strong_hand_jams(self):
        bot = TournamentEquityBot()

        action = bot.get_action(state(hero_equity=0.70, stack_size=160, call_amount=20, min_raise=20))

        self.assertEqual(action, ("raise", 140))

    def test_bubble_pressure_folds_marginal_call(self):
        bot = TournamentEquityBot()

        action = bot.get_action(
            state(
                pot_size=100,
                call_amount=50,
                hero_equity=0.39,
                players_left=31,
                paid_places=30,
                next_prize_gain_pct=0.02,
            )
        )

        self.assertEqual(action, ("fold", 0))

    def test_committed_strong_hand_can_raise_all_in(self):
        bot = TournamentEquityBot()

        action = bot.get_action(state(hero_equity=0.95, stack_size=1000, call_amount=750, min_raise=20))

        self.assertEqual(action, ("raise", 250))


if __name__ == "__main__":
    unittest.main()
