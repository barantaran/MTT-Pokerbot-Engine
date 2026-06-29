from __future__ import annotations

import unittest
from unittest.mock import patch

from players.range_policy_bot import RangePolicyBot


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
        "preflop_spot_type": "srp",
    }
    base.update(overrides)
    return base


class RangePolicyBotTests(unittest.TestCase):
    def test_uses_range_aware_equity_by_default(self):
        bot = RangePolicyBot(variant="test")
        table_stats = {"sample_quality": 1.0, "vpip": 0.40, "pfr": 0.24, "three_bet_rate": 0.10}

        with patch("players.range_policy_bot.estimate_equity", return_value=0.40) as equity:
            action = bot.get_action(state(call_amount=40, table_stats=table_stats))

        self.assertEqual(action, ("call", 0))
        self.assertTrue(equity.call_args.kwargs["use_preflop_spot_range"])
        self.assertEqual(equity.call_args.kwargs["preflop_spot_type"], "srp")
        self.assertEqual(equity.call_args.kwargs["table_stats"], table_stats)

    def test_uses_provided_hero_equity_without_recomputing(self):
        bot = RangePolicyBot(raise_threshold=0.60)

        with patch("players.range_policy_bot.estimate_equity", side_effect=AssertionError("should not compute")):
            action = bot.get_action(state(hero_equity=0.70))

        self.assertEqual(action, ("raise", 50))

    def test_folds_when_price_is_not_justified(self):
        bot = RangePolicyBot(call_margin=0.04)

        action = bot.get_action(state(call_amount=80, hero_equity=0.20))

        self.assertEqual(action, ("fold", 0))


if __name__ == "__main__":
    unittest.main()
