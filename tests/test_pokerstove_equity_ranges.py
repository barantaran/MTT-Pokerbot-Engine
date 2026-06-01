from __future__ import annotations

import unittest

from treys import Card

from engine.pokerstove_equity import _range_filtered_hands, clear_equity_cache, equity_cache_info, estimate_equity, normalize_range_pct


class PokerstoveEquityRangeTests(unittest.TestCase):
    def test_preflop_spot_maps_to_stronger_range_percent(self):
        self.assertEqual(normalize_range_pct(preflop_spot_type="unknown", use_preflop_spot_range=True), 1.0)
        self.assertLess(
            normalize_range_pct(preflop_spot_type="three_bet", use_preflop_spot_range=True),
            normalize_range_pct(preflop_spot_type="srp", use_preflop_spot_range=True),
        )
        self.assertLess(
            normalize_range_pct(preflop_spot_type="all_in_pressure", use_preflop_spot_range=True),
            normalize_range_pct(preflop_spot_type="three_bet", use_preflop_spot_range=True),
        )

    def test_preflop_spot_range_is_opt_in(self):
        self.assertEqual(normalize_range_pct(preflop_spot_type="three_bet"), 1.0)

    def test_explicit_range_percent_accepts_percent_or_fraction(self):
        self.assertEqual(normalize_range_pct(opponent_range_pct=25), 0.25)
        self.assertEqual(normalize_range_pct(opponent_range_pct=0.25), 0.25)

    def test_range_filter_keeps_strongest_starting_hands(self):
        available_cards = [f"{rank}{suit}" for rank in "23456789TJQKA" for suit in "cdhs"]

        hands = _range_filtered_hands(available_cards, 0.01)

        self.assertIn(("Ac", "Ad"), hands)
        self.assertNotIn(("2c", "7d"), hands)

    def test_equity_estimates_are_cached_by_public_card_context(self):
        clear_equity_cache()
        hole_cards = [Card.new("As"), Card.new("Kd")]

        estimate_equity(hole_cards, [], 6, iterations=3, opponent_range_pct=35)
        before = equity_cache_info()["equity"]
        estimate_equity(hole_cards, [], 6, iterations=3, opponent_range_pct=35)
        after = equity_cache_info()["equity"]

        self.assertEqual(before["misses"], 1)
        self.assertEqual(after["hits"], before["hits"] + 1)


if __name__ == "__main__":
    unittest.main()
