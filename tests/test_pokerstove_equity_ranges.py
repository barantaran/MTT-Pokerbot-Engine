from __future__ import annotations

import unittest

from treys import Card

from engine.pokerstove_equity import (
    _range_filtered_hands,
    clear_equity_cache,
    equity_cache_info,
    estimate_equity,
    normalize_range_pct,
    player_range,
)


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

    def test_five_bet_plus_uses_tight_range(self):
        self.assertEqual(
            normalize_range_pct(preflop_spot_type="five_bet_plus", use_preflop_spot_range=True),
            0.05,
        )

    def test_adaptive_profile_uses_softer_default_ranges(self):
        self.assertGreater(
            normalize_range_pct(
                preflop_spot_type="three_bet",
                use_preflop_spot_range=True,
                range_profile="adaptive",
            ),
            normalize_range_pct(preflop_spot_type="three_bet", use_preflop_spot_range=True),
        )
        self.assertGreater(
            normalize_range_pct(
                preflop_spot_type="all_in_pressure",
                use_preflop_spot_range=True,
                range_profile="adaptive",
            ),
            normalize_range_pct(preflop_spot_type="all_in_pressure", use_preflop_spot_range=True),
        )

    def test_adaptive_profile_widens_short_stack_pressure_ranges(self):
        base = normalize_range_pct(
            preflop_spot_type="all_in_pressure",
            use_preflop_spot_range=True,
            range_profile="adaptive",
            stack_bb=30,
        )
        short = normalize_range_pct(
            preflop_spot_type="all_in_pressure",
            use_preflop_spot_range=True,
            range_profile="adaptive",
            stack_bb=7,
        )

        self.assertGreater(short, base)

    def test_adaptive_profile_tightens_bubble_pressure_ranges(self):
        base = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="adaptive",
            players_left=80,
            starting_field=200,
            paid_places=30,
            itm_distance=0.50,
        )
        bubble = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="adaptive",
            players_left=31,
            starting_field=200,
            paid_places=30,
            itm_distance=0.01,
        )

        self.assertLess(bubble, base)

    def test_player_range_maps_public_rates_to_spot_ranges(self):
        nit = player_range(vpip=0.12, pfr=0.07, three_bet=0.02)
        lag = player_range(vpip=0.42, pfr=0.28, three_bet=0.13)

        self.assertLess(nit["srp"], lag["srp"])
        self.assertLess(nit["three_bet"], lag["three_bet"])
        self.assertLess(nit["all_in_pressure"], lag["all_in_pressure"])

    def test_player_range_keeps_position_neutral(self):
        same_stats = {"vpip": 0.28, "pfr": 0.18, "three_bet": 0.08}

        utg = player_range(position="UTG", **same_stats)
        button = player_range(position="BTN", **same_stats)

        self.assertEqual(utg["srp"], button["srp"])
        self.assertEqual(utg["three_bet"], button["three_bet"])

    def test_player_profile_keeps_aggressor_position_neutral(self):
        opponent = {"vpip": 0.28, "pfr": 0.18, "three_bet_rate": 0.08, "sample_quality": 1.0}

        utg = normalize_range_pct(
            preflop_spot_type="srp",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=opponent,
            opponent_position="UTG",
        )
        button = normalize_range_pct(
            preflop_spot_type="srp",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=opponent,
            opponent_position="BTN",
        )

        self.assertEqual(utg, button)

    def test_player_profile_blends_public_rates_by_sample_quality(self):
        tight_opponent = {"vpip": 0.12, "pfr": 0.06, "three_bet_rate": 0.02, "sample_quality": 1.0}
        loose_opponent = {"vpip": 0.55, "pfr": 0.36, "three_bet_rate": 0.18, "sample_quality": 1.0}

        base = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats={"sample_quality": 0.0},
        )
        tight = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=tight_opponent,
        )
        loose = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=loose_opponent,
        )

        self.assertLess(tight, base)
        self.assertGreater(loose, base)

    def test_player_profile_sampling_can_be_disabled(self):
        loose_opponent = {"vpip": 0.55, "pfr": 0.36, "three_bet_rate": 0.18, "sample_quality": 0.20}

        sampled = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=loose_opponent,
            player_range_sampling=True,
        )
        unsampled = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=loose_opponent,
            player_range_sampling=False,
        )
        full_sample = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats={**loose_opponent, "sample_quality": 1.0},
            player_range_sampling=True,
        )

        self.assertGreater(unsampled, sampled)
        self.assertEqual(unsampled, full_sample)

    def test_player_profile_sample_config_delays_pressure_reads(self):
        config = {"pressure_start_hands": 40, "pressure_full_hands": 120}
        loose_pressure = {
            "vpip": 0.55,
            "pfr": 0.36,
            "three_bet_rate": 0.18,
            "sample_quality": 1.0,
            "player_hands_observed": 20,
        }

        base = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats={"sample_quality": 0.0},
        )
        guarded = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=loose_pressure,
            player_range_sample_config=config,
        )
        unguarded = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats={**loose_pressure, "player_hands_observed": 120},
            player_range_sample_config=config,
        )

        self.assertEqual(guarded, base)
        self.assertGreater(unguarded, guarded)

    def test_player_profile_sample_config_trusts_tight_reads_earlier_than_loose_reads(self):
        config = {
            "tight_start_hands": 5,
            "tight_full_hands": 20,
            "loose_start_hands": 25,
            "loose_full_hands": 75,
        }
        tight_opponent = {
            "vpip": 0.12,
            "pfr": 0.06,
            "three_bet_rate": 0.02,
            "sample_quality": 1.0,
            "player_hands_observed": 20,
        }
        loose_opponent = {
            "vpip": 0.70,
            "pfr": 0.55,
            "three_bet_rate": 0.18,
            "sample_quality": 1.0,
            "player_hands_observed": 20,
        }

        base = normalize_range_pct(
            preflop_spot_type="srp",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats={"sample_quality": 0.0},
        )
        tight = normalize_range_pct(
            preflop_spot_type="srp",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=tight_opponent,
            player_range_sample_config=config,
        )
        loose = normalize_range_pct(
            preflop_spot_type="srp",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=loose_opponent,
            player_range_sample_config=config,
        )

        self.assertLess(tight, base)
        self.assertEqual(loose, base)

    def test_player_profile_widens_pressure_range_against_short_stack_aggressor(self):
        opponent = {"vpip": 0.22, "pfr": 0.14, "three_bet_rate": 0.06, "sample_quality": 1.0}

        normal_stack = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=opponent,
            opponent_stack_bb=35,
            stack_bb=35,
        )
        short_stack = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=opponent,
            opponent_stack_bb=8,
            stack_bb=35,
        )

        self.assertGreater(short_stack, normal_stack)

    def test_player_profile_widens_pressure_range_against_covering_late_aggressor(self):
        opponent = {"vpip": 0.22, "pfr": 0.14, "three_bet_rate": 0.06, "sample_quality": 1.0}

        normal_stack = normalize_range_pct(
            preflop_spot_type="srp",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=opponent,
            opponent_stack_bb=35,
            stack_bb=35,
            players_left=80,
            starting_field=100,
            paid_places=26,
            itm_distance=0.7,
        )
        covering_bubble_stack = normalize_range_pct(
            preflop_spot_type="srp",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats=opponent,
            opponent_stack_bb=70,
            stack_bb=35,
            players_left=27,
            starting_field=100,
            paid_places=26,
            itm_distance=0.01,
        )

        self.assertGreater(covering_bubble_stack, normal_stack)

    def test_player_profile_uses_population_range_without_sample(self):
        self.assertEqual(
            normalize_range_pct(
                preflop_spot_type="srp",
                use_preflop_spot_range=True,
                range_profile="player",
                opponent_stats={"vpip": 0.55, "pfr": 0.36, "three_bet_rate": 0.18, "sample_quality": 0.0},
            ),
            normalize_range_pct(
                preflop_spot_type="srp",
                use_preflop_spot_range=True,
                range_profile="adaptive",
            ),
        )

    def test_player_profile_ignores_table_rates_for_personal_range(self):
        loose_table = {"vpip": 0.55, "pfr": 0.36, "three_bet_rate": 0.18, "sample_quality": 1.0}

        self.assertEqual(
            normalize_range_pct(
                preflop_spot_type="three_bet",
                use_preflop_spot_range=True,
                range_profile="player",
                table_stats=loose_table,
                opponent_stats={"sample_quality": 0.0},
            ),
            normalize_range_pct(
                preflop_spot_type="three_bet",
                use_preflop_spot_range=True,
                range_profile="adaptive",
            ),
        )

    def test_range_influence_can_soften_selected_pressure_spots(self):
        full = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats={"vpip": 0.12, "pfr": 0.06, "three_bet_rate": 0.02, "sample_quality": 1.0},
            range_influence={"default": 1.0, "three_bet": 1.0},
        )
        softened = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            range_profile="player",
            opponent_stats={"vpip": 0.12, "pfr": 0.06, "three_bet_rate": 0.02, "sample_quality": 1.0},
            range_influence={"default": 1.0, "three_bet": 0.25},
        )
        unopened = normalize_range_pct(
            preflop_spot_type="unknown",
            use_preflop_spot_range=True,
            range_profile="player",
            range_influence={"default": 1.0, "three_bet": 0.25},
        )

        self.assertGreater(softened, full)
        self.assertLess(softened, 1.0)
        self.assertEqual(unopened, 1.0)

    def test_zero_range_influence_matches_no_range_for_derived_ranges(self):
        self.assertEqual(
            normalize_range_pct(
                preflop_spot_type="all_in_pressure",
                use_preflop_spot_range=True,
                range_profile="player",
                opponent_stats={"vpip": 0.12, "pfr": 0.06, "three_bet_rate": 0.02, "sample_quality": 1.0},
                range_influence=0.0,
            ),
            1.0,
        )

    def test_preflop_spot_range_is_opt_in(self):
        loose_stats = {"vpip": 0.55, "pfr": 0.36, "three_bet_rate": 0.18, "sample_quality": 1.0}
        self.assertEqual(normalize_range_pct(preflop_spot_type="three_bet", table_stats=loose_stats), 1.0)

    def test_table_stats_adjust_range_when_enabled(self):
        tight_stats = {"vpip": 0.12, "pfr": 0.06, "three_bet_rate": 0.02, "sample_quality": 1.0}
        loose_stats = {"vpip": 0.55, "pfr": 0.36, "three_bet_rate": 0.18, "sample_quality": 1.0}

        base = normalize_range_pct(preflop_spot_type="three_bet", use_preflop_spot_range=True)
        tight = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            table_stats=tight_stats,
        )
        loose = normalize_range_pct(
            preflop_spot_type="three_bet",
            use_preflop_spot_range=True,
            table_stats=loose_stats,
        )

        self.assertLess(tight, base)
        self.assertGreater(loose, base)

    def test_low_sample_table_stats_do_not_adjust_range(self):
        loose_stats = {"vpip": 0.55, "pfr": 0.36, "three_bet_rate": 0.18, "sample_quality": 0.0}

        self.assertEqual(
            normalize_range_pct(
                preflop_spot_type="three_bet",
                use_preflop_spot_range=True,
                table_stats=loose_stats,
            ),
            normalize_range_pct(preflop_spot_type="three_bet", use_preflop_spot_range=True),
        )

    def test_missing_table_stat_rates_are_neutral(self):
        self.assertEqual(
            normalize_range_pct(
                preflop_spot_type="three_bet",
                use_preflop_spot_range=True,
                table_stats={"sample_quality": 1.0},
            ),
            normalize_range_pct(preflop_spot_type="three_bet", use_preflop_spot_range=True),
        )


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
