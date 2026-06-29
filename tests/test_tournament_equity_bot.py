from __future__ import annotations

import unittest
from unittest.mock import patch

from players.tournament_equity_bot import (
    AdaptiveTournamentICMEquityBot,
    ButtonStealTournamentICMEquityBot,
    ConfiguredTournamentEquityBot,
    TournamentEquityBot,
    TournamentEquityBotV2,
    TournamentICMEquityBot,
)


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
        self.assertEqual(equity.call_args.kwargs["range_profile"], "legacy")

    def test_configured_range_profile_is_passed_to_equity_estimator(self):
        bot = TournamentEquityBot(range_profile="adaptive")

        with patch("players.tournament_equity_bot.estimate_equity", return_value=0.50) as equity:
            bot.get_action(
                state(
                    call_amount=40,
                    preflop_spot_type="three_bet",
                    position="BB",
                    opponent_position="BTN",
                    opponent_stack_bb=12,
                    players_left=31,
                    paid_places=30,
                    itm_distance=0.01,
                )
            )

        self.assertEqual(equity.call_args.kwargs["range_profile"], "adaptive")
        self.assertEqual(equity.call_args.kwargs["position"], "BB")
        self.assertEqual(equity.call_args.kwargs["opponent_position"], "BTN")
        self.assertEqual(equity.call_args.kwargs["opponent_stack_bb"], 12)
        self.assertEqual(equity.call_args.kwargs["players_left"], 31)

    def test_configured_range_influence_is_passed_to_equity_estimator(self):
        range_influence = {"default": 1.0, "three_bet": 0.30, "all_in_pressure": 0.40}
        bot = ConfiguredTournamentEquityBot(range_profile="player", range_influence=range_influence)

        with patch("players.tournament_equity_bot.estimate_equity", return_value=0.50) as equity:
            bot.get_action(state(call_amount=40, preflop_spot_type="three_bet"))

        self.assertEqual(equity.call_args.kwargs["range_profile"], "player")
        self.assertEqual(equity.call_args.kwargs["range_influence"], range_influence)

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

    def test_v2_tightens_only_marginal_preflop_reraise(self):
        old_bot = TournamentEquityBot()
        new_bot = TournamentEquityBotV2()

        old_action = old_bot.get_action(state(hero_equity=0.56, call_amount=40, preflop_spot_type="srp"))
        new_action = new_bot.get_action(state(hero_equity=0.56, call_amount=40, preflop_spot_type="srp"))

        self.assertEqual(old_action, ("raise", 60))
        self.assertEqual(new_action, ("call", 0))

    def test_v2_keeps_open_raise_threshold_unchanged(self):
        old_bot = TournamentEquityBot()
        new_bot = TournamentEquityBotV2()

        old_action = old_bot.get_action(state(hero_equity=0.51, position="BTN", call_amount=0))
        new_action = new_bot.get_action(state(hero_equity=0.51, position="BTN", call_amount=0))

        self.assertEqual(old_action, ("raise", 44))
        self.assertEqual(new_action, ("raise", 44))

    def test_icm_variant_keeps_tournament_value_sizing(self):
        bot = TournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(hero_equity=0.80, stack_size=3000, call_amount=0))

        self.assertEqual(action, ("raise", 46))

    def test_base_tournament_bot_does_not_bluff_minor_raise_miss(self):
        bot = TournamentEquityBot()

        action = bot.get_action(state(hero_equity=0.50, position="HJ", call_amount=0))

        self.assertEqual(action, ("call", 0))

    def test_icm_variant_does_not_bluff_preflop_minor_open_raise_miss(self):
        bot = TournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(hero_equity=0.50, position="HJ", call_amount=0))

        self.assertEqual(action, ("call", 0))

    def test_icm_variant_does_not_bluff_large_equity_miss(self):
        bot = TournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            board_cards=[2, 3, 4, 5, 6],
            hero_equity=0.55,
            active_players=2,
            call_amount=0,
        ))

        self.assertEqual(action, ("call", 0))

    def test_icm_variant_does_not_bluff_river_minor_bet_miss_heads_up(self):
        bot = TournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            board_cards=[2, 3, 4, 5, 6],
            pot_size=100,
            min_raise=20,
            stack_size=1000,
            hero_equity=0.64,
            active_players=2,
            call_amount=0,
        ))

        self.assertEqual(action, ("call", 0))

    def test_icm_variant_does_not_bluff_river_multiway(self):
        bot = TournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            board_cards=[2, 3, 4, 5, 6],
            hero_equity=0.64,
            active_players=3,
            call_amount=0,
        ))

        self.assertEqual(action, ("call", 0))

    def test_icm_variant_does_not_bluff_river_facing_bet(self):
        bot = TournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            board_cards=[2, 3, 4, 5, 6],
            pot_size=100,
            call_amount=50,
            min_raise=20,
            stack_size=1000,
            hero_equity=0.64,
            active_players=2,
        ))

        self.assertEqual(action, ("call", 0))

    def test_icm_variant_does_not_bluff_river_under_high_payout_pressure(self):
        bot = TournamentICMEquityBot()

        action = bot.get_action(state(
            board_cards=[2, 3, 4, 5, 6],
            hero_equity=0.64,
            active_players=2,
            call_amount=0,
            players_left=31,
            paid_places=30,
            itm_distance=0.01,
            next_prize_gain_pct=0.02,
        ))

        self.assertEqual(action, ("call", 0))

    def test_icm_variant_does_not_bluff_flop_or_turn(self):
        bot = TournamentICMEquityBot(use_icm=False)

        flop_action = bot.get_action(state(
            board_cards=[2, 3, 4],
            pot_size=100,
            call_amount=0,
            min_raise=20,
            stack_size=1000,
            hero_equity=0.57,
            active_players=2,
        ))
        turn_action = bot.get_action(state(
            board_cards=[2, 3, 4, 5],
            pot_size=100,
            call_amount=0,
            min_raise=20,
            stack_size=1000,
            hero_equity=0.60,
            active_players=2,
        ))

        self.assertEqual(flop_action, ("call", 0))
        self.assertEqual(turn_action, ("call", 0))

    def test_icm_variant_fallback_pressure_folds_bubble_call(self):
        bot = TournamentICMEquityBot()

        action = bot.get_action(
            state(
                pot_size=100,
                call_amount=50,
                hero_equity=0.39,
                players_left=31,
                paid_places=30,
                itm_distance=0.01,
                next_prize_gain_pct=0.02,
            )
        )

        self.assertEqual(action, ("fold", 0))

    def test_icm_variant_exact_pressure_uses_calculator(self):
        bot = TournamentICMEquityBot()

        with patch(
            "players.tournament_equity_bot.calculate_exact_icm",
            side_effect=[
                [0.20, 0.20, 0.15],
                [0.20, 0.20, 0.00],
                [0.20, 0.20, 0.16],
            ],
        ) as icm:
            pressure = bot._payout_pressure(
                state(
                    stack_size=120,
                    players_left=3,
                    paid_places=2,
                    table_stacks=[1000, 1000, 120],
                    hero_table_index=2,
                    payouts={1: 0.65, 2: 0.35},
                    itm_distance=0.0,
                )
            )

        self.assertGreater(pressure, 0.7)
        self.assertEqual(icm.call_count, 3)

    def test_adaptive_icm_ignores_low_sample_table_stats(self):
        bot = AdaptiveTournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            hero_equity=0.49,
            position="BTN",
            table_stats={
                "sample_quality": 0.10,
                "vpip": 0.60,
                "pfr": 0.05,
                "three_bet_rate": 0.0,
            },
        ))

        self.assertEqual(action, ("call", 0))

    def test_adaptive_icm_does_not_widen_only_because_table_is_tight(self):
        bot = AdaptiveTournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            hero_equity=0.48,
            position="BTN",
            call_amount=0,
            table_stats={
                "sample_quality": 1.0,
                "vpip": 0.18,
                "pfr": 0.08,
                "three_bet_rate": 0.02,
            },
        ))

        self.assertEqual(action, ("call", 0))

    def test_adaptive_icm_value_raises_extreme_loose_passive_table(self):
        bot = AdaptiveTournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            hero_equity=0.535,
            position="HJ",
            call_amount=0,
            table_stats={
                "sample_quality": 1.0,
                "vpip": 0.58,
                "pfr": 0.08,
                "three_bet_rate": 0.02,
            },
        ))

        self.assertEqual(action, ("raise", 46))

    def test_adaptive_icm_tightens_reraise_on_aggressive_table(self):
        bot = AdaptiveTournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            hero_equity=0.55,
            call_amount=40,
            preflop_spot_type="srp",
            position="HJ",
            table_stats={
                "sample_quality": 1.0,
                "vpip": 0.34,
                "pfr": 0.34,
                "three_bet_rate": 0.24,
            },
        ))

        self.assertEqual(action, ("call", 0))

    def test_configured_bot_applies_icm_as_tool(self):
        bot = ConfiguredTournamentEquityBot()

        action = bot.get_action(
            state(
                pot_size=100,
                call_amount=50,
                hero_equity=0.39,
                players_left=31,
                paid_places=30,
                itm_distance=0.01,
                next_prize_gain_pct=0.02,
            )
        )

        self.assertEqual(action, ("fold", 0))

    def test_configured_bot_applies_table_adaptation_as_tool(self):
        bot = ConfiguredTournamentEquityBot(tools=[{"type": "table_adaptation"}])

        action = bot.get_action(state(
            hero_equity=0.535,
            position="HJ",
            call_amount=0,
            table_stats={
                "sample_quality": 1.0,
                "vpip": 0.58,
                "pfr": 0.08,
                "three_bet_rate": 0.02,
            },
        ))

        self.assertEqual(action, ("raise", 46))

    def test_configured_bot_can_disable_default_tools(self):
        bot = ConfiguredTournamentEquityBot(tools=[])

        action = bot.get_action(state(
            hero_equity=0.47,
            position="BTN",
            call_amount=0,
            active_players=3,
            table_stats={
                "sample_quality": 1.0,
                "vpip": 0.18,
                "pfr": 0.08,
                "three_bet_rate": 0.02,
            },
        ))

        self.assertEqual(action, ("call", 0))

    def test_endgame_conversion_tool_raises_final_table_big_stack(self):
        bot = ConfiguredTournamentEquityBot(tools=[{"type": "endgame_conversion"}])

        action = bot.get_action(state(
            hero_equity=0.525,
            players_left=9,
            paid_places=26,
            stack_size=2400,
            table_stacks=[2400, 1800, 1600, 1400, 1200, 1000, 900, 800, 700],
            hero_table_index=0,
        ))

        self.assertEqual(action, ("raise", 46))

    def test_endgame_conversion_tool_preserves_bubble_discipline(self):
        bot = ConfiguredTournamentEquityBot(tools=[{"type": "endgame_conversion"}])

        action = bot.get_action(state(
            hero_equity=0.525,
            players_left=9,
            paid_places=8,
            stack_size=2400,
            table_stacks=[2400, 1800, 1600, 1400, 1200, 1000, 900, 800, 700],
            hero_table_index=0,
        ))

        self.assertEqual(action, ("call", 0))

    def test_endgame_conversion_tool_requires_playable_stack(self):
        bot = ConfiguredTournamentEquityBot(tools=[{"type": "endgame_conversion"}])

        action = bot.get_action(state(
            hero_equity=0.525,
            players_left=9,
            paid_places=26,
            stack_size=300,
            table_stacks=[300, 280, 260, 240, 220, 200, 180, 160, 140],
            hero_table_index=0,
        ))

        self.assertEqual(action, ("call", 0))

    def test_button_steal_icm_widens_unopened_button_on_tight_table(self):
        bot = ButtonStealTournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            hero_equity=0.47,
            position="BTN",
            call_amount=0,
            active_players=3,
            table_stats={
                "sample_quality": 1.0,
                "vpip": 0.18,
                "pfr": 0.08,
                "three_bet_rate": 0.02,
            },
        ))

        self.assertEqual(action, ("raise", 44))

    def test_configured_tool_widens_unopened_button_on_tight_table(self):
        bot = TournamentICMEquityBot(
            use_icm=False,
            tools=[
                {
                    "type": "button_steal",
                    "sample_quality_min": 0.50,
                    "max_vpip": 0.30,
                    "max_pfr": 0.16,
                    "max_three_bet_rate": 0.08,
                    "base_discount": 0.018,
                    "tightness_multiplier": 0.12,
                    "max_discount": 0.040,
                }
            ],
        )

        action = bot.get_action(state(
            hero_equity=0.47,
            position="BTN",
            call_amount=0,
            active_players=3,
            table_stats={
                "sample_quality": 1.0,
                "vpip": 0.18,
                "pfr": 0.08,
                "three_bet_rate": 0.02,
            },
        ))

        self.assertEqual(action, ("raise", 44))

    def test_button_steal_icm_does_not_widen_cutoff(self):
        bot = ButtonStealTournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            hero_equity=0.48,
            position="CO",
            call_amount=0,
            active_players=4,
            table_stats={
                "sample_quality": 1.0,
                "vpip": 0.18,
                "pfr": 0.08,
                "three_bet_rate": 0.02,
            },
        ))

        self.assertEqual(action, ("call", 0))

    def test_button_steal_icm_does_not_widen_without_table_sample(self):
        bot = ButtonStealTournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            hero_equity=0.47,
            position="BTN",
            call_amount=0,
            active_players=3,
            table_stats={
                "sample_quality": 0.20,
                "vpip": 0.18,
                "pfr": 0.08,
                "three_bet_rate": 0.02,
            },
        ))

        self.assertEqual(action, ("call", 0))

    def test_button_steal_icm_does_not_widen_on_aggressive_table(self):
        bot = ButtonStealTournamentICMEquityBot(use_icm=False)

        action = bot.get_action(state(
            hero_equity=0.47,
            position="BTN",
            call_amount=0,
            active_players=3,
            table_stats={
                "sample_quality": 1.0,
                "vpip": 0.28,
                "pfr": 0.18,
                "three_bet_rate": 0.10,
            },
        ))

        self.assertEqual(action, ("call", 0))


if __name__ == "__main__":
    unittest.main()
