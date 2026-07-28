from __future__ import annotations

from unittest.mock import patch

from players.icm_tight_bot import ICMTightBot


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
        "position": "HJ",
        "preflop_spot_type": "unknown",
        "players_left": 50,
        "starting_field": 200,
        "paid_places": 30,
    }
    base.update(overrides)
    return base


def test_icm_tight_bot_uses_provided_hero_equity_without_recomputing():
    bot = ICMTightBot()

    with patch("players.icm_tight_bot.estimate_equity", side_effect=AssertionError("should not compute")):
        action = bot.get_action(state(hero_equity=0.90))

    assert action == ("raise", 48)


def test_icm_tight_bot_checks_free_big_blind_with_weak_equity():
    bot = ICMTightBot()

    action = bot.get_action(state(hero_equity=0.10, position="BB", call_amount=0))

    assert action == ("call", 0)


def test_icm_tight_bot_non_icm_mode_calls_marginal_chip_spot():
    bot = ICMTightBot(use_icm=False)

    action = bot.get_action(state(pot_size=100, call_amount=50, hero_equity=0.43))

    assert action == ("call", 0)


def test_icm_tight_bot_fallback_icm_pressure_folds_marginal_bubble_call():
    bot = ICMTightBot()

    action = bot.get_action(
        state(
            pot_size=100,
            call_amount=50,
            hero_equity=0.43,
            players_left=31,
            paid_places=30,
            payouts={31: 0.0, 30: 0.02},
        )
    )

    assert action == ("fold", 0)


def test_icm_tight_bot_exact_icm_pressure_folds_marginal_final_table_call():
    bot = ICMTightBot()

    action = bot.get_action(
        state(
            pot_size=100,
            call_amount=50,
            hero_equity=0.43,
            players_left=3,
            paid_places=2,
            table_stacks=[1000, 1000, 120],
            hero_table_index=2,
            payouts={1: 0.65, 2: 0.35},
        )
    )

    assert action == ("fold", 0)


def test_icm_tight_bot_short_stack_strong_hand_can_jam():
    bot = ICMTightBot()

    action = bot.get_action(
        state(
            hero_equity=0.80,
            stack_size=160,
            call_amount=20,
            min_raise=20,
            players_left=50,
            paid_places=30,
        )
    )

    assert action == ("raise", 140)


def test_icm_tight_bot_deep_stack_value_raise_is_sized_not_all_in():
    bot = ICMTightBot()

    action = bot.get_action(state(hero_equity=0.88, stack_size=3000, call_amount=0))

    assert action == ("raise", 48)
