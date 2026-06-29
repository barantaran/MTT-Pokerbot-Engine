import unittest

from engine.player_state import PlayerState
from engine.table import (
    PREFLOP_SPOT_ALL_IN_PRESSURE,
    PREFLOP_SPOT_FIVE_BET_PLUS,
    PREFLOP_SPOT_FOUR_BET,
    PREFLOP_SPOT_LIMPED,
    PREFLOP_SPOT_SRP,
    PREFLOP_SPOT_THREE_BET,
    PREFLOP_SPOT_UNKNOWN,
    Table,
)


class _Bot:
    def __init__(self, name):
        self.name = name


class _RecordingBot:
    def __init__(self, name):
        self.name = name
        self.states = []

    def get_action(self, game_state):
        self.states.append(dict(game_state))
        return ("fold", 0)


class _ScriptedBot:
    def __init__(self, name, actions):
        self.name = name
        self.actions = list(actions)
        self.states = []

    def get_action(self, game_state):
        self.states.append(dict(game_state))
        if self.actions:
            return self.actions.pop(0)
        return ("fold", 0)


class TablePreflopSpotTypeTests(unittest.TestCase):
    def _table(self):
        table = Table(table_id=1)
        for index in range(3):
            player = PlayerState(_Bot(f"P{index}"), 1000)
            player.is_active = True
            table.add_player(player)
        return table

    def test_early_positions_face_big_blind_preflop_not_free_check(self):
        table = Table(table_id=1)
        bots = [_RecordingBot(f"P{index}") for index in range(9)]
        for bot in bots:
            player = PlayerState(bot, 1000)
            player.is_active = True
            table.add_player(player)

        _active_players, _events = table.play_hand({"small": 10, "big": 20})

        first_states = [bot.states[0] for bot in bots if bot.states]
        utg_state = next(state for state in first_states if state["position"] == "UTG")
        non_blind_preflop_states = [
            state
            for state in first_states
            if state["board_cards"] == [] and state["position"] not in {"SB", "BB"}
        ]

        self.assertEqual(utg_state["call_amount"], 20)
        self.assertTrue(non_blind_preflop_states)
        self.assertTrue(all(state["call_amount"] > 0 for state in non_blind_preflop_states))

    def test_bot_state_includes_public_stack_and_payout_context(self):
        table = Table(table_id=1)
        table.players_left = 3
        table.paid_places = 2
        table.payouts = {1: 0.65, 2: 0.35}
        bots = [_RecordingBot(f"P{index}") for index in range(3)]
        for index, bot in enumerate(bots):
            player = PlayerState(bot, 1000 + index * 500)
            player.is_active = True
            table.add_player(player)

        _active_players, _events = table.play_hand({"small": 10, "big": 20})

        recorded_states = [state for bot in bots for state in bot.states]
        self.assertTrue(recorded_states)
        for state in recorded_states:
            self.assertIn("table_stacks", state)
            self.assertIn("hero_table_index", state)
            self.assertIn("payouts", state)
            self.assertIn("table_stats", state)
            self.assertEqual(state["payouts"], {1: 0.65, 2: 0.35})
            self.assertEqual(len(state["table_stacks"]), 3)
            self.assertGreaterEqual(state["hero_table_index"], 0)
            self.assertLess(state["hero_table_index"], 3)
            self.assertNotIn("opponent_hole_cards", state)
            self.assertNotIn("cards", state["table_stats"])

    def test_table_stats_are_public_and_chronological(self):
        table = Table(table_id=1)
        bots = [_RecordingBot(f"P{index}") for index in range(3)]
        for bot in bots:
            player = PlayerState(bot, 1000)
            player.is_active = True
            table.add_player(player)

        _active_players, _events = table.play_hand({"small": 10, "big": 20})

        recorded_states = [state for bot in bots for state in bot.states]
        self.assertTrue(recorded_states)
        first_state = next(state for state in recorded_states if state["table_stats"]["action_total"] == 0)
        later_state = next(state for state in recorded_states if state["table_stats"]["action_total"] > 0)

        self.assertEqual(first_state["table_stats"]["hands_observed"], 1)
        self.assertEqual(first_state["table_stats"]["player_hands_observed"], 3)
        self.assertEqual(first_state["table_stats"]["action_total"], 0)
        self.assertGreater(later_state["table_stats"]["action_total"], 0)

    def test_state_includes_current_preflop_aggressor_public_stats(self):
        table = Table(table_id=1)
        bots = [
            _ScriptedBot("HJ", [("fold", 0)]),
            _ScriptedBot("SB", [("fold", 0)]),
            _ScriptedBot("BB", [("fold", 0)]),
            _ScriptedBot("BTN", [("fold", 0)]),
            _ScriptedBot("UTG", [("raise", 40)]),
        ]
        for bot in bots:
            player = PlayerState(bot, 1000)
            player.is_active = True
            table.add_player(player)

        _active_players, _events = table.play_hand({"small": 10, "big": 20})

        hj_state = bots[0].states[0]
        self.assertEqual(hj_state["opponent_id"], "UTG")
        self.assertEqual(hj_state["opponent_position"], "UTG")
        self.assertEqual(hj_state["opponent_stack_size"], 940)
        self.assertEqual(hj_state["opponent_stack_bb"], 47.0)
        self.assertEqual(hj_state["opponent_stats"]["hands_observed"], 1)
        self.assertEqual(hj_state["opponent_stats"]["vpip"], 1.0)
        self.assertEqual(hj_state["opponent_stats"]["pfr"], 1.0)

    def test_separates_unknown_from_limped(self):
        table = self._table()

        unknown = table._preflop_spot_type(
            board=[],
            blinds={"big": 20},
            current_highest_bet=20,
            raise_count=0,
            limp_count=0,
            call_amount=20,
            player_stack=1000,
        )
        limped = table._preflop_spot_type(
            board=[],
            blinds={"big": 20},
            current_highest_bet=20,
            raise_count=0,
            limp_count=1,
            call_amount=20,
            player_stack=1000,
        )

        self.assertEqual(unknown, PREFLOP_SPOT_UNKNOWN)
        self.assertEqual(limped, PREFLOP_SPOT_LIMPED)

    def test_maps_raise_count_to_preflop_aggression(self):
        table = self._table()

        self.assertEqual(
            table._preflop_spot_type(
                board=[],
                blinds={"big": 20},
                current_highest_bet=60,
                raise_count=1,
                limp_count=0,
                call_amount=40,
                player_stack=1000,
            ),
            PREFLOP_SPOT_SRP,
        )
        self.assertEqual(
            table._preflop_spot_type(
                board=[],
                blinds={"big": 20},
                current_highest_bet=180,
                raise_count=2,
                limp_count=0,
                call_amount=120,
                player_stack=1000,
            ),
            PREFLOP_SPOT_THREE_BET,
        )
        self.assertEqual(
            table._preflop_spot_type(
                board=[],
                blinds={"big": 20},
                current_highest_bet=420,
                raise_count=3,
                limp_count=0,
                call_amount=240,
                player_stack=1000,
            ),
            PREFLOP_SPOT_FOUR_BET,
        )
        self.assertEqual(
            table._preflop_spot_type(
                board=[],
                blinds={"big": 20},
                current_highest_bet=900,
                raise_count=4,
                limp_count=0,
                call_amount=480,
                player_stack=1000,
            ),
            PREFLOP_SPOT_FIVE_BET_PLUS,
        )

    def test_all_in_pressure_overrides_raise_count(self):
        table = self._table()
        table.players[1].is_all_in = True
        table.players[1].current_bet = 300

        self.assertEqual(
            table._preflop_spot_type(
                board=[],
                blinds={"big": 20},
                current_highest_bet=300,
                raise_count=1,
                limp_count=0,
                call_amount=240,
                player_stack=1000,
            ),
            PREFLOP_SPOT_ALL_IN_PRESSURE,
        )

    def test_postflop_is_unknown(self):
        table = self._table()

        self.assertEqual(
            table._preflop_spot_type(
                board=[1, 2, 3],
                blinds={"big": 20},
                current_highest_bet=60,
                raise_count=1,
                limp_count=0,
                call_amount=40,
                player_stack=1000,
            ),
            PREFLOP_SPOT_UNKNOWN,
        )

    def test_postflop_carries_preflop_spot_type(self):
        table = self._table()
        table.current_preflop_spot_type = PREFLOP_SPOT_THREE_BET

        self.assertEqual(
            table._preflop_spot_type(
                board=[1, 2, 3],
                blinds={"big": 20},
                current_highest_bet=0,
                raise_count=0,
                limp_count=0,
                call_amount=0,
                player_stack=1000,
            ),
            PREFLOP_SPOT_THREE_BET,
        )

    def test_next_prize_gain_uses_next_bustout_pay_jump(self):
        table = self._table()
        table.players_left = 10
        table.payouts = {
            9: 0.012,
            10: 0.0,
        }

        self.assertEqual(table._next_prize_gain_pct(), 0.012)

    def test_next_prize_gain_is_zero_without_pay_jump(self):
        table = self._table()
        table.players_left = 12
        table.payouts = {
            9: 0.012,
            10: 0.0,
        }

        self.assertEqual(table._next_prize_gain_pct(), 0.0)


if __name__ == "__main__":
    unittest.main()
