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


class TablePreflopSpotTypeTests(unittest.TestCase):
    def _table(self):
        table = Table(table_id=1)
        for index in range(3):
            player = PlayerState(_Bot(f"P{index}"), 1000)
            player.is_active = True
            table.add_player(player)
        return table

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


if __name__ == "__main__":
    unittest.main()
