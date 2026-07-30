"""Schema tests for the event stream Table emits.

The stream is consumed outside the engine — the resume log, the stats
summarizers, and the arena's replay exporter all read these dicts — but nothing
asserted its shape, which is how two card-encoding bugs survived: the runout
board and the showdown reveal both emitted raw treys integers while every other
card field emitted strings.
"""
import unittest

from treys import Card, Deck

from engine.player_state import PlayerState
from engine.pot import PotManager
from engine.table import Table


class _Bot:
    def __init__(self, name):
        self.name = name


class _ScriptedBot:
    def __init__(self, name, actions):
        self.name = name
        self.actions = list(actions)

    def get_action(self, game_state):
        if self.actions:
            return self.actions.pop(0)
        return ("fold", 0)


def _card_values(events):
    """Every card value the stream carries, from any event that has `cards`."""
    return [card for event in events if "cards" in event for card in event["cards"]]


class TableEventCardEncodingTests(unittest.TestCase):
    def _table(self, count=6, stack=1000, actions=None):
        table = Table(table_id=1, tournament_id=7)
        for index in range(count):
            bot = _ScriptedBot(f"P{index}", (actions or {}).get(index, []))
            player = PlayerState(bot, stack)
            player.is_active = True
            table.add_player(player)
        return table

    def test_no_card_value_is_ever_an_int(self):
        # A full hand: deal, flop, turn, river and (with callers) showdown.
        table = self._table(actions={index: [("call", 0)] * 4 for index in range(6)})

        _busted, events = table.play_hand({"small": 10, "big": 20})

        cards = _card_values(events)
        self.assertTrue(cards, "expected the hand to emit card events")
        for card in cards:
            self.assertIsInstance(card, str, f"card {card!r} is not a string")
            self.assertEqual(len(card), 2, f"card {card!r} is not a two-char code")

    def test_showdown_reveals_strings_and_a_readable_rank_class(self):
        table = self._table(count=2, actions={0: [("call", 0)] * 4, 1: [("call", 0)] * 4})

        _busted, events = table.play_hand({"small": 10, "big": 20})

        showdowns = [event for event in events if event["type"] == "showdown"]
        self.assertTrue(showdowns, "two callers should reach showdown")
        for event in showdowns:
            self.assertEqual(len(event["cards"]), 2)
            self.assertTrue(all(isinstance(card, str) for card in event["cards"]))
            self.assertIsInstance(event["rank"], int)
            # A human-readable class so a consumer never needs treys to render it.
            self.assertIsInstance(event["rank_class"], str)
            self.assertTrue(event["rank_class"])

    def test_runout_board_cards_are_strings(self):
        # The runout path only fires when _showdown is reached with a short
        # board, which play_hand cannot currently produce (every street deals
        # while two players are active). Drive _showdown directly so the branch
        # is still covered.
        table = self._table(count=2)
        for player in table.players:
            player.hole_cards = []
        deck = Deck()
        for player in table.players:
            player.hole_cards = deck.draw(2)
            player.total_bet = 100
        events = []

        table._showdown([], PotManager(), deck, events)

        runouts = [event for event in events if event.get("street") == "runout"]
        self.assertEqual(len(runouts), 5, "an empty board runs out five cards")
        for event in runouts:
            self.assertTrue(all(isinstance(card, str) for card in event["cards"]))
            # and they are real cards, not str() of an int
            for card in event["cards"]:
                self.assertIsInstance(Card.new(card), int)


class TableEventAddressingTests(unittest.TestCase):
    def _table(self, count=6, stack=1000, actions=None):
        table = Table(table_id=3, tournament_id=9)
        for index in range(count):
            bot = _ScriptedBot(f"P{index}", (actions or {}).get(index, []))
            player = PlayerState(bot, stack)
            player.is_active = True
            table.add_player(player)
        return table

    def test_every_event_carries_the_hand_key_and_a_sequence_number(self):
        table = self._table(actions={index: [("call", 0)] * 4 for index in range(6)})

        _busted, events = table.play_hand({"small": 10, "big": 20})

        self.assertTrue(events)
        for index, event in enumerate(events):
            self.assertEqual(event["table_id"], 3, event)
            self.assertEqual(event["tournament_id"], 9, event)
            self.assertEqual(event["hand_id"], 1, event)
            self.assertEqual(event["seq"], index, event)

    def test_post_blind_is_addressable(self):
        table = self._table(count=3)

        _busted, events = table.play_hand({"small": 10, "big": 20})

        blinds = [event for event in events if event["type"] == "post_blind"]
        self.assertEqual([event["blind"] for event in blinds], ["small", "big"])
        for event in blinds:
            self.assertEqual(event["street"], "preflop")
            self.assertEqual(event["hand_id"], 1)
            self.assertEqual(event["tournament_id"], 9)
            self.assertIsInstance(event["seat"], int)
            self.assertEqual(table.players[event["seat"]].name, event["player"])

    def test_board_events_name_their_hand(self):
        table = self._table(actions={index: [("call", 0)] * 4 for index in range(6)})

        _busted, events = table.play_hand({"small": 10, "big": 20})

        boards = [event for event in events if event["type"] == "board"]
        self.assertEqual([event["street"] for event in boards], ["flop", "turn", "river"])
        for event in boards:
            self.assertEqual(event["hand_id"], 1)
            self.assertEqual(event["tournament_id"], 9)

    def test_sequence_restarts_each_hand(self):
        table = self._table(count=3, actions={index: [("call", 0)] * 8 for index in range(3)})

        table.play_hand({"small": 10, "big": 20})
        _busted, events = table.play_hand({"small": 10, "big": 20})

        self.assertEqual(events[0]["seq"], 0)
        self.assertTrue(all(event["hand_id"] == 2 for event in events))


class HandStartDescribesTheTableTests(unittest.TestCase):
    def _table(self, count=6, stack=1000, level=0):
        table = Table(table_id=3, tournament_id=9)
        table.level = level
        for index in range(count):
            player = PlayerState(_Bot(f"P{index}"), stack)
            player.is_active = True
            table.add_player(player)
        return table

    def _hand_start(self, events):
        return next(event for event in events if event["type"] == "hand_start")

    def test_hand_start_names_the_button_and_labels_every_seat(self):
        table = self._table(level=4)

        _busted, events = table.play_hand({"small": 10, "big": 20})

        start = self._hand_start(events)
        self.assertEqual(start["level"], 4)
        self.assertEqual(start["blinds"], {"small": 10, "big": 20})
        self.assertEqual(start["button_seat"], table.button_idx)
        self.assertEqual(start["button_player"], table.players[table.button_idx].name)
        self.assertTrue(start["started_at"].endswith("+00:00"))

        seats = start["players"]
        self.assertEqual([seat["seat"] for seat in seats], list(range(len(seats))))
        for index, seat in enumerate(seats):
            self.assertEqual(seat["position"], table._position_label(index, len(seats)))
        positions = [seat["position"] for seat in seats]
        self.assertIn("BTN", positions)
        self.assertIn("SB", positions)
        self.assertIn("BB", positions)

    def test_hand_start_keeps_the_legacy_name_and_stack_keys(self):
        # game_stats and the summarizers read players[].name / .stack; the seat
        # and position fields are additive.
        table = self._table()

        _busted, events = table.play_hand({"small": 10, "big": 20})

        for seat in self._hand_start(events)["players"]:
            self.assertIn("name", seat)
            self.assertIsInstance(seat["stack"], int)

    def test_button_advances_one_seat_per_hand(self):
        table = self._table(count=5, stack=100000)

        first = self._hand_start(table.play_hand({"small": 10, "big": 20})[1])
        second = self._hand_start(table.play_hand({"small": 10, "big": 20})[1])

        self.assertEqual(second["button_seat"], (first["button_seat"] + 1) % 5)

    def test_hand_end_is_timestamped(self):
        table = self._table()

        _busted, events = table.play_hand({"small": 10, "big": 20})

        end = next(event for event in events if event["type"] == "hand_end")
        self.assertTrue(end["ended_at"].endswith("+00:00"))
        self.assertGreaterEqual(end["ended_at"], self._hand_start(events)["started_at"])


if __name__ == "__main__":
    unittest.main()
