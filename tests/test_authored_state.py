"""The authored seam ships only what the author cannot compute.

``_normalize_state`` (``engine/authored_loader.py``) is the whole author-facing
contract, and it is also the redaction boundary between the table's event log
and an untrusted seat. These tests pin both halves: what is present (irreducible
table truth plus the event log), what is deliberately absent (every measure the
author can derive), and what must never cross (other players' hole cards, other
bots' tool payloads).
"""

from __future__ import annotations

import json
import unittest

from treys import Card

from engine.authored_loader import _normalize_state


def _engine_state():
    """The dict engine/table.py builds for one decision — treys ints and all."""
    return {
        "hole_cards": [Card.new("As"), Card.new("Kd")],
        "board_cards": [Card.new("Qh"), Card.new("Jc"), Card.new("2s")],
        "pot_size": 450,
        "stack_size": 1500,
        "table_stacks": [900, 1500, 3000],
        "hero_table_index": 1,
        "button_seat": 2,
        "call_amount": 100,
        "min_raise": 200,
        "blinds": {"small": 25, "big": 50},
        "active_players": 3,
        "players_left": 40,
        "starting_field": 100,
        "paid_places": 15,
        "payouts": {1: 0.5, 2: 0.3},
        "player_id": "hero",
        "table_id": 1,
        "hand_id": 12,
        "tournament_id": 3,
        "position": "SB",
        "preflop_spot_type": "three_bet",
        "table_stats": {"vpip": 0.4},
        "opponent_id": "villainX",
        "opponent_position": "BTN",
        "opponent_stack_size": 3000,
        "opponent_stack_bb": 60.0,
        "opponent_stats": {"vpip": 0.31, "pfr": 0.24},
        "_hand_events": _hand_events(),
    }


def _hand_events():
    """A hand's worth of the table's own log, including the two event shapes the
    seam must never forward: the deal, and a rival's tool payload."""
    return [
        {
            "type": "hand_start", "table_id": 1, "hand_id": 12, "tournament_id": 3,
            "started_at": "2026-08-11T00:00:00+00:00", "level": 3,
            "blinds": {"small": 25, "big": 50},
            "button_seat": 2, "button_player": "villainY",
            "players": [
                {"seat": 0, "name": "villainX", "stack": 900, "position": "SB"},
                {"seat": 1, "name": "hero", "stack": 1500, "position": "BB"},
                {"seat": 2, "name": "villainY", "stack": 3000, "position": "BTN"},
            ],
        },
        {"type": "post_blind", "player": "villainX", "seat": 0, "blind": "small",
         "amount": 25, "street": "preflop", "table_id": 1, "hand_id": 12},
        {"type": "deal", "player": "villainX", "cards": ["7c", "7d"],
         "table_id": 1, "hand_id": 12, "tournament_id": 3},
        {"type": "deal", "player": "hero", "cards": ["As", "Kd"],
         "table_id": 1, "hand_id": 12, "tournament_id": 3},
        {"type": "action", "player": "villainX", "action": "raise", "amount": 150,
         "street": "preflop", "position": "SB", "call_amount": 50, "pot_size": 75,
         "tool_event": {"tool": "bluff_pressure", "equity": 0.19},
         "tool_events": [{"tool": "bluff_pressure", "equity": 0.19}]},
        {"type": "board", "cards": ["Qh", "Jc", "2s"], "street": "flop",
         "table_id": 1, "hand_id": 12},
    ]


class NormalizeStateShape(unittest.TestCase):
    def test_top_level_is_the_irreducible_set(self):
        state = _normalize_state(_engine_state())
        self.assertEqual(
            sorted(state),
            ["blinds", "board", "button_seat", "events", "hero", "live_opponents",
             "pot", "stats", "table_stacks", "tournament"],
        )
        self.assertEqual(sorted(state["hero"]), ["call_amount", "hole", "min_raise", "seat"])

    def test_facts_survive_the_translation(self):
        state = _normalize_state(_engine_state())
        self.assertEqual(state["hero"]["hole"], ["As", "Kd"])
        self.assertEqual(state["board"], ["Qh", "Jc", "2s"])
        self.assertEqual(state["hero"]["seat"], 1)
        self.assertEqual(state["button_seat"], 2)
        self.assertEqual(state["table_stacks"], [900, 1500, 3000])
        self.assertEqual(state["pot"], 450)
        self.assertEqual(state["blinds"], {"small": 25, "big": 50})
        self.assertEqual(state["live_opponents"], 2)
        self.assertEqual(state["tournament"]["players_left"], 40)
        self.assertEqual(state["tournament"]["payouts"], {1: 0.5, 2: 0.3})

    def test_derived_measures_are_not_shipped(self):
        """Each of these is one line of author code over what is already there;
        shipping it would put the engine back in the judgement business.

        Scoped to the top level on purpose: ``street`` stays *on an event*,
        where it says when that action happened and is not a restatement of
        anything else in the dict, while a top-level ``street`` would be
        ``len(board)`` spelled twice."""
        state = _normalize_state(_engine_state())
        top = json.dumps({key: value for key, value in state.items() if key != "events"})
        for absent in ("stack_bb", "position", "street", "spot_type",
                       "villain_hands", "history", "equity", "table_stats"):
            self.assertNotIn(absent, top, f"{absent} is derivable and must not be shipped")
        self.assertNotIn("stack", state["hero"])  # table_stacks[seat]

    def test_stats_are_keyed_by_player(self):
        state = _normalize_state(_engine_state())
        self.assertEqual(state["stats"], {"villainX": {"vpip": 0.31, "pfr": 0.24}})

    def test_stats_are_empty_when_nobody_is_faced(self):
        engine_state = _engine_state()
        engine_state.update({"opponent_id": "", "opponent_stats": None})
        self.assertEqual(_normalize_state(engine_state)["stats"], {})

    def test_a_thin_state_normalizes_without_raising(self):
        """table.py is not the only caller — tests and adapters hand over less."""
        state = _normalize_state({"hole_cards": ["As", "Kd"], "blinds": {"big": 100}})
        self.assertEqual(state["hero"]["hole"], ["As", "Kd"])
        self.assertEqual(state["table_stacks"], [])
        self.assertEqual(state["events"], [])
        self.assertEqual(state["live_opponents"], 0)


class NormalizeStateEvents(unittest.TestCase):
    def test_the_hand_is_replayable_from_events(self):
        events = _normalize_state(_engine_state())["events"]
        self.assertEqual(
            [event["type"] for event in events],
            ["hand_start", "post_blind", "action", "board"],
        )
        start = events[0]
        self.assertEqual(start["button_seat"], 2)
        self.assertEqual([seat["name"] for seat in start["players"]],
                         ["villainX", "hero", "villainY"])
        self.assertEqual(events[2]["amount"], 150)
        self.assertEqual(events[3]["cards"], ["Qh", "Jc", "2s"])

    def test_no_rival_hole_cards_cross_the_seam(self):
        flat = json.dumps(_normalize_state(_engine_state()))
        self.assertNotIn("7c", flat)
        self.assertNotIn("7d", flat)
        self.assertNotIn("deal", flat)

    def test_no_rival_tool_payloads_cross_the_seam(self):
        flat = json.dumps(_normalize_state(_engine_state()))
        self.assertNotIn("bluff_pressure", flat)
        self.assertNotIn("tool_event", flat)

    def test_per_event_derivables_are_dropped(self):
        action = _normalize_state(_engine_state())["events"][2]
        self.assertEqual(sorted(action), ["action", "amount", "player", "street", "type"])

    def test_unknown_event_types_are_not_forwarded(self):
        """The projection is an allowlist: a new event type reaches no author
        until it is named, so adding one upstream cannot leak by default."""
        engine_state = _engine_state()
        engine_state["_hand_events"] = [
            {"type": "showdown", "hands": {"villainX": ["7c", "7d"]}},
            {"type": "award_pot", "player": "villainX", "amount": 450},
        ]
        self.assertEqual(_normalize_state(engine_state)["events"], [])


if __name__ == "__main__":
    unittest.main()
