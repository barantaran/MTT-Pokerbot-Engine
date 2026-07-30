"""Schema tests for the tournament-scope events.

Table events carry their hand key (see test_table_event_stream). The events
Tournament emits around them did not: a level_up in a merged multi-tournament
stream named no tournament, and a seat named no seat.
"""
import random
import unittest

from engine.bot_factory import build_configurable_bots
from engine.fixed_bot_mtt_evaluation import default_engine_config
from engine.tournament import Tournament

TOURNAMENT_SCOPE_TYPES = {
    "tournament_start",
    "seat",
    "table_broken",
    "knockout",
    "level_up",
    "tournament_win",
    "tournament_stopped_max_hands",
}


def _run(seed=20260730, lineup=None):
    random.seed(seed)
    bots, _names = build_configurable_bots(
        lineup or {"call": 6, "aggressive_no_equity": 6}, default_engine_config()
    )
    tournament = Tournament(bots, tournament_id=4)
    results, events = tournament.play()
    return results, events


class TournamentEventStreamTests(unittest.TestCase):
    def setUp(self):
        self.results, self.events = _run()

    def _of_type(self, event_type):
        return [event for event in self.events if event["type"] == event_type]

    def test_every_tournament_scope_event_names_its_tournament(self):
        scoped = [e for e in self.events if e["type"] in TOURNAMENT_SCOPE_TYPES]
        self.assertTrue(scoped)
        for event in scoped:
            self.assertEqual(event["tournament_id"], 4, event)

    def test_seat_events_name_the_seat(self):
        seats = self._of_type("seat")
        self.assertTrue(seats)
        for event in seats:
            self.assertIsInstance(event["seat"], int)
            self.assertGreaterEqual(event["seat"], 0)
            self.assertIn("table_id", event)

    def test_level_up_names_the_level(self):
        levels = self._of_type("level_up")
        self.assertTrue(levels, "the seeded run should raise blinds at least once")
        self.assertEqual([e["level"] for e in levels], list(range(2, len(levels) + 2)))
        for event in levels:
            self.assertIn("small", event["blinds"])

    def test_knockout_finish_position_matches_the_results_table(self):
        positions = {row["name"]: row["position"] for row in self.results}
        knockouts = self._of_type("knockout")
        self.assertTrue(knockouts)
        for event in knockouts:
            self.assertEqual(event["finish_position"], positions[event["player"]], event)

    def test_the_winner_is_the_one_player_never_knocked_out(self):
        win = self._of_type("tournament_win")
        self.assertEqual(len(win), 1)
        busted = {event["player"] for event in self._of_type("knockout")}
        self.assertNotIn(win[0]["player"], busted)


if __name__ == "__main__":
    unittest.main()
