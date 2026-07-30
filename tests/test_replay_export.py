"""Tests for the redacted replay export.

These run a real (tiny) marathon rather than hand-building events: the point of
the exporter is that it survives the engine's actual output, and a synthetic
fixture would happily keep passing after the event schema drifted.
"""
import json
import tempfile
import unittest
from pathlib import Path

from engine import replay_export
from engine.fixed_bot_mtt_evaluation import run_fixed_bot_evaluation
from engine.replay_export import SchemaError, build_hand, export, redact, split_hands

ENGINE_ROOT = Path(__file__).resolve().parents[1]


def _config(root: Path) -> dict:
    return {
        "run_id": "replay_export_test",
        "artifact_root": str(root),
        "mtt_count": 2,
        "workers": 2,
        "mtt_seed_start": 4242,
        "resume_log": True,
        "write_events": True,
        "lineup": {"call": 6, "aggressive_no_equity": 6},
        "engine": {
            "equity_source": "constant",
            "equity_fallback_source": "constant",
            "starting_stack": 500,
            "max_players_per_table": 6,
            "hands_per_level": 3,
        },
    }


class ReplayExportTests(unittest.TestCase):
    """One marathon, exported once; every case reads the same artifacts."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        root = Path(cls._tmp.name)
        cls.artifact_root = root / "runs" / "replay_export_test"
        cls.report = run_fixed_bot_evaluation(_config(cls.artifact_root), engine_root=ENGINE_ROOT)
        cls.out = cls.artifact_root / "replay"
        cls.fanout = root / "replays"
        cls.manifest = export(
            artifact_root=cls.artifact_root,
            report=cls.report,
            run_id="replay_export_test",
            out_root=cls.out,
            fanout_root=cls.fanout,
            engine_sha="testsha",
            page_size=5,
        )
        cls.hero = "call"
        cls.villain = "aggressive_no_equity"

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    # -- helpers ----------------------------------------------------------

    def _index(self, nick):
        with (self.out / nick / "index.json").open(encoding="utf-8") as handle:
            return json.load(handle)

    def _hands(self, nick):
        index = self._index(nick)
        out = []
        for page in range(index["pages"]):
            with (self.out / nick / f"hands_{page:03d}.json").open(encoding="utf-8") as handle:
                out.extend(json.load(handle)["hands"])
        return out

    def _source_events(self):
        events = []
        for tournament_id in (1, 2):
            events.extend(replay_export.load_events(self.artifact_root, tournament_id))
        return events

    # -- structure --------------------------------------------------------

    def test_manifest_lists_every_population(self):
        nicks = {row["nick"] for row in self.manifest["nicks"]}
        self.assertEqual(nicks, {self.hero, self.villain})
        self.assertEqual(self.manifest["engine_sha"], "testsha")
        self.assertEqual(self.manifest["starting_stack"], 500)
        self.assertTrue(self.manifest["blinds_schedule"])

    def test_manifest_carries_no_cards(self):
        blob = json.dumps(self.manifest)
        for name, population in self.report["name_to_population"].items():
            self.assertNotIn('"cards"', blob, f"{name}/{population}")

    def test_paging_is_consistent_with_the_index(self):
        index = self._index(self.hero)
        hands = self._hands(self.hero)
        self.assertEqual(len(hands), index["hands"])
        self.assertEqual(len(index["rows"]), index["hands"])
        by_ref = {hand["ref"]: hand for hand in hands}
        for row in index["rows"]:
            with (self.out / self.hero / f"hands_{row['page']:03d}.json").open(encoding="utf-8") as fh:
                page = json.load(fh)["hands"]
            self.assertEqual(page[row["i"]]["ref"], row["ref"])
            self.assertEqual(by_ref[row["ref"]]["pot_total"], row["pot_total"])

    def test_fanout_pointer_matches_the_index(self):
        with (self.fanout / self.hero / "replay_export_test.json").open(encoding="utf-8") as handle:
            pointer = json.load(handle)
        index = self._index(self.hero)
        self.assertEqual(pointer["run_id"], "replay_export_test")
        self.assertEqual(pointer["hands"], index["hands"])
        self.assertEqual(pointer["pages"], index["pages"])

    # -- selection --------------------------------------------------------

    def test_hand_selection_is_exactly_the_hands_the_hero_was_dealt_into(self):
        hero_names = {
            name for name, population in self.report["name_to_population"].items()
            if population == self.hero
        }
        expected = set()
        for event in self._source_events():
            if event.get("type") == "deal" and event.get("player") in hero_names:
                expected.add("t{}.b{}.h{}".format(
                    event["tournament_id"], event["table_id"], event["hand_id"]))
        self.assertTrue(expected)
        self.assertEqual({hand["ref"] for hand in self._hands(self.hero)}, expected)

    def test_every_exported_hand_has_a_hero_seat(self):
        for hand in self._hands(self.hero):
            self.assertTrue(hand["hero_seats"], hand["ref"])
            heroes = [seat for seat in hand["seats"] if seat["is_hero"]]
            self.assertTrue(all(seat["cards"] for seat in heroes))

    # -- redaction --------------------------------------------------------

    def test_no_foreign_hole_cards_survive_outside_a_showdown(self):
        """The load-bearing test: an opponent's dealt hand must not appear.

        Collect what every non-hero was dealt from the raw log, then assert the
        hero's whole slice never shows it — except where that opponent reached
        showdown, which the table saw too.
        """
        hero_names = {
            name for name, population in self.report["name_to_population"].items()
            if population == self.hero
        }
        dealt = {}       # (ref, player) -> cards
        shown = set()    # (ref, player)
        for event in self._source_events():
            if "hand_id" not in event:
                continue
            ref = "t{}.b{}.h{}".format(
                event["tournament_id"], event["table_id"], event["hand_id"])
            if event["type"] == "deal" and event["player"] not in hero_names:
                dealt[(ref, event["player"])] = event["cards"]
            elif event["type"] == "showdown":
                shown.add((ref, event["player"]))

        leaked = []
        for hand in self._hands(self.hero):
            blob = json.dumps(hand)
            for (ref, player), cards in dealt.items():
                if ref != hand["ref"] or (ref, player) in shown:
                    continue
                for card in cards:
                    # a bare rank+suit string is only ever a card
                    if f'"{card}"' in blob and card not in hand["board"]["final"]:
                        leaked.append((ref, player, card))
        self.assertEqual(leaked, [], "opponent hole cards leaked into the slice")

    def test_folded_opponents_stay_hidden(self):
        hidden_seen = False
        for hand in self._hands(self.hero):
            for seat in hand["seats"]:
                if seat["is_hero"] or seat["revealed"]:
                    continue
                self.assertIsNone(seat["cards"], (hand["ref"], seat["name"]))
                self.assertTrue(seat["hidden"])
                self.assertIsNone(seat["rank"])
                hidden_seen = True
        self.assertTrue(hidden_seen, "expected at least one hidden opponent")

    def test_showdown_reveals_are_kept(self):
        revealed = [
            seat
            for hand in self._hands(self.hero)
            for seat in hand["seats"]
            if seat["revealed"]
        ]
        self.assertTrue(revealed, "the run should contain showdowns")
        for seat in revealed:
            self.assertEqual(len(seat["cards"]), 2)
            self.assertTrue(seat["rank_class"])

    def test_opponent_tool_events_are_stripped_and_the_heros_are_kept(self):
        hand = _hand_with_tool_events()
        record = redact(hand, ["hero"])
        actions = {a["name"]: a for street in record["streets"] for a in street["actions"]}
        self.assertEqual(actions["hero"]["tool_events"], [{"tool": "t", "secret": "AhKh"}])
        self.assertIsNone(actions["villain"]["tool_events"])

    # -- arithmetic -------------------------------------------------------

    def test_pot_tracks_the_bets_and_awards_never_exceed_it(self):
        for hand in self._hands(self.hero):
            posted = sum(row["amount"] for row in hand["blinds_posted"])
            bet = sum(a["amount"] for s in hand["streets"] for a in s["actions"])
            self.assertEqual(hand["pot_total"], posted + bet, hand["ref"])
            self.assertLessEqual(hand["awarded_total"], hand["pot_total"], hand["ref"])
            # the only shortfall is the split-pot remainder
            self.assertLess(
                hand["pot_total"] - hand["awarded_total"],
                max(1, len(hand["payouts"])),
                hand["ref"],
            )

    def test_first_preflop_action_faces_the_posted_blinds(self):
        checked = 0
        for hand in self._hands(self.hero):
            preflop = hand["streets"][0]
            if not preflop["actions"]:
                continue
            posted = sum(row["amount"] for row in hand["blinds_posted"])
            self.assertEqual(preflop["actions"][0]["pot_before"], posted, hand["ref"])
            checked += 1
        self.assertTrue(checked)

    def test_seat_stacks_and_net_reconcile(self):
        for hand in self._hands(self.hero):
            for seat in hand["seats"]:
                self.assertEqual(seat["net"], seat["won"] - seat["invested"], hand["ref"])
                self.assertEqual(
                    seat["stack_end"],
                    seat["stack_start"] - seat["invested"] + seat["won"],
                    (hand["ref"], seat["name"]),
                )

    def test_streets_are_ordered_and_boards_grow(self):
        for hand in self._hands(self.hero):
            names = [street["street"] for street in hand["streets"]]
            self.assertEqual(names[0], "preflop")
            order = [replay_export.STREET_ORDER.index(name) for name in names]
            self.assertEqual(order, sorted(order), hand["ref"])
            sizes = [len(street["board"]) for street in hand["streets"]]
            self.assertEqual(sizes, sorted(sizes), hand["ref"])

    def test_demux_keeps_each_hand_on_one_table(self):
        for hand in self._hands(self.hero):
            seats = {seat["seat"] for seat in hand["seats"]}
            self.assertEqual(len(seats), len(hand["seats"]))
            seqs = [a["seq"] for s in hand["streets"] for a in s["actions"]]
            self.assertEqual(seqs, sorted(seqs), hand["ref"])


class ReplayExportGuardTests(unittest.TestCase):
    def test_a_pre_schema_log_is_rejected(self):
        legacy = [{"type": "hand_start", "table_id": 1, "hand_id": 1,
                   "tournament_id": 1, "players": [{"name": "a", "stack": 100}]}]
        with self.assertRaises(SchemaError) as caught:
            replay_export.check_schema(legacy, "legacy.json")
        self.assertIn("predates", str(caught.exception))

    def test_a_report_without_name_to_population_is_rejected(self):
        with self.assertRaises(SchemaError):
            export(
                artifact_root=Path("/nonexistent"),
                report={"mtt_count": 1},
                run_id="x",
                out_root=Path("/nonexistent/out"),
            )

    def test_a_negative_action_amount_is_flagged_not_hidden(self):
        # engine/table.py can emit a negative call amount when a betting round
        # early-returns without resetting current_bet (docs/EVENT_STREAM.md).
        # The exporter must stay faithful and say so.
        events = _minimal_hand_events()
        events[3]["amount"] = -200
        hand = build_hand(events)
        self.assertTrue(hand["anomalies"])
        self.assertIn("negative amount -200", hand["anomalies"][0])
        self.assertEqual(redact(hand, ["hero"])["anomalies"], hand["anomalies"])

    def test_a_clean_hand_has_no_anomalies(self):
        self.assertEqual(build_hand(_minimal_hand_events())["anomalies"], [])

    def test_truncation_keeps_the_biggest_swings_and_says_so(self):
        hero = replay_export.HeroSlice("h", cap=2)
        for net in (5, 100, 1, 50):
            hero.add(_hand_record_with_net(net))
        with tempfile.TemporaryDirectory() as tmp:
            index = replay_export.write_slice(
                hero, run_id="r", out_root=Path(tmp), page_size=10
            )
        self.assertTrue(index["truncated"])
        self.assertEqual(index["hands"], 2)
        self.assertEqual(index["hands_total"], 4)
        self.assertEqual(sorted(row["hero_net"] for row in index["rows"]), [50, 100])


def _hand_with_tool_events():
    return build_hand(_minimal_hand_events())


def _minimal_hand_events():
    return [
        {"type": "hand_start", "seq": 0, "table_id": 1, "hand_id": 1, "tournament_id": 1,
         "level": 1, "blinds": {"small": 10, "big": 20}, "button_seat": 0,
         "started_at": "2026-01-01T00:00:00+00:00",
         "players": [{"seat": 0, "name": "hero", "stack": 100, "position": "BTN"},
                     {"seat": 1, "name": "villain", "stack": 100, "position": "SB"}]},
        {"type": "deal", "seq": 1, "table_id": 1, "hand_id": 1, "tournament_id": 1,
         "player": "hero", "cards": ["Ah", "Kh"]},
        {"type": "deal", "seq": 2, "table_id": 1, "hand_id": 1, "tournament_id": 1,
         "player": "villain", "cards": ["2c", "7d"]},
        {"type": "action", "seq": 3, "table_id": 1, "hand_id": 1, "tournament_id": 1,
         "player": "hero", "action": "raise", "amount": 20, "street": "preflop",
         "tool_events": [{"tool": "t", "secret": "AhKh"}]},
        {"type": "action", "seq": 4, "table_id": 1, "hand_id": 1, "tournament_id": 1,
         "player": "villain", "action": "fold", "amount": 0, "street": "preflop",
         "tool_event": {"tool": "t", "secret": "2c7d"}},
        {"type": "hand_end", "seq": 5, "table_id": 1, "hand_id": 1, "tournament_id": 1,
         "ended_at": "2026-01-01T00:00:01+00:00",
         "players": [{"name": "hero", "stack": 100}, {"name": "villain", "stack": 100}]},
    ]


def _hand_record_with_net(net):
    hand = _hand_with_tool_events()
    hand["seats"][0]["net"] = net
    return redact(hand, ["hero"])


if __name__ == "__main__":
    unittest.main()
