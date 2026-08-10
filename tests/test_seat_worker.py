"""Hostile-seat fixtures for the out-of-process authored seat.

Each test writes a throwaway ``plugins/<nick>/`` folder and asserts what the seat
does with it. The fixtures mirror the verification table in
``service/INTEGRITY.md``: a spinner, a printer, a raiser, a crasher, a folder
that will not import.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from engine.seat_worker import DEFAULT_ACTION, KIND_TOOLSTACK, SeatWorker


def _state():
    """A minimal engine-side game state; _normalize_state fills the rest."""
    return {
        "hole_cards": ["As", "Kd"],
        "board_cards": [],
        "pot_size": 150,
        "stack_size": 10000,
        "call_amount": 100,
        "min_raise": 200,
        "blinds": {"small": 50, "big": 100},
        "active_players": 3,
        "payouts": {1: 0.5, 2: 0.3},
        "player_id": "hero",
        "opponent_id": "villain",
    }


def _toolstack_state():
    """The raw dict engine/table.py hands a bot — the tool-stack contract takes
    this shape verbatim, with no _normalize_state in between.

    Note the cards: table.py deals treys-encoded ints (``deck.draw``), and only
    _normalize_state turns those into the "As" strings the authored contract
    shows. The tool stack sees the ints, so this fixture must too.
    """
    from treys import Card

    state = _state()
    state.update(
        {
            "hole_cards": [Card.new("As"), Card.new("Kd")],
            "table_stacks": [10000, 8000, 12000],
            "hero_table_index": 0,
            "players_left": 12,
            "starting_field": 27,
            "paid_places": 4,
            "table_id": 1,
            "hand_id": 7,
            "tournament_id": 3,
            "position": "BTN",
            "preflop_spot_type": "open",
            "table_stats": {},
            "opponent_position": "BB",
            "opponent_stack_size": 8000,
            "opponent_stack_bb": 80.0,
            "opponent_stats": None,
            "_hand_events": [],
        }
    )
    return state


# A tool stack is the arena's contract: tool.py registers tool classes on import,
# bot.json names a registry bot plus the tools to stack on it.
PROBE_TOOL = """
from engine.bot_tools import register_bot_tool


@register_bot_tool
class ProbeTool:
    name = "probe_tool"
    priority = 10

    def apply(self, context, game_state):
        return context.with_forced_action("raise", 400).with_tool_event(
            {"tool": "probe_tool", "decision": "force_raise"}
        )
"""

PROBE_SPEC = {
    "type": "configured_tournament_equity",
    "params": {"tools": [{"type": "probe_tool"}]},
}


class SeatWorkerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.seats = []

    def tearDown(self):
        for seat in self.seats:
            seat.close()
        self._tmp.cleanup()

    def _seat(self, nick: str, source: str, *, bot: str = "bot", **kwargs) -> SeatWorker:
        folder = self.root / nick
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{bot}.py").write_text(source, encoding="utf-8")
        seat = SeatWorker(
            nick,
            str(folder),
            bot,
            f"{nick}.{bot}",
            timeout_ms=kwargs.pop("timeout_ms", 2000),
            **kwargs,
        )
        self.seats.append(seat)
        return seat

    def test_returns_the_bots_action(self):
        seat = self._seat(
            "raiser",
            "def get_action(state, api):\n"
            "    return 'raise', state['hero']['min_raise'] * 2\n",
        )
        self.assertEqual(seat.get_action(_state()), ("raise", 400))
        self.assertEqual(seat.strikes, 0)
        self.assertIsNone(seat.incident_summary())

    def test_state_reaches_the_bot_intact(self):
        # int-keyed payouts are why the parent pickles instead of sending JSON.
        seat = self._seat(
            "reader",
            "def get_action(state, api):\n"
            "    payouts = state['tournament']['payouts']\n"
            "    ok = 1 in payouts and state['hero']['hole'] == ['As', 'Kd']\n"
            "    return ('call', 0) if ok else ('fold', 0)\n",
        )
        self.assertEqual(seat.get_action(_state()), ("call", 0))

    def test_infinite_loop_is_killed_and_folds(self):
        seat = self._seat(
            "spinner",
            "def get_action(state, api):\n    while True:\n        pass\n",
            timeout_ms=300,
        )
        started = time.monotonic()
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        # The point of the change: bounded, and the process is gone rather than
        # joined by an executor shutdown.
        self.assertLess(time.monotonic() - started, 20.0)
        self.assertEqual(seat.timeouts, 1)
        self.assertEqual(seat.strikes, 1)
        self.assertIsNone(seat.proc)
        self.assertFalse(seat.disabled)

    def test_strike_cap_shuts_the_seat_off(self):
        seat = self._seat(
            "staller",
            "def get_action(state, api):\n    while True:\n        pass\n",
            timeout_ms=200,
            max_strikes=2,
        )
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        self.assertTrue(seat.disabled)
        self.assertEqual(seat.spawns, 2)

        # Disabled means no further spawns at all: folds are now free.
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        self.assertEqual(seat.spawns, 2)
        self.assertEqual(seat.incident_summary()["nick"], "staller")

    def test_memory_persists_across_decisions_in_one_process(self):
        seat = self._seat(
            "rememberer",
            "COUNT = 0\n"
            "def get_action(state, api):\n"
            "    global COUNT\n"
            "    COUNT += 1\n"
            "    return 'raise', COUNT\n",
        )
        self.assertEqual(seat.get_action(_state()), ("raise", 1))
        self.assertEqual(seat.get_action(_state()), ("raise", 2))
        self.assertEqual(seat.spawns, 1)

    def test_respawn_loses_cross_hand_memory(self):
        # The marker survives the kill; module state does not. First decision
        # spins and is killed, second answers from a fresh interpreter.
        marker = self.root / "spun_once"
        seat = self._seat(
            "forgetful",
            "import os\n"
            "COUNT = 0\n"
            "def get_action(state, api):\n"
            "    global COUNT\n"
            "    COUNT += 1\n"
            f"    if not os.path.exists({str(marker)!r}):\n"
            f"        open({str(marker)!r}, 'w').write('x')\n"
            "        while True:\n"
            "            pass\n"
            "    return 'raise', COUNT\n",
            timeout_ms=300,
        )
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        # COUNT is back to 1 rather than 2 — that reset is the penalty.
        self.assertEqual(seat.get_action(_state()), ("raise", 1))
        self.assertEqual(seat.spawns, 2)

    def test_printing_does_not_corrupt_the_protocol(self):
        seat = self._seat(
            "chatty",
            "import sys\n"
            "print('hello from the seat')\n"
            "def get_action(state, api):\n"
            "    print('deciding')\n"
            "    sys.stdout.write('more noise\\n')\n"
            "    return 'call', 0\n",
        )
        self.assertEqual(seat.get_action(_state()), ("call", 0))
        self.assertEqual(seat.get_action(_state()), ("call", 0))

    def test_exception_folds_without_a_strike(self):
        seat = self._seat(
            "crasher",
            "def get_action(state, api):\n    raise ValueError('boom')\n",
        )
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        self.assertEqual(seat.strikes, 0)
        self.assertEqual(seat.errors, 1)
        # Same process still serving — an exception is not a protocol breach.
        self.assertEqual(seat.spawns, 1)
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        self.assertEqual(seat.spawns, 1)

    def test_garbage_return_folds(self):
        seat = self._seat(
            "liar",
            "def get_action(state, api):\n    return 'teleport', 5\n",
        )
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)

    def test_self_exit_folds_and_respawns(self):
        seat = self._seat(
            "quitter",
            "import os\n"
            "def get_action(state, api):\n    os._exit(0)\n",
        )
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        self.assertEqual(seat.strikes, 1)
        self.assertEqual(seat.timeouts, 0)

    def test_unimportable_folder_folds_the_seat_instead_of_raising(self):
        seat = self._seat("broken", "raise RuntimeError('bad import')\n")
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        self.assertTrue(seat.disabled)
        self.assertIn("bad import", seat.last_error)

    def test_missing_bot_name_folds_the_seat(self):
        folder = self.root / "toolsonly"
        folder.mkdir()
        (folder / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
        seat = SeatWorker("toolsonly", str(folder), "bot", "toolsonly.bot", timeout_ms=1000)
        self.seats.append(seat)
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        self.assertTrue(seat.disabled)
        self.assertIn("has no bot", seat.last_error)

    def test_module_level_hang_is_caught_by_the_handshake(self):
        seat = self._seat(
            "importspinner",
            "while True:\n    pass\n\ndef get_action(state, api):\n    return 'call', 0\n",
            spawn_timeout_ms=1500,
        )
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        self.assertEqual(seat.timeouts, 1)
        self.assertIsNone(seat.proc)

    def test_forked_child_dies_with_the_seat(self):
        marker = self.root / "child_alive"
        seat = self._seat(
            "forker",
            "import os, time\n"
            "def get_action(state, api):\n"
            "    if os.fork() == 0:\n"
            f"        open({str(marker)!r}, 'w').write(str(os.getpid()))\n"
            "        time.sleep(60)\n"
            "        os._exit(0)\n"
            "    while True:\n"
            "        pass\n",
            timeout_ms=500,
        )
        self.assertEqual(seat.get_action(_state()), DEFAULT_ACTION)
        deadline = time.monotonic() + 5.0
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue(marker.exists(), "fixture never forked")
        child_pid = int(marker.read_text())
        time.sleep(0.2)
        with self.assertRaises(OSError):
            # Killing the process group takes the fork with it.
            os.kill(child_pid, 0)

    # -- the arena's contract: tool stacks ---------------------------------- #

    def _toolstack_seat(self, nick: str, source: str, **kwargs) -> SeatWorker:
        folder = self.root / nick
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "tool.py").write_text(source, encoding="utf-8")
        seat = SeatWorker(
            nick,
            str(folder),
            "",
            nick,
            timeout_ms=kwargs.pop("timeout_ms", 10000),
            kind=KIND_TOOLSTACK,
            module="tool",
            spec=kwargs.pop("spec", PROBE_SPEC),
            engine_config={},
            **kwargs,
        )
        self.seats.append(seat)
        return seat

    def test_toolstack_seat_plays(self):
        seat = self._toolstack_seat("stacker", PROBE_TOOL)
        self.assertEqual(seat.get_action(_toolstack_state()), ("raise", 400))
        self.assertEqual(seat.strikes, 0)
        self.assertIsNone(seat.incident_summary())

    def test_tool_events_survive_the_pipe(self):
        # The tool stack reports by writing into the state dict it was handed.
        # Across a pipe that dict is a copy, so the worker has to send the report
        # back and the parent has to write it into the caller's dict — table.py
        # reads it off there once get_action returns, and it feeds the replay.
        seat = self._toolstack_seat("reporter", PROBE_TOOL)
        state = _toolstack_state()
        seat.get_action(state)
        self.assertEqual(state["_bot_tool_event"]["tool"], "probe_tool")
        self.assertEqual(state["_bot_tool_event"]["decision"], "force_raise")
        self.assertEqual(
            [row["tool"] for row in state["_bot_tool_events"]], ["probe_tool"]
        )

    def test_a_tool_stack_never_reaches_the_parent(self):
        # The whole point of the seat: this process must not end up holding the
        # nick's code. If the tool registers here, containment is decorative.
        from engine.bot_tools import TOOL_REGISTRY

        seat = self._toolstack_seat("private", PROBE_TOOL)
        seat.get_action(_toolstack_state())
        self.assertNotIn("probe_tool", TOOL_REGISTRY)
        self.assertNotIn("ProbeTool", TOOL_REGISTRY)

    def test_unserializable_tool_event_still_pays_the_decision(self):
        # A tool that puts a non-JSON value in its event must lose the report,
        # not the action it legally made.
        seat = self._toolstack_seat(
            "sloppy",
            PROBE_TOOL.replace(
                '{"tool": "probe_tool", "decision": "force_raise"}',
                '{"tool": "probe_tool", "blob": object()}',
            ),
        )
        state = _toolstack_state()
        self.assertEqual(seat.get_action(state), ("raise", 400))
        self.assertNotIn("_bot_tool_event", state)

    def test_broken_tool_stack_folds_the_seat(self):
        seat = self._toolstack_seat("badstack", "raise RuntimeError('no import')\n")
        self.assertEqual(seat.get_action(_toolstack_state()), DEFAULT_ACTION)
        self.assertTrue(seat.disabled)
        self.assertIn("no import", seat.last_error)

    def test_unknown_tool_in_the_spec_folds_the_seat(self):
        seat = self._toolstack_seat(
            "ghosttool",
            PROBE_TOOL,
            spec={"type": "configured_tournament_equity",
                  "params": {"tools": [{"type": "not_a_tool"}]}},
        )
        self.assertEqual(seat.get_action(_toolstack_state()), DEFAULT_ACTION)
        self.assertTrue(seat.disabled)

    # -- identity does not leak through argv -------------------------------- #

    def test_argv_carries_no_identity(self):
        # /proc/<pid>/cmdline is world-readable: an argv roster is one `ps` away
        # from every other seat on the box.
        seat = self._seat("secretive", "def get_action(state, api):\n    return 'call', 0\n")
        seat.get_action(_state())
        self.assertIsNotNone(seat.proc)
        cmdline = Path(f"/proc/{seat.proc.pid}/cmdline").read_bytes().decode()
        self.assertIn("engine.seat_worker", cmdline)
        self.assertNotIn("secretive", cmdline)
        self.assertNotIn(str(self.root), cmdline)

    @unittest.skipUnless(os.geteuid() == 0, "uid drop needs a root parent")
    def test_uid_drop_takes_the_gid_with_it(self):
        # user= alone leaves the child at gid 0 — CPython only calls setregid
        # when group= is passed — so every root:root 0640 file stays readable.
        import pwd

        try:
            account = pwd.getpwnam("mttbot000")
        except KeyError:
            self.skipTest("mttbot000 not provisioned on this host")
        seat = self._seat(
            "dropped",
            "def get_action(state, api):\n    return 'call', 0\n",
            user="mttbot000",
        )
        seat.get_action(_state())
        self.assertIsNotNone(seat.proc)
        status = Path(f"/proc/{seat.proc.pid}/status").read_text()
        uid_line = next(line for line in status.splitlines() if line.startswith("Uid:"))
        gid_line = next(line for line in status.splitlines() if line.startswith("Gid:"))
        self.assertEqual(int(uid_line.split()[1]), account.pw_uid)
        self.assertEqual(int(gid_line.split()[1]), account.pw_gid)
        groups = next(line for line in status.splitlines() if line.startswith("Groups:"))
        self.assertEqual(groups.split()[1:], [])

    def test_close_reaps_the_process(self):
        seat = self._seat("polite", "def get_action(state, api):\n    return 'call', 0\n")
        seat.get_action(_state())
        proc = seat.proc
        self.assertIsNotNone(proc)
        seat.close()
        self.assertIsNone(seat.proc)
        self.assertIsNotNone(proc.poll())


if __name__ == "__main__":
    unittest.main()
