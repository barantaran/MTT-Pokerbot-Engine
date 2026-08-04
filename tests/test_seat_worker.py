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

from engine.seat_worker import DEFAULT_ACTION, SeatWorker


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
