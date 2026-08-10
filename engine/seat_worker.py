"""Out-of-process seats for self-service authored bots (``service/INTEGRITY.md``).

An authored bot is third-party code the arena runs on its own VM. Loaded
in-process it can read every other nick's modules out of ``sys.modules``, read
the resolved run config off the disk, write the fuse-mounted results bucket —
and, because a Python thread cannot be killed, hang the entire run with
``while True: pass``. (``engine/table.py`` used to wrap the call in a
``ThreadPoolExecutor``; ``future.result(timeout=)`` stops *waiting* but leaving
the ``with`` block calls ``shutdown(wait=True)``, which then joins the runaway
thread forever.)

So each authored seat gets its own subprocess, spawned lazily, living for the
tournament, answering one decision at a time over a pipe::

    parent                                           seat worker
      state -> _normalize_state -> pickle   ------>  fn(state, api)
      (action, amount)  <---  JSON, validated  <---

The directions are deliberately asymmetric. The parent pickles *to* the worker
(cheap, and it preserves the int-keyed ``payouts`` map the author contract
promises); the worker answers in JSON, which the parent type-checks — unpickling
whatever a hostile seat produced would hand it the parent process.

Two authoring contracts ride the same pipe, selected by the handshake ``kind``:

``authored``
    A bare ``get_action(state, api)`` per module (``engine/authored_loader.py``).
    The parent normalizes the state first.

``toolstack``
    The arena's contract: ``tool.py`` registers tool classes via
    ``register_bot_tool`` and ``bot.json`` names a registry bot plus its tool
    list. The worker imports the nick's module and builds exactly one bot from
    the spec, then calls ``bot.get_action(state)`` on the raw engine dict.
    Its tools report through ``state["_bot_tool_event(s)"]`` — a back-channel
    that does not survive a pipe, so the worker returns those in the reply and
    the parent writes them into the caller's dict. ``engine/table.py`` reads
    them off the state *after* ``get_action`` returns and they feed the replay
    slices; drop them and every replay silently loses ``tool_events``.

Identity (nick, folder, bot, spec) travels in that first frame rather than in
argv: ``/proc/<pid>/cmdline`` is world-readable, so an argv roster is one ``ps``
away from every other seat on the box.

Because the parent normalizes before sending, this pipe is also the single
chokepoint where opponent names will be swapped for per-tournament aliases
(INTEGRITY.md change 3); nothing downstream of the engine ever sees them.

A worker that misses its deadline is SIGKILLed — the whole process group, since a
bot may have forked — the seat folds, and the next decision respawns it with
fresh memory. Losing cross-hand state *is* the penalty. After ``max_strikes`` the
seat stops respawning and folds out the rest of the tournament.
"""

from __future__ import annotations

import json
import os
import pickle
import select
import signal
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

_ENGINE_ROOT = Path(__file__).resolve().parents[1]
_LEN = struct.Struct(">I")
_MAX_FRAME = 8 << 20  # a normalized state is a few KB; this is a sanity bound

# Facing a bet this folds, facing none table.py's default branch checks it —
# the same thing the in-process path did on timeout or exception.
DEFAULT_ACTION: Tuple[str, int] = ("fold", 0)
_ACTIONS = frozenset({"fold", "call", "check", "raise"})

KIND_AUTHORED = "authored"
KIND_TOOLSTACK = "toolstack"
_KINDS = frozenset({KIND_AUTHORED, KIND_TOOLSTACK})

DEFAULT_MAX_STRIKES = 5
# Import of the nick's folder plus pokerstove (~0.05s measured) — generous, and
# it is what catches module-level `while True: pass`.
DEFAULT_SPAWN_TIMEOUT_MS = 10_000


class _SeatTimeout(Exception):
    """Worker missed its deadline. Always fatal to that process."""


class _SeatDead(Exception):
    """Worker exited, crashed, or broke the protocol."""


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #


def _read_exact(fd: int, count: int, deadline: float) -> bytes:
    chunks = []
    got = 0
    while got < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            raise _SeatTimeout()
        try:
            block = os.read(fd, count - got)
        except OSError as exc:
            raise _SeatDead(f"read failed: {exc}") from exc
        if not block:
            raise _SeatDead("worker closed its pipe")
        chunks.append(block)
        got += len(block)
    return b"".join(chunks)


def _read_frame(fd: int, deadline: float) -> bytes:
    (size,) = _LEN.unpack(_read_exact(fd, _LEN.size, deadline))
    if size > _MAX_FRAME:
        raise _SeatDead(f"frame too large: {size}")
    return _read_exact(fd, size, deadline)


def _write_frame(fd: int, payload: bytes, deadline: float) -> None:
    view = memoryview(_LEN.pack(len(payload)) + payload)
    sent = 0
    while sent < len(view):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([], [fd], [], remaining)[1]:
            raise _SeatTimeout()
        try:
            sent += os.write(fd, view[sent:])
        except OSError as exc:
            raise _SeatDead(f"write failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# parent side
# --------------------------------------------------------------------------- #


class SeatWorker:
    """One authored seat, run in its own process. Quacks like ``_AuthoredBot``.

    ``PlayerState`` only needs ``.name`` and ``.get_action(game_state)``.
    ``self_timing`` tells ``engine/table.py`` the deadline is enforced here, so it
    must not wrap the call in its own (unkillable) thread.
    """

    self_timing = True

    def __init__(
        self,
        nick: str,
        folder: str,
        bot_name: str,
        name: str,
        *,
        timeout_ms: int,
        max_strikes: int = DEFAULT_MAX_STRIKES,
        spawn_timeout_ms: int = DEFAULT_SPAWN_TIMEOUT_MS,
        user: str | None = None,
        kind: str = KIND_AUTHORED,
        module: str = "",
        spec: Mapping[str, Any] | None = None,
        engine_config: Mapping[str, Any] | None = None,
    ):
        if kind not in _KINDS:
            raise ValueError(f"unknown seat kind {kind!r}; expected one of {sorted(_KINDS)}")
        self.nick = nick
        self.folder = str(folder)
        self.bot_name = bot_name
        self.name = name
        self.timeout_ms = int(timeout_ms)
        self.max_strikes = int(max_strikes)
        self.spawn_timeout_ms = int(spawn_timeout_ms)
        # Dropping the seat to an unprivileged uid (INTEGRITY.md change 1)
        # requires a root parent: Popen(user=) calls setuid in the child, and
        # _kill's killpg needs to outrank the uid it signals. With no user set
        # the worker inherits the engine's own.
        self.user = user
        self.kind = kind
        self.module = module
        self.spec = dict(spec or {})
        self.engine_config = dict(engine_config or {})

        self.proc: subprocess.Popen | None = None
        self.strikes = 0
        self.timeouts = 0
        self.errors = 0
        self.spawns = 0
        self.disabled = False
        self.last_error = ""

    # -- process lifecycle -------------------------------------------------- #

    def _spawn(self) -> bool:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(_ENGINE_ROOT), env.get("PYTHONPATH", "")) if part
        )
        # The worker loads exactly one nick's folder; registry plugins are the
        # parent's business and must not be imported next to untrusted code.
        env.pop("MTT_PLUGINS", None)

        # A read-only plugin dir (change 1 chowns them root:mttbotNNN 0640) makes
        # __pycache__ writes fail noisily for nothing.
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        kwargs: Dict[str, Any] = {}
        if self.user:
            # user= alone is not containment: CPython only calls setregid when
            # group= is passed and setgroups when extra_groups= is, so the child
            # would keep gid 0 and read every root:root 0640 file on the box.
            kwargs.update(
                user=self.user,
                group=self.user,
                extra_groups=[],
                umask=0o077,
            )
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "engine.seat_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                cwd=str(_ENGINE_ROOT),
                env=env,
                close_fds=True,
                # Own process group, so a kill takes any children the bot forked.
                start_new_session=True,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001 - spawn failure must not kill the run
            self.proc = None
            return self._strike(f"spawn failed: {exc}")

        self.spawns += 1
        handshake = pickle.dumps(
            {
                "kind": self.kind,
                "nick": self.nick,
                "folder": self.folder,
                "bot": self.bot_name,
                "name": self.name,
                "module": self.module,
                "spec": self.spec,
                "engine_config": self.engine_config,
            },
            protocol=4,
        )
        try:
            reply = self._exchange(handshake, self.spawn_timeout_ms)
        except _SeatTimeout:
            self.timeouts += 1
            return self._strike("hung during import")
        except _SeatDead as exc:
            return self._strike(f"died during import: {exc}")
        if not reply.get("ok"):
            # A folder that cannot import, or a bot name that is not there. One
            # broken nick must not take the run down, so the seat simply folds.
            self._kill()
            self.strikes = self.max_strikes
            self.disabled = True
            self.last_error = str(reply.get("error", "load failed"))
            print(
                f"seat {self.name}: {self.last_error} — seat folds this tournament",
                flush=True,
            )
            return False
        return True

    def _kill(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except OSError:
                pass
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 - reaping is best effort
            pass

    def _strike(self, reason: str) -> bool:
        """Record a fatal incident, kill the process, decide on respawn."""
        self._kill()
        self.strikes += 1
        self.last_error = reason
        if self.strikes >= self.max_strikes:
            self.disabled = True
            print(
                f"seat {self.name}: {reason} (strike {self.strikes}/{self.max_strikes}) "
                "— seat folds the rest of the tournament",
                flush=True,
            )
        else:
            print(
                f"seat {self.name}: {reason} (strike {self.strikes}/{self.max_strikes}) "
                "— respawning with fresh memory",
                flush=True,
            )
        return False

    def close(self) -> None:
        self._kill()

    def __del__(self):  # pragma: no cover - interpreter teardown
        try:
            self._kill()
        except Exception:  # noqa: BLE001
            pass

    # -- one round trip ----------------------------------------------------- #

    def _exchange(self, payload: bytes | None, timeout_ms: int) -> Dict[str, Any]:
        """Send one frame (the handshake, or a state) and read one reply."""
        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        deadline = time.monotonic() + timeout_ms / 1000.0
        if payload is not None:
            _write_frame(self.proc.stdin.fileno(), payload, deadline)
        raw = _read_frame(self.proc.stdout.fileno(), deadline)
        try:
            reply = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            raise _SeatDead(f"unparseable reply: {exc}") from exc
        if not isinstance(reply, dict):
            raise _SeatDead(f"reply is {type(reply).__name__}, expected object")
        return reply

    def get_action(self, game_state) -> Tuple[str, Any]:
        if self.disabled:
            return DEFAULT_ACTION

        if self.kind == KIND_TOOLSTACK:
            # The tool-stack contract takes the raw engine dict; table.py builds
            # it fresh per decision, so there is nothing to normalize.
            payload = pickle.dumps(game_state, protocol=4)
        else:
            # Normalizing here (not in the worker) keeps the author-facing shape —
            # and the future alias substitution — on the trusted side of the pipe.
            from engine.authored_loader import _normalize_state

            payload = pickle.dumps(_normalize_state(game_state), protocol=4)

        if self.proc is None and not self._spawn():
            return DEFAULT_ACTION
        try:
            reply = self._exchange(payload, self.timeout_ms)
        except _SeatTimeout:
            self.timeouts += 1
            self._strike(f"timed out (> {self.timeout_ms}ms)")
            return DEFAULT_ACTION
        except _SeatDead as exc:
            self._strike(str(exc))
            return DEFAULT_ACTION

        if not reply.get("ok"):
            # The bot raised. Same handling as in-process: fold this decision and
            # keep the seat — an exception is not a protocol violation, and the
            # worker is still sitting on the pipe.
            self.errors += 1
            self.last_error = str(reply.get("error", "bot raised"))
            print(f"Bot {self.name} raised exception: {self.last_error}. Defaulting to fold.", flush=True)
            return DEFAULT_ACTION

        action = reply.get("action")
        amount = reply.get("amount")
        if not isinstance(action, str) or action not in _ACTIONS:
            self.errors += 1
            return DEFAULT_ACTION
        if amount is not None and not isinstance(amount, int):
            # json gives back int/float; a float chip count was never legal.
            self.errors += 1
            return DEFAULT_ACTION

        # Restore the back-channel the pipe broke: a tool stack reports what it
        # decided by writing into the state dict it was handed, and table.py
        # reads that off *this* dict once we return (engine/table.py, the
        # `_bot_tool_event` lookup right after get_action). Without this the
        # replay slices lose every tool_event.
        if self.kind == KIND_TOOLSTACK and isinstance(game_state, dict):
            event = reply.get("tool_event")
            if isinstance(event, dict):
                game_state["_bot_tool_event"] = event
            events = reply.get("tool_events")
            if isinstance(events, list) and all(isinstance(row, dict) for row in events):
                game_state["_bot_tool_events"] = events

        return action, amount

    def incident_summary(self) -> Dict[str, Any] | None:
        """Per-seat trouble for the run report, or None when the seat behaved."""
        if not (self.timeouts or self.errors or self.disabled or self.strikes):
            return None
        return {
            "nick": self.nick,
            "timeouts": self.timeouts,
            "errors": self.errors,
            "strikes": self.strikes,
            "spawns": self.spawns,
            "disabled": self.disabled,
            "last_error": self.last_error,
        }


# --------------------------------------------------------------------------- #
# worker side
# --------------------------------------------------------------------------- #


def _blocking_read_frame(fd: int) -> bytes | None:
    header = b""
    while len(header) < _LEN.size:
        block = os.read(fd, _LEN.size - len(header))
        if not block:
            return None  # parent closed the pipe: tournament over
        header += block
    (size,) = _LEN.unpack(header)
    body = b""
    while len(body) < size:
        block = os.read(fd, size - len(body))
        if not block:
            return None
        body += block
    return body


def _reply(fd: int, obj: Dict[str, Any]) -> None:
    payload = json.dumps(obj).encode("utf-8")
    os.write(fd, _LEN.pack(len(payload)) + payload)


def _coerce_amount(amount):
    if amount is None:
        return None
    return int(amount)


def _load_authored(handshake: Dict[str, Any]):
    """The bare ``get_action(state, api)`` contract — one function per module."""
    from engine.authored_api import AuthoredApi
    from engine.authored_loader import load_authored_bots

    nick = str(handshake.get("nick") or "")
    bot_name = str(handshake.get("bot") or "")
    api = AuthoredApi()
    bots = load_authored_bots(nick, str(handshake.get("folder") or ""), api)
    if bot_name not in bots:
        raise KeyError(f"nick {nick!r} has no bot {bot_name!r}; bots: {sorted(bots)}")
    fn = bots[bot_name].fn
    return lambda state: fn(state, api)


def _load_toolstack(handshake: Dict[str, Any]):
    """The arena contract — ``tool.py`` registers tool classes on import, and the
    spec from ``bot.json`` names a registry bot plus the tools to stack on it.

    Exactly one bot: a seat is one entry, and ``count`` is the parent's business.
    """
    from engine.bot_factory import build_configurable_bots
    from engine.plugins import load_plugin

    module = str(handshake.get("module") or "")
    if not module:
        raise ValueError("a toolstack seat needs a module name")
    # Registration is a side effect of this import — the whole point is that the
    # parent never performs it.
    load_plugin(f"{handshake.get('folder') or ''}::{module}")

    spec = dict(handshake.get("spec") or {})
    spec["count"] = 1
    bots, _ = build_configurable_bots(
        {}, dict(handshake.get("engine_config") or {}), extra_specs=[spec]
    )
    if len(bots) != 1:
        raise ValueError(f"spec built {len(bots)} bots, expected exactly 1")
    bot = bots[0]
    name = str(handshake.get("name") or "")
    if name:
        bot.name = name
    return bot.get_action


_LOADERS = {KIND_AUTHORED: _load_authored, KIND_TOOLSTACK: _load_toolstack}


def _tool_events_for_reply(state: Any) -> Dict[str, Any]:
    """A tool stack's reports, but only if they survive JSON.

    A tool that stuffs a non-serializable value into its event would otherwise
    take down ``_reply`` mid-frame, and the seat would lose a decision it had
    already made legally. Dropping the report is the cheaper failure.
    """
    if not isinstance(state, dict):
        return {}
    out: Dict[str, Any] = {}
    event = state.get("_bot_tool_event")
    if isinstance(event, dict):
        out["tool_event"] = event
    events = state.get("_bot_tool_events")
    if isinstance(events, list) and all(isinstance(row, dict) for row in events):
        out["tool_events"] = events
    try:
        json.dumps(out)
    except (TypeError, ValueError):
        return {}
    return out


def _serve() -> int:
    # Take private copies of the protocol fds, then point the inherited stdio
    # somewhere harmless: a bot that prints (or reads input) must not be able to
    # corrupt or consume the frame stream.
    proto_in = os.dup(0)
    proto_out = os.dup(1)
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.close(devnull)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    # Identity arrives in the first frame, not argv — /proc/<pid>/cmdline is
    # world-readable and would hand every seat the roster.
    frame = _blocking_read_frame(proto_in)
    if frame is None:
        return 0

    kind = KIND_AUTHORED
    try:
        handshake = pickle.loads(frame)
        if not isinstance(handshake, dict):
            raise TypeError(f"handshake is {type(handshake).__name__}, expected dict")
        kind = str(handshake.get("kind") or KIND_AUTHORED)
        loader = _LOADERS.get(kind)
        if loader is None:
            raise ValueError(f"unknown seat kind {kind!r}")
        decide = loader(handshake)
    except BaseException as exc:  # noqa: BLE001 - report, do not traceback into the pipe
        _reply(proto_out, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1
    _reply(proto_out, {"ok": True})

    while True:
        frame = _blocking_read_frame(proto_in)
        if frame is None:
            return 0
        state = pickle.loads(frame)
        try:
            action, amount = decide(state)
            reply = {"ok": True, "action": str(action), "amount": _coerce_amount(amount)}
            if kind == KIND_TOOLSTACK:
                reply.update(_tool_events_for_reply(state))
        except BaseException as exc:  # noqa: BLE001 - a bad seat folds, it never kills the hand
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _reply(proto_out, reply)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        print(
            "usage: python -m engine.seat_worker  (identity arrives on stdin)",
            file=sys.stderr,
        )
        return 2
    return _serve()


if __name__ == "__main__":
    raise SystemExit(main())
