"""Load self-service authored bots into the engine (see docs/BOT_ARCHITECTURE.md).

An authored bot is a bare ``get_action(state, api)`` function in a ``.py`` file
under ``plugins/<nick>/``; a tool is any sibling file without ``get_action``.
Composition is code — the bot imports and orders its tools with bare sibling
imports (``from shove_short import shove_short``). This loader makes those bare
imports resolve to the author's own folder and nowhere else, then wraps each bot
function in the ``_AuthoredBot`` adapter that injects the card-math ``api``.

Namespacing is required because a 2-max duel puts both nicks in one worker
process: if nick A and nick B both ship ``shove_short.py``, a bare
``import shove_short`` would collide in the global ``sys.modules``. Each nick is
loaded with its folder on ``sys.path``, its modules bound at import time, then
the nick-owned modules are moved under a private ``authored.<nick>.*`` namespace
so the next nick starts from a clean bare-name space. Stdlib modules pulled in
along the way (e.g. ``math``) are shared and left untouched.

Contrast with ``engine.plugins``: that loads *legacy* registry plugins whose
import self-registers a class into ``bot_factory`` / ``bot_tools``. Authored bots
register nothing — they are plain functions discovered by presence of
``get_action``.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from treys import Card

from engine.authored_api import AuthoredApi

_NAMESPACE_ROOT = "authored"


def _cards_to_str(cards):
    """treys ints (engine-internal) -> string cards (``"As"``) the author sees.

    The engine holds cards as treys ints; authored bots import nothing native, so
    the seam decodes them here. Pass-through for anything already a string."""
    if not cards:
        return []
    return [Card.int_to_str(c) if isinstance(c, int) else c for c in cards]


def _events(events):
    """The hand so far, in order — blinds, board runouts and actions.

    Everything the state no longer spells out is derived from this list, so it
    carries more than actions: position labels come from ``hand_start``, the
    preflop spot shape from the raise sequence, per-street contributions from
    the amounts.

    It is also the redaction boundary. The engine's event list is the *table's*
    log: it holds a ``deal`` event with every player's hole cards, and each
    action can carry the deciding tool's debug payload. So this is an allowlist
    projection — a new event type reaches no author until it is named here —
    and per-event keys the author can derive (``position``, ``call_amount``,
    ``pot_size``) are left out with the rest."""
    if not events:
        return []
    out = []
    for event in events:
        kind = event.get("type")
        if kind == "hand_start":
            out.append({
                "type": "hand_start",
                "hand_id": event.get("hand_id"),
                "level": event.get("level"),
                "blinds": dict(event.get("blinds") or {}),
                "button_seat": event.get("button_seat"),
                "players": [
                    {"seat": seat.get("seat"), "name": seat.get("name"),
                     "stack": seat.get("stack")}
                    for seat in event.get("players") or []
                ],
            })
        elif kind == "post_blind":
            out.append({
                "type": "post_blind",
                "player": event.get("player"),
                "seat": event.get("seat"),
                "blind": event.get("blind"),
                "amount": event.get("amount"),
                "street": event.get("street"),
            })
        elif kind == "board":
            out.append({
                "type": "board",
                "cards": list(event.get("cards") or []),
                "street": event.get("street"),
            })
        elif kind == "action":
            out.append({
                "type": "action",
                "player": event.get("player"),
                "action": event.get("action"),
                "amount": event.get("amount"),
                "street": event.get("street"),
            })
    return out


def _normalize_state(state):
    """Translate the engine's internal ``state`` (treys ints, flat, built for the
    legacy bot classes in ``engine/table.py``) into the raw-facts dict authored
    bots are promised (``docs/BOT_ARCHITECTURE.md``). Legacy bots never go
    through here; they read the internal shape directly.

    **Only what the author cannot compute.** The engine ships current table
    truth it alone holds, plus the event log everything else follows from. It
    does not ship anything the author can work out from those, which is why
    there is no ``stack_bb`` (``table_stacks[seat] / blinds["big"]``), no
    ``street`` (``len(board)``), no ``position`` (``button_seat`` and the
    ``hand_start`` seating), no ``spot_type`` (the raise sequence in
    ``events``), and no equity or threshold of any kind — equity is the
    author's, computed through ``api`` against the pokerstove evaluator.

    Two things stay that a determined author could reconstruct:

    * ``call_amount`` / ``min_raise`` — engine *rulings*, not measurements. A
      seat that derives them wrong returns an illegal action every time, so the
      legal bounds of the reply come from the side that enforces them.
    * ``stats`` — a seat is only called when it is to act, so it never observes
      hands it sat out or streets after it folded, and a timeout kill wipes any
      accumulator it kept. Keyed by player name: the author decides who matters
      (the ``events`` name the aggressor), the engine only supplies the counts
      it alone could keep. Today the tracker prices one snapshot per decision,
      so the map holds the faced aggressor.
    """
    blinds = state.get("blinds") or {}
    active = int(state.get("active_players", 1) or 1)
    opponent_id = str(state.get("opponent_id", "") or "")
    opponent_stats = state.get("opponent_stats")
    return {
        "hero": {
            "hole": _cards_to_str(state.get("hole_cards")),
            "seat": int(state.get("hero_table_index", 0) or 0),
            "call_amount": int(state.get("call_amount", 0) or 0),
            "min_raise": int(state.get("min_raise", 0) or 0),
        },
        "board": _cards_to_str(state.get("board_cards")),
        "pot": int(state.get("pot_size", 0) or 0),
        "blinds": dict(blinds),
        "table_stacks": [int(chips or 0) for chips in state.get("table_stacks") or []],
        "button_seat": int(state.get("button_seat", 0) or 0),
        # Villains are hidden, so this is the count, not the hands: the author
        # builds `[None] * live_opponents` for `api.deal` / `api.equity`, which
        # samples each unknown hand from the live deck every runout.
        "live_opponents": max(0, active - 1),
        "events": _events(state.get("_hand_events")),
        "tournament": {
            "players_left": state.get("players_left"),
            "starting_field": state.get("starting_field"),
            "paid_places": state.get("paid_places"),
            "payouts": dict(state.get("payouts") or {}),
        },
        "stats": {opponent_id: opponent_stats} if opponent_id and opponent_stats else {},
    }


class _AuthoredBot:
    """Bridge the engine's 1-arg ``bot.get_action(game_state)`` call convention
    (``engine/player_state.py``) to the author's 2-arg module function, injecting
    the card-math ``api`` and normalizing the engine ``state`` to the author-facing
    shape. Internal — authors never see it."""

    def __init__(self, fn, api):
        self.fn = fn
        self.api = api

    def get_action(self, game_state):
        return self.fn(_normalize_state(game_state), self.api)


def _ensure_package(name: str) -> None:
    """Register an empty namespace package in ``sys.modules`` if absent, so a
    dotted ``authored.<nick>`` key is a valid, importable package."""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = []  # marks it a package; nothing is imported through it
    sys.modules[name] = module


def _reparent_nick_modules(nick: str, folder: Path, before: set[str]) -> None:
    """Move modules newly imported *from this nick's folder* out of the bare
    namespace into ``authored.<nick>.*`` so the next nick's identically-named
    files load fresh instead of hitting a stale cache. Only nick-owned files are
    moved; transitively-imported stdlib stays shared under its bare name."""
    _ensure_package(_NAMESPACE_ROOT)
    _ensure_package(f"{_NAMESPACE_ROOT}.{nick}")
    folder = folder.resolve()
    for name in set(sys.modules) - before:
        module = sys.modules.get(name)
        file = getattr(module, "__file__", None)
        if file and Path(file).resolve().parent == folder:
            sys.modules.pop(name)
            sys.modules[f"{_NAMESPACE_ROOT}.{nick}.{name}"] = module


def load_authored_bots(nick: str, folder, api: AuthoredApi | None = None) -> dict:
    """Import every ``.py`` in ``plugins/<nick>/`` and return the bots it defines.

    Returns ``{bot_name: _AuthoredBot}`` for each file that defines a callable
    ``get_action``; files without one are tools, imported (so bots can reference
    them) but not returned as entries. ``api`` is built once per worker and shared
    across bots — pass a single instance to reuse it, or let each call make one.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"plugin folder not found: {folder}")
    if api is None:
        api = AuthoredApi()

    py_files = sorted(p for p in folder.glob("*.py") if not p.name.startswith("_"))
    before = set(sys.modules)
    sys.path.insert(0, str(folder))
    modules = {}
    try:
        for path in py_files:
            modules[path.stem] = importlib.import_module(path.stem)
    finally:
        try:
            sys.path.remove(str(folder))
        except ValueError:
            pass

    bots = {}
    for stem, module in modules.items():
        fn = getattr(module, "get_action", None)
        if callable(fn):
            bots[stem] = _AuthoredBot(fn, api)

    _reparent_nick_modules(nick, folder, before)
    return bots
