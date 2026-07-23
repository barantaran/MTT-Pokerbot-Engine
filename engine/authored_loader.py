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


def _history(events):
    """Actions taken so far this hand, in order. Filters the engine's mixed event
    log (``engine/table.py``) down to a stable per-action shape; board/blind/
    showdown events are dropped — the author reads board and pot from the top
    level, not by replaying them."""
    if not events:
        return []
    out = []
    for event in events:
        if event.get("type") == "action":
            out.append({
                "player": event.get("player"),
                "action": event.get("action"),
                "amount": event.get("amount"),
                "street": event.get("street"),
                "position": event.get("position"),
            })
    return out


def _normalize_state(state):
    """Translate the engine's internal ``state`` (treys ints, flat, built for the
    legacy bot classes in ``engine/table.py``) into the raw-facts dict authored
    bots are promised (``docs/BOT_ARCHITECTURE.md`` — cards as strings, hero and
    opponent grouped, stacks in chips *and* bb, action history, ``villain_hands``
    ready for ``api``). Legacy bots never go through here; they read the internal
    shape directly. Villains are hidden, so ``villain_hands`` is one ``None`` per
    live opponent — the ``api`` samples them each runout."""
    blinds = state.get("blinds") or {}
    big = blinds.get("big", 0) or 0
    stack = int(state.get("stack_size", 0) or 0)
    active = int(state.get("active_players", 1) or 1)
    return {
        "hero": {
            "hole": _cards_to_str(state.get("hole_cards")),
            "stack": stack,
            "stack_bb": stack / big if big else 0.0,
            "position": state.get("position", ""),
            "call_amount": int(state.get("call_amount", 0) or 0),
            "min_raise": int(state.get("min_raise", 0) or 0),
        },
        "board": _cards_to_str(state.get("board_cards")),
        "pot": int(state.get("pot_size", 0) or 0),
        "blinds": dict(blinds),
        "villain_hands": [None] * max(0, active - 1),
        "opponent": {
            "id": state.get("opponent_id", ""),
            "position": state.get("opponent_position", ""),
            "stack": int(state.get("opponent_stack_size", 0) or 0),
            "stack_bb": float(state.get("opponent_stack_bb", 0.0) or 0.0),
            "stats": state.get("opponent_stats"),
        },
        "history": _history(state.get("_hand_events")),
        "tournament": {
            "players_left": state.get("players_left"),
            "starting_field": state.get("starting_field"),
            "paid_places": state.get("paid_places"),
            "payouts": dict(state.get("payouts") or {}),
        },
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
