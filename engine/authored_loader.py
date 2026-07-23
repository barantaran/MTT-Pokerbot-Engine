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

from engine.authored_api import AuthoredApi

_NAMESPACE_ROOT = "authored"


class _AuthoredBot:
    """Bridge the engine's 1-arg ``bot.get_action(game_state)`` call convention
    (``engine/player_state.py``) to the author's 2-arg module function, injecting
    the card-math ``api``. Internal — authors never see it."""

    def __init__(self, fn, api):
        self.fn = fn
        self.api = api

    def get_action(self, game_state):
        return self.fn(game_state, self.api)


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
