"""Load author-private bot/tool plugins into the engine registries.

A plugin is an ordinary Python module that lives *outside* the engine repo (on
the runner host, pre-provisioned before a run). On import it calls
``engine.bot_tools.register_bot_tool`` and/or ``engine.bot_factory.register_bot``
to populate the two engine registries. Loading is explicit opt-in per run — never
a scan of a shared directory — so other engine users never discover it.

Entry syntax (one string per plugin):
- ``"module"``              -- an already-importable module name.
- ``"/path/to/dir::module"`` -- ``dir`` is added to ``sys.path`` first, then
  ``module`` imported from it.

Entries come from the ``MTT_PLUGINS`` environment variable (newline-joined,
inherited by both fork and spawn workers) and/or the run-config ``plugins`` list.
A newline is used as the outer separator because an entry's own ``dir::module``
syntax already contains ``:``, so ``os.pathsep`` would be ambiguous on POSIX.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ENV_VAR = "MTT_PLUGINS"
ENTRY_SEP = "\n"
_SEP = "::"

# Entries already imported this process — makes double-loading (parent import +
# fork inheritance + worker initializer) harmless.
_LOADED: set[str] = set()


def _add_path(root: str | Path) -> None:
    resolved = str(Path(root).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def load_plugin(entry: str) -> None:
    """Import one plugin entry (``module`` or ``dir::module``); registration is a
    side effect of the import."""
    entry = entry.strip()
    if not entry or entry in _LOADED:
        return
    if _SEP in entry:
        root, module = entry.split(_SEP, 1)
        _add_path(root)
    else:
        module = entry
    importlib.import_module(module.strip())
    _LOADED.add(entry)


def _entries_from_env() -> list[str]:
    raw = os.environ.get(ENV_VAR, "")
    return [entry for entry in raw.split(ENTRY_SEP) if entry.strip()]


def load_plugins(entries=None) -> list[str]:
    """Merge ``MTT_PLUGINS`` env entries with ``entries`` (preserving order,
    de-duplicated) and import each. Returns the merged entry list."""
    merged: list[str] = []
    for entry in _entries_from_env() + list(entries or []):
        entry = entry.strip()
        if entry and entry not in merged:
            merged.append(entry)
    for entry in merged:
        load_plugin(entry)
    return merged
