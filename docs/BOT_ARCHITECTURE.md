# Bot architecture — self-service authoring

How the abstractions in `SELF_SERVICE_GLOSSARY.md` are realized as code. The
glossary is the *what* (roles, contracts); this is the *how* (files, the entry
seam, the injected card-math `api`). Implementation vocabulary lives only here
and never leaks back into the glossary.

Converged with user 2026-07-23.

## One idea

A **bot is one function**, `get_action`, that receives the game state and returns
a decision. A **tool** is any other function the bot calls. Composition is code:
the bot imports its tools and orders their calls. There is no config file, no
class hierarchy, no engine-side tool registry.

```python
# plugins/yourchev/aggro.py   — a bot
from shove_short import shove_short
from value_bet   import value_bet

DESCRIPTION = "push-fold short, value-bet deep"

def get_action(state, api):
    return (shove_short(state, api)
            or value_bet(state, api)
            or ("fold", None))
```

```python
# plugins/yourchev/shove_short.py   — a tool
def shove_short(state, api):
    if state["hero"]["stack_bb"] > 12:
        return None                       # not my spot, defer
    eq = api.equity(state["hero"]["hole"], state["board"],
                    state["villain_hands"], n=500)
    if eq > 0.42:
        return ("raise", state["hero"]["stack"])   # jam
    return None
```

## Files

Everything a nick owns lives under one folder; the folder *is* the nick.

```
plugins/<nick>/
  shove_short.py     # tool  — one function, no get_action
  value_bet.py       # tool
  my_icm.py          # tool
  aggro.py           # bot   — has get_action, imports tools
  nitty.py           # bot   — different composition of the same tools
```

- **One tool, one file, one function.** A tool is independently uploaded and
  independently validated: editing `value_bet.py` re-checks only that file. A
  monolithic toolbox would re-upload and re-validate the whole thing on every
  tweak.
- **A bot is a tool file that defines `get_action`.** Nothing else distinguishes
  them — no prefix, no manifest. The classification is static and obvious:

  | file defines `get_action`? | it is a | upload route |
  |---|---|---|
  | yes | bot (an entry) | `POST /bot` |
  | no  | tool           | `POST /tool` |

  `POST /bot` rejects a file with no `get_action` ("no get_action"); `POST /tool`
  rejects a file that has one.
- **Names are per-nick.** Filenames are unique within `plugins/<nick>/`, so tool
  and bot names are unique per nick and the nick namespaces them. Two authors may
  both have `shove_short.py`; they never collide.

## The entry seam

```
get_action(state, api) -> (action, amount)
```

- `action` is one of `"fold"`, `"call"`, `"raise"`. `amount` is chips for
  `"raise"`, ignored (use `0` or `None`) for `"fold"` / `"call"`.
- The engine clamps the return to a legal action for the seat (an over-jam
  becomes all-in, an illegal fold-to-check becomes check). The author never has
  to know the exact legal bounds; returning intent is enough.
- This mirrors the legacy `Bot.get_action` (`engine/player_interface.py`), which
  is why the name was kept. The difference: no class, no `self`, no engine-owned
  threshold machinery in between.

### `state` — raw facts, no strategy

A plain dict of what happened, not what to do. Hole cards, board, pot, stacks in
chips and bb, positions, the full action history this hand, and per-opponent
stats. The engine owns card facts; the author owns every judgement made from
them. There is no `DecisionContext`, no precomputed equity, no thresholds — those
were the old middle layer and are gone from the self-service surface.

## The `api` — injected card math

`api` is **injected**, not imported. It is the one capability the sandbox grants,
and it is the only way to reach pokerstove (a native cp310 C++ wheel that cannot
be reimplemented in pure Python). Injection keeps the no-imports rule intact:
author code imports nothing native.

Surface (raw primitives plus one convenience):

| call | returns | use |
|---|---|---|
| `api.showdown(hero7, villain7)` | `1.0` / `0.5` / `0.0` | who wins a fixed 7-card matchup |
| `api.deal(hole, board, villain_hands, n)` | seeded iterator of runouts | roll your own Monte Carlo |
| `api.equity(hole, board, villain_hands, n)` | float 0–1 | equity now, without writing the loop |

`equity` is `deal` + `showdown` averaged — convenience, not a different
capability. An author who wants a custom estimator (weighted ranges, early exit)
uses `deal` and `showdown` directly.

**Seeded and deterministic.** `deal` / `equity` seed their RNG from the call
arguments, so the same spot yields the same estimate in every worker and on every
SPOT resume. Authors compute ICM, ranges, and equity themselves — the engine
hands over card math, never a strategy.

## Import rule (validation contract)

A bot or tool file may import **only its siblings in its own `plugins/<nick>/`
folder**. Everything else is denied statically, before the file is ever run:

- **Denied:** `engine`, `pokerstove`, `os` / `sys` / `subprocess` / `socket`,
  any third-party package, any other nick's folder.
- **Allowed:** sibling files in the same nick folder; optionally a `math`
  allowlist for pure numerics.
- Card math and randomness come through the injected `api` only.

## Why validation is static-only

`POST /validate` (and the whole public Cloud Run service) **never imports or
executes author code.** The service holds `objectAdmin` on `gs://mtt-results`;
importing an uploaded module would be RCE against that bucket. Validation is pure
`ast` inspection: does it parse, does a bot define `get_action`, does a tool avoid
one, are all imports siblings-only.

Anything that needs the file to actually run — a bad runtime value, an
`api` misuse — is checked **box-side**, where the wheel already exists and there
are no bucket credentials. The two contexts are enforced separately:

- **validate** (public service) — static only, never runs author code.
- **run / dryrun** (box) — author code executes by design, isolated, no bucket
  creds.

Restricted `exec` is not a security boundary and is not relied on as one.

## Legacy

The ~24 bot **classes** (`RandomBot`, `CallBot`, the EV bots, `phase*.py`) stay
in the engine as research baselines for `fixed_bot_mtt_evaluation`. "Legacy" means
removed from the *self-service surface* (no catalog, no `/bot`, no `/validate`
path) — not deleted from the engine. New authored bots are functions; the classes
remain the yardstick they race against.
