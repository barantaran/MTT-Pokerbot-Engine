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

A plain dict of what happened, not what to do. The engine owns card facts; the
author owns every judgement made from them. There is no `DecisionContext`, no
precomputed equity, no thresholds — those were the old middle layer and are gone
from the self-service surface.

The engine's internal state (treys-int cards, flat, built for the legacy bot
classes) is translated to this author-facing shape at the seam by
`_normalize_state` (`engine/authored_loader.py`) — the author never sees a treys
int, and legacy bots never see this dict.

```python
state = {
  "hero": {
    "hole": ["As", "Kd"],   # string cards, always 2
    "stack": 1500,          # chips
    "stack_bb": 30.0,       # chips / big blind
    "position": "BTN",
    "call_amount": 100,     # chips to call (0 = can check)
    "min_raise": 200,       # chips
  },
  "board": ["Qh", "Jc", "2s"],       # 0-5 string cards
  "pot": 300,
  "blinds": {"small": 25, "big": 50},
  "villain_hands": [None, None],     # one per live opponent; api samples them
  "opponent": {                      # the aggressor hero faces, if any
    "id": "villainX", "position": "SB",
    "stack": 900, "stack_bb": 18.0,
    "stats": {...} or None,
  },
  "history": [                       # actions so far this hand, in order
    {"player": "villainX", "action": "raise", "amount": 150,
     "street": "preflop", "position": "SB"},
  ],
  "tournament": {
    "players_left": 40, "starting_field": 100,
    "paid_places": 15, "payouts": {...},
  },
}
```

`villain_hands` is exactly what `api.deal` / `api.equity` want: villains are
hidden, so each entry is `None` and the `api` draws them from the live deck.

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

## One process per seat

**Your bot runs alone, in its own process, for the length of one tournament**
(`engine/seat_worker.py`; rationale in `service/INTEGRITY.md`). Nothing else is
in there — not the engine, not another nick, not your own second entry. The
engine sends the state over a pipe and reads back `(action, amount)`.

What that means for an author:

- **The decision deadline is real.** Miss `bot_decision_timeout_ms` (500 ms by
  default) and the process is killed outright; the seat folds that decision
  (checks if it is free). There is no partial credit and no way to overrun it.
- **A kill costs you memory.** The next decision starts a fresh process, so any
  module-level state you were carrying across hands is gone. That reset *is* the
  penalty for stalling.
- **Five kills and the seat is done** — it stops respawning and folds out the
  rest of that tournament, blinds included. Timeouts are counted per nick in the
  run report.
- **An exception is cheaper than a stall.** Raising folds that one decision and
  keeps the process (same as before); only a timeout or a crash costs a strike.
- **A folder that will not import folds the whole seat** for that tournament
  instead of failing the run. Catch that at `POST /validate`, not in a match.
- **`print()` is safe.** Worker stdout is redirected to stderr, so it lands in
  the run log and cannot corrupt the pipe.
- **Your entries do not know each other.** Two seats of the same nick are two
  processes and share nothing.

**The call is unchanged.** The engine holds a bot object and calls
`bot.get_action(game_state)` (`engine/player_state.py`). The author's two-arg
function reaches that one-arg method through a thin adapter that injects `api`
— in-process it is `_AuthoredBot`, out-of-process it is `SeatWorker` on the
engine side and a bare `fn(state, api)` call inside the worker:

```python
class _AuthoredBot:
    def __init__(self, fn, api): self.fn, self.api = fn, api
    def get_action(self, game_state): return self.fn(game_state, self.api)
```

The adapter is internal; authors never see it.

## Loading and namespacing

**Import happens once per seat process, per tournament.** The seat process loads
your folder — and only your folder — on its first decision, then reuses the
imported `get_action` for every later decision, with one `api` built alongside it
(card math is stateless).

**Per-nick namespacing.** Historically both seats of a 2-max duel lived in one
worker, so if nick A and nick B both shipped `shove_short.py`, a bare
`import shove_short` would collide in the global `sys.modules` — one would shadow
the other. Per-seat processes remove that collision; the namespacing stays as the
second layer, and it is still what makes the in-process debug path
(`authored_isolation: false`) behave the same as a real run.

The loader prevents this: each nick's folder is loaded as a **private module
namespace** (e.g. `authored.<nick>.*`) so a bare sibling import inside nick A
resolves to nick A's tool, and nick B's identically-named file is a different
module. The author writes bare imports (`from shove_short import shove_short`);
the loader — not the author — guarantees they resolve to the author's own folder
and nowhere else.

**Responsibility split:**

| owner | responsible for |
|---|---|
| author | *which* tools to import and calling them right — names a real sibling, uses it correctly |
| engine / loader | *making* a correct sibling import resolve — nick folder is the import root, isolated from other nicks; author never touches `sys.path` or packaging |
| validator | *catching* mistakes before a match — a non-sibling import (`engine`, `os`, another nick) or a missing sibling fails at `POST /validate`, not mid-hand |

A consequence, made mechanical: an author **cannot** import another author's tool.
Reuse is within a nick only — the "per-nick private toolbox" rule enforced by the
namespace, not by author discipline.

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
