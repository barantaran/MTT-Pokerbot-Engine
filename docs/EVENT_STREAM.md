# Event stream

Every tournament the engine plays produces one ordered list of event dicts.
It is the engine's only structured output about *what happened* — the reports
say who won, the event stream says how. It feeds the durable resume log
(`engine/event_log.py`), the action summarizers, the live table stats, and the
arena's replay exporter.

It is a contract. Consumers outside this repo read it. Adding a key is safe;
renaming or removing one is not.

## Where it comes from

| producer | events |
|---|---|
| `engine/table.py` (`Table.play_hand`) | everything inside a hand |
| `engine/tournament.py` (`Tournament.play`) | everything around hands |

`Tournament.play` runs one hand on every table in turn and extends a single flat
list (`engine/tournament.py`), so **events from different tables interleave**.
A consumer that cares about one table must demux by `table_id`.

## Where it lands

| path | format | written when |
|---|---|---|
| `runs/<run_id>/<ts>/events/mtt_<id>/chunk_NNNNN.jsonl` | JSONL, one event per line, one file per table-hand | `resume_log: true` (the default) |
| `runs/<run_id>/<ts>/events/tournament_NNNN_events.json` | a single JSON array | `write_events: true` |
| `logs/sim_<id>.json` | `{simulation_id, results, events}` | `main.py` |

Read a chunk log back with `read_event_log(events_dir)` (`engine/event_log.py`).

## Addressing

Every event emitted inside a hand carries four addressing keys:

| key | meaning |
|---|---|
| `tournament_id` | which tournament |
| `table_id` | which table |
| `hand_id` | per-table, monotonic, 1-based |
| `seq` | 0-based position **within that hand** |

The hand key is the triple `(tournament_id, table_id, hand_id)`. `hand_id` is
per table, so it is not unique on its own.

`seq` exists because array position stops being an ordering once tables
interleave. Within a hand, `seq` is total and gapless. It restarts at 0 each
hand.

Tournament-scope events (`seat`, `table_broken`, `knockout`, `level_up`,
`tournament_win`, `tournament_stopped_max_hands`, `tournament_start`) carry
`tournament_id`, and `table_id` where one applies, but no `hand_id`/`seq` — they
happen between hands.

**There are no per-event timestamps.** Wall clock appears twice per hand:
`hand_start.started_at` and `hand_end.ended_at`, both ISO-8601 UTC. A marathon
tournament emits ~35k events; a timestamp on each would cost about a megabyte
and tell a consumer nothing `seq` does not.

## Cards

Every card in every event is a **two-character treys string** — rank then suit,
e.g. `"Ah"`, `"Td"`, `"2c"`. Never an integer. Build one back with
`Card.new("Ah")` if you need the treys value.

Hole cards appear in exactly two events: `deal` (the owner only) and `showdown`
(players who reached showdown). Anything reconstructing a hand for a viewer who
is not that player must not read `deal`.

## Hand lifecycle

```
hand_start
post_blind × 2
deal × (players dealt in)
  [preflop]  action × n
board street=flop
  [flop]     action × n
board street=turn
  [turn]     action × n
board street=river
  [river]    action × n
board street=runout × k     (only if the board is short at showdown)
showdown × (players who got there)
award_pot × (one per side pot won)
hand_end
```

`hand_start` and `hand_end` pair 1:1 on every non-degenerate path. A table with
fewer than two players returns an empty event list and does not advance
`hand_id` — such a hand is invisible, not half-emitted.

A street that is checked through emits only its `board` event; an all-in
preflop emits boards with no actions between them. **Detect street boundaries
from `board` events, not from gaps in the action list.**

## Event reference

### `tournament_start`
`tournament_id`, `blinds{small,big}`, `players[]` — a list of **names**, not
dicts.

### `seat`
`tournament_id`, `table_id`, `player`, `seat`. Emitted at initial seating and
again whenever a broken table's players are redistributed.

### `table_broken`
`tournament_id`, `table_id`. The table's players get fresh `seat` events
immediately after.

### `level_up`
`tournament_id`, `level` (1-based blind index), `blinds{small,big}`.

### `knockout`
`tournament_id`, `table_id`, `player`, `finish_position` — the player's final
tournament placement, matching the `position` in the results table.

### `tournament_win`
`tournament_id`, `player`.

### `tournament_stopped_max_hands`
`tournament_id`, `max_hands`. **Mutually exclusive with `tournament_win`**: when
the hand cap fires, no winner event is emitted at all. Remaining players are
placed by stack size.

### `hand_start`
`started_at`, `level`, `blinds{small,big}`, `button_seat`, `button_player`, and
`players[]` of `{seat, name, stack, position}` in seating order, where `stack`
is the stack *before* the blinds are posted.

`position` is one of `BTN`, `SB`, `BB`, `UTG`, `UTG_1`, `UTG_2`, `LJ`, `HJ`,
`CO`. **Heads-up is non-standard**: at two players the button posts the big
blind, so the labels are `BTN` and `SB` and no `BB` is emitted.

### `post_blind`
`player`, `seat`, `blind` (`"small"` / `"big"`), `amount`, `street`
(`"preflop"`). `amount` is the chips actually deducted — a short stack posts
what it has.

Blinds are **not** `action` events. A consumer that folds only over `action`
loses the blinds and will mis-total the pot.

### `deal`
`player`, `cards[]` — that player's two hole cards.

### `action`
`player`, `action` (`fold` / `call` / `check` / `raise`), `amount`, `street`
(`"preflop"` / `"flop"` / `"turn"` / `"river"`), `position`, `call_amount`,
`pot_size`, and optionally `tool_event` / `tool_events`.

- `amount` is **incremental** — the chips this action deducted, not the running
  total. Summing `post_blind.amount` and `action.amount` gives the pot.
- `pot_size` is the pot **before** this action.
- `tool_event` is the last BotTool decision behind this action; `tool_events` is
  all of them. Both are **author-controlled free-form dicts**
  (`engine/bot_tools.py`) — an author can put anything in there, including their
  own hole cards. Anything redacting the stream for one viewer must strip them
  from every other player's actions.

### `board`
`cards[]`, `street` — `"flop"` (3 cards), `"turn"`, `"river"`, or `"runout"`
(1 card per event, dealt at showdown when the board is short).

### `showdown`
`player`, `cards[]`, `rank` (treys hand rank, **lower is better**, 1 = royal
flush), `rank_class` (human-readable, e.g. `"Full House"`). Emitted only for
players still active at showdown, i.e. exactly the hands that are public
information at a real table.

### `award_pot`
`player`, `amount`, `showdown` (bool — `false` when everyone folded).

One event **per side pot**, so a hand can emit several. A split pot divides
`pot.amount // len(winners)` and **drops the remainder**, so
`sum(award_pot.amount)` can be a few chips short of the real pot. Track the pot
from the bets, not from the awards.

### `hand_end`
`ended_at`, `players[]` of `{name, stack}` — stacks *after* the hand, emitted
**before** busted players are removed from the table, so a player at 0 is still
listed. That is how you detect a bust from the stream alone.

## Known bugs

- **A negative `action.amount` is possible.** `_betting_round` returns early
  when at most one player can act, and that early return happens *before* the
  end-of-round `current_bet = 0` reset. The stale per-street bet then survives
  into the next street, where `call_amount = current_highest_bet -
  player.current_bet` can come out negative and `PlayerState.bet` deducts a
  negative amount — handing chips back and shrinking the pot.

  Reproduces reliably heads-up when the small blind is all in for less than the
  big blind: the short stack wins back only its own blind instead of doubling.
  Any consumer summing `amount` must expect this. The replay exporter flags such
  hands in `anomalies` rather than hiding them.

## Known quirks, deliberately unfixed

- **Heads-up positions** — the button posts the big blind (see `hand_start`).
  Fixing it changes play and invalidates every existing result.
- **Split-pot remainder** — see `award_pot`.
- **The `runout` branch is currently unreachable from `play_hand`**: every
  street deals while two or more players are active, so the board is always
  complete by showdown. The code path and its encoding are kept correct anyway.
