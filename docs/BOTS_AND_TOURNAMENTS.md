# Bots & Tournaments — Architecture Guide

This document describes how bots are built, configured, and evaluated inside the
MTT-Pokerbot-Engine, and how tournament simulations are structured, run, and
reported. It complements `BATCH_DUEL_MARATHON_RUNBOOK.md` (operational how-to)
with an architectural map of the code.

---

## 1. Big picture

```
┌────────────────────────── poker-ai workspace ──────────────────────────┐
│                                                                        │
│  recognition/            MTT-Pokerbot-Engine/          pokerstove/    │
│  (screen → spot state) ──► players/  engine/  ◄──────  (C++ equity    │
│  tools/tournament_bot_     │           │                 wheel)        │
│  decision.py bridge        │           │                               │
│                            ▼           ▼                               │
│                     Bot classes   Tournament / Table                   │
│                            │           │                               │
│                            └─────┬─────┘                               │
│                                  ▼                                     │
│               fixed_bot_mtt_evaluation (duels / batches / marathons)   │
│                                  ▼                                     │
│                       runs/<run_id>/report.json → --report tables      │
└────────────────────────────────────────────────────────────────────────┘
```

- **Bots** implement one method: `get_action(game_state) -> (action, amount)`
  (`engine/player_interface.py:4`). Everything a bot knows arrives in the
  `game_state` dict built by the table each decision.
- **Tournaments** are multi-table MTTs: `Tournament` orchestrates tables,
  blinds, rebalancing, and payouts; `Table` runs individual hands.
- **Evaluation** happens by seating bot populations into the same MTT field
  and comparing tournament finishes → payout share → ROI.

---

## 2. The engine

### 2.1 Layering

| Layer | Class / module | Responsibility |
|---|---|---|
| Tournament | `engine/tournament.py:7` (`Tournament`) | Multi-table orchestration: random seating, per-iteration hand across active tables, blind level advance every `hands_per_level`, table coalescing/breaking, bustout placement, `max_hands_per_tournament` early stop. |
| Table | `engine/table.py:19` (`Table`) | Single-hand NLHE loop: button move, blinds, deal (`treys.Deck`), preflop→river via `_betting_round()` (`table.py:247`), showdown with `treys.Evaluator`, event stream emission (schema: [EVENT_STREAM.md](EVENT_STREAM.md)). |
| Pots | `engine/pot.py` (`Pot`, `PotManager`) | Main + side pots rebuilt from per-player `total_bet` at unique all-in thresholds; dead money folded forward. |
| Player state | `engine/player_state.py` | Stack, current/total bet, active/all-in flags; wraps the bot object. |
| Config | `engine/config.py` + `config.json` | Starting stack (10,000), 9-max tables, 12 hands/level, 14-level blind schedule, top-7 payout fractions. |
| ICM | `engine/icm.py:9` (`calculate_exact_icm`) | Exact Malmuth-Harville via bitmask DP, capped at 30 players. |
| Stats | `engine/game_stats.py` (`TableStatsTracker`) | Per-player VPIP / PFR / 3-bet / postflop aggregates, exposed to bots via snapshots. |

### 2.2 The decision loop

Each turn, `Table` builds a rich `state` dict (`table.py:326-366`) containing:

- cards, board, pot, stacks, legal actions and amounts;
- seating: `hero_table_index`, `button_seat`, and the precomputed `position`
  label (`_position_label`, `table.py:517`) that legacy bots read;
- preflop spot classification (`_preflop_spot_type`, `table.py:198`):
  `limped / srp / three_bet / four_bet / five_bet_plus / all_in_pressure`;
- ICM context: `payouts`, `players_left`, `paid_places`, `starting_field`
  (derived measures like ITM distance and next-prize gain live in
  `engine/bot_tools.py` `derive_*` helpers, not in the state dict);
- table stats snapshot from `TableStatsTracker`.

The bot's `get_action` is called under a per-decision timeout governed by
`bot_decision_timeout_ms`. The table itself just calls the bot (`table.py:374`);
the deadline is enforced by the seat, because a Python thread cannot be killed
and the old `ThreadPoolExecutor` deadlocked the run on shutdown. Untrusted
authored seats run in their own process and enforce their own budget
(`engine/seat_worker.py`); house bots are trusted code and are called directly.
Timeout or exception ⇒ fold. This keeps one broken bot from stalling a
simulation.

### 2.3 Equity

`engine/pokerstove_equity.py` loads the PokerStove wheel from
`pokerstove/dist/*.whl` and provides Monte Carlo equity vs a *modeled opponent
range*:

- static preflop spot ranges (`_PREFLOP_SPOT_RANGE_PCT`) and adaptive variants;
- `player_range()` — builds a range from the opponent's observed VPIP/PFR/3-bet;
- MTT-stage, stack-pressure, and profile adjustments layered on top.

Equity source is selected per run via the engine config block
(`equity_source: "pokerstove"`).

---

## 3. Bots

### 3.1 Contract and registry

All bots subclass `Bot` (`engine/player_interface.py:4`).
`engine/bot_factory.py` owns the registry:

- `BOT_DEFINITIONS` / `BOT_REGISTRY` (`bot_factory.py:37,72`) — alias → class
  mapping with population names and default params;
- `build_configurable_bots()` (`bot_factory.py:135`) — instantiates lineups
  from JSON specs, merges/validates params, and appends tool-set signatures to
  population names.

### 3.2 Bot roster

**Baselines and exploit archetypes** (deliberately flawed opponents used to
test exploitation):

| Bot | File | Behavior |
|---|---|---|
| `RandomBot` | `players/random_bot.py` | Random legal action — noise floor. |
| `CallBot` | `players/call_bot.py` | Pure calling station. |
| `AggressiveNoEquityBot` | `players/aggressive_no_equity_bot.py` | Card-blind pressure from stack/pot ratios. |
| `ThreeBetManiacBot` | `players/three_bet_maniac_bot.py` | Relentless card-blind preflop re-raiser. |
| `StickyCallStationBot` | `players/sticky_call_station_bot.py` | Equity-aware never-raise station; punishes bluffers. |

**Equity heuristics:**

| Bot | File | Behavior |
|---|---|---|
| `TightEquityBot` | `players/tight_equity_bot.py` | Conservative equity bot; raises only strong equity or short-stack value. |
| `AggressiveBot` | `players/aggressive_bot.py` | Looser equity pressure, jams wider when short. |
| `NoisyEquityBot` | `players/noisy_equity_bot.py` | TightEquityBot with noisy equity perception and occasional mistakes. |
| `EVReactionBot` → `EVInitiativeBot` → `EVFormulaBot` | `players/ev_reaction_bot.py` | Chip-EV cascade: call/fold on immediate EV → add raises → add fold-equity-approximated bet EV. |
| `ICMTightBot` | `players/icm_tight_bot.py` | Conservative equity bot with optional ICM risk pressure. |
| `RangePolicyBot` | `players/range_policy_bot.py` | Configurable range-aware policy bot for policy search. |

**The tournament-equity family** (`players/tournament_equity_bot.py`).
Two generations live side by side:

*Legacy generation — behavior baked into subclasses.* Each variant hardcodes
one strategic addition on top of the base equity engine:

```
TournamentEquityBot                       (:26)  base MTT equity engine + risk controls
 ├─ TournamentEquityBotV2                 (:388) + tighter preflop reraise threshold
 └─ TournamentICMEquityBot                (:442) + exact ICM pressure with payout state
     ├─ AdaptiveTournamentICMEquityBot    (:557) + VPIP/PFR/3-bet adaptation
     └─ ButtonStealTournamentICMEquityBot (:725) + default button-steal tool
```

These remain in the registry as fixed opponents and marathon field filler, but
they are not where development happens anymore.

*Current generation — composable tools.* `ConfiguredTournamentEquityBot`
(`:665`) inherits the base engine directly and replaces the subclass-per-feature
pattern with a declarative **tool stack**: it zeroes out built-in payout
pressure (`:721`) so *all* strategic pressure comes from tools listed in the
JSON config (§3.4). What used to require a new subclass (ICM pressure, button
steal, adaptation) is now a config entry — every `bot_configs/conf_mtt_eq_*.json`
variant, including the champion, is this one class with a different tool list.

### 3.3 Decision flow (TournamentEquityBot family)

`get_action` (`tournament_equity_bot.py:61-203`):

1. Parse state; compute street, stack in BB, `required_equity = pot_odds(call, pot)`.
2. **Estimate equity** via PokerStove Monte Carlo against the modeled range.
3. Compute dynamic thresholds — `_call_margin`, `_raise_threshold`,
   `_jam_threshold` — adjusted by position, spot tightness table
   (`tournament_equity_bot.py:9-23`), and payout/ICM pressure
   (`_exact_icm_pressure`, `:492`).
4. Build a `DecisionContext` (`bot_tools.py:15`) and run the **tool stack**
   (`_apply_tools`, `:319`). Tools can mutate equity/thresholds, force an
   action, or emit telemetry events.
5. Decision cascade: forced action → jam → fold if equity below
   required + margin → sized value raise (pot-bucket sizing) → call/check →
   fold.

There is no external solver. Decisions = equity + heuristic thresholds + tool
overlays.

### 3.4 Decision tools

`engine/bot_tools.py` (~1,300 lines) provides composable strategy modules,
applied in priority order:

| Tool | Class (line) | Purpose |
|---|---|---|
| `icm_pressure` | `ICMPressureTool` (`:64`) | Exact-ICM risk premium (current vs loss vs gain ICM). |
| `preflop_reraise` | `PreflopReraiseTightnessTool` (`:179`) | Tightens facing 3-bet+ spots. |
| `table_adaptation` | `TableAdaptationTool` (`:195`) | Adjusts to observed table stats. |
| `button_steal` | `ButtonStealTool` (`:271`) | Late-position steal raises. |
| `endgame_conversion` | `EndgameConversionTool` (`:364`) | Payout-jump conversion near/at final table. |
| `cbet_pressure` | `ContinuationPressureTool` (`:479`) | Continuation betting as aggressor. |
| `bluff_pressure` | `BluffPressureTool` (`:641`) | Bluffing incl. river scare-card mode. |

Registry: `available_decision_tools()` (`:1117`), `build_bot_tools()` (`:1252`).
Tool telemetry lands in evaluation reports under `population_action_summary`
(`tool_action_counts`, `tool_reject_reason_counts`, `tool_response_rates`).

### 3.5 Configuring a bot

Bot specs live in `bot_configs/*.json`:

```json
{
  "type": "configured_tournament_equity",
  "population": "…",
  "include_tool_set_in_name": true,
  "params": {
    "range_profile": "player",
    "tools": [
      { "name": "icm_pressure", … },
      { "name": "preflop_reraise", … },
      { "name": "button_steal", … },
      { "name": "bluff_pressure", "min_fold_equity": …, "require_river_scare_card": true, … }
    ]
  }
}
```

Key constructor params (`tournament_equity_bot.py:35-59`):
`use_preflop_spot_range`, `range_profile` (`legacy` / `adaptive` / `player`),
`range_influence`, `player_range_sampling` + `player_range_sample_config`,
`raise_sizing` (`legacy` / `pot_buckets` / `postflop_pot_buckets`),
`pot_size_buckets`, `pot_bucket_edge_step`, `pot_bucket_all_in_spr`, `tools`.

### 3.6 Current champion

**`conf_mtt_eq_playerrange_samplecurve_icm_reraise_steal_scarebluff`**
(`bot_configs/conf_mtt_eq_playerrange_samplecurve_icm_reraise_steal_scarebluff.json`)
— a `ConfiguredTournamentEquityBot` with:

- `range_profile: "player"` (opponent range from observed stats);
- sample-curve range quality;
- tools: `icm_pressure` + `preflop_reraise` + `button_steal` +
  `bluff_pressure` (river scare-card forced bluff).

Latest exploit benchmarks (`history/PHASE164_165_EXPLOIT_BATCH_RESULTS.md`, 600
entries per matchup): **ROI +99.8 % vs call station, +90.9 % vs 3-bet maniac.**

Champion evolution across phase runs:
legacy range → player range → dampened / postflop-pot-capped → cbet →
bluff / bluffstrong / bluffleverage → samplecurve → survivalbluff →
**scarebluff** (phases 159–165). Ongoing: sticky-call-station (phases
167/168) and 3-bet-maniac (166/169) exploit tests.

### 3.7 Neural bots (poker-ai-basemodel)

The workspace also trains neural policy bots (see root `README.md` phase
table, phases 1–36). The current best is **reduced v5** — an 11-feature
reduced-observation model (8 base features + `stack_over_avg_table_stack` +
`field_pct_remaining` + `itm_distance`) cloned from equity-bot teachers and
evaluated in this engine via adapter classes (phase 21–36 configs). The
engine-side wiring lives in the `phase21..phase32_*.py` scripts and their
matching configs; scripted bots above remain the strength benchmark.

---

## 4. Running tournaments

Three separate entry points — no unified CLI; the module you invoke is the mode.

### 4.1 `main.py` — legacy config-driven runner

```bash
python3 main.py           # driven entirely by config.json, no argparse
```

- Dynamically imports every `players/*.py` after an AST import-allowlist
  security scan (`main.py:13-62`).
- Runs `simulation_count` tournaments in a `ProcessPoolExecutor`.
- Writes `logs/sim_{id}.json` (results + full event stream) and
  `simulation_results.csv` (`simulation_id, position, name, bot_class,
  payout_pct`).

### 4.2 `engine/fixed_bot_mtt_evaluation.py` — primary benchmark runner ★

```bash
python3 -m engine.fixed_bot_mtt_evaluation --config configs/<run>.json
python3 -m engine.fixed_bot_mtt_evaluation --report runs/<run>/<ts>/fixed_bot_evaluation_report.json
```

Vocabulary (from the runbook):

| Term | Meaning |
|---|---|
| **MTT** | One tournament. |
| **Entry** | One bot seat in one MTT. |
| **Duel** | 2 populations, typically 50+50 seats in a 100-entry MTT. |
| **Batch** | 8–20 MTT smoke run. |
| **Marathon** | 100+ MTT ranking run. |
| **Workers** | Parallel MTT processes — speed only, never rules. |

Run config shape (`configs/*.json`, ~150 phase fixtures):

```json
{
  "run_id": "phase166_3betmaniac_vs_champion_20x100",
  "mtt_count": 20,
  "workers": 8,
  "mtt_seed_start": 42,
  "bot_config_dir": "bot_configs",
  "named_lineup": [ { "bot": "…", "count": 50 }, … ],
  "engine": {
    "equity_source": "pokerstove",
    "bot_decision_timeout_ms": …,
    "starting_stack": 10000,
    "max_players_per_table": 9,
    "hands_per_level": 12,
    "payouts": { … }
  }
}
```

#### Mode recipes (batch vs duel vs marathon)

The three modes use the same runner and config schema — only `mtt_count` and
`named_lineup` shape differ:

| Mode | `mtt_count` | `named_lineup` shape | Entries/bot | Purpose |
|---|---:|---|---:|---|
| **Batch** | 8 | up to 20 populations × 10 seats (200/MTT) | 80 | Smoke test: unknown bot names, broken configs, tool init errors, perf regressions. Do not read rankings from it. |
| **Duel** | 20 | 2 populations × 50 seats (100/MTT) | 1,000 | Challenger vs baseline — directional feature verdict. |
| **Marathon** | 100 | ≤20 populations × 10 seats (200/MTT) | 1,000 | Mixed-field ranking — champion decisions. A duel winner can still underperform here. |

Duel example (20×100, 1,000 entries per bot):

```json
{
  "run_id": "phaseXXX_challenger_vs_baseline_20x100",
  "mtt_count": 20,
  "workers": 8,
  "mtt_seed_start": 161001,
  "bot_config_dir": "bot_configs",
  "named_lineup": [
    {"bot": "conf_mtt_eq_adaptiverange_icm_reraise_steal", "count": 50},
    {"bot": "conf_mtt_eq_adaptiverange_icm_reraise_steal_scarebluff", "count": 50}
  ],
  "engine": { "…": "copy engine block + full payouts from an existing duel config" }
}
```

Batch: same shape, `mtt_count: 8`, marathon lineup. Marathon: `mtt_count: 100`,
all-non-trash lineup — copy engine block from
`configs/phase160_all_nontrash_scarebluff_100x200.json`. Full walkthroughs with
complete lineups: `BATCH_DUEL_MARATHON_RUNBOOK.md` §"Launch A Duel" /
"Launch A Small Batch" / "Launch An All-Non-Trash Marathon".

Mechanics (`fixed_bot_mtt_evaluation.py:332`):

- One unique seed per MTT (`mtt_seed_start` sequential, or explicit
  `tournament_seeds` — uniqueness enforced).
- `named_lineup` expands against the `bot_configs/` library; each bot gets
  `mtt_count × count` entries.
- Each worker seeds its RNG, builds bots, shuffles seating, plays one
  `Tournament`.
- Bots compete purely by finish position → payout share; **ROI** is measured
  against the equal-share expected payout.

### 4.3 `engine/evolutionary_reduced_mtt.py` — evolutionary trainer

```bash
python3 -m engine.evolutionary_reduced_mtt --config … --mode {initial,mutation,steady} \
  --champion-checkpoint … --mtts N --candidates N --entries-per-candidate N …
```

Also exports shared summary helpers used by the fixed-bot runner.

### 4.4 Other harnesses

- `engine/ev_initiative_head_to_head.py` — duel harness;
- `engine/ev_initiative_evolution.py`, `engine/ev_reaction_mtt_evaluation.py`;
- historical `engine/phase21..phase32_*.py` scripts (each with a matching
  `tests/test_phase*.py`).

---

## 5. Results & reporting

### 5.1 Artifacts

| Location | Producer | Content |
|---|---|---|
| `logs/sim_*.json` | `main.py` | `{simulation_id, results, events}` per simulation. |
| `simulation_results*.csv` | `main.py` | Flat per-seat finish/payout rows. |
| `runs/<run_id>/<utc_ts>/partial_summary.json` | benchmark runner | Live snapshot, rewritten each progress tick. |
| `runs/<run_id>/<utc_ts>/fixed_bot_evaluation_report.json` | benchmark runner | Final report: lineup, seeds, engine block, `population_summary`, `population_action_summary` (tool telemetry), per-tournament summaries, failures, runtime. |
| `runs/…/events/tournament_NNNN_events.json` | benchmark runner | Full event streams, only with `write_events: true`. Schema: [EVENT_STREAM.md](EVENT_STREAM.md). |
| `runs/…/checkpoints/`, `*_generation_report.json` | evolutionary trainer | Generation artifacts. |

Generated `runs/` are not committed;
`MTT-Pokerbot-Engine-runs-20260630.tar.gz` is an archived snapshot.

### 5.2 Ranking tables

`--report` prints (and merges) a stable markdown table
(`format_population_table`, `merge_population_summaries`). Columns: payout,
avg payout, ROI, score, wins, top3, final tables, ITM, avg position. Ranking
key: payout → score → wins → avg-pos → name.

### 5.3 Analysis & visualization

- `stats.py` — CLI log analyzer: `python stats.py logs/sim_*.json` — event
  histogram, action breakdown, winners/ITM.
- `visualizer/` (`index.html`, `app.js`) — browser event-stream replayer:
  per-table boards/pots, leaderboard, blinds, alive count. Serve with
  `python server.py` (port 8000) and load a `sim_*.json`.

### 5.4 Recommended benchmark pipeline (runbook)

1. Edit / add spec in `bot_configs/`.
2. Focused pytest.
3. 8-MTT batch (smoke).
4. 20-MTT duel — 1,000 entries per bot.
5. 20-population all-non-trash marathon.
6. `--report` for the stable ranking table.

Keep `workers` at 8 on this machine (14 froze it).

---

## 6. Live play bridge

`recognition/tools/tournament_bot_decision.py` connects screen recognition to
the same bots used in simulation:

```
screenshot frame
  → extract_frame (recognition)          # OCR + templates → spot state (BB units)
  → spot_state_to_tournament_game_state  # BB → engine chips (chips_per_bb, default 100)
  → TournamentEquityBot.get_action()     # the real simulation bot class
  → engine_action + bb_action
```

```bash
python -m recognition.tools.tournament_bot_decision <frame> [--chips-per-bb N] [--no-preflop-ranges]
```

Equity in the live path also uses PokerStove (`equity_source: "pokerstove"`).
The bridge monkeypatches `estimate_equity` to record the equity call for
inspection.

**Note on `icm-calculator/`**: the TypeScript package in the workspace root is
a *parallel* implementation of Malmuth-Harville ICM (for JS/production use).
The engine never imports it — Python ICM lives independently in
`engine/icm.py`.

---

## 7. Quick reference

| I want to… | Do this |
|---|---|
| Add a new scripted bot | Subclass `Bot`, drop in `players/`, register in `engine/bot_factory.py`. |
| Tune the champion | Edit tool params in `bot_configs/conf_mtt_eq_playerrange_samplecurve_icm_reraise_steal_scarebluff.json`. |
| Create a new bot variant | New JSON in `bot_configs/` (`type` + `params.tools`). |
| Run a duel | Copy a `configs/phase*_*_vs_*_20x100.json`, adjust `named_lineup`, run `python3 -m engine.fixed_bot_mtt_evaluation --config …`. |
| See rankings | `… --report runs/<run>/<ts>/fixed_bot_evaluation_report.json` (multiple reports merge). |
| Watch a hand | `python server.py` → visualizer → load `logs/sim_*.json`. |
| Ask a bot about a real screenshot | `python -m recognition.tools.tournament_bot_decision <frame>`. |
