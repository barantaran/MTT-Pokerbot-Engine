# Batch, Duel, And Marathon Runbook

Use this runbook from the engine root:

```bash
cd MTT-Pokerbot-Engine
```

The main fixed-bot tournament runner is:

```bash
python3 -m engine.fixed_bot_mtt_evaluation --config configs/<config>.json
```

It writes artifacts under:

```text
runs/<run_id>/<timestamp>/
```

Important files:

- `partial_summary.json`: live progress snapshot while a run is active.
- `fixed_bot_evaluation_report.json`: final report after the run completes.
- `events/`: only written when the config has `"write_events": true`.

Generated `runs/` output is benchmark data. Do not commit it unless it is explicitly intended as a fixture.

## Terms

- **MTT**: one tournament simulation.
- **Entry**: one bot seat in one MTT.
- **Duel**: usually 2 bot populations, commonly `50 + 50` seats for a 100-entry MTT.
- **Batch**: a small run used as a smoke or quick comparison, often `8` or `20` MTTs.
- **Marathon**: a larger run used for ranking, commonly `100` MTTs or more.
- **Workers**: number of MTTs run in parallel. It changes speed, not tournament rules.
- **Seed**: one unique tournament seed per MTT. Do not reuse one seed for multiple MTTs in the same run.

## Config Shape

Minimal config:

```json
{
  "run_id": "phaseXXX_name",
  "mtt_count": 20,
  "workers": 4,
  "mtt_seed_start": 161001,
  "bot_config_dir": "bot_configs",
  "named_lineup": [
    {"bot": "bot_a", "count": 50},
    {"bot": "bot_b", "count": 50}
  ],
  "engine": {
    "equity_source": "pokerstove",
    "equity_fallback_source": "constant",
    "fixed_bots_use_preflop_spot_range": false,
    "bot_decision_timeout_ms": 1500,
    "starting_stack": 5000,
    "max_players_per_table": 9,
    "hands_per_level": 10,
    "max_hands_per_tournament": null,
    "payouts": {
      "1": 0.248949579832,
      "2": 0.157563025210,
      "3": 0.108193277311
    }
  }
}
```

Use the full payout table from a recent config when creating a real benchmark. Example configs:

- `configs/phase161_adaptiverange_bluff_vs_base_20x100.json`
- `configs/phase160_all_nontrash_scarebluff_100x200.json`

## Seeds

Preferred:

```json
"mtt_seed_start": 161001
```

This produces unique seeds:

```text
161001, 161002, 161003, ...
```

Manual explicit seeds are also supported:

```json
"tournament_seeds": [161001, 161002, 161003]
```

Rules:

- `tournament_seeds` length must equal `mtt_count`.
- Seeds must be unique.
- One seed belongs to one MTT only.

### Cross-Run Seed Bands

Uniqueness above is *within* one run. To keep separate runs statistically independent, give each phase its own seed band that does not overlap any prior run:

```text
phase170  mtt_seed_start 170001  -> seeds 170001..170100
phase171  mtt_seed_start 171001  -> seeds 171001..171100
phase172  mtt_seed_start 172001  -> seeds 172001..172100
```

Convention: `mtt_seed_start = <phase_number>001`. With `mtt_count <= 1000` the bands never collide. Reusing a prior band re-draws the same tournaments (reproducible A/B), but for a fresh independent field always bump to a new band.

## Workers

Set workers based on CPU and stability:

```json
"workers": 4
```

Local default used in recent runs: `4`.

Notes:

- More workers run more MTTs in parallel.
- More workers can increase memory and CPU pressure.
- If a run crashes or the machine becomes unresponsive, reduce workers.
- Results should not depend on worker count because each MTT has its own seed.

## Compose A New Bot From Sibling Configs

Configured bots are a `type` + a list of `tools` ordered by `priority`. A "feature" suffix in the filename is usually just one extra tool instance:

- `_cbet`       -> a `cbet_pressure` tool (flop continuation). A real, self-standing tool type.
- `_turnbarrel` -> a `bluff_pressure` tool gated to the turn (`allowed_streets: [4]`, `turn_size`). Not its own type.
- `_scarebluff` -> a `bluff_pressure` tool gated to the river (`allowed_streets: [5]`, `require_river_scare_card`).

Registered tool types live in `engine/bot_tools.py` (`cbet_pressure -> ContinuationPressureTool`, `bluff_pressure -> BluffPressureTool`). Street codes: `3` flop, `4` turn, `5` river.

To build a combined bot, take a base config and graft the extra tool(s) from existing siblings, then re-sort by `priority` and give the bot a unique `name` and `population`:

```python
import json
base = json.load(open('bot_configs/BASE.json'))
cbet = json.load(open('bot_configs/BASE_cbet.json'))
tb   = json.load(open('bot_configs/BASE_turnbarrel.json'))

cbet_tool = [t for t in cbet['params']['tools'] if t['type'] == 'cbet_pressure']
tb_tool   = [t for t in tb['params']['tools']
             if t['type'] == 'bluff_pressure' and t.get('allowed_streets') == [4]]

new = json.loads(json.dumps(base))              # deep copy
new['name'] = 'BASE_cbet_turnbarrel'
new['population'] = 'BASE_pop_cbet_turnbarrel'  # must be unique across the field
tools = new['params']['tools'] + cbet_tool + tb_tool
tools.sort(key=lambda t: t['priority'])
prios = [t['priority'] for t in tools]
assert prios == sorted(prios) and len(prios) == len(set(prios)), prios  # no priority clash
new['params']['tools'] = tools
json.dump(new, open('bot_configs/BASE_cbet_turnbarrel.json', 'w'), indent=2)
```

Rules:

- `name` and `population` must be unique — duplicate `name` breaks the factory.
- No two tools may share a `priority`.
- Stacking pressure tools compounds aggression across streets (flop cbet -> turn barrel -> river scare). Each still gates on its own equity / fold-equity / SPR mins, but validate the combo in a marathon; do not assume additive edges stack.

## Launch A Duel

Use a duel when testing one challenger against one baseline.

Example: 20 MTTs, 50 seats each, 1000 entries per bot:

```json
{
  "run_id": "phaseXXX_challenger_vs_baseline_20x100",
  "mtt_count": 20,
  "workers": 4,
  "mtt_seed_start": 161001,
  "bot_config_dir": "bot_configs",
  "named_lineup": [
    {"bot": "conf_mtt_eq_adaptiverange_icm_reraise_steal", "count": 50},
    {"bot": "conf_mtt_eq_adaptiverange_icm_reraise_steal_scarebluff", "count": 50}
  ],
  "engine": {
    "...": "copy engine block and full payouts from an existing duel config"
  }
}
```

Run it:

```bash
python3 -m engine.fixed_bot_mtt_evaluation --config configs/phaseXXX_challenger_vs_baseline_20x100.json
```

Interpretation:

- Each bot gets `mtt_count * count` entries.
- In the example: `20 * 50 = 1000` entries per bot.
- Use duel results to decide whether a feature is directionally useful.
- Validate promising duel winners in a mixed field before calling them champions.

## Launch A Small Batch

Use a batch for quick smoke testing or checking that a config runs.

Example: 8 MTTs, 20 populations, 10 seats each:

```json
{
  "run_id": "phaseXXX_all_nontrash_test_batch_8x200",
  "mtt_count": 8,
  "workers": 4,
  "mtt_seed_start": 162001,
  "bot_config_dir": "bot_configs",
  "named_lineup": [
    {"bot": "bot_1", "count": 10},
    {"bot": "bot_2", "count": 10}
  ],
  "engine": {
    "...": "copy engine block and full payouts from an existing all-non-trash config"
  }
}
```

Run it:

```bash
python3 -m engine.fixed_bot_mtt_evaluation --config configs/phaseXXX_all_nontrash_test_batch_8x200.json
```

Use batches to catch:

- Unknown bot names.
- Broken bot configs.
- Tool initialization errors.
- Severe performance regressions.

Do not over-read the ranking from a small batch.

## Launch An All-Non-Trash Marathon

Use a marathon to rank bots in a mixed field.

Current preferred all-non-trash shape:

- At most `20` bot populations.
- `20` seats per population (recent phase163/170/171/172 shape; `10` is the older lighter shape).
- `400` total entries per MTT (`20 pops x 20 seats`; `200` for the `10`-seat shape).
- `100` MTTs for a serious run — `100 x 400 = 2000` entries per population.

Pick `10` seats for a faster/cheaper marathon, `20` seats for the current serious ranking runs. Keep `count` equal across all populations so the field stays balanced.

Example:

```json
{
  "run_id": "phaseXXX_all_nontrash_100x200",
  "mtt_count": 100,
  "workers": 4,
  "mtt_seed_start": 160001,
  "bot_config_dir": "bot_configs",
  "named_lineup": [
    {"bot": "aggressive_equity", "count": 10},
    {"bot": "tight_equity", "count": 10},
    {"bot": "range_policy", "count": 10},
    {"bot": "ev_initiative", "count": 10},
    {"bot": "ev_formula", "count": 10},
    {"bot": "icm_tight", "count": 10},
    {"bot": "tournament_equity", "count": 10},
    {"bot": "tournament_equity_v2", "count": 10},
    {"bot": "tournament_icm_equity", "count": 10},
    {"bot": "adaptive_tournament_icm_equity", "count": 10},
    {"bot": "button_steal_tournament_icm_equity", "count": 10},
    {"bot": "conf_mtt_eq_icm_reraise_steal", "count": 10},
    {"bot": "conf_mtt_eq_legacyrange_icm_reraise_steal", "count": 10},
    {"bot": "conf_mtt_eq_adaptiverange_icm_reraise_steal", "count": 10},
    {"bot": "conf_mtt_eq_playerrange_icm_reraise_steal", "count": 10},
    {"bot": "conf_mtt_eq_playerrange_samplecurve_icm_reraise_steal", "count": 10},
    {"bot": "conf_mtt_eq_playerrange_samplecurve_icm_reraise_steal_scarebluff", "count": 10},
    {"bot": "conf_mtt_eq_playerrange_damped_icm_reraise_steal", "count": 10},
    {"bot": "conf_mtt_eq_playerrange_damped_postfloppotcapped_icm_reraise_steal", "count": 10},
    {"bot": "conf_mtt_eq_playerrange_damped_postfloppotcapped_icm_reraise_steal_cbet", "count": 10}
  ],
  "engine": {
    "...": "copy engine block and full payouts from configs/phase160_all_nontrash_scarebluff_100x200.json"
  }
}
```

Run it:

```bash
python3 -m engine.fixed_bot_mtt_evaluation --config configs/phaseXXX_all_nontrash_100x200.json
```

Interpretation:

- Each bot gets `100 * 10 = 1000` entries.
- ROI is relative to equal-share expected payout for the reported field.
- Ranking is by total payout, then score, wins, average position, and name.
- A duel winner can still underperform in a mixed field.

## Launch In Background (Long Marathons)

A `100 x 400` marathon runs ~50-60 min. Launch detached, record the main PID for a clean kill path, and tail the log instead of blocking:

```bash
LOG=/tmp/phaseXXX.log
nohup python3 -m engine.fixed_bot_mtt_evaluation \
  --config configs/phaseXXX_all_nontrash_100x400.json > "$LOG" 2>&1 &

# main runner PID (parent of the worker pool) — kill path:
pgrep -af "fixed_bot_mtt_evaluation.*phaseXXX" | grep -v "bash -c"
#   -> kill <main_pid>   tears down the whole worker pool

tail -f "$LOG"   # heartbeat/progress lines
```

Notes:

- The first PID is the parent runner; the rest are the `workers` pool children. Kill the parent.
- Do not raise `workers` to go faster on a long run — oversubscription has frozen the box. Run longer, not wider.
- Watch for `completed_mtts=N/M failures=0`. Non-zero `failures` usually means pool pressure — reduce workers and rerun.

## Live Progress

The runner prints progress lines:

```text
heartbeat: completed_mtts=8/100 failures=0 elapsed_s=229.2 since_last_completion_s=30.0 partial=runs/.../partial_summary.json
progress: completed_mtts=9/100 failures=0 elapsed_s=267.4 since_last_completion_s=0.0 partial=runs/.../partial_summary.json
```

Meanings:

- `heartbeat`: no MTT finished during the last progress interval, but the run is alive.
- `progress`: one or more MTTs finished.
- `failures`: failed MTT count.
- `partial`: live JSON summary path.

Long MTTs can cause heartbeat-only periods. Do not assume a stall unless `since_last_completion_s` is unusually large for the field size and machine.

## Print A Stable Table

After a run finishes, print the complete table from the report:

```bash
python3 -m engine.fixed_bot_mtt_evaluation --report runs/<run_id>/<timestamp>/fixed_bot_evaluation_report.json
```

Output columns:

- `payout`: total payout share won by that population.
- `avg payout`: payout per entry.
- `ROI`: payout versus equal-share expected payout for that report set.
- `score`: tournament score used by the runner.
- `wins`, `top3`, `FT`, `ITM`: conversion counts.
- `avg pos`: average finishing position.

## Merge Multiple Reports

You can combine reports into one stable table:

```bash
python3 -m engine.fixed_bot_mtt_evaluation --report \
  runs/run_a/timestamp_a/fixed_bot_evaluation_report.json \
  runs/run_b/timestamp_b/fixed_bot_evaluation_report.json
```

Only merge reports when the populations and tournament shapes are comparable. Merging unrelated fields can produce misleading ROI.

## Inspect Tool Telemetry

The final report contains:

```json
"population_action_summary": {
  "bot_name": {
    "tool_action_counts": {},
    "tool_reject_reason_counts": {},
    "tool_response_rates": {}
  }
}
```

Useful for tools such as `bluff_pressure`:

- `tool_action_counts.bluff_pressure`: how often the tool fired.
- `tool_reject_reason_counts.bluff_pressure`: why it did not fire.
- `tool_response_rates.bluff_pressure.opponent_fold_rate`: opponent fold response.
- `tool_response_rates.bluff_pressure.immediate_chip_delta_per_raise_estimate`: immediate chip estimate per bluff.

Important: tool telemetry is usually chip-level or spot-level. It does not prove tournament payout EV by itself. Always compare with payout, ROI, ITM, final-table, and top-3 results.

## Common Checks Before Running

Run focused tests after changing bot tools or configs:

```bash
python3 -m pytest tests/test_bot_factory.py tests/test_tournament_equity_bot.py
```

Check that the bot name exists:

```bash
rg -n '"name": "conf_mtt_eq_adaptiverange_icm_reraise_steal"' bot_configs
```

Check recent configs for copyable payout and engine blocks:

```bash
ls configs
```

## Common Failure Modes

- Unknown bot name: add the config to `bot_configs/` or use a registered legacy bot name.
- Duplicate bot config name: two JSON specs define the same `"name"`.
- Too many workers: run becomes slow, memory-heavy, or unstable. Reduce `"workers"`.
- Reused seed: runner rejects duplicate `tournament_seeds`.
- `max_hands_per_tournament` set: tournament can stop by hand count. Use `null` for real MTT finish.
- Missing full payout table: benchmark no longer matches recent marathon structure.

## Recommended Workflow

1. Create or edit the bot config in `bot_configs/`.
2. Add or update focused tests if bot logic changed.
3. Run focused tests.
4. Run a small duel or 8-MTT batch.
5. If promising, run a 20-MTT duel for 1000 entries per bot.
6. If still promising, add the bot to the 20-population all-non-trash marathon.
7. Print the stable report table with `--report`.
8. Commit source/config/doc changes, not generated `runs/`.
