# Phase 22 Implementation Plan: Small MTT Engine Simulation

## Goal

Phase 22 runs the Phase 20-promoted checkpoint, already accepted by Phase 21 engine wiring, through a bounded real MTT engine simulation with mixed opponents.

This phase is the first actual engine tournament run for the promoted model bot. It should prove runtime compatibility and produce tournament evidence, not claim model strength or start training.

The Phase 22 loop is:

```text
read latest accepted Phase 21 engine wiring report
-> require small_mtt_engine_simulation_allowed: true
-> load the promoted checkpoint through BaselineModelEngineBot
-> build a fixed small MTT lineup with model, random, equity, and fixed baseline bots
-> run bounded tournaments through the engine Tournament API
-> validate action/runtime/result integrity
-> write a Phase 22 small simulation report
-> decide whether larger MTT engine simulation may start
```

## Scope

Included in Phase 22:

- A config for small, deterministic engine simulations
- Phase 21 prerequisite selection and gating
- A direct simulation runner that uses the engine `Tournament` API
- A mixed opponent lineup:
  - `BaselineModelEngineBot` from the promoted checkpoint
  - `RandomBot`
  - `AggressiveBot` as the equity-aware opponent
  - `AggressiveNoEquityBot` and/or `CallBot` as fixed baselines
- Generated tournament event logs under `runs/phase22_small_mtt_engine_simulation/<run>/`
- A structured report with result summaries, placement summaries, runtime summaries, fallback counts, and failure gates
- Unit tests for report gating, lineup construction, config behavior, and failure handling
- Root and engine README updates

Excluded from Phase 22:

- PPO training
- Checkpoint promotion
- Large tournament campaigns
- Player-loader integration through `main.py`
- Changes to observation schema or action ids
- Raw hidden card features in model observations

## Inputs

Phase 22 consumes the latest accepted Phase 21 report:

```text
MTT-Pokerbot-Engine/runs/phase21_engine_bot_wiring_smoke_test/*/engine_wiring_report.json
```

Required Phase 21 fields:

```text
phase21_engine_wiring_status: accepted
small_mtt_engine_simulation_allowed: true
promoted_checkpoint_path
phase21_engine_wiring_report_path
source_phase20_report_path
```

The promoted checkpoint path may be absolute or relative to `poker-ai-basemodel/`.

## Outputs

Phase 22 writes:

```text
MTT-Pokerbot-Engine/runs/phase22_small_mtt_engine_simulation/<run>/small_mtt_engine_simulation_report.json
MTT-Pokerbot-Engine/runs/phase22_small_mtt_engine_simulation/<run>/tournament_<id>_events.json
MTT-Pokerbot-Engine/runs/phase22_small_mtt_engine_simulation/<run>/tournament_results.json
```

Primary report fields:

```text
phase22_small_engine_simulation_id
phase22_small_engine_simulation_status
phase22_small_engine_simulation_failures
source_phase21_report_path
source_phase21_status
source_small_mtt_engine_simulation_allowed
promoted_checkpoint_path
lineup_summary
tournament_count
completed_tournament_count
stopped_max_hands_count
result_summary
placement_summary_by_bot_class
model_bot_summary
runtime_summary
bot_fallback_summary
event_log_paths
result_artifact_path
larger_mtt_engine_simulation_allowed
next_phase_recommendation
```

Accepted Phase 22 reports should set:

```text
phase22_small_engine_simulation_status: accepted
larger_mtt_engine_simulation_allowed: true
next_phase_recommendation: launch_phase23_larger_mtt_engine_simulation
```

Rejected Phase 22 reports should set:

```text
phase22_small_engine_simulation_status: rejected
larger_mtt_engine_simulation_allowed: false
next_phase_recommendation: resolve_phase22_small_engine_simulation_failures
```

## Proposed Config

Add:

```text
configs/phase22_small_mtt_engine_simulation.json
```

Suggested fields:

```json
{
  "phase21_report_glob": "runs/phase21_engine_bot_wiring_smoke_test/*/engine_wiring_report.json",
  "expected_phase21_status": "accepted",
  "required_small_mtt_engine_simulation_allowed": true,
  "output_dir": "runs/phase22_small_mtt_engine_simulation",
  "random_seed": 2200,
  "tournament_count": 3,
  "max_hands_per_tournament": 250,
  "starting_stack": 1500,
  "max_players_per_table": 9,
  "hands_per_level": 10,
  "bot_decision_timeout_ms": 500,
  "lineup": {
    "model": 1,
    "random": 3,
    "equity_aggressive": 2,
    "aggressive_no_equity": 2,
    "call": 1
  },
  "blinds_schedule": [
    {"small": 10, "big": 20},
    {"small": 15, "big": 30},
    {"small": 25, "big": 50},
    {"small": 50, "big": 100},
    {"small": 100, "big": 200}
  ],
  "payouts": {
    "1": 0.5,
    "2": 0.3,
    "3": 0.2
  },
  "acceptance": {
    "min_completed_tournaments": 1,
    "max_model_inference_errors": 0,
    "max_model_illegal_actions": 0,
    "max_model_timeout_fallbacks": 0,
    "require_event_logs": true,
    "require_result_artifact": true
  }
}
```

The lineup totals 9 players so the first implementation can stay single-table while still using the MTT engine. Later phases can increase table count.

## Implementation Steps

## Step 1: Phase 22 Config

- [x] Add `configs/phase22_small_mtt_engine_simulation.json`.
- [x] Include Phase 21 report discovery and explicit prerequisite gates.
- [x] Include bounded tournament count, max hands, seeds, and engine config overrides.
- [x] Include lineup counts for model, random, equity-aware, no-equity aggressive, and call bots.
- [x] Include acceptance thresholds.

Acceptance checks:

- [x] Missing Phase 21 reports produce a rejected Phase 22 report.
- [x] A rejected Phase 21 report blocks simulation.
- [x] A Phase 21 report without `small_mtt_engine_simulation_allowed: true` blocks simulation.
- [x] The config can be parsed by `python3 -m json.tool`.

## Step 2: Simulation Runner

- [x] Add `engine/phase22_small_mtt_engine_simulation.py`.
- [x] Resolve the latest accepted Phase 21 report unless an explicit path is passed.
- [x] Resolve and validate the promoted checkpoint.
- [x] Instantiate `BaselineModelEngineBot`.
- [x] Instantiate opponents from existing engine bot classes:
  - `players.random_bot.RandomBot`
  - `players.aggressive_bot.AggressiveBot`
  - `players.aggressive_no_equity_bot.AggressiveNoEquityBot`
  - `players.call_bot.CallBot`
- [x] Run tournaments through `engine.tournament.Tournament`.
- [x] Apply temporary engine config overrides during the run and restore them afterward.
- [x] Seed each tournament deterministically.
- [x] Write tournament event logs and a combined result artifact.
- [x] Write a structured report and exit nonzero on rejection.

Acceptance checks:

- [x] The runner does not use `main.py` player auto-loading.
- [x] The runner uses only visible engine `game_state` for runtime bots.
- [x] Every model action still passes the engine action contract through the Phase 21 adapter.
- [x] Bounded runs stop safely if `max_hands_per_tournament` is reached.
- [x] Generated artifacts stay under ignored `runs/`.

## Step 3: Report Metrics And Gates

- [x] Summarize completed tournament count and max-hand stops.
- [x] Summarize placements by bot class.
- [x] Summarize model bot placements, payouts, and survival evidence.
- [x] Summarize model bot fallback counts:
  - inference errors
  - illegal actions
  - no-model fallbacks
  - timeout fallbacks
- [x] Summarize model bot equity counts:
  - computed
  - fallbacks
- [x] Record event log paths and result artifact path.
- [x] Set `larger_mtt_engine_simulation_allowed` only when all gates pass.

Acceptance checks:

- [x] Illegal model actions reject the phase.
- [x] Model inference errors reject the phase.
- [x] Model timeout fallbacks reject the phase for this small diagnostic run.
- [x] Missing result artifacts reject the phase.
- [x] At least one tournament must complete or safely stop with valid placements.

## Step 4: Tests

- [x] Add focused Phase 22 tests under `tests/`.
- [x] Cover accepted Phase 21 prerequisite selection.
- [x] Cover rejected or disallowed Phase 21 prerequisite blocking.
- [x] Cover lineup construction counts and bot classes.
- [x] Cover config override restore behavior.
- [x] Cover report acceptance and rejection decisions.
- [x] Cover artifact path writing using a temporary directory.

Acceptance checks:

- [x] Phase 22 tests pass with `pytest`.
- [x] Existing Phase 21 tests still pass.
- [x] Syntax validation passes for new runner and bot imports.

## Step 5: Root Wrapper And Docs

- [x] Add `scripts/phase22_small_mtt_engine_simulation.sh`.
- [x] Update root `README.md` phase table.
- [x] Update root complete workspace testing pipeline to include Phase 22 after Phase 21.
- [x] Update `MTT-Pokerbot-Engine/README.md` with the Phase 22 command and expected report.
- [x] Keep generated Phase 22 artifacts ignored.

Acceptance checks:

- [x] The wrapper works from the workspace root.
- [x] The engine command works from `MTT-Pokerbot-Engine/`.
- [x] Docs state Phase 22 is a small diagnostic engine simulation, not a training phase or promotion proof.

## Validation Commands

From `MTT-Pokerbot-Engine/`:

```bash
python3 -m json.tool configs/phase22_small_mtt_engine_simulation.json
../poker-ai-basemodel/.venv/bin/python -m py_compile engine/phase22_small_mtt_engine_simulation.py
../poker-ai-basemodel/.venv/bin/python -m pytest tests/test_phase22_small_mtt_engine_simulation.py
../poker-ai-basemodel/.venv/bin/python -m engine.phase21_engine_bot_wiring --config configs/phase21_engine_bot_wiring_smoke_test.json
../poker-ai-basemodel/.venv/bin/python -m engine.phase22_small_mtt_engine_simulation --config configs/phase22_small_mtt_engine_simulation.json
```

From the workspace root:

```bash
./scripts/phase22_small_mtt_engine_simulation.sh
```

Expected accepted CLI summary:

```text
phase22_small_engine_simulation_status: accepted
larger_mtt_engine_simulation_allowed: true
next_phase_recommendation: launch_phase23_larger_mtt_engine_simulation
```

## Handoff To Phase 23

Phase 23 may start only when the Phase 22 report says:

```text
phase22_small_engine_simulation_status: accepted
larger_mtt_engine_simulation_allowed: true
```

Phase 23 should then expand from a single-table diagnostic lineup to a larger MTT engine simulation campaign with more tournaments, multi-table lineups, richer result aggregation, and stricter regression gates.
