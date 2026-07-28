# Phase 23 Implementation Plan: Larger MTT Engine Simulation

## Goal

Phase 23 expands the accepted Phase 22 diagnostic run into a larger engine-side MTT simulation campaign.

The purpose is to determine whether the promoted model bot's weak Phase 22 placements were small-sample noise or a repeatable performance problem against random, equity-aware, and fixed baseline opponents.

This phase is still evaluation, not training. It should produce enough tournament evidence to decide whether to:

- proceed to engine-backed rollout collection for training,
- run another larger or adjusted simulation,
- or return to model training/evaluation fixes before more engine play.

The Phase 23 loop is:

```text
read latest accepted Phase 22 small engine simulation report
-> require larger_mtt_engine_simulation_allowed: true
-> load the same promoted checkpoint through BaselineModelEngineBot
-> build larger mixed MTT fields with model, random, equity, and fixed bots
-> run a bounded multi-tournament simulation campaign through Tournament
-> aggregate model placement, payout, bustout, action, runtime, and fallback diagnostics
-> write a Phase 23 larger simulation report
-> recommend engine-backed rollout collection, more evaluation, or training repair
```

## Scope

Included in Phase 23:

- A larger MTT simulation config
- Phase 22 prerequisite selection and gating
- A campaign runner that reuses the Phase 22 engine tournament harness patterns
- Larger mixed lineups, including:
  - `BaselineModelEngineBot`
  - `RandomBot`
  - equity-aware `AggressiveBot`
  - `AggressiveNoEquityBot`
  - `CallBot`
- Multi-table fields when lineup size exceeds `max_players_per_table`
- Aggregated result summaries over many tournaments
- Model placement distribution, payout/ROI-style summary, and opponent-class comparison
- Model fallback and equity-count summaries
- Event/result artifacts under `runs/phase23_larger_mtt_engine_simulation/<run>/`
- Focused tests for prerequisite gates, config parsing, lineup generation, aggregation, and report decisions
- Root and engine README updates

Excluded from Phase 23:

- PPO training
- Replay-buffer or rollout dataset generation
- Checkpoint promotion
- Observation/action schema changes
- Hidden-card or training-only state exposure to runtime bots
- Broad simulator rules changes unless the larger run exposes a correctness bug

## Inputs

Phase 23 consumes the latest accepted Phase 22 report:

```text
MTT-Pokerbot-Engine/runs/phase22_small_mtt_engine_simulation/*/small_mtt_engine_simulation_report.json
```

Required Phase 22 fields:

```text
phase22_small_engine_simulation_status: accepted
larger_mtt_engine_simulation_allowed: true
promoted_checkpoint_path
source_phase21_report_path
lineup_summary
bot_fallback_summary
```

## Outputs

Phase 23 writes:

```text
MTT-Pokerbot-Engine/runs/phase23_larger_mtt_engine_simulation/<run>/larger_mtt_engine_simulation_report.json
MTT-Pokerbot-Engine/runs/phase23_larger_mtt_engine_simulation/<run>/tournament_results.json
MTT-Pokerbot-Engine/runs/phase23_larger_mtt_engine_simulation/<run>/event_summaries.json
```

Event logs may be optional or sampled to avoid excessive artifact size:

```text
MTT-Pokerbot-Engine/runs/phase23_larger_mtt_engine_simulation/<run>/events/tournament_<id>_events.json
```

Primary report fields:

```text
phase23_larger_engine_simulation_id
phase23_larger_engine_simulation_status
phase23_larger_engine_simulation_failures
source_phase22_report_path
source_phase22_status
source_larger_mtt_engine_simulation_allowed
promoted_checkpoint_path
lineup_summary
tournament_count
completed_tournament_count
stopped_max_hands_count
model_entry_count
model_average_position
model_median_position
model_top_3_rate
model_itm_rate
model_win_rate
model_total_payout_pct
placement_summary_by_bot_class
payout_summary_by_bot_class
bustout_summary
runtime_summary
bot_fallback_summary
model_equity_summary
artifact_paths
engine_rollout_collection_allowed
next_phase_recommendation
```

Accepted Phase 23 reports should set:

```text
phase23_larger_engine_simulation_status: accepted
```

The next recommendation depends on the measured result:

```text
next_phase_recommendation: launch_phase24_engine_rollout_collection
```

only if runtime gates pass and the report provides enough evidence to justify training-data collection.

If the model remains clearly weak but runtime is stable:

```text
next_phase_recommendation: plan_phase24_engine_rollout_collection_for_retraining
```

If runtime or simulator integrity fails:

```text
next_phase_recommendation: resolve_phase23_engine_simulation_failures
```

## Proposed Config

Add:

```text
configs/phase23_larger_mtt_engine_simulation.json
```

Suggested fields:

```json
{
  "phase22_report_glob": "runs/phase22_small_mtt_engine_simulation/*/small_mtt_engine_simulation_report.json",
  "expected_phase22_status": "accepted",
  "required_larger_mtt_engine_simulation_allowed": true,
  "output_dir": "runs/phase23_larger_mtt_engine_simulation",
  "random_seed": 2300,
  "tournament_count": 50,
  "max_hands_per_tournament": 600,
  "starting_stack": 1500,
  "max_players_per_table": 9,
  "hands_per_level": 10,
  "bot_decision_timeout_ms": 500,
  "lineup": {
    "model": 3,
    "random": 12,
    "equity_aggressive": 9,
    "aggressive_no_equity": 9,
    "call": 3
  },
  "event_logging": {
    "write_full_event_logs": false,
    "write_full_event_logs_for_first_n": 3,
    "write_event_summaries": true
  },
  "blinds_schedule": [
    {"small": 10, "big": 20},
    {"small": 15, "big": 30},
    {"small": 25, "big": 50},
    {"small": 50, "big": 100},
    {"small": 100, "big": 200},
    {"small": 200, "big": 400},
    {"small": 400, "big": 800}
  ],
  "payouts": {
    "1": 0.35,
    "2": 0.22,
    "3": 0.16,
    "4": 0.11,
    "5": 0.08,
    "6": 0.05,
    "7": 0.03
  },
  "acceptance": {
    "min_completed_tournaments": 45,
    "max_stopped_max_hands_rate": 0.1,
    "max_model_inference_errors": 0,
    "max_model_illegal_actions": 0,
    "max_model_timeout_fallbacks": 0,
    "require_result_artifact": true,
    "require_event_summary_artifact": true
  },
  "decision_thresholds": {
    "min_model_entries_for_decision": 100,
    "weak_model_average_position_quantile": 0.6,
    "positive_model_itm_rate": 0.15
  }
}
```

The suggested lineup totals 36 players, creating a real multi-table MTT field while keeping runtime bounded.

## Implementation Steps

## Step 1: Phase 23 Config

- [x] Add `configs/phase23_larger_mtt_engine_simulation.json`.
- [x] Include Phase 22 report discovery and prerequisite gates.
- [x] Include larger tournament count, max-hand cap, deterministic seed, and engine config overrides.
- [x] Include multi-table lineup counts.
- [x] Include artifact-size controls for event logs.
- [x] Include acceptance thresholds and decision thresholds.

Acceptance checks:

- [x] Missing Phase 22 reports produce a rejected Phase 23 report.
- [x] Rejected Phase 22 reports block simulation.
- [x] Phase 22 reports without `larger_mtt_engine_simulation_allowed: true` block simulation.
- [x] The config parses with `python3 -m json.tool`.

## Step 2: Larger Simulation Runner

- [x] Add `engine/phase23_larger_mtt_engine_simulation.py`.
- [x] Resolve the latest accepted Phase 22 report unless an explicit path is passed.
- [x] Resolve and validate the promoted checkpoint.
- [x] Reuse or factor shared helpers from Phase 22 where appropriate without broad refactoring.
- [x] Instantiate multiple model bots and mixed opponents with unique names.
- [x] Run bounded tournaments through `engine.tournament.Tournament`.
- [x] Apply temporary engine config overrides and restore them after the run.
- [x] Write combined tournament results.
- [x] Write event summaries for every tournament.
- [x] Write full event logs only according to the event logging config.
- [x] Exit nonzero on rejection.

Acceptance checks:

- [x] The runner does not use `main.py` player auto-loading.
- [x] Runtime bots receive only visible game state.
- [x] All model bots go through `BaselineModelEngineBot`.
- [x] Multi-table seating is exercised by lineup size.
- [x] Generated artifacts stay under ignored `runs/`.

## Step 3: Aggregation And Diagnostics

- [x] Summarize tournament completion and max-hand stop rate.
- [x] Summarize placement distribution by bot class.
- [x] Summarize payout distribution by bot class.
- [x] Summarize model placement metrics:
  - average position
  - median position
  - best/worst position
  - top-3 rate
  - in-the-money rate
  - win rate
  - total payout percentage
- [x] Summarize model bustout timing where event data is available.
- [x] Summarize model fallback counts.
- [x] Summarize model equity counts.
- [x] Record runtime and artifact integrity.
- [x] Make the next-phase recommendation from diagnostics, not from a single tournament outcome.

Acceptance checks:

- [x] Illegal model actions reject the phase.
- [x] Model inference errors reject the phase.
- [x] Model timeout fallbacks reject the phase.
- [x] Too many max-hand stops reject the phase.
- [x] Missing result or event-summary artifacts reject the phase.
- [x] Report clearly distinguishes runtime acceptance from model-strength conclusions.

## Step 4: Tests

- [x] Add `tests/test_phase23_larger_mtt_engine_simulation.py`.
- [x] Cover accepted Phase 22 prerequisite selection.
- [x] Cover rejected/disallowed Phase 22 prerequisite blocking.
- [x] Cover multi-table lineup construction counts.
- [x] Cover event log sampling decisions.
- [x] Cover aggregation metrics for model placements and payouts.
- [x] Cover report acceptance and rejection decisions.
- [x] Cover artifact path writing with a temporary directory.

Acceptance checks:

- [x] Phase 23 tests pass with `pytest`.
- [x] Phase 22 tests still pass.
- [x] Syntax validation passes for the new runner.

## Step 5: Root Wrapper And Docs

- [x] Add `scripts/phase23_larger_mtt_engine_simulation.sh`.
- [x] Update root `README.md` phase table.
- [x] Update root complete workspace testing pipeline to include Phase 23 after Phase 22.
- [x] Update `MTT-Pokerbot-Engine/README.md` with the Phase 23 command and expected report.
- [x] Document that Phase 23 is evaluation, not training.

Acceptance checks:

- [ ] The wrapper works from the workspace root.
- [x] The engine command works from `MTT-Pokerbot-Engine/`.
- [x] Docs state that Phase 24 should collect proper per-decision rollouts before training.

## Validation Commands

From `MTT-Pokerbot-Engine/`:

```bash
python3 -m json.tool configs/phase23_larger_mtt_engine_simulation.json
../poker-ai-basemodel/.venv/bin/python -m py_compile engine/phase23_larger_mtt_engine_simulation.py
../poker-ai-basemodel/.venv/bin/python -m pytest tests/test_phase22_small_mtt_engine_simulation.py tests/test_phase23_larger_mtt_engine_simulation.py
../poker-ai-basemodel/.venv/bin/python -m engine.phase23_larger_mtt_engine_simulation --config configs/phase23_larger_mtt_engine_simulation.json
```

From the workspace root:

```bash
./scripts/phase23_larger_mtt_engine_simulation.sh
```

Expected accepted CLI summary:

```text
phase23_larger_engine_simulation_status: accepted
engine_rollout_collection_allowed: true
next_phase_recommendation: launch_phase24_engine_rollout_collection
```

or, if the model is weak but runtime is clean:

```text
phase23_larger_engine_simulation_status: accepted
engine_rollout_collection_allowed: true
next_phase_recommendation: plan_phase24_engine_rollout_collection_for_retraining
```

## Handoff To Phase 24

Phase 24 should not train directly from Phase 23 result summaries.

Phase 24 should add engine-backed rollout collection with per-decision records:

```text
observation
legal action mask
chosen action id
engine action
reward components
terminal placement/payout
episode/tournament metadata
```

Training can start only after that data contract is implemented and validated.
