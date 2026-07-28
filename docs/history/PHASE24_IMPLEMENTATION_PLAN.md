# Phase 24 Implementation Plan: Engine Rollout Collection For Retraining

## Goal

Phase 24 turns accepted engine tournament play into a training-ready rollout dataset.

Phase 23 proved that the promoted model bot can participate in larger real engine MTTs with random, equity-aware, no-equity aggressive, and call opponents. It also showed that model strength is weak enough that the next useful step is not another summary-only evaluation. The next useful step is collecting per-decision records from actual engine play so the basemodel trainer can learn from real MTT decisions.

This phase is data collection, not PPO training. It should produce a validated artifact that Phase 25 can consume for retraining.

The Phase 24 loop is:

```text
read latest accepted Phase 23 larger engine simulation report
-> require engine_rollout_collection_allowed: true
-> load the same promoted checkpoint through BaselineModelEngineBot
-> run bounded mixed-opponent MTTs through the engine Tournament API
-> record every model-bot decision as a training sample
-> attach terminal tournament outcomes and reward fields
-> validate observation, mask, action, privacy, and artifact integrity
-> write a Phase 24 rollout collection report
-> decide whether engine-backed PPO training may start
```

## Scope

Included in Phase 24:

- A rollout collection config
- Phase 23 prerequisite selection and gating
- A collector that records model-bot decisions during real engine tournaments
- Per-decision records for:
  - 62-float model observation
  - 9-way legal-action mask
  - selected action id
  - selected action probability/logprob when available
  - value estimate when available
  - mapped engine action and amount
  - tournament, hand, table, player, blind, stack, pot, and street metadata
  - fallback and inference diagnostics
- Terminal outcome records:
  - placement
  - payout percentage
  - in-the-money flag
  - final stack
  - bustout hand when available
- Reward fields suitable for later training:
  - immediate chip delta if available
  - terminal payout reward
  - terminal placement reward
  - configurable combined reward placeholder
- Artifact validation for finite observations, mask/action consistency, privacy, and loadability
- A compact JSONL or NPZ-style rollout artifact plus a manifest/report
- Focused tests for collector schema, prerequisite gates, privacy, validation, and report decisions
- Root and engine README updates

Excluded from Phase 24:

- PPO optimization
- Checkpoint promotion
- Model architecture changes
- Observation schema changes
- Raw hidden-card model features
- Training-only privileged information passed into runtime bot decisions
- Large unbounded tournament campaigns
- Broad simulator mechanics changes unless a correctness bug blocks collection

## Inputs

Phase 24 consumes the latest accepted Phase 23 report:

```text
MTT-Pokerbot-Engine/runs/phase23_larger_mtt_engine_simulation/*/larger_mtt_engine_simulation_report.json
```

Required Phase 23 fields:

```text
phase23_larger_engine_simulation_status: accepted
engine_rollout_collection_allowed: true
promoted_checkpoint_path
lineup_summary
bot_fallback_summary
artifact_paths
next_phase_recommendation
```

The promoted checkpoint path may be absolute or relative to `poker-ai-basemodel/`.

## Outputs

Phase 24 writes:

```text
MTT-Pokerbot-Engine/runs/phase24_engine_rollout_collection/<run>/engine_rollout_collection_report.json
MTT-Pokerbot-Engine/runs/phase24_engine_rollout_collection/<run>/rollouts/model_decisions.jsonl
MTT-Pokerbot-Engine/runs/phase24_engine_rollout_collection/<run>/rollouts/terminal_outcomes.jsonl
MTT-Pokerbot-Engine/runs/phase24_engine_rollout_collection/<run>/rollouts/manifest.json
MTT-Pokerbot-Engine/runs/phase24_engine_rollout_collection/<run>/tournament_results.json
MTT-Pokerbot-Engine/runs/phase24_engine_rollout_collection/<run>/event_summaries.json
```

Event logs should remain optional or sampled:

```text
MTT-Pokerbot-Engine/runs/phase24_engine_rollout_collection/<run>/events/tournament_<id>_events.json
```

Primary report fields:

```text
phase24_engine_rollout_collection_id
phase24_engine_rollout_collection_status
phase24_engine_rollout_collection_failures
source_phase23_report_path
source_phase23_status
source_engine_rollout_collection_allowed
promoted_checkpoint_path
lineup_summary
tournament_count
completed_tournament_count
stopped_max_hands_count
model_decision_count
model_terminal_outcome_count
rollout_schema_version
rollout_artifact_paths
rollout_validation_summary
observation_validation_summary
legal_mask_validation_summary
action_mapping_validation_summary
privacy_validation_summary
reward_summary
runtime_summary
bot_fallback_summary
artifact_integrity_summary
phase25_engine_training_allowed
next_phase_recommendation
```

Accepted Phase 24 reports should set:

```text
phase24_engine_rollout_collection_status: accepted
phase25_engine_training_allowed: true
next_phase_recommendation: launch_phase25_engine_rollout_training
```

Rejected Phase 24 reports should set:

```text
phase24_engine_rollout_collection_status: rejected
phase25_engine_training_allowed: false
next_phase_recommendation: resolve_phase24_rollout_collection_failures
```

## Rollout Record Contract

Each line in `model_decisions.jsonl` should be one model-bot decision:

```json
{
  "schema_version": 1,
  "phase": 24,
  "tournament_id": 0,
  "hand_id": 12,
  "table_id": 1,
  "decision_index": 184,
  "player_name": "Phase24ModelBot_1",
  "player_seat": 4,
  "street": "flop",
  "small_blind": 25,
  "big_blind": 50,
  "stack_before": 1320,
  "pot_before": 275,
  "call_amount": 50,
  "min_raise": 100,
  "observation": [0.0],
  "legal_action_mask": [1, 0, 1, 1, 1, 1, 1, 1, 1],
  "selected_action_id": 5,
  "selected_action_name": "Bet_1",
  "selected_action_probability": 0.22,
  "selected_action_logprob": -1.51,
  "value_estimate": -0.03,
  "engine_action": "raise",
  "engine_amount": 275,
  "fallback_used": false,
  "fallback_reason": null,
  "inference_error": null
}
```

The example observation is abbreviated. Actual records must contain exactly 62 finite floats.

Each line in `terminal_outcomes.jsonl` should attach final tournament outcome for each model bot:

```json
{
  "schema_version": 1,
  "phase": 24,
  "tournament_id": 0,
  "player_name": "Phase24ModelBot_1",
  "placement": 14,
  "field_size": 36,
  "payout_pct": 0.0,
  "in_the_money": false,
  "final_stack": 0,
  "bustout_hand_id": 93,
  "terminal_reward": -0.36
}
```

## Proposed Config

Add:

```text
configs/phase24_engine_rollout_collection.json
```

Suggested fields:

```json
{
  "phase23_report_glob": "runs/phase23_larger_mtt_engine_simulation/*/larger_mtt_engine_simulation_report.json",
  "expected_phase23_status": "accepted",
  "required_engine_rollout_collection_allowed": true,
  "output_dir": "runs/phase24_engine_rollout_collection",
  "random_seed": 2400,
  "tournament_count": 10,
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
  "rollout_collection": {
    "schema_version": 1,
    "write_jsonl": true,
    "include_action_probabilities": true,
    "include_value_estimates": true,
    "include_terminal_outcomes": true,
    "include_reward_fields": true,
    "flush_every_decisions": 100
  },
  "event_logging": {
    "write_full_event_logs": false,
    "write_full_event_logs_for_first_n": 1,
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
    "min_completed_tournaments": 8,
    "min_model_decisions": 50,
    "max_stopped_max_hands_rate": 0.2,
    "max_model_inference_errors": 0,
    "max_model_illegal_actions": 0,
    "max_model_timeout_fallbacks": 0,
    "require_rollout_artifact": true,
    "require_terminal_outcomes": true,
    "require_manifest": true,
    "require_no_privacy_violations": true
  }
}
```

## Implementation Steps

## Step 1: Phase 24 Config

- [ ] Add `configs/phase24_engine_rollout_collection.json`.
- [ ] Include Phase 23 report discovery and prerequisite gates.
- [ ] Include bounded tournament count, max hands, deterministic seed, and engine config overrides.
- [ ] Include mixed lineup counts matching Phase 23 unless a smaller smoke budget is needed.
- [ ] Include rollout schema and artifact-writing options.
- [ ] Include acceptance thresholds for tournament completion, model decision count, fallbacks, and artifact integrity.

Acceptance checks:

- [ ] Missing Phase 23 reports produce a rejected Phase 24 report.
- [ ] Rejected Phase 23 reports block collection.
- [ ] Phase 23 reports without `engine_rollout_collection_allowed: true` block collection.
- [ ] The config parses with `python3 -m json.tool`.

## Step 2: Rollout Collector Contract

- [ ] Add a small collector module or class that owns rollout record construction and artifact writing.
- [ ] Keep the collector separate from betting/pot logic.
- [ ] Record only model-bot decisions, not every opponent decision, unless explicitly configured.
- [ ] Record exactly 62 finite observation floats.
- [ ] Record exactly 9 legal-action mask entries.
- [ ] Record selected action id and mapped engine action together.
- [ ] Record fallback status and inference diagnostics.
- [ ] Write JSONL records incrementally so a long run does not keep all decisions in memory.
- [ ] Write a manifest with schema version, counts, source report, config snapshot, and artifact checksums or sizes.

Acceptance checks:

- [ ] A malformed observation rejects the phase.
- [ ] A non-finite observation rejects the phase.
- [ ] A selected action id outside the 9-way action space rejects the phase.
- [ ] A selected action not legal under the mask rejects the phase unless it was explicitly corrected through the fallback path.
- [ ] Rollout files can be loaded back without custom runtime state.

## Step 3: Model Bot Instrumentation

- [ ] Extend `BaselineModelEngineBot` or wrap it with a Phase 24 collector hook.
- [ ] Capture the model observation before inference.
- [ ] Capture legal-action mask before sampling or deterministic selection.
- [ ] Capture selected action id before mapping to engine action.
- [ ] Capture selected action probability/logprob and value estimate if the basemodel wrapper exposes them.
- [ ] If probability/logprob/value are not currently exposed, record `null` and add a clear report field explaining the limitation.
- [ ] Preserve conservative fallback behavior for inference errors, illegal actions, and timeouts.
- [ ] Do not pass hidden opponent cards, future board cards, deck state, or terminal labels into runtime model inference.

Acceptance checks:

- [ ] Existing Phase 21 action-contract tests still pass.
- [ ] Existing Phase 22 and Phase 23 simulation tests still pass.
- [ ] Runtime bot-facing state remains visible-state only.
- [ ] Instrumentation does not change engine action legality.

## Step 4: Collection Runner

- [ ] Add `engine/phase24_engine_rollout_collection.py`.
- [ ] Resolve the latest accepted Phase 23 report unless an explicit path is passed.
- [ ] Resolve and validate the promoted checkpoint.
- [ ] Instantiate model bots with collection enabled.
- [ ] Instantiate mixed opponents:
  - `RandomBot`
  - equity-aware `AggressiveBot`
  - `AggressiveNoEquityBot`
  - `CallBot`
- [ ] Run bounded tournaments through `engine.tournament.Tournament`.
- [ ] Apply temporary engine config overrides and restore them after the run.
- [ ] Write tournament results and event summaries.
- [ ] Attach terminal outcomes to model-bot episodes.
- [ ] Write the final collection report and exit nonzero on rejection.

Acceptance checks:

- [ ] The runner does not use `main.py` player auto-loading.
- [ ] Multi-table seating is exercised by lineup size.
- [ ] Generated artifacts stay under ignored `runs/`.
- [ ] Report clearly distinguishes data-collection acceptance from model-quality conclusions.

## Step 5: Reward And Outcome Attachment

- [ ] Define reward fields without committing to a final PPO objective.
- [ ] Attach terminal payout and placement rewards to each model bot.
- [ ] Record enough metadata for Phase 25 to compute returns by tournament episode.
- [ ] If immediate chip deltas are not available without simulator changes, record `null` and document the gap.
- [ ] Avoid hidden-information leakage in any reward or terminal fields used during runtime decisions.

Acceptance checks:

- [ ] Every model bot with decisions has a terminal outcome record unless the tournament stops at max hands.
- [ ] Terminal records match `tournament_results.json`.
- [ ] Reward fields are numeric or explicitly `null`; no mixed string/numeric values.

## Step 6: Tests

- [ ] Add `tests/test_phase24_engine_rollout_collection.py`.
- [ ] Cover accepted Phase 23 prerequisite selection.
- [ ] Cover rejected/disallowed Phase 23 prerequisite blocking.
- [ ] Cover rollout record schema validation.
- [ ] Cover 62-float finite observation validation.
- [ ] Cover 9-way mask validation.
- [ ] Cover action id to engine action consistency.
- [ ] Cover privacy rejection for hidden-card or training-only fields.
- [ ] Cover terminal outcome attachment.
- [ ] Cover report acceptance and rejection decisions.
- [ ] Cover artifact path writing with a temporary directory.

Acceptance checks:

- [ ] Phase 24 tests pass with `pytest`.
- [ ] Phase 21, Phase 22, and Phase 23 tests still pass.
- [ ] Syntax validation passes for the new runner and collector.

## Step 7: Root Wrapper And Docs

- [ ] Add `scripts/phase24_engine_rollout_collection.sh`.
- [ ] Update root `README.md` phase table.
- [ ] Update root complete workspace testing pipeline to include Phase 24 after Phase 23 once implemented.
- [ ] Update `MTT-Pokerbot-Engine/README.md` with the Phase 24 command and expected report.
- [ ] Document that Phase 24 creates training data but does not train.

Acceptance checks:

- [ ] The wrapper works from the workspace root.
- [ ] The engine command works from `MTT-Pokerbot-Engine/`.
- [ ] Docs state that Phase 25 should consume the validated rollout artifact for training.

## Validation Commands

From `MTT-Pokerbot-Engine/`:

```bash
python3 -m json.tool configs/phase24_engine_rollout_collection.json
../poker-ai-basemodel/.venv/bin/python -m py_compile engine/phase24_engine_rollout_collection.py
../poker-ai-basemodel/.venv/bin/python -m pytest tests/test_phase21_engine_bot_wiring.py tests/test_phase22_small_mtt_engine_simulation.py tests/test_phase23_larger_mtt_engine_simulation.py tests/test_phase24_engine_rollout_collection.py
../poker-ai-basemodel/.venv/bin/python -m engine.phase24_engine_rollout_collection --config configs/phase24_engine_rollout_collection.json
```

From the workspace root:

```bash
./scripts/phase24_engine_rollout_collection.sh
```

Expected accepted CLI summary:

```text
phase24_engine_rollout_collection_status: accepted
model_decision_count: <positive count>
phase25_engine_training_allowed: true
next_phase_recommendation: launch_phase25_engine_rollout_training
```

## Handoff To Phase 25

Phase 25 should start only after Phase 24 writes an accepted report with a loadable rollout artifact.

Phase 25 should consume the Phase 24 records, compute returns or advantages, run a bounded PPO update, evaluate the candidate against fixed opponents and the previous checkpoint, and write a promotion or rejection report. Phase 25 should not depend on Phase 23 event summaries because those summaries do not contain enough per-decision training data.
