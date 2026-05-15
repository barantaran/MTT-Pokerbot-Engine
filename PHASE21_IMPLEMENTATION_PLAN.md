# Phase 21 Implementation Plan: Engine Bot Wiring Smoke Test

Root roadmap entry: [Engine Bot Wiring Smoke Test](../README.md#implementation-phases).

## Goal

Phase 21 proves that the Phase 20-promoted checkpoint can be loaded as an engine-compatible MTT bot before any real engine simulation campaign starts.

Phase 20 accepted the best Phase 19 larger-training checkpoint and wrote `mtt_engine_bot_simulation_allowed: true`. Phase 21 should not train and should not run an open-ended tournament batch. It should wire the promoted checkpoint through the existing engine `Bot` API, exercise deterministic visible-state decisions, verify fallback behavior, and write a structured report that decides whether the first small MTT engine simulation may start.

The Phase 21 loop is:

```text
read the latest accepted Phase 20 promotion report
-> require phase20_promotion_status: accepted
-> require mtt_engine_bot_simulation_allowed: true
-> resolve promoted_checkpoint_path
-> load the basemodel policy through an engine-compatible bot adapter
-> pass representative engine game_state dictionaries through the adapter
-> verify observation encoding, legal-action masking, model action mapping, timeouts, and fallbacks
-> verify bot-facing state does not expose hidden opponent cards
-> write a Phase 21 engine wiring report
-> decide whether the first small MTT engine simulation may start
```

## Boundaries

Included in Phase 21:

- A config that locates the latest accepted Phase 20 promotion report
- Phase 20 prerequisite checks
- A thin engine bot adapter for the promoted basemodel checkpoint
- Checkpoint loading through the existing basemodel model/checkpoint utilities
- Deterministic smoke states for check, call, raise, short-stack, and malformed-state paths
- Action contract checks for exactly `("fold", 0)`, `("call", 0)`, or `("raise", amount)`
- Verification that `("call", 0)` remains the engine representation for check when `call_amount == 0`
- Conservative fallback when model loading, inference, illegal action selection, malformed state, or timeout occurs
- Bot-facing privacy checks for no opponent hole cards or training-only privileged fields
- Runtime checks against `bot_decision_timeout_ms`
- A structured report with `small_mtt_engine_simulation_allowed`
- A root wrapper command
- Tests for accepted, rejected, missing-report, missing-checkpoint, action-contract, fallback, timeout, and privacy cases

Excluded from Phase 21:

- More training updates
- Full MTT simulation campaigns
- Tournament result quality claims
- New model architecture
- Raw-card model features
- PokerStove as a required dependency
- Broad simulator mechanics changes
- Table balancing, pot, side-pot, blind, or payout refactors unless a correctness bug blocks wiring
- Generated simulation logs, CSVs, checkpoints, or datasets committed to git

## Contracts

Phase 21 consumes these Phase 20 fields:

```text
phase20_promotion_eval_id
phase20_promotion_status
phase20_promotion_failures
phase20_promotion_report_path
source_phase19_report_path
candidate_checkpoint_path
baseline_checkpoint_path
promoted_checkpoint_path
equity_source
equity_fallback_source
phase20_gate_results
mtt_engine_bot_simulation_allowed
next_phase_recommendation
```

Phase 21 writes these fields:

```text
phase21_engine_wiring_id
phase21_engine_wiring_status
phase21_engine_wiring_failures
phase21_engine_wiring_report_path
source_phase20_report_path
source_phase20_status
source_mtt_engine_bot_simulation_allowed
promoted_checkpoint_path
engine_adapter_path
basemodel_import_path
checkpoint_load_status
checkpoint_load_error
smoke_state_count
smoke_action_outputs
fallback_action_outputs
timeout_probe_result
privacy_probe_result
observation_probe_result
action_contract_probe_result
runtime_summary
artifact_integrity_summary
phase21_gate_results
small_mtt_engine_simulation_allowed
next_phase_recommendation
```

Metrics can be `null` when a probe is not applicable, but report keys should remain stable.

## Result Metrics

Phase 21 should preserve the Phase 20 decision evidence:

```text
source_phase20_status
source_mtt_engine_bot_simulation_allowed
source_candidate_checkpoint_path
source_promoted_checkpoint_path
source_score_delta
source_roi_delta
source_phase20_failures
```

Additional Phase 21 metrics:

```text
source_accepted_gate_passed
source_engine_sim_allowed_gate_passed
promoted_checkpoint_exists_gate_passed
basemodel_import_gate_passed
engine_adapter_import_gate_passed
checkpoint_load_gate_passed
observation_size_gate_passed
observation_finite_gate_passed
legal_action_mask_gate_passed
engine_action_shape_gate_passed
engine_action_legal_gate_passed
check_call_contract_gate_passed
fallback_action_gate_passed
timeout_fallback_gate_passed
privacy_gate_passed
runtime_gate_passed
artifact_report_write_gate_passed
small_mtt_engine_simulation_allowed
```

## Proposed Files

```text
configs/
  phase21_engine_bot_wiring_smoke_test.json

engine/
  baseline_model_bot.py
  phase21_engine_bot_wiring.py

tests/
  test_phase21_engine_bot_wiring.py

scripts/
  phase21_engine_bot_wiring_smoke_test.sh
```

Likely existing files to touch:

```text
README.md
main.py
players/
../README.md
../poker-ai-basemodel/poker_ai/bot.py
../poker-ai-basemodel/poker_ai/checkpoint.py
../poker-ai-basemodel/tests/test_bot_smoke.py
```

Prefer a thin adapter in the engine tree that imports the basemodel package through an explicit configured path. Keep the existing `players/` directory usable for user bots and avoid weakening the engine import allow-list for arbitrary third-party bot scripts unless the model adapter needs a narrowly documented exception.

## Step 1: Phase 21 Smoke Config

- [x] Add `configs/phase21_engine_bot_wiring_smoke_test.json`.
- [x] Include a glob for Phase 20 reports, defaulting to `../poker-ai-basemodel/runs/phase20_mtt_promotion_evaluation/*/promotion_evaluation_report.json`.
- [x] Include expected Phase 20 status and permission gates.
- [x] Include the basemodel import root, timeout budget, deterministic decision mode, and report output template.
- [x] Include representative smoke states that cover check, call, raise, all-in pressure, and malformed inputs.

Acceptance checks:

- [x] Config parses with `python3 -m json.tool`.
- [x] Missing or rejected Phase 20 reports fail with a clear report field and non-destructive exit.
- [x] The selected report is the latest accepted timestamped Phase 20 report unless an explicit path is provided.

## Step 2: Engine Basemodel Adapter

- [x] Add a thin adapter that subclasses `engine.player_interface.Bot`.
- [x] Load the promoted checkpoint into the existing basemodel actor-critic model.
- [x] Use `poker_ai.bot.NeuralBaselineBot` or the existing basemodel action/observation utilities rather than duplicating model logic in the engine.
- [x] Keep model inference deterministic for the smoke test.
- [x] Return only engine-compatible actions:

```text
("fold", 0)
("call", 0)
("raise", amount)
```

- [x] Treat `("call", 0)` as the check representation when `call_amount == 0`.
- [x] Keep fallback conservative and legal on load failure, inference failure, malformed state, timeout, or invalid model action.

Acceptance checks:

- [x] A promoted checkpoint loads from the Phase 20 report.
- [x] The adapter produces a valid engine action for every configured smoke state.
- [x] Invalid or missing model state returns a conservative action rather than raising into the engine loop.
- [x] Adapter code does not expose opponent hole cards or privileged tournament-training fields to the model.

## Step 3: Privacy And State Contract Probes

- [x] Define the allowed bot-facing state keys for runtime play.
- [x] Probe the engine table state shape currently passed to `get_action`.
- [x] Reject state dictionaries containing opponent hole-card keys, full deck state, evaluator internals, future board cards, or training-only privileged labels.
- [x] Confirm the model observation remains exactly 62 finite floats.
- [x] Confirm legal-action masks and selected action ids agree before mapping to engine output.

Acceptance checks:

- [x] Privacy probe passes for normal engine states.
- [x] Privacy probe fails for injected hidden-card fields.
- [x] Observation probe fails clearly if the basemodel schema changes.
- [x] Legal-action probe fails clearly if the action contract changes.

## Step 4: Phase 21 Report Runner

- [x] Add a command-line runner for the Phase 21 smoke test.
- [x] Resolve and validate Phase 20 prerequisites.
- [x] Run adapter import, checkpoint load, observation, action, fallback, timeout, privacy, and artifact probes.
- [x] Write a timestamped report under `runs/phase21_engine_bot_wiring_smoke_test/<run>/`.
- [x] Set `phase21_engine_wiring_status: accepted` only when all required gates pass.
- [x] Set `small_mtt_engine_simulation_allowed: true` only when accepted.
- [x] Set `next_phase_recommendation: launch_phase22_small_mtt_engine_simulation` when accepted.

Acceptance checks:

- [x] Accepted report includes stable Phase 21 schema fields.
- [x] Rejected report records all failed gates.
- [x] Missing checkpoint reports `small_mtt_engine_simulation_allowed: false`.
- [x] Runner exits successfully for expected rejection cases where a report is written.

## Step 5: Tests

- [x] Add focused Phase 21 unit tests in the engine tree.
- [x] Cover accepted Phase 20 prerequisite selection.
- [x] Cover rejected/disallowed Phase 20 prerequisites.
- [x] Cover missing promoted checkpoint.
- [x] Cover checkpoint load success using a tiny test checkpoint or monkeypatched loader.
- [x] Cover action output shape and legal raise amounts.
- [x] Cover check-as-call behavior.
- [x] Cover malformed-state fallback.
- [x] Cover timeout fallback.
- [x] Cover hidden-card privacy rejection.
- [x] Cover report schema and `small_mtt_engine_simulation_allowed`.

Acceptance checks:

- [x] Engine Phase 21 tests pass.
- [x] Existing basemodel bot/action/observation tests still pass.
- [x] Engine syntax/import validation still passes.

## Step 6: Root Wrapper And Docs

- [x] Add `scripts/phase21_engine_bot_wiring_smoke_test.sh`.
- [x] Document the Phase 21 command in the root README.
- [x] Document the engine adapter path and expected report in the engine README.
- [x] Keep generated Phase 21 reports ignored unless intentionally promoted to fixtures.

Acceptance checks:

- [x] Root wrapper can be run from `/home/varantaran/poker-ai`.
- [x] README commands match the implemented CLI.
- [x] Docs clearly state Phase 21 is not a tournament-quality evaluation.

## Runbook

From the workspace root:

```bash
./scripts/phase21_engine_bot_wiring_smoke_test.sh
```

Or from the engine tree after implementation:

```bash
cd MTT-Pokerbot-Engine
python3 -m engine.phase21_engine_bot_wiring --config configs/phase21_engine_bot_wiring_smoke_test.json
```

Expected accepted result:

```text
phase21_engine_wiring_status: accepted
small_mtt_engine_simulation_allowed: true
next_phase_recommendation: launch_phase22_small_mtt_engine_simulation
```

## Validation

Narrow validation:

```bash
cd MTT-Pokerbot-Engine
python3 -m json.tool configs/phase21_engine_bot_wiring_smoke_test.json
python3 -m pytest tests/test_phase21_engine_bot_wiring.py
python3 -m py_compile main.py server.py engine/*.py players/*.py
```

Cross-contract validation:

```bash
cd poker-ai-basemodel
.venv/bin/python -m pytest tests/test_bot_smoke.py tests/test_actions.py tests/test_observations.py
```

Phase run:

```bash
./scripts/phase21_engine_bot_wiring_smoke_test.sh
```

## Handoff To Phase 22

Phase 22 may start only when the Phase 21 report says:

```text
phase21_engine_wiring_status: accepted
small_mtt_engine_simulation_allowed: true
```

Phase 22 should then run a bounded small MTT engine simulation using the wired bot, collect actual engine results/log evidence, and decide whether larger MTT engine bot simulation can start.
