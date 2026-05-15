# Phase 26 Implementation Plan: Engine Retraining Evaluation

## Goal

Compare the Phase 25 engine-rollout-trained candidate checkpoint against the Phase 24 source checkpoint inside the engine mixed-opponent MTT lineup before any promotion decision.

## Scope

- Read the latest accepted Phase 25 report.
- Require `phase26_engine_evaluation_allowed: true`.
- Resolve both `source_checkpoint_path` and `candidate_checkpoint_path`.
- Run paired source and candidate campaigns with the same deterministic tournament seeds and opponent lineup.
- Record tournament outcomes, model action mix, fallback counts, equity counts, and runtime artifacts per checkpoint.
- Accept only when the candidate has clean runtime behavior and does not regress beyond configured placement, ITM, and payout margins.

## Files

- `engine/phase26_engine_retraining_evaluation.py`
- `configs/phase26_engine_retraining_evaluation.json`
- `tests/test_phase26_engine_retraining_evaluation.py`
- `../scripts/phase26_engine_retraining_evaluation.sh`

## Acceptance

Phase 26 is accepted when:

- Phase 25 prerequisite checks pass.
- Source and candidate campaigns complete the required tournament count.
- Model inference and timeout fallbacks stay within limits.
- Stopped-max-hands rate stays within limits.
- Candidate average placement, ITM rate, and payout do not regress beyond configured margins.
- Candidate action mix is measurable.

Acceptance allows Phase 27 to make a promotion decision. It does not promote the checkpoint by itself.
