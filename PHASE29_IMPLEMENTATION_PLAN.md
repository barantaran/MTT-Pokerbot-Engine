# Phase 29 Implementation Plan: Reduced Clone Engine Evaluation

## Goal

Evaluate the Phase 28 reduced-observation imitation checkpoint inside real engine MTT tournaments before any promotion or fine-tuning decision.

## Scope

- Add an engine adapter for reduced-observation checkpoints.
- Load the Phase 28 checkpoint with `observation_size=8`.
- Encode live engine state through the reduced observation encoder.
- Keep the existing 9-action legal mask and engine action mapper.
- Run the reduced clone in a mixed 36-player MTT lineup.
- Compare reduced clone results against random and equity-aggressive bots.
- Gate only on runtime cleanliness, beating random by average placement, and staying close to the equity bot.

## Non-Goals

- Do not promote the reduced checkpoint in this phase.
- Do not replace the existing 62-feature engine model bot.
- Do not run PPO fine-tuning.
- Do not treat one short evaluation as proof that the clone is stronger than the equity bot.

## Implemented Files

- `engine/reduced_model_bot.py`
- `engine/phase29_reduced_clone_engine_evaluation.py`
- `configs/phase29_reduced_clone_engine_evaluation.json`
- `tests/test_phase29_reduced_clone_engine_evaluation.py`
- `../scripts/phase29_reduced_clone_engine_evaluation.sh`

## Acceptance Checks

- Latest accepted Phase 28 training report is resolved.
- Candidate reduced checkpoint exists.
- Engine tournaments complete without runtime failures.
- Reduced model fallback counts stay clean.
- Reduced model produces at least the configured minimum number of entries.
- Reduced model average placement beats random bots.
- Reduced model average placement stays within the configured gap to the equity bot.
- Action mix is reported and passes configured diversity limits.

## Result

The accepted Phase 29 run completed 10 tournaments with 30 reduced-model entries. The reduced clone beat random bots by average placement and stayed ahead of the equity-aggressive bot by average placement in this short sample, though the equity bot still won more total payout.

## Next Phase

Phase 30 should make a conservative reduced-clone decision. The current evidence supports larger paired evaluation or another imitation collection pass, not immediate replacement of the main promoted model.
