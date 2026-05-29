# Phase 30 Implementation Plan: PokerStove Reduced Bootstrap Decision

## Goal

Make the reduced model path the official active learning path by validating the Phase 28 reduced clone with PokerStove-backed equity in larger paired engine evaluations. This phase is a decision gate before adding features, PPO fine-tuning, or replacing any promoted 62-feature checkpoint.

## Current Position

The 62-feature model remains a frozen modern reference and future feature source. Its current learning signal is too weak for the available rollout volume, so the active training path is the 8-feature reduced observation contract trained by supervised equity-bot cloning.

"Cloning" means the model learns the equity-aware bot's action policy from labels. PokerStove supplies the equity feature; it does not supply the full action policy.

## Scope

- Use `pokerstove` as the primary equity source for reduced-model engine evaluation.
- Keep an explicit fallback source only for local development and test robustness.
- Re-run larger paired evaluations against random, equity-aggressive, and tight-equity opponents.
- Report placement, ITM, payout, fallback counts, action mix, and equity-source fallback counts.
- Decide whether to collect more supervised labels, train reduced v2, or keep the current checkpoint only as a baseline.

## Non-Goals

- Do not add new observation features in this phase.
- Do not launch PPO fine-tuning in this phase.
- Do not promote the reduced clone from supervised loss alone.
- Do not remove or mutate the frozen 62-feature model contract.

## Checklist

- [ ] Treat reduced 8-feature model as the active learning path.
- [ ] Keep the 62-feature model frozen as the future feature source.
- [ ] Use PokerStove-backed equity for reduced engine evaluation.
- [ ] Keep `constant` fallback explicit and count every fallback.
- [ ] Run the larger paired Phase 29/30-style evaluation profile.
- [ ] Verify reduced model beats random by average placement.
- [ ] Verify reduced model approaches the equity teacher on placement and payout.
- [ ] Verify zero model inference errors, timeouts, and illegal actions.
- [ ] Verify action mix and top-action-rate reports are present.
- [ ] Decide one next action: collect more labels, train reduced v2, or start one-feature ablation.

## Acceptance Checks

- PokerStove equity is available or every fallback is explicitly reported.
- Completed tournaments meet the configured minimum.
- Reduced model entries meet the configured minimum.
- Reduced model average placement beats random.
- Reduced model average placement stays within the configured equity-bot gap.
- Model inference errors, timeout fallbacks, and illegal actions are zero.
- No new feature is accepted without a separate ablation plan.

## Result

The PokerStove-backed 50-tournament reduced evaluation completed and was accepted:

- report: `runs/phase29_larger_reduced_one_tight_engine_evaluation/20260529_144509_039814/reduced_clone_engine_evaluation_report.json`
- completed tournaments: 50
- reduced model entries: 450
- equity source: `pokerstove`
- model equity computations: 31,345
- model equity fallbacks: 0
- model inference errors, timeouts, illegal actions: 0
- reduced model average position: 10.8244
- equity-aggressive bot average position: 11.8356
- random bot average position: 21.6689
- model minus equity average position: -1.0111
- model minus random average position: -10.8444
- reduced model total payout pct: 24.86
- equity-aggressive total payout pct: 18.94
- random total payout pct: 1.0
- model actions: fold 12,803, call 11,171, raise 7,371

All configured Phase 29/30 gates passed, including PokerStove fallback, random-baseline, equity-baseline, action-mix, runtime, and legal-action gates.

## Next Phase

Phase 31 should collect a larger PokerStove-backed supervised dataset from mostly TightEquityBot labels and train reduced v2 before any feature ablation. PPO should wait until the reduced supervised policy is stable across at least one more paired evaluation.
