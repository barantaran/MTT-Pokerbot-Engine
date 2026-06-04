from __future__ import annotations

import tempfile
import unittest
import json
import random
from pathlib import Path

from engine.evolutionary_reduced_mtt import (
    Candidate,
    basemodel_root_from_engine,
    create_replaced_population,
    create_random_checkpoint,
    default_engine_config,
    load_population_manifest,
    load_population_next_candidate_index,
    merge_candidate_action_summaries,
    mutate_checkpoint,
    observation_contract,
    ranked_candidates,
    run_initial_generation,
    summarize_candidate_actions,
    summarize_candidate_results,
    tournament_winner_candidate_ids,
)
from engine.ev_initiative_evolution import (
    PARAM_LIMITS,
    mutate_params,
    rank_candidate_summary,
    random_params,
)


class EvolutionaryReducedMttTests(unittest.TestCase):
    def test_candidate_summary_scores_deep_runs(self):
        results = [
            {"name": "candidate_001_entry_001", "position": 1, "payout_pct": 0.5},
            {"name": "candidate_001_entry_002", "position": 9, "payout_pct": 0.0},
            {"name": "candidate_002_entry_001", "position": 2, "payout_pct": 0.3},
            {"name": "candidate_002_entry_002", "position": 15, "payout_pct": 0.0},
        ]
        summary = summarize_candidate_results(
            results,
            {
                "candidate_001_entry_001": "candidate_001",
                "candidate_001_entry_002": "candidate_001",
                "candidate_002_entry_001": "candidate_002",
                "candidate_002_entry_002": "candidate_002",
            },
        )

        self.assertEqual(summary["candidate_001"]["entries"], 2)
        self.assertEqual(summary["candidate_001"]["wins"], 1)
        self.assertEqual(summary["candidate_001"]["final_table"], 2)
        self.assertEqual(summary["candidate_001"]["score"], 13.0)
        self.assertEqual(summary["candidate_002"]["top3"], 1)

    def test_ranked_candidates_prefers_score_then_payout(self):
        ranking = ranked_candidates(
            {
                "candidate_001": {"score": 4.0, "total_payout_pct": 0.2, "average_position": 10.0},
                "candidate_002": {"score": 4.0, "total_payout_pct": 0.4, "average_position": 12.0},
                "candidate_003": {"score": 3.0, "total_payout_pct": 0.8, "average_position": 2.0},
            }
        )

        self.assertEqual([row["candidate_id"] for row in ranking], ["candidate_002", "candidate_001", "candidate_003"])

    def test_ev_initiative_ranking_prefers_money_over_wins(self):
        ranking = rank_candidate_summary(
            {
                "evg00_001": {"total_payout_pct": 1.0, "wins": 0, "top3": 0, "final_table": 0},
                "evg00_002": {"total_payout_pct": 0.5, "wins": 3, "top3": 3, "final_table": 3},
            }
        )

        self.assertEqual([row["candidate_id"] for row in ranking], ["evg00_001", "evg00_002"])

    def test_ev_initiative_mutation_keeps_params_in_bounds(self):
        rng = random.Random(123)
        params = random_params(rng)
        mutated = mutate_params(params, rng, sigma=10.0)

        for name, value in mutated.items():
            lo, hi = PARAM_LIMITS[name]
            self.assertGreaterEqual(value, lo)
            self.assertLessEqual(value, hi)

    def test_candidate_action_summary_counts_and_rates(self):
        events = [
            {"type": "action", "player": "candidate_001_entry_001", "action": "raise", "amount": 120},
            {"type": "action", "player": "candidate_001_entry_001", "action": "call", "amount": 40},
            {"type": "action", "player": "candidate_001_entry_001", "action": "call", "amount": 0},
            {"type": "action", "player": "candidate_001_entry_001", "action": "fold", "amount": 0},
            {"type": "action", "player": "candidate_002_entry_001", "action": "check", "amount": 0},
            {"type": "hand_end", "player": "candidate_001_entry_001", "action": "ignored"},
        ]

        summary = summarize_candidate_actions(
            events,
            {
                "candidate_001_entry_001": "candidate_001",
                "candidate_002_entry_001": "candidate_002",
            },
        )

        self.assertEqual(summary["candidate_001"]["action_total"], 4)
        self.assertEqual(summary["candidate_001"]["action_counts"], {"raise": 1, "call": 1, "check": 1, "fold": 1})
        self.assertAlmostEqual(summary["candidate_001"]["action_rates"]["raise"], 1 / 4)
        self.assertEqual(summary["candidate_001"]["average_raise_amount"], 120.0)
        self.assertEqual(summary["candidate_001"]["vpip_hands"], 0)
        self.assertEqual(summary["candidate_001"]["vpip"], 0.0)

    def test_candidate_action_summary_counts_vpip_once_per_preflop_hand(self):
        events = [
            {"type": "deal", "table_id": 1, "hand_id": 1, "player": "candidate_001_entry_001"},
            {
                "type": "action",
                "table_id": 1,
                "hand_id": 1,
                "player": "candidate_001_entry_001",
                "action": "raise",
                "amount": 120,
                "street": "preflop",
            },
            {
                "type": "action",
                "table_id": 1,
                "hand_id": 1,
                "player": "candidate_001_entry_001",
                "action": "call",
                "amount": 300,
                "street": "preflop",
            },
            {"type": "deal", "table_id": 1, "hand_id": 2, "player": "candidate_001_entry_001"},
            {
                "type": "action",
                "table_id": 1,
                "hand_id": 2,
                "player": "candidate_001_entry_001",
                "action": "check",
                "amount": 0,
                "street": "preflop",
            },
            {"type": "deal", "table_id": 1, "hand_id": 3, "player": "candidate_001_entry_001"},
            {
                "type": "action",
                "table_id": 1,
                "hand_id": 3,
                "player": "candidate_001_entry_001",
                "action": "call",
                "amount": 200,
                "street": "flop",
            },
        ]

        summary = summarize_candidate_actions(events, {"candidate_001_entry_001": "candidate_001"})

        self.assertEqual(summary["candidate_001"]["vpip_hands"], 3)
        self.assertEqual(summary["candidate_001"]["vpip_count"], 1)
        self.assertAlmostEqual(summary["candidate_001"]["vpip"], 1 / 3)
        self.assertEqual(summary["candidate_001"]["pfr_count"], 1)
        self.assertAlmostEqual(summary["candidate_001"]["pfr"], 1 / 3)
        self.assertEqual(summary["candidate_001"]["preflop_call_count"], 1)
        self.assertEqual(summary["candidate_001"]["postflop_call_count"], 1)
        self.assertEqual(summary["candidate_001"]["postflop_raise_count"], 0)
        self.assertEqual(summary["candidate_001"]["street_action_counts"]["preflop"], {"raise": 1, "call": 1, "check": 1})
        self.assertEqual(summary["candidate_001"]["street_action_counts"]["flop"], {"call": 1})

    def test_candidate_action_summary_reports_hud_context_stats(self):
        events = [
            {"type": "deal", "table_id": 1, "hand_id": 1, "tournament_id": 1, "player": "opener"},
            {"type": "deal", "table_id": 1, "hand_id": 1, "tournament_id": 1, "player": "threebettor"},
            {"type": "action", "table_id": 1, "hand_id": 1, "tournament_id": 1, "player": "opener", "action": "raise", "amount": 100, "street": "preflop", "call_amount": 20},
            {"type": "action", "table_id": 1, "hand_id": 1, "tournament_id": 1, "player": "threebettor", "action": "raise", "amount": 300, "street": "preflop", "call_amount": 100},
            {"type": "action", "table_id": 1, "hand_id": 1, "tournament_id": 1, "player": "opener", "action": "fold", "amount": 0, "street": "preflop", "call_amount": 200},
            {"type": "deal", "table_id": 1, "hand_id": 2, "tournament_id": 1, "player": "opener"},
            {"type": "deal", "table_id": 1, "hand_id": 2, "tournament_id": 1, "player": "caller"},
            {"type": "action", "table_id": 1, "hand_id": 2, "tournament_id": 1, "player": "opener", "action": "raise", "amount": 100, "street": "preflop", "call_amount": 20},
            {"type": "action", "table_id": 1, "hand_id": 2, "tournament_id": 1, "player": "caller", "action": "call", "amount": 100, "street": "preflop", "call_amount": 100},
            {"type": "action", "table_id": 1, "hand_id": 2, "tournament_id": 1, "player": "caller", "action": "check", "amount": 0, "street": "flop", "call_amount": 0},
            {"type": "action", "table_id": 1, "hand_id": 2, "tournament_id": 1, "player": "opener", "action": "raise", "amount": 120, "street": "flop", "call_amount": 0},
            {"type": "action", "table_id": 1, "hand_id": 2, "tournament_id": 1, "player": "caller", "action": "call", "amount": 120, "street": "flop", "call_amount": 120},
            {"type": "showdown", "table_id": 1, "hand_id": 2, "tournament_id": 1, "player": "opener"},
            {"type": "showdown", "table_id": 1, "hand_id": 2, "tournament_id": 1, "player": "caller"},
            {"type": "award_pot", "table_id": 1, "hand_id": 2, "tournament_id": 1, "player": "opener", "amount": 500, "showdown": True},
        ]

        summary = summarize_candidate_actions(
            events,
            {
                "opener": "opener",
                "threebettor": "threebettor",
                "caller": "caller",
            },
        )

        self.assertEqual(summary["threebettor"]["three_bet_opportunity_count"], 1)
        self.assertEqual(summary["threebettor"]["three_bet_count"], 1)
        self.assertEqual(summary["threebettor"]["three_bet_rate"], 1.0)
        self.assertEqual(summary["opener"]["faced_three_bet_count"], 1)
        self.assertEqual(summary["opener"]["folded_to_three_bet_count"], 1)
        self.assertEqual(summary["opener"]["fold_to_three_bet_rate"], 1.0)
        self.assertEqual(summary["opener"]["cbet_opportunity_count"], 1)
        self.assertEqual(summary["opener"]["cbet_count"], 1)
        self.assertEqual(summary["opener"]["cbet_rate"], 1.0)
        self.assertEqual(summary["opener"]["showdown_count"], 1)
        self.assertEqual(summary["opener"]["won_showdown_count"], 1)
        self.assertEqual(summary["opener"]["wsd"], 1.0)
        self.assertEqual(summary["caller"]["saw_flop_count"], 1)
        self.assertEqual(summary["caller"]["showdown_count"], 1)
        self.assertEqual(summary["caller"]["wtsd"], 1.0)
        self.assertEqual(summary["caller"]["wsd"], 0.0)

    def test_merge_candidate_action_summaries_recomputes_rates(self):
        merged = merge_candidate_action_summaries(
            [
                {
                    "candidate_001": {
                        "action_total": 2,
                        "action_counts": {"raise": 1, "call": 1},
                        "amount_total": 150,
                        "raise_amount_total": 100,
                        "raise_count": 1,
                        "vpip_hands": 2,
                        "vpip_count": 1,
                        "pfr_count": 1,
                        "preflop_call_count": 0,
                        "postflop_raise_count": 1,
                        "postflop_call_count": 1,
                        "street_action_counts": {"preflop": {"raise": 1}, "flop": {"call": 1}},
                    }
                },
                {
                    "candidate_001": {
                        "action_total": 1,
                        "action_counts": {"raise": 1},
                        "amount_total": 200,
                        "raise_amount_total": 200,
                        "raise_count": 1,
                        "vpip_hands": 1,
                        "vpip_count": 1,
                        "pfr_count": 0,
                        "preflop_call_count": 1,
                        "postflop_raise_count": 1,
                        "postflop_call_count": 0,
                        "street_action_counts": {"preflop": {"call": 1}, "turn": {"raise": 1}},
                    }
                },
            ]
        )

        self.assertEqual(merged["candidate_001"]["action_total"], 3)
        self.assertEqual(merged["candidate_001"]["action_counts"], {"raise": 2, "call": 1})
        self.assertAlmostEqual(merged["candidate_001"]["action_rates"]["raise"], 2 / 3)
        self.assertEqual(merged["candidate_001"]["average_raise_amount"], 150.0)
        self.assertEqual(merged["candidate_001"]["vpip_hands"], 3)
        self.assertEqual(merged["candidate_001"]["vpip_count"], 2)
        self.assertAlmostEqual(merged["candidate_001"]["vpip"], 2 / 3)
        self.assertEqual(merged["candidate_001"]["pfr_count"], 1)
        self.assertAlmostEqual(merged["candidate_001"]["pfr"], 1 / 3)
        self.assertEqual(merged["candidate_001"]["preflop_call_count"], 1)
        self.assertAlmostEqual(merged["candidate_001"]["preflop_call_rate"], 1 / 3)
        self.assertEqual(merged["candidate_001"]["postflop_raise_count"], 2)
        self.assertEqual(merged["candidate_001"]["postflop_call_count"], 1)
        self.assertEqual(merged["candidate_001"]["postflop_aggression_factor"], 2.0)
        self.assertEqual(merged["candidate_001"]["street_action_counts"]["preflop"], {"raise": 1, "call": 1})

    def test_mutate_checkpoint_changes_policy_head_only(self):
        engine_root = Path(__file__).resolve().parents[1]
        basemodel_root = basemodel_root_from_engine(engine_root)
        observation_size, _fields = observation_contract(basemodel_root, "reduced_v6")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = create_random_checkpoint(
                root / "source.pt",
                basemodel_root=basemodel_root,
                observation_size=observation_size,
                seed=1,
            )
            target = mutate_checkpoint(
                source,
                root / "target.pt",
                basemodel_root=basemodel_root,
                observation_size=observation_size,
                seed=2,
                sigma=0.10,
                policy_head_only=True,
            )

            import torch

            source_state = torch.load(source, map_location="cpu")["model_state"]
            target_state = torch.load(target, map_location="cpu")["model_state"]

        self.assertFalse(torch.equal(source_state["policy_head.weight"], target_state["policy_head.weight"]))
        self.assertTrue(torch.equal(source_state["trunk.0.weight"], target_state["trunk.0.weight"]))

    def test_tiny_initial_generation_smoke(self):
        engine_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            engine = default_engine_config()
            engine.update(
                {
                    "equity_source": "constant",
                    "equity_fallback_source": "constant",
                    "starting_stack": 200,
                    "hands_per_level": 2,
                    "max_hands_per_tournament": 20,
                    "blinds_schedule": [{"small": 5, "big": 10}, {"small": 10, "big": 20}],
                    "payouts": {"1": 1.0},
                }
            )
            report = run_initial_generation(
                {
                    "artifact_root": str(Path(tmp) / "run"),
                    "candidate_count": 2,
                    "entries_per_candidate": 1,
                    "mtt_count": 1,
                    "random_seed": 123,
                    "observation_schema": "reduced_v6",
                    "engine": engine,
                },
                engine_root=engine_root,
            )

        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(report["mtt_count"], 1)
        self.assertEqual(len(report["ranking"]), 2)
        self.assertFalse(report["simulation"]["failures"])

    def test_replaced_population_mutates_worst_from_best(self):
        engine_root = Path(__file__).resolve().parents[1]
        basemodel_root = basemodel_root_from_engine(engine_root)
        observation_size, _fields = observation_contract(basemodel_root, "reduced_v6")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = []
            for index in range(3):
                checkpoint = create_random_checkpoint(
                    root / f"candidate_{index + 1:03d}" / "latest.pt",
                    basemodel_root=basemodel_root,
                    observation_size=observation_size,
                    seed=index + 10,
                )
                candidates.append(Candidate(f"candidate_{index + 1:03d}", checkpoint))

            next_population = create_replaced_population(
                artifact_root=root,
                candidates=candidates,
                ranking=[
                    {"candidate_id": "candidate_001", "score": 9},
                    {"candidate_id": "candidate_002", "score": 5},
                    {"candidate_id": "candidate_003", "score": 1},
                ],
                basemodel_root=basemodel_root,
                observation_size=observation_size,
                replace_count=1,
                seed=99,
                sigma=0.02,
                policy_head_only=True,
            )

        self.assertEqual(next_population["best_candidate_ids"], ["candidate_001"])
        self.assertEqual(next_population["worst_candidate_ids"], ["candidate_003"])
        self.assertEqual(next_population["replacements"][0]["parent_candidate_id"], "candidate_001")
        self.assertEqual(next_population["replacements"][0]["new_candidate_id"], "candidate_004")
        self.assertNotIn("candidate_003", next_population["next_population_checkpoints"])
        self.assertIn("candidate_004", next_population["next_population_checkpoints"])
        self.assertEqual(next_population["lineage"]["candidate_004"]["parent_id"], "candidate_001")
        self.assertEqual(next_population["lineage"]["candidate_004"]["root_id"], "candidate_001")
        self.assertEqual(next_population["lineage"]["candidate_004"]["generation"], 1)
        self.assertEqual(next_population["next_candidate_index"], 5)

    def test_replaced_population_can_use_duplicate_winner_parents(self):
        engine_root = Path(__file__).resolve().parents[1]
        basemodel_root = basemodel_root_from_engine(engine_root)
        observation_size, _fields = observation_contract(basemodel_root, "reduced_v6")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = []
            for index in range(4):
                checkpoint = create_random_checkpoint(
                    root / f"candidate_{index + 1:03d}" / "latest.pt",
                    basemodel_root=basemodel_root,
                    observation_size=observation_size,
                    seed=index + 20,
                )
                candidates.append(Candidate(f"candidate_{index + 1:03d}", checkpoint))

            next_population = create_replaced_population(
                artifact_root=root,
                candidates=candidates,
                ranking=[
                    {"candidate_id": "candidate_001", "score": 9},
                    {"candidate_id": "candidate_002", "score": 5},
                    {"candidate_id": "candidate_003", "score": 2},
                    {"candidate_id": "candidate_004", "score": 1},
                ],
                basemodel_root=basemodel_root,
                observation_size=observation_size,
                replace_count=2,
                seed=99,
                sigma=0.02,
                policy_head_only=True,
                parent_candidate_ids=["candidate_001", "candidate_001"],
            )

        self.assertEqual(next_population["best_candidate_ids"], ["candidate_001", "candidate_001"])
        self.assertEqual([row["parent_candidate_id"] for row in next_population["replacements"]], ["candidate_001", "candidate_001"])
        self.assertEqual([row["new_candidate_id"] for row in next_population["replacements"]], ["candidate_005", "candidate_006"])

    def test_tournament_winner_candidate_ids_preserves_duplicate_wins(self):
        winners = tournament_winner_candidate_ids(
            {
                "tournaments": [
                    {
                        "candidate_summary": {
                            "candidate_001": {"average_position": 1.0},
                            "candidate_002": {"average_position": 2.0},
                        }
                    },
                    {
                        "candidate_summary": {
                            "candidate_001": {"average_position": 1.0},
                            "candidate_003": {"average_position": 2.0},
                        }
                    },
                ]
            }
        )

        self.assertEqual(winners, ["candidate_001", "candidate_001"])

    def test_load_population_manifest_uses_next_population_checkpoints(self):
        engine_root = Path(__file__).resolve().parents[1]
        basemodel_root = basemodel_root_from_engine(engine_root)
        observation_size, _fields = observation_contract(basemodel_root, "reduced_v6")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = create_random_checkpoint(
                root / "candidate_001" / "latest.pt",
                basemodel_root=basemodel_root,
                observation_size=observation_size,
                seed=42,
            )
            manifest_path = root / "next_population.json"
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump({"next_population_checkpoints": {"candidate_001": str(checkpoint)}}, handle)

            candidates = load_population_manifest(manifest_path, engine_root=engine_root)

        self.assertEqual(candidates, [Candidate("candidate_001", checkpoint, generation=0, root_id="candidate_001")])

    def test_load_population_manifest_preserves_lineage(self):
        engine_root = Path(__file__).resolve().parents[1]
        basemodel_root = basemodel_root_from_engine(engine_root)
        observation_size, _fields = observation_contract(basemodel_root, "reduced_v6")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = create_random_checkpoint(
                root / "candidate_093" / "latest.pt",
                basemodel_root=basemodel_root,
                observation_size=observation_size,
                seed=44,
            )
            manifest_path = root / "next_population.json"
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "next_population_checkpoints": {"candidate_093": str(checkpoint)},
                        "lineage": {
                            "candidate_093": {
                                "generation": 1,
                                "parent_id": "candidate_107",
                                "root_id": "candidate_107",
                            }
                        },
                    },
                    handle,
                )

            candidates = load_population_manifest(manifest_path, engine_root=engine_root)

        self.assertEqual(
            candidates,
            [Candidate("candidate_093", checkpoint, generation=1, parent_id="candidate_107", root_id="candidate_107")],
        )

    def test_load_population_next_candidate_index_preserves_manifest_counter(self):
        engine_root = Path(__file__).resolve().parents[1]
        basemodel_root = basemodel_root_from_engine(engine_root)
        observation_size, _fields = observation_contract(basemodel_root, "reduced_v6")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = create_random_checkpoint(
                root / "candidate_004" / "latest.pt",
                basemodel_root=basemodel_root,
                observation_size=observation_size,
                seed=45,
            )
            manifest_path = root / "next_population.json"
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "next_candidate_index": 6,
                        "next_population_checkpoints": {"candidate_004": str(checkpoint)},
                    },
                    handle,
                )

            candidates = load_population_manifest(manifest_path, engine_root=engine_root)
            next_id = load_population_next_candidate_index(
                manifest_path,
                engine_root=engine_root,
                candidates=candidates,
            )

        self.assertEqual(next_id, 6)


if __name__ == "__main__":
    unittest.main()
