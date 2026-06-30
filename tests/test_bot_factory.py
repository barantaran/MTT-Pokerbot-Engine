import unittest

from engine.bot_factory import available_bot_tools, build_configurable_bots, population_for_spec


class BotFactoryTests(unittest.TestCase):
    def test_available_bot_tools_lists_constructor_switches(self):
        tools = available_bot_tools()

        self.assertIn("ev_formula", tools)
        self.assertIn("base_single_fold_prob", tools["ev_formula"]["params"])
        self.assertIn("use_preflop_spot_range", tools["range_policy"]["params"])
        self.assertIn("icm_strength", tools["tournament_icm_equity"]["params"])
        self.assertIn("configured_tournament_equity", tools)
        self.assertIn("tools", tools["configured_tournament_equity"]["params"])
        self.assertIn("raise_sizing", tools["configured_tournament_equity"]["params"])
        self.assertIn("pot_size_buckets", tools["configured_tournament_equity"]["params"])
        self.assertIn("tool:icm_pressure", tools)
        self.assertIn("strength", tools["tool:icm_pressure"]["params"])
        self.assertIn("tool:table_adaptation", tools)
        self.assertIn("sample_quality_min", tools["tool:table_adaptation"]["params"])
        self.assertIn("tool:preflop_reraise_tightness", tools)
        self.assertIn("tightness", tools["tool:preflop_reraise_tightness"]["params"])
        self.assertIn("tool:button_steal", tools)
        self.assertIn("max_vpip", tools["tool:button_steal"]["params"])
        self.assertIn("tool:endgame_conversion", tools)
        self.assertIn("final_table_players", tools["tool:endgame_conversion"]["params"])
        self.assertIn("tool:bluff_pressure", tools)
        self.assertIn("min_fold_equity", tools["tool:bluff_pressure"]["params"])

    def test_build_configurable_bots_keeps_legacy_lineup_counts(self):
        bots, name_to_population = build_configurable_bots(
            {"ev_reaction": 1, "tight_equity": 1},
            {"fixed_bots_use_preflop_spot_range": True},
        )

        self.assertEqual([bot.__class__.__name__ for bot in bots], ["EVReactionBot", "TightEquityBot"])
        self.assertTrue(bots[0].use_preflop_spot_range)
        self.assertTrue(bots[1].use_preflop_spot_range)
        self.assertEqual(name_to_population, {"ev_reaction_001": "ev_reaction", "tight_equity_001": "tight_equity"})

    def test_build_configurable_bots_accepts_arbitrary_parameterized_specs(self):
        bots, name_to_population = build_configurable_bots(
            {},
            {"fixed_bots_use_preflop_spot_range": False},
            extra_specs=[
                {
                    "type": "ev_formula",
                    "population": "formula_tight_ranges",
                    "count": 2,
                    "params": {
                        "use_preflop_spot_range": True,
                        "base_single_fold_prob": 0.70,
                        "raise_ev_margin_fraction": 0.20,
                    },
                },
                {
                    "type": "TournamentEquityBotV2",
                    "name": "mtt_v2_baseline",
                    "params": {"use_preflop_spot_range": False, "preflop_reraise_tightness": 0.12},
                },
            ],
        )

        self.assertEqual([bot.__class__.__name__ for bot in bots], ["EVFormulaBot", "EVFormulaBot", "TournamentEquityBotV2"])
        self.assertEqual([bot.name for bot in bots], ["formula_tight_ranges_001", "formula_tight_ranges_002", "mtt_v2_baseline"])
        self.assertTrue(bots[0].use_preflop_spot_range)
        self.assertAlmostEqual(bots[0].base_single_fold_prob, 0.70)
        self.assertAlmostEqual(bots[0].raise_ev_margin_fraction, 0.20)
        self.assertFalse(bots[2].use_preflop_spot_range)
        self.assertAlmostEqual(bots[2].preflop_reraise_tightness, 0.12)
        self.assertEqual(name_to_population["formula_tight_ranges_001"], "formula_tight_ranges")
        self.assertEqual(name_to_population["mtt_v2_baseline"], "tournament_equity_v2")

    def test_build_configurable_bots_accepts_decision_tools(self):
        bots, _name_to_population = build_configurable_bots(
            {},
            {},
            extra_specs=[
                {
                    "type": "tournament_icm_equity",
                    "params": {
                        "tools": [
                            {
                                "type": "button_steal",
                                "sample_quality_min": 0.75,
                                "max_vpip": 0.22,
                            }
                        ]
                    },
                },
            ],
        )

        self.assertEqual(len(bots[0].tools), 1)
        self.assertEqual(bots[0].tools[0].name, "button_steal")
        self.assertAlmostEqual(bots[0].tools[0].sample_quality_min, 0.75)
        self.assertAlmostEqual(bots[0].tools[0].max_vpip, 0.22)

    def test_build_configurable_bots_can_name_generated_population_by_tool_set(self):
        bots, name_to_population = build_configurable_bots(
            {},
            {},
            extra_specs=[
                {
                    "type": "configured_tournament_equity",
                    "count": 2,
                    "params": {
                        "tools": [
                            {"type": "icm_pressure", "priority": 10},
                            {"type": "button_steal", "priority": 20},
                        ]
                    },
                },
                {
                    "type": "configured_tournament_equity",
                    "params": {"tools": []},
                },
            ],
        )

        self.assertEqual(
            [bot.name for bot in bots],
            [
                "conf_mtt_eq_icm_steal_001",
                "conf_mtt_eq_icm_steal_002",
                "conf_mtt_eq_none_001",
            ],
        )
        self.assertEqual(
            name_to_population,
            {
                "conf_mtt_eq_icm_steal_001": "conf_mtt_eq_icm_steal",
                "conf_mtt_eq_icm_steal_002": "conf_mtt_eq_icm_steal",
                "conf_mtt_eq_none_001": "conf_mtt_eq_none",
            },
        )
        self.assertEqual(
            population_for_spec(
                {
                    "type": "configured_tournament_equity",
                    "include_tool_set_in_name": True,
                    "params": {
                        "tools": [
                            {"type": "icm_pressure", "priority": 10},
                            {"type": "button_steal", "priority": 20},
                        ]
                    },
                }
            ),
            "conf_mtt_eq_icm_steal",
        )

    def test_build_configurable_bots_accepts_fully_configured_tournament_bot(self):
        bots, name_to_population = build_configurable_bots(
            {},
            {},
            extra_specs=[
                {
                    "type": "configured_tournament_equity",
                    "name": "assembled_champion",
                    "params": {
                        "tools": [
                            {"type": "icm_pressure", "priority": 10, "strength": 0.8},
                            {"type": "table_adaptation", "priority": 20},
                            {"type": "button_steal", "priority": 30},
                            {"type": "endgame_conversion", "priority": 40},
                            {"type": "bluff_pressure", "priority": 50},
                        ]
                    },
                },
            ],
        )

        self.assertEqual(bots[0].__class__.__name__, "ConfiguredTournamentEquityBot")
        self.assertEqual(bots[0].name, "assembled_champion")
        self.assertEqual(
            [tool.name for tool in bots[0].tools],
            ["icm_pressure", "table_adaptation", "button_steal", "endgame_conversion", "bluff_pressure"],
        )
        self.assertEqual(name_to_population["assembled_champion"], "configured_tournament_equity")

    def test_unknown_params_fail_with_clear_message(self):
        with self.assertRaisesRegex(ValueError, "does not accept params"):
            build_configurable_bots(
                {},
                {},
                extra_specs=[{"type": "random", "params": {"use_preflop_spot_range": True}}],
            )


if __name__ == "__main__":
    unittest.main()
