import unittest

from engine.bot_factory import available_bot_tools, build_configurable_bots


class BotFactoryTests(unittest.TestCase):
    def test_available_bot_tools_lists_constructor_switches(self):
        tools = available_bot_tools()

        self.assertIn("ev_formula", tools)
        self.assertIn("base_single_fold_prob", tools["ev_formula"]["params"])
        self.assertIn("use_preflop_spot_range", tools["range_policy"]["params"])
        self.assertIn("icm_strength", tools["tournament_icm_equity"]["params"])

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
                    "name": "mtt_v2_no_ranges",
                    "params": {"use_preflop_spot_range": False, "preflop_reraise_tightness": 0.12},
                },
            ],
        )

        self.assertEqual([bot.__class__.__name__ for bot in bots], ["EVFormulaBot", "EVFormulaBot", "TournamentEquityBotV2"])
        self.assertEqual([bot.name for bot in bots], ["formula_tight_ranges_001", "formula_tight_ranges_002", "mtt_v2_no_ranges"])
        self.assertTrue(bots[0].use_preflop_spot_range)
        self.assertAlmostEqual(bots[0].base_single_fold_prob, 0.70)
        self.assertAlmostEqual(bots[0].raise_ev_margin_fraction, 0.20)
        self.assertFalse(bots[2].use_preflop_spot_range)
        self.assertAlmostEqual(bots[2].preflop_reraise_tightness, 0.12)
        self.assertEqual(name_to_population["formula_tight_ranges_001"], "formula_tight_ranges")
        self.assertEqual(name_to_population["mtt_v2_no_ranges"], "tournament_equity_v2")

    def test_unknown_params_fail_with_clear_message(self):
        with self.assertRaisesRegex(ValueError, "does not accept params"):
            build_configurable_bots(
                {},
                {},
                extra_specs=[{"type": "random", "params": {"use_preflop_spot_range": True}}],
            )


if __name__ == "__main__":
    unittest.main()
