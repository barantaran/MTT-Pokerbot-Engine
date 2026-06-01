import random

from engine.player_interface import Bot
from engine.pokerstove_equity import estimate_equity, pot_odds


class AggressiveBot(Bot):
    """
    A looser, pressure-oriented bot.
    It raises more often, accepts thinner spots, and jams wider when short.
    """

    def __init__(self, use_preflop_spot_range=False):
        super().__init__("AggressiveBot")
        self.use_preflop_spot_range = bool(use_preflop_spot_range)

    def get_action(self, game_state):
        hole_cards = game_state.get("hole_cards", [])
        board_cards = game_state.get("board_cards", [])
        pot_size = game_state.get("pot_size", 0)
        stack_size = game_state.get("stack_size", 0)
        call_amount = game_state.get("call_amount", 0)
        min_raise = game_state.get("min_raise", 0)
        active_players = game_state.get("active_players", 2)
        big_blind = game_state.get("blinds", {}).get("big", 1) or 1

        street = len(board_cards)
        stack_bb = stack_size / float(big_blind)
        if "hero_equity" in game_state:
            equity = float(game_state["hero_equity"])
        else:
            equity = estimate_equity(
                hole_cards=hole_cards,
                board_cards=board_cards,
                active_players=active_players,
                opponent_range_pct=game_state.get("opponent_range_pct"),
                preflop_spot_type=game_state.get("preflop_spot_type"),
                use_preflop_spot_range=self.use_preflop_spot_range,
            )
        required_equity = pot_odds(call_amount, pot_size)

        max_raise_extra = stack_size - call_amount
        can_raise = max_raise_extra >= min_raise

        jam_threshold = {
            0: 0.50,
            3: 0.56,
            4: 0.60,
            5: 0.64,
        }.get(street, 0.56)
        raise_threshold = {
            0: 0.44,
            3: 0.50,
            4: 0.54,
            5: 0.58,
        }.get(street, 0.50)
        defend_threshold = max(0.16, required_equity - 0.05)

        if stack_bb <= 12 and can_raise and equity >= jam_threshold:
            return ("raise", max_raise_extra)

        if call_amount > 0 and equity < defend_threshold:
            return ("fold", 0)

        if can_raise:
            semibluff_window = street > 0 and 0.34 <= equity < raise_threshold
            open_pressure = call_amount == 0 and street == 0 and equity >= 0.34
            rng = random.random()

            if equity >= raise_threshold or (semibluff_window and rng < 0.28) or (open_pressure and rng < 0.45):
                if street == 0:
                    target_raise = max(min_raise, int(3.5 * big_blind))
                else:
                    target_raise = max(min_raise, int(max(big_blind, pot_size * 0.9)))
                return ("raise", min(max_raise_extra, target_raise))

        if call_amount == 0:
            return ("call", 0)

        if equity >= defend_threshold:
            return ("call", 0)

        return ("fold", 0)
