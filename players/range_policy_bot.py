from __future__ import annotations

from engine.player_interface import Bot
from engine.pokerstove_equity import estimate_equity, pot_odds


class RangePolicyBot(Bot):
    """
    Configurable range-aware policy bot used for policy search.

    It is intentionally not a teacher clone target. The useful role is to test
    families of public-state range policies as opponents or future data sources.
    """

    def __init__(
        self,
        *,
        variant: str = "balanced",
        use_preflop_spot_range: bool = True,
        call_margin: float = 0.02,
        raise_threshold: float = 0.56,
        jam_threshold: float = 0.64,
        pressure_threshold: float = 0.46,
        raise_pot_fraction: float = 0.65,
        open_bb: float = 2.5,
        multiway_tightness: float = 0.02,
        short_stack_bb: float = 12.0,
    ):
        super().__init__("RangePolicyBot")
        self.variant = str(variant or "balanced")
        self.use_preflop_spot_range = bool(use_preflop_spot_range)
        self.call_margin = float(call_margin)
        self.raise_threshold = float(raise_threshold)
        self.jam_threshold = float(jam_threshold)
        self.pressure_threshold = float(pressure_threshold)
        self.raise_pot_fraction = float(raise_pot_fraction)
        self.open_bb = float(open_bb)
        self.multiway_tightness = float(multiway_tightness)
        self.short_stack_bb = float(short_stack_bb)

    def get_action(self, game_state):
        hole_cards = game_state.get("hole_cards", [])
        board_cards = game_state.get("board_cards", [])
        pot_size = int(game_state.get("pot_size", 0) or 0)
        stack_size = int(game_state.get("stack_size", 0) or 0)
        call_amount = int(game_state.get("call_amount", 0) or 0)
        min_raise = int(game_state.get("min_raise", 0) or 0)
        active_players = int(game_state.get("active_players", 2) or 2)
        big_blind = int(game_state.get("blinds", {}).get("big", 1) or 1)

        street = len(board_cards)
        stack_bb = stack_size / float(big_blind)
        max_raise_extra = max(0, stack_size - call_amount)
        can_raise = max_raise_extra >= max(1, min_raise)

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
                table_stats=game_state.get("table_stats"),
            )

        required_equity = pot_odds(call_amount, pot_size)
        multiway_penalty = max(0, active_players - 2) * self.multiway_tightness
        call_threshold = required_equity + self.call_margin + multiway_penalty
        raise_threshold = self.raise_threshold + multiway_penalty
        pressure_threshold = self.pressure_threshold + multiway_penalty
        jam_threshold = self.jam_threshold + multiway_penalty

        if call_amount > 0 and equity < call_threshold:
            return ("fold", 0)

        if can_raise and stack_bb <= self.short_stack_bb and equity >= jam_threshold:
            return ("raise", max_raise_extra)

        if can_raise and equity >= raise_threshold:
            if street == 0:
                target_raise = max(min_raise, int(self.open_bb * big_blind))
            else:
                target_raise = max(min_raise, int(max(big_blind, pot_size * self.raise_pot_fraction)))
            return ("raise", min(max_raise_extra, target_raise))

        if can_raise and call_amount == 0 and equity >= pressure_threshold:
            target_raise = max(min_raise, int(self.open_bb * big_blind))
            return ("raise", min(max_raise_extra, target_raise))

        return ("call", 0) if call_amount == 0 or equity >= call_threshold else ("fold", 0)
