from engine.player_interface import Bot
from engine.pokerstove_equity import estimate_equity, pot_odds


class TightEquityBot(Bot):
    """
    Equity-aware conservative bot.
    It avoids bluff pressure, calls when the price is justified, and raises only
    with strong equity or short-stack value.
    """

    def __init__(
        self,
        equity_estimator=None,
        use_preflop_spot_range=False,
        *,
        call_margin_shift=0.0,
        raise_threshold_shift=0.0,
        preflop_vpip_gate_shift=None,
    ):
        super().__init__("TightEquityBot")
        self.equity_estimator = equity_estimator
        self.use_preflop_spot_range = bool(use_preflop_spot_range)
        self.call_margin_shift = float(call_margin_shift)
        self.raise_threshold_shift = float(raise_threshold_shift)
        self.preflop_vpip_gate_shift = (
            None if preflop_vpip_gate_shift is None else float(preflop_vpip_gate_shift)
        )

    def get_action(self, game_state):
        hole_cards = game_state.get("hole_cards", [])
        board_cards = game_state.get("board_cards", [])
        pot_size = game_state.get("pot_size", 0)
        stack_size = game_state.get("stack_size", 0)
        call_amount = game_state.get("call_amount", 0)
        min_raise = game_state.get("min_raise", 0)
        active_players = game_state.get("active_players", 2)
        big_blind = game_state.get("blinds", {}).get("big", 1) or 1
        position = str(game_state.get("position", "") or "").upper()

        street = len(board_cards)
        stack_bb = stack_size / float(big_blind)
        estimator = self.equity_estimator or estimate_equity
        if self.equity_estimator is None and "hero_equity" in game_state:
            equity = float(game_state["hero_equity"])
        else:
            equity = estimator(
                hole_cards=hole_cards,
                board_cards=board_cards,
                active_players=active_players,
                opponent_range_pct=game_state.get("opponent_range_pct"),
                preflop_spot_type=game_state.get("preflop_spot_type"),
                use_preflop_spot_range=self.use_preflop_spot_range,
                table_stats=game_state.get("table_stats"),
            )
        required_equity = pot_odds(call_amount, pot_size)

        if street == 0 and self.preflop_vpip_gate_shift is not None:
            vpip_threshold = self._preflop_vpip_threshold(position) + self.preflop_vpip_gate_shift
            if position != "BB" and equity < max(0.0, min(0.95, vpip_threshold)):
                return ("fold", 0)

        max_raise_extra = stack_size - call_amount
        can_raise = max_raise_extra >= min_raise

        call_margin = {
            0: 0.06,
            3: 0.05,
            4: 0.04,
            5: 0.03,
        }.get(street, 0.05)
        call_margin += self.call_margin_shift
        raise_threshold = {
            0: 0.62,
            3: 0.68,
            4: 0.72,
            5: 0.76,
        }.get(street, 0.70)
        raise_threshold += self.raise_threshold_shift

        if active_players >= 5:
            raise_threshold += 0.04
            call_margin += 0.02

        call_margin = max(-0.05, min(0.40, call_margin))
        raise_threshold = max(0.20, min(0.95, raise_threshold))

        if call_amount > 0 and equity < required_equity + call_margin:
            return ("fold", 0)

        if can_raise and equity >= raise_threshold:
            if stack_bb <= 8:
                return ("raise", max_raise_extra)
            if street == 0:
                value_raise = max(min_raise, int(2.5 * big_blind))
            else:
                value_raise = max(min_raise, int(max(big_blind, pot_size * 0.5)))
            return ("raise", min(max_raise_extra, value_raise))

        if call_amount == 0:
            return ("call", 0)

        return ("call", 0)

    def _preflop_vpip_threshold(self, position):
        return {
            "UTG": 0.26,
            "UTG_1": 0.25,
            "UTG_2": 0.24,
            "LJ": 0.23,
            "HJ": 0.22,
            "CO": 0.20,
            "BTN": 0.18,
            "SB": 0.20,
            "BB": 0.00,
        }.get(str(position or "").upper(), 0.22)
