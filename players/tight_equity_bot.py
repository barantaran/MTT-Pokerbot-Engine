from engine.player_interface import Bot
from engine.pokerstove_equity import estimate_equity, pot_odds


class TightEquityBot(Bot):
    """
    Equity-aware conservative bot.
    It avoids bluff pressure, calls when the price is justified, and raises only
    with strong equity or short-stack value.
    """

    def __init__(self):
        super().__init__("TightEquityBot")

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
        equity = estimate_equity(
            hole_cards=hole_cards,
            board_cards=board_cards,
            active_players=active_players,
        )
        required_equity = pot_odds(call_amount, pot_size)

        max_raise_extra = stack_size - call_amount
        can_raise = max_raise_extra >= min_raise

        call_margin = {
            0: 0.06,
            3: 0.05,
            4: 0.04,
            5: 0.03,
        }.get(street, 0.05)
        raise_threshold = {
            0: 0.62,
            3: 0.68,
            4: 0.72,
            5: 0.76,
        }.get(street, 0.70)

        if active_players >= 5:
            raise_threshold += 0.04
            call_margin += 0.02

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
