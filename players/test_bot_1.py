from engine.player_interface import Bot
from engine.pokerstove_equity import estimate_equity, pot_odds

class TestBot1(Bot):
    def __init__(self):
        super().__init__("TestBot_1")

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

        call_buffer = 0.04 + (0.015 * max(active_players - 2, 0))
        raise_threshold = {
            0: 0.58,
            3: 0.64,
            4: 0.68,
            5: 0.72,
        }.get(street, 0.65)

        if stack_bb <= 12:
            raise_threshold -= 0.06

        if call_amount > 0 and equity + call_buffer < required_equity:
            return ("fold", 0)

        max_raise_extra = stack_size - call_amount
        can_raise = max_raise_extra >= min_raise

        if can_raise and equity >= raise_threshold:
            if stack_bb <= 10:
                return ("raise", max_raise_extra)

            if street == 0:
                raise_extra = min(max_raise_extra, max(min_raise, 3 * big_blind))
            else:
                target_pot_bet = max(big_blind, int(pot_size * 0.75))
                raise_extra = min(max_raise_extra, max(min_raise, target_pot_bet))
            return ("raise", raise_extra)

        if call_amount == 0:
            return ("call", 0)

        if equity >= required_equity:
            return ("call", 0)

        return ("fold", 0)
