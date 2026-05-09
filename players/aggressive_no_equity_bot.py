import random

from engine.player_interface import Bot


class AggressiveNoEquityBot(Bot):
    """
    Pressure-oriented bot with no hand equity calculation.
    Decisions are driven by stack pressure, pot size, and random aggression.
    """

    def __init__(self):
        super().__init__("AggressiveNoEquityBot")

    def get_action(self, game_state):
        board_cards = game_state.get("board_cards", [])
        pot_size = game_state.get("pot_size", 0)
        stack_size = game_state.get("stack_size", 0)
        call_amount = game_state.get("call_amount", 0)
        min_raise = game_state.get("min_raise", 0)
        big_blind = game_state.get("blinds", {}).get("big", 1) or 1
        active_players = game_state.get("active_players", 2)

        street = len(board_cards)
        stack_bb = stack_size / float(big_blind)
        max_raise_extra = stack_size - call_amount
        can_raise = max_raise_extra >= min_raise

        if stack_bb <= 10 and can_raise and random.random() < 0.45:
            return ("raise", max_raise_extra)

        if call_amount == 0:
            if can_raise:
                open_raise_chance = {
                    0: 0.55,
                    3: 0.42,
                    4: 0.34,
                    5: 0.28,
                }.get(street, 0.35)
                if active_players <= 3:
                    open_raise_chance += 0.08
                if random.random() < open_raise_chance:
                    if street == 0:
                        target_raise = max(min_raise, int(3.5 * big_blind))
                    else:
                        target_raise = max(min_raise, int(max(big_blind, pot_size * 0.8)))
                    return ("raise", min(max_raise_extra, target_raise))
            return ("call", 0)

        if call_amount >= max(big_blind * 6, stack_size * 0.35):
            if can_raise and stack_bb <= 12 and random.random() < 0.22:
                return ("raise", max_raise_extra)
            return ("fold", 0)

        defend_chance = {
            0: 0.62,
            3: 0.52,
            4: 0.46,
            5: 0.40,
        }.get(street, 0.45)

        if active_players <= 3:
            defend_chance += 0.08

        if can_raise and call_amount <= big_blind * 3 and random.random() < 0.20:
            if street == 0:
                target_raise = max(min_raise, int(4 * big_blind))
            else:
                target_raise = max(min_raise, int(max(big_blind, pot_size)))
            return ("raise", min(max_raise_extra, target_raise))

        if random.random() < defend_chance:
            return ("call", 0)

        return ("fold", 0)
