import random

from engine.player_interface import Bot


class ThreeBetManiacBot(Bot):
    """
    Relentless preflop re-raiser built to attack fold-heavy strategies.
    Card-blind: pressure comes from position-independent aggression, not hand
    strength. Re-raises most raises, keeps barreling as the aggressor, and
    rarely surrenders to a single bet.
    """

    def __init__(
        self,
        *,
        three_bet_chance=0.70,
        four_bet_chance=0.45,
        open_chance=0.35,
        barrel_chance=0.65,
    ):
        super().__init__("ThreeBetManiacBot")
        self.three_bet_chance = float(three_bet_chance)
        self.four_bet_chance = float(four_bet_chance)
        self.open_chance = float(open_chance)
        self.barrel_chance = float(barrel_chance)
        self._preflop_aggressor = False

    def get_action(self, game_state):
        board_cards = game_state.get("board_cards", [])
        pot_size = game_state.get("pot_size", 0)
        stack_size = game_state.get("stack_size", 0)
        call_amount = game_state.get("call_amount", 0)
        min_raise = game_state.get("min_raise", 0)
        big_blind = game_state.get("blinds", {}).get("big", 1) or 1

        street = len(board_cards)
        stack_bb = stack_size / float(big_blind)
        max_raise_extra = stack_size - call_amount
        can_raise = max_raise_extra >= min_raise

        if street == 0:
            self._preflop_aggressor = False
            return self._preflop(
                call_amount, big_blind, min_raise, max_raise_extra, can_raise, stack_bb
            )
        return self._postflop(
            call_amount, pot_size, big_blind, min_raise, max_raise_extra, can_raise
        )

    def _preflop(self, call_amount, big_blind, min_raise, max_raise_extra, can_raise, stack_bb):
        if call_amount == 0:
            if can_raise and random.random() < self.open_chance:
                self._preflop_aggressor = True
                target = max(min_raise, int(3 * big_blind))
                return ("raise", min(max_raise_extra, target))
            return ("call", 0)

        facing_three_bet_or_more = call_amount > big_blind * 6

        if facing_three_bet_or_more:
            if can_raise and random.random() < self.four_bet_chance:
                self._preflop_aggressor = True
                if stack_bb <= 30:
                    return ("raise", max_raise_extra)
                target = max(min_raise, int(call_amount * 2.5))
                return ("raise", min(max_raise_extra, target))
            return ("call", 0)

        if can_raise and random.random() < self.three_bet_chance:
            self._preflop_aggressor = True
            target = max(min_raise, int(call_amount * 3))
            return ("raise", min(max_raise_extra, target))
        return ("call", 0)

    def _postflop(self, call_amount, pot_size, big_blind, min_raise, max_raise_extra, can_raise):
        if call_amount == 0:
            if can_raise and random.random() < self.barrel_chance:
                target = max(min_raise, int(max(big_blind, pot_size * 0.75)))
                return ("raise", min(max_raise_extra, target))
            return ("call", 0)

        # Facing a bet: raise-bluff some, call small bets often, fold air to
        # huge pressure occasionally so it is not a pure station postflop.
        if can_raise and random.random() < 0.25:
            target = max(min_raise, int(max(big_blind, pot_size)))
            return ("raise", min(max_raise_extra, target))
        if call_amount <= pot_size * 0.5 or random.random() < 0.45:
            return ("call", 0)
        return ("fold", 0)
