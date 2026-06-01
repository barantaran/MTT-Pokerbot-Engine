import random

from engine.player_interface import Bot
from players.tight_equity_bot import TightEquityBot


class NoisyEquityBot(Bot):
    """
    Imperfect equity-aware opponent for evaluation.
    It follows TightEquityBot using a noisy equity perception, with occasional
    legal nearby mistakes.
    """

    def __init__(
        self,
        adjacent_noise_rate=0.15,
        mistake_rate=0.05,
        real_equity_weight=0.60,
        use_preflop_spot_range=False,
    ):
        super().__init__("NoisyEquityBot")
        self.real_equity_weight = float(real_equity_weight)
        self.use_preflop_spot_range = bool(use_preflop_spot_range)
        self.tight = TightEquityBot(
            equity_estimator=self._estimate_perceived_equity,
            use_preflop_spot_range=self.use_preflop_spot_range,
        )
        self.adjacent_noise_rate = float(adjacent_noise_rate)
        self.mistake_rate = float(mistake_rate)

    def _estimate_perceived_equity(
        self,
        *,
        hole_cards,
        board_cards,
        active_players,
        opponent_range_pct=None,
        preflop_spot_type=None,
        use_preflop_spot_range=False,
    ):
        real_equity = self.tight_equity(
            hole_cards=hole_cards,
            board_cards=board_cards,
            active_players=active_players,
            opponent_range_pct=opponent_range_pct,
            preflop_spot_type=preflop_spot_type,
            use_preflop_spot_range=use_preflop_spot_range,
        )
        random_equity = random.uniform(0.15, 0.85)
        real_weight = min(1.0, max(0.0, self.real_equity_weight))
        return real_weight * real_equity + (1.0 - real_weight) * random_equity

    def tight_equity(
        self,
        *,
        hole_cards,
        board_cards,
        active_players,
        opponent_range_pct=None,
        preflop_spot_type=None,
        use_preflop_spot_range=False,
    ):
        from players.tight_equity_bot import estimate_equity

        return estimate_equity(
            hole_cards=hole_cards,
            board_cards=board_cards,
            active_players=active_players,
            opponent_range_pct=opponent_range_pct,
            preflop_spot_type=preflop_spot_type,
            use_preflop_spot_range=use_preflop_spot_range,
        )

    def get_action(self, game_state):
        base_action = self.tight.get_action(game_state)
        roll = random.random()
        if roll < self.mistake_rate:
            return self._bigger_mistake(base_action, game_state)
        if roll < self.mistake_rate + self.adjacent_noise_rate:
            return self._adjacent_mistake(base_action, game_state)
        return base_action

    def _adjacent_mistake(self, action, game_state):
        action_name, _amount = action
        call_amount = int(game_state.get("call_amount", 0) or 0)
        if action_name == "fold":
            return ("call", 0)
        if action_name == "call":
            return self._small_raise(game_state) if self._can_raise(game_state) else ("call", 0)
        if action_name == "raise":
            return ("call", 0)
        return ("call", 0) if call_amount == 0 else ("fold", 0)

    def _bigger_mistake(self, action, game_state):
        action_name, _amount = action
        call_amount = int(game_state.get("call_amount", 0) or 0)
        if action_name == "fold":
            return self._small_raise(game_state) if self._can_raise(game_state) and random.random() < 0.35 else ("call", 0)
        if action_name == "call":
            return self._large_raise(game_state) if self._can_raise(game_state) else ("fold", 0)
        if action_name == "raise":
            return ("fold", 0) if call_amount > 0 else ("call", 0)
        return ("call", 0)

    def _can_raise(self, game_state):
        stack_size = int(game_state.get("stack_size", 0) or 0)
        call_amount = int(game_state.get("call_amount", 0) or 0)
        min_raise = int(game_state.get("min_raise", 0) or 0)
        return stack_size - call_amount >= min_raise

    def _small_raise(self, game_state):
        pot_size = int(game_state.get("pot_size", 0) or 0)
        stack_size = int(game_state.get("stack_size", 0) or 0)
        call_amount = int(game_state.get("call_amount", 0) or 0)
        min_raise = int(game_state.get("min_raise", 0) or 0)
        big_blind = int(game_state.get("blinds", {}).get("big", 1) or 1)
        max_raise_extra = max(0, stack_size - call_amount)
        target = max(min_raise, int(max(big_blind, pot_size * 0.5)))
        return ("raise", min(max_raise_extra, target))

    def _large_raise(self, game_state):
        pot_size = int(game_state.get("pot_size", 0) or 0)
        stack_size = int(game_state.get("stack_size", 0) or 0)
        call_amount = int(game_state.get("call_amount", 0) or 0)
        min_raise = int(game_state.get("min_raise", 0) or 0)
        big_blind = int(game_state.get("blinds", {}).get("big", 1) or 1)
        max_raise_extra = max(0, stack_size - call_amount)
        target = max(min_raise, int(max(big_blind * 3, pot_size * 1.25)))
        return ("raise", min(max_raise_extra, target))
