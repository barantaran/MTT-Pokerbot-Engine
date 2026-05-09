import random
from engine.player_interface import Bot

class RandomBot(Bot):
    """
    A bot that makes random valid decisions:
    - Folds 20% of the time, calls 60% of the time, raises 20% of the time.
    """
    def __init__(self):
        super().__init__("RandomBot")

    def get_action(self, game_state):
        action_map = random.choices(
            population=['fold', 'call', 'raise'],
            weights=[0.2, 0.6, 0.2],
            k=1
        )[0]
        
        call_amount = game_state.get('call_amount', 0)
        stack_size = game_state.get('stack_size', 0)
        min_raise = game_state.get('min_raise', 0)

        if action_map == 'fold':
            if call_amount == 0:
                return ('call', 0)  # No reason to fold if we can check
            return ('fold', 0)

        elif action_map == 'call':
            return ('call', 0)

        elif action_map == 'raise':
            # We can only raise if we have enough stack beyond the call amount
            effective_stack = stack_size - call_amount
            if effective_stack < min_raise:
                if call_amount > 0:
                     return ('call', 0)
                else:
                    # Going all in as a raise
                    return ('raise', effective_stack)
            else:
                raise_amount = random.randint(min_raise, effective_stack)
                return ('raise', raise_amount)
        
        return ('fold', 0)
