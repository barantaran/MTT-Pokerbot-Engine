from engine.player_interface import Bot

class CallBot(Bot):
    """
    A bot that acts as a "calling station". It will always call or check.
    """
    def __init__(self):
        super().__init__("CallBot")

    def get_action(self, game_state):
        # We only need to specify 'call' with 0 amount. 
        # The engine will automatically deduce the amount to withdraw from the stack.
        # If call_amount is 0, this acts as a 'check'.
        return ('call', 0)
