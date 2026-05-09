class PlayerState:
    """
    Represents an active player in the tournament.
    Wraps the bot instance to track state like chips, status, and table position.
    """
    def __init__(self, bot_instance, starting_stack: int):
        self.bot = bot_instance
        self.name = bot_instance.name
        self.stack = starting_stack
        
        # State for the current hand
        self.hole_cards = []
        self.current_bet = 0     # Amount bet in the current round
        self.total_bet = 0       # Total amount bet across the entire hand
        self.is_active = True    # False if folded or busted
        self.is_all_in = False
        self.has_acted = False   # True if they have acts in the current betting round

    def setup_new_hand(self):
        self.hole_cards = []
        self.current_bet = 0
        self.total_bet = 0
        if self.stack > 0:
            self.is_active = True
            self.is_all_in = False
        else:
            self.is_active = False
            self.is_all_in = False
        self.has_acted = False

    def get_action(self, game_state):
        return self.bot.get_action(game_state)

    def bet(self, amount: int) -> int:
        """
        Deducts chips from stack, capping at their available stack.
        Returns the actual amount bet.
        """
        actual_bet = min(amount, self.stack)
        self.stack -= actual_bet
        self.current_bet += actual_bet
        self.total_bet += actual_bet
        
        if self.stack == 0:
            self.is_all_in = True
            
        return actual_bet

    def win(self, amount: int):
        self.stack += amount
        if self.stack > 0:
            # Note: they might have been active in the hand but are now no longer all-in overall
            # (though practically this only matters for the NEXT hand)
            pass

    def __repr__(self):
        return f"<{self.name} (Stack: {self.stack})>"
