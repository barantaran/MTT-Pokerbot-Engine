from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

class Bot(ABC):
    """
    Base class for all Pokerbots.
    Runner scripts must inherit from this and implement `get_action()`.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_action(self, game_state: Dict[str, Any]) -> Tuple[str, Optional[int]]:
        """
        Determines the action the bot will take based on the current game state.

        Args:
            game_state: A dictionary containing all visible information:
                - "hole_cards": list of ints (treys integer representation)
                - "board_cards": list of ints (treys integer representation)
                - "pot_size": int, total chips in the main pot
                - "stack_size": int, bot's current remaining stack
                - "call_amount": int, the amount to call the current bet (0 if checking is allowed)
                - "min_raise": int, the minimum an additional raise must be
                - "blinds": {"small": int, "big": int}
                - "active_players": int
                - "position": string (SB, BB, BTN, etc) or integer index

        Returns:
            A tuple of (action_type, amount):
                - ("fold", 0) or ("fold", None)
                - ("call", 0) -> calls the `call_amount`
                - ("raise", amount) -> raises by the specific `amount` (total bet = call_amount + amount)
        """
        pass
