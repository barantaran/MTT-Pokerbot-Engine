from typing import List
from engine.player_state import PlayerState

class Pot:
    """
    Tracks a specific pot (main or side) and the players eligible for it.
    """
    def __init__(self, amount=0):
        self.amount = amount
        self.eligible_players: List[PlayerState] = []

    def add(self, amount: int):
        self.amount += amount

    def add_eligible(self, player: PlayerState):
        if player not in self.eligible_players:
            self.eligible_players.append(player)
            
class PotManager:
    """
    Handles calculating side pots at the end of a round.
    """
    def __init__(self):
        self.pots: List[Pot] = [Pot()] # Main pot is always at index 0
        
    def get_total_amount(self):
        return sum(p.amount for p in self.pots)

    def collect_bets(self, players: List[PlayerState]):
        """
        Collects all bets (total_bet) from the given players for the entire hand so far
        and splits them into correct side pots based on all-in amounts.
        Note: This is best called at the end of the hand, tracking the total investement 
        of each player over all rounds. 
        """
        
        # We recalculate pots from scratch based on the total_bet of each player.
        # So we reset them first.
        self.pots = []
        
        # Players with non-zero bets
        bettors = [p for p in players if p.total_bet > 0]
        if not bettors:
            # Just create an empty main pot
            self.pots.append(Pot())
            return
            
        # Get unique bet amounts, sorted ascending
        unique_bets = sorted(list(set(p.total_bet for p in bettors)))
        
        previous_bet_threshold = 0
        
        for bet_threshold in unique_bets:
            current_pot_chunk = Pot()
            chunk_size = bet_threshold - previous_bet_threshold
            
            for p in bettors:
                if p.total_bet >= bet_threshold:
                    # They contributed at least this chunk size
                    current_pot_chunk.add(chunk_size)
                    # They are eligible if they are still active or all-in
                    if p.is_active or p.is_all_in:
                        current_pot_chunk.add_eligible(p)
                elif p.total_bet > previous_bet_threshold:
                    # They contributed partially to this chunk (but this shouldn't happen 
                    # due to unique_bets iterating exact thresholds)
                    partial_chunk = p.total_bet - previous_bet_threshold
                    current_pot_chunk.add(partial_chunk)
                    
            if current_pot_chunk.amount > 0:
                self.pots.append(current_pot_chunk)
                
            previous_bet_threshold = bet_threshold

        # If everyone folded and pots were created, the single remaining active player
        # might be eligible for all pots they contributed to.
        # But if no one is eligible for a side pot because it consisted of folded players' dead money,
        # we roll it into the previous pot or give it to the remaining active player.
        self._ensure_eligibility()
        
    def _ensure_eligibility(self):
        """
        Dead money (pots with no eligible players) goes to the most recent pot
        that has eligible players.
        """
        for i in range(len(self.pots) - 1, -1, -1):
            if not self.pots[i].eligible_players:
                if i > 0:
                    self.pots[i-1].add(self.pots[i].amount)
                self.pots[i].amount = 0

        # Remove empty pots
        self.pots = [p for p in self.pots if p.amount > 0 or p == self.pots[0]]
        if not self.pots:
             self.pots = [Pot()]

