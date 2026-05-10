import random
from typing import List, Dict, Any, Tuple
from treys import Deck, Evaluator, Card
from engine.player_state import PlayerState
from engine.pot import PotManager
from engine.config import config
import concurrent.futures

class Table:
    """
    Manages a single poker table.
    Handles dealing, betting rounds, and winner evaluation.
    """
    def __init__(self, table_id: int, tournament_id: int = 0):
        self.table_id = table_id
        self.tournament_id = tournament_id
        self.hand_id = 0
        self.players: List[PlayerState] = []
        self.button_idx = 0
        self.evaluator = Evaluator()

    def add_player(self, player: PlayerState):
        self.players.append(player)

    def remove_player(self, player: PlayerState):
        if player in self.players:
            self.players.remove(player)

    def _draw_one(self, deck: Deck) -> int:
        drawn = deck.draw(1)
        return drawn if isinstance(drawn, int) else drawn[0]

    def play_hand(self, blinds: Dict[str, int]) -> Tuple[List[PlayerState], List[Dict]]:
        """
        Runs a single hand of Texas Hold'em.
        Returns a tuple of (busted_players, events).
        """
        events = []
        if len(self.players) < 2:
            return [], events
        self.hand_id += 1

        # 1. Setup
        for p in self.players:
            p.setup_new_hand()
            
        events.append({
            "type": "hand_start",
            "table_id": self.table_id,
            "hand_id": self.hand_id,
            "tournament_id": self.tournament_id,
            "players": [{"name": p.name, "stack": p.stack} for p in self.players]
        })
            
        deck = Deck()
        board = []
        pot_manager = PotManager()
        
        # Determine positions
        num_players = len(self.players)
        # Move button
        self.button_idx = (self.button_idx + 1) % num_players
        sb_idx = (self.button_idx + 1) % num_players
        bb_idx = (self.button_idx + 2) % num_players if num_players > 2 else self.button_idx

        # Post Blinds
        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]
        
        sb_val = sb_player.bet(blinds['small'])
        bb_val = bb_player.bet(blinds['big'])
        
        events.append({"type": "post_blind", "table_id": self.table_id, "player": sb_player.name, "amount": sb_val})
        events.append({"type": "post_blind", "table_id": self.table_id, "player": bb_player.name, "amount": bb_val})

        # Deal hole cards
        for p in self.players:
            if p.is_active:
                p.hole_cards = deck.draw(2)
                events.append({"type": "deal", "table_id": self.table_id, "player": p.name, "cards": [Card.int_to_str(c) for c in p.hole_cards]})

        # 2. Pre-Flop Betting
        action_start_idx = (bb_idx + 1) % num_players
        self._betting_round(blinds, board, pot_manager, action_start_idx, bb_player.current_bet, events)
        
        # 3. Flop
        if self._active_players_count() > 1:
            drawn = deck.draw(3)
            board.extend(drawn)
            events.append({"type": "board", "table_id": self.table_id, "cards": [Card.int_to_str(c) for c in drawn], "street": "flop"})
            self._betting_round(blinds, board, pot_manager, sb_idx, 0, events)
            
        # 4. Turn
        if self._active_players_count() > 1:
            drawn = self._draw_one(deck)
            board.append(drawn)
            events.append({"type": "board", "table_id": self.table_id, "cards": [Card.int_to_str(drawn)], "street": "turn"})
            self._betting_round(blinds, board, pot_manager, sb_idx, 0, events)

        # 5. River
        if self._active_players_count() > 1:
            drawn = self._draw_one(deck)
            board.append(drawn)
            events.append({"type": "board", "table_id": self.table_id, "cards": [Card.int_to_str(drawn)], "street": "river"})
            self._betting_round(blinds, board, pot_manager, sb_idx, 0, events)

        # 6. Showdown and Pot Distribution
        self._showdown(board, pot_manager, deck, events)
        
        # 7. Cleanup
        busted_players = [p for p in self.players if p.stack == 0]
        for p in busted_players:
            self.remove_player(p)
            
        return busted_players, events

    def _active_players_count(self) -> int:
        return sum(1 for p in self.players if p.is_active and not p.is_all_in) + \
               sum(1 for p in self.players if p.is_active and p.is_all_in)

    def _players_who_can_act(self) -> int:
         return sum(1 for p in self.players if p.is_active and not p.is_all_in)

    def _betting_round(self, blinds: Dict[str, int], board: List[int], pot_manager: PotManager, start_idx: int, current_highest_bet: int, events: List[Dict]):
        if self._players_who_can_act() <= 1 and all(p.current_bet == current_highest_bet for p in self.players if p.is_active and not p.is_all_in):
            return # No betting round needed if only 1 can act and they match the highest bet

        for p in self.players:
             p.has_acted = False
             
        min_raise = blinds['big']
        idx = start_idx
        num_players = len(self.players)
        
        while True:
            player = self.players[idx]
            
            if player.is_active and not player.is_all_in:
                # Need to act if haven't acted, OR if current bet doesn't match highest bet
                if not player.has_acted or player.current_bet < current_highest_bet:
                    call_amount = current_highest_bet - player.current_bet
                    
                    state = {
                        "hole_cards": player.hole_cards,
                        "board_cards": board,
                        "pot_size": sum(p.current_bet for p in self.players) + pot_manager.get_total_amount(),
                        "stack_size": player.stack,
                        "call_amount": call_amount,
                        "min_raise": min_raise,
                        "blinds": blinds,
                        "active_players": self._active_players_count(),
                        "player_id": player.name,
                        "table_id": self.table_id,
                        "hand_id": self.hand_id,
                        "tournament_id": self.tournament_id,
                        "position": self._position_label(idx, num_players),
                    }
                    
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(player.get_action, state)
                            action, amount = future.result(timeout=config.bot_decision_timeout_ms / 1000.0)
                    except concurrent.futures.TimeoutError:
                        print(f"Bot {player.name} timed out (> {config.bot_decision_timeout_ms}ms). Defaulting to fold.")
                        action, amount = "fold", 0
                    except Exception as e:
                        print(f"Bot {player.name} raised exception: {e}. Defaulting to fold.")
                        action, amount = "fold", 0

                    if action == "fold" and call_amount > 0:
                        player.is_active = False
                        events.append({"type": "action", "table_id": self.table_id, "player": player.name, "action": "fold", "amount": 0})
                    elif action == "call":
                        actual_bet = player.bet(call_amount)
                        events.append({"type": "action", "table_id": self.table_id, "player": player.name, "action": "call", "amount": actual_bet})
                    elif action == "raise":
                        if amount is None or amount < min_raise:
                            amount = min_raise
                        total_to_put_in = call_amount + amount
                        actual_bet = player.bet(total_to_put_in)
                        
                        events.append({"type": "action", "table_id": self.table_id, "player": player.name, "action": "raise", "amount": actual_bet})

                        if player.current_bet > current_highest_bet:
                            # Valid raise increases the highest bet
                            raise_size = player.current_bet - current_highest_bet
                            if raise_size > min_raise:
                                min_raise = raise_size
                            current_highest_bet = player.current_bet
                            
                            # Reset has_acted for others since the bet increased
                            for p in self.players:
                                if p != player:
                                    p.has_acted = False
                    else:
                        # Default check/fold
                        if call_amount > 0:
                            player.is_active = False
                            events.append({"type": "action", "table_id": self.table_id, "player": player.name, "action": "fold", "amount": 0})
                        else:
                            events.append({"type": "action", "table_id": self.table_id, "player": player.name, "action": "check", "amount": 0})
                            
                    player.has_acted = True
            
            idx = (idx + 1) % num_players
            
            # End condition: all active non-all-in players have acted AND their bets match the highest bet
            all_acted = True
            all_matched = True
            for p in self.players:
                if p.is_active and not p.is_all_in:
                    if not p.has_acted:
                         all_acted = False
                    if p.current_bet < current_highest_bet:
                         all_matched = False
                         
            if all_acted and all_matched:
                break
                
            if self._active_players_count() <= 1:
                break
                
        # End of betting round: reset current_bet for next round
        for p in self.players:
             p.current_bet = 0

    def _position_label(self, idx: int, num_players: int) -> str:
        if idx == self.button_idx:
            return "BTN"
        if idx == (self.button_idx + 1) % num_players:
            return "SB"
        if idx == ((self.button_idx + 2) % num_players if num_players > 2 else self.button_idx):
            return "BB"
        labels = ["UTG", "UTG_1", "UTG_2", "LJ", "HJ", "CO"]
        return labels[(idx - self.button_idx - 3) % len(labels)]

    def _showdown(self, board: List[int], pot_manager: PotManager, deck: Deck, events: List[Dict]):
        pot_manager.collect_bets(self.players)
        
        active_players = [p for p in self.players if p.is_active]
        if len(active_players) == 1:
            # Everyone else folded, unchallenged winner
            winner = active_players[0]
            for pot in pot_manager.pots:
                winner.win(pot.amount)
                if pot.amount > 0:
                    events.append({"type": "award_pot", "table_id": self.table_id, "player": winner.name, "amount": pot.amount})
            return
            
        # Draw remaining cards if they went all in early
        while len(board) < 5:
            drawn = self._draw_one(deck)
            board.append(drawn)
            events.append({"type": "board", "table_id": self.table_id, "cards": [drawn], "street": "runout"})

        # Showdown for real
        player_ranks = {p: self.evaluator.evaluate(board, p.hole_cards) for p in active_players}
        
        for p in active_players:
             events.append({"type": "showdown", "table_id": self.table_id, "player": p.name, "cards": p.hole_cards, "rank": player_ranks[p]})
        
        # Distribute pots
        for pot in pot_manager.pots:
            if pot.amount == 0:
                continue
                
            eligible = [p for p in pot.eligible_players if p in active_players]
            if not eligible:
                continue
                
            # Find best rank among eligible
            best_rank = min(player_ranks[p] for p in eligible)
            winners = [p for p in eligible if player_ranks[p] == best_rank]
            
            # Split pot if necessary
            split_amount = pot.amount // len(winners)
            for w in winners:
                w.win(split_amount)
                events.append({"type": "award_pot", "table_id": self.table_id, "player": w.name, "amount": split_amount})
