import random
from typing import List, Dict, Any, Tuple
from engine.player_state import PlayerState
from engine.table import Table
from engine.config import config

class Tournament:
    """
    Orchestrates the entire multi-table tournament.
    """
    def __init__(self, bots: List[Any], tournament_id: int = 0):
        self.tournament_id = tournament_id
        self.starting_field = len(bots)
        self.paid_places = len(config.payouts)
        self.players = [PlayerState(bot, config.starting_stack) for bot in bots]
        self.tables: List[Table] = []
        self.placements: List[PlayerState] = []
        
        self.current_blind_idx = 0
        self.hands_played = 0
        
        self.events = []
        
        self.events.append({
            "type": "tournament_start",
            "tournament_id": self.tournament_id,
            "blinds": config.blinds_schedule[self.current_blind_idx],
            "players": [p.name for p in self.players]
        })
        
        self._seat_players_randomly()

    def _seat_players_randomly(self):
        random.shuffle(self.players)
        max_p = config.max_players_per_table
        
        table_id = 1
        for i in range(0, len(self.players), max_p):
            table_players = self.players[i:i+max_p]
            table = Table(table_id, tournament_id=self.tournament_id)
            table.starting_field = self.starting_field
            table.paid_places = self.paid_places
            for p in table_players:
                table.add_player(p)
                self.events.append({"type": "seat", "player": p.name, "table_id": table_id})
            self.tables.append(table)
            table_id += 1

    def _coalesce_tables(self):
        """
        Balances the tables after players bust out.
        """
        # Remove empty tables
        self.tables = [t for t in self.tables if len(t.players) > 0]
        
        if len(self.tables) <= 1:
            return
            
        max_p = config.max_players_per_table
        total_active = sum(len(t.players) for t in self.tables)
        needed_tables = (total_active + max_p - 1) // max_p
        
        if len(self.tables) > needed_tables:
            # We can collapse a table
            tables_sorted = sorted(self.tables, key=lambda t: len(t.players))
            table_to_break = tables_sorted[0]
            
            self.tables.remove(table_to_break)
            self.events.append({"type": "table_broken", "table_id": table_to_break.table_id})

            # Distribute players
            for player in table_to_break.players:
                # Find table with most seats available
                target_table = min(self.tables, key=lambda t: len(t.players))
                target_table.add_player(player)
                self.events.append({"type": "seat", "player": player.name, "table_id": target_table.table_id})
                
            # Recursive check
            self._coalesce_tables()

    def play(self) -> Tuple[List[Dict[str, Any]], List[Dict]]:
        """
        Runs the tournament until 1 player remains.
        Returns (placements_list, events_list).
        """
        max_hands = getattr(config, "max_hands_per_tournament", None)
        total_hands_played = 0
        while sum(len(t.players) for t in self.tables) > 1:
            if max_hands is not None and total_hands_played >= max_hands:
                remaining = sorted(
                    [p for table in self.tables for p in table.players],
                    key=lambda p: p.stack,
                    reverse=True,
                )
                self.placements.extend(reversed(remaining))
                self.events.append({
                    "type": "tournament_stopped_max_hands",
                    "tournament_id": self.tournament_id,
                    "max_hands": max_hands,
                })
                break

            blinds = config.blinds_schedule[self.current_blind_idx]
            
            # Play one hand on all active tables
            players_left = sum(len(t.players) for t in self.tables)
            for table in self.tables:
                table.starting_field = self.starting_field
                table.paid_places = self.paid_places
                table.players_left = players_left
                busted, table_events = table.play_hand(blinds)
                self.events.extend(table_events)
                
                # Busted players
                if busted:
                    for p in busted:
                        self.events.append({"type": "knockout", "player": p.name, "table_id": table.table_id})
                    self.placements.extend(busted)
            
            self.hands_played += 1
            total_hands_played += 1
            if self.hands_played >= config.hands_per_level:
                self.hands_played = 0
                if self.current_blind_idx < len(config.blinds_schedule) - 1:
                    self.current_blind_idx += 1
                    new_blinds = config.blinds_schedule[self.current_blind_idx]
                    self.events.append({"type": "level_up", "blinds": new_blinds})
                    
            self._coalesce_tables()
            
        # Add the winner to the end of the list
        if sum(len(t.players) for t in self.tables) == 1 and self.tables and self.tables[0].players:
            winner = self.tables[0].players[0]
            self.placements.append(winner)
            self.events.append({"type": "tournament_win", "player": winner.name})
             
        # Return reversed: 1st place is at 0, 2nd at 1, etc.
        self.placements.reverse()
        
        results = []
        for i, p in enumerate(self.placements):
            pos = i + 1
            payout_pct = config.payouts.get(pos, 0.0)
            results.append({
                "position": pos,
                "name": p.name,
                "bot_class": p.bot.__class__.__name__,
                "payout_pct": payout_pct
            })
            
        return results, self.events
