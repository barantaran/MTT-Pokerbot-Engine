import random
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple
from treys import Deck, Evaluator, Card
from engine.player_state import PlayerState
from engine.pot import PotManager
from engine.config import config
from engine.game_stats import TableStatsTracker
import concurrent.futures


PREFLOP_SPOT_UNKNOWN = "unknown"
PREFLOP_SPOT_LIMPED = "limped"
PREFLOP_SPOT_SRP = "srp"
PREFLOP_SPOT_THREE_BET = "three_bet"
PREFLOP_SPOT_FOUR_BET = "four_bet"
PREFLOP_SPOT_FIVE_BET_PLUS = "five_bet_plus"
PREFLOP_SPOT_ALL_IN_PRESSURE = "all_in_pressure"

class Table:
    """
    Manages a single poker table.
    Handles dealing, betting rounds, and winner evaluation.
    """
    def __init__(self, table_id: int, tournament_id: int = 0):
        self.table_id = table_id
        self.tournament_id = tournament_id
        self.hand_id = 0
        self.level = 0
        self.players: List[PlayerState] = []
        self.button_idx = 0
        self.evaluator = Evaluator()
        self.starting_field = 0
        self.players_left = 0
        self.paid_places = 0
        self.payouts: Dict[int, float] = {}
        self.current_preflop_spot_type = PREFLOP_SPOT_UNKNOWN
        self.stats_tracker = TableStatsTracker()

    def add_player(self, player: PlayerState):
        self.players.append(player)

    def remove_player(self, player: PlayerState):
        if player in self.players:
            self.players.remove(player)

    def _draw_one(self, deck: Deck) -> int:
        drawn = deck.draw(1)
        return drawn if isinstance(drawn, int) else drawn[0]

    def _street_label(self, board: List[int]) -> str:
        if not board:
            return "preflop"
        if len(board) == 3:
            return "flop"
        if len(board) == 4:
            return "turn"
        return "river"

    def _blind_event(self, player: PlayerState, seat: int, blind: str, amount: int) -> Dict[str, Any]:
        return {
            "type": "post_blind",
            "table_id": self.table_id,
            "hand_id": self.hand_id,
            "tournament_id": self.tournament_id,
            "player": player.name,
            "seat": seat,
            "blind": blind,
            "amount": amount,
            "street": "preflop",
        }

    def _board_event(self, cards: List[str], street: str) -> Dict[str, Any]:
        return {
            "type": "board",
            "table_id": self.table_id,
            "hand_id": self.hand_id,
            "tournament_id": self.tournament_id,
            "cards": cards,
            "street": street,
        }

    def _action_event(
        self,
        player_name: str,
        action: str,
        amount: int,
        street: str,
        *,
        position: str = "",
        call_amount: int = 0,
        pot_size: int = 0,
        tool_event: Dict[str, Any] | None = None,
        tool_events: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        event = {
            "type": "action",
            "table_id": self.table_id,
            "hand_id": self.hand_id,
            "tournament_id": self.tournament_id,
            "player": player_name,
            "action": action,
            "amount": amount,
            "street": street,
            "position": position,
            "call_amount": call_amount,
            "pot_size": pot_size,
        }
        if tool_event:
            event["tool_event"] = dict(tool_event)
        if tool_events:
            event["tool_events"] = [dict(item) for item in tool_events]
        return event

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
        self.current_preflop_spot_type = PREFLOP_SPOT_UNKNOWN
            
        # Determine positions. This runs before hand_start is built so the event
        # can name the button and label every seat — a replayer cannot lay out a
        # table without knowing where the button is.
        num_players = len(self.players)
        # Move button
        self.button_idx = (self.button_idx + 1) % num_players
        sb_idx = (self.button_idx + 1) % num_players
        bb_idx = (self.button_idx + 2) % num_players if num_players > 2 else self.button_idx

        hand_start_event = {
            "type": "hand_start",
            "table_id": self.table_id,
            "hand_id": self.hand_id,
            "tournament_id": self.tournament_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "level": self.level,
            "blinds": {"small": blinds["small"], "big": blinds["big"]},
            "button_seat": self.button_idx,
            "button_player": self.players[self.button_idx].name,
            "players": [
                {
                    "seat": index,
                    "name": p.name,
                    "stack": p.stack,
                    "position": self._position_label(index, num_players),
                }
                for index, p in enumerate(self.players)
            ],
        }
        events.append(hand_start_event)
        self.stats_tracker.observe_event(hand_start_event)

        deck = Deck()
        board = []
        pot_manager = PotManager()

        # Post Blinds
        sb_player = self.players[sb_idx]
        bb_player = self.players[bb_idx]
        
        sb_val = sb_player.bet(blinds['small'])
        bb_val = bb_player.bet(blinds['big'])
        
        events.append(self._blind_event(sb_player, sb_idx, "small", sb_val))
        events.append(self._blind_event(bb_player, bb_idx, "big", bb_val))

        # Deal hole cards
        for p in self.players:
            if p.is_active:
                p.hole_cards = deck.draw(2)
                events.append({
                    "type": "deal",
                    "table_id": self.table_id,
                    "hand_id": self.hand_id,
                    "tournament_id": self.tournament_id,
                    "player": p.name,
                    "cards": [Card.int_to_str(c) for c in p.hole_cards],
                })

        # 2. Pre-Flop Betting
        action_start_idx = (bb_idx + 1) % num_players
        self._betting_round(blinds, board, pot_manager, action_start_idx, bb_player.current_bet, events)
        
        # 3. Flop
        if self._active_players_count() > 1:
            drawn = deck.draw(3)
            board.extend(drawn)
            events.append(self._board_event([Card.int_to_str(c) for c in drawn], "flop"))
            self._betting_round(blinds, board, pot_manager, sb_idx, 0, events)
            
        # 4. Turn
        if self._active_players_count() > 1:
            drawn = self._draw_one(deck)
            board.append(drawn)
            events.append(self._board_event([Card.int_to_str(drawn)], "turn"))
            self._betting_round(blinds, board, pot_manager, sb_idx, 0, events)

        # 5. River
        if self._active_players_count() > 1:
            drawn = self._draw_one(deck)
            board.append(drawn)
            events.append(self._board_event([Card.int_to_str(drawn)], "river"))
            self._betting_round(blinds, board, pot_manager, sb_idx, 0, events)

        # 6. Showdown and Pot Distribution
        self._showdown(board, pot_manager, deck, events)
        events.append({
            "type": "hand_end",
            "table_id": self.table_id,
            "hand_id": self.hand_id,
            "tournament_id": self.tournament_id,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "players": [{"name": p.name, "stack": p.stack} for p in self.players],
        })
        
        # 7. Cleanup
        busted_players = [p for p in self.players if p.stack == 0]
        for p in busted_players:
            self.remove_player(p)

        # Stamp the hand keys and a per-hand sequence number on everything. The
        # tournament merges one flat event list from every table in turn, so
        # array position is not an ordering within a hand once the streams are
        # interleaved — seq is. setdefault leaves the keys the emitters already
        # set untouched and only fills the gaps.
        for index, event in enumerate(events):
            event.setdefault("table_id", self.table_id)
            event.setdefault("hand_id", self.hand_id)
            event.setdefault("tournament_id", self.tournament_id)
            event["seq"] = index

        return busted_players, events

    def _active_players_count(self) -> int:
        return sum(1 for p in self.players if p.is_active and not p.is_all_in) + \
               sum(1 for p in self.players if p.is_active and p.is_all_in)

    def _players_who_can_act(self) -> int:
         return sum(1 for p in self.players if p.is_active and not p.is_all_in)

    def _visible_pot_size(self) -> int:
        return sum(max(0, int(p.total_bet)) for p in self.players)

    def _preflop_spot_type(
        self,
        *,
        board: List[int],
        blinds: Dict[str, int],
        current_highest_bet: int,
        raise_count: int,
        limp_count: int,
        call_amount: int,
        player_stack: int,
    ) -> str:
        if board:
            return self.current_preflop_spot_type
        if call_amount > 0 and (
            call_amount >= player_stack
            or any(p.is_active and p.is_all_in and p.current_bet >= current_highest_bet for p in self.players)
        ):
            return PREFLOP_SPOT_ALL_IN_PRESSURE
        if raise_count >= 4:
            return PREFLOP_SPOT_FIVE_BET_PLUS
        if raise_count == 3:
            return PREFLOP_SPOT_FOUR_BET
        if raise_count == 2:
            return PREFLOP_SPOT_THREE_BET
        if raise_count == 1:
            return PREFLOP_SPOT_SRP
        if limp_count > 0:
            return PREFLOP_SPOT_LIMPED
        return PREFLOP_SPOT_UNKNOWN

    def _preflop_spot_from_raise_count(self, raise_count: int) -> str:
        if raise_count >= 4:
            return PREFLOP_SPOT_FIVE_BET_PLUS
        if raise_count == 3:
            return PREFLOP_SPOT_FOUR_BET
        if raise_count == 2:
            return PREFLOP_SPOT_THREE_BET
        if raise_count == 1:
            return PREFLOP_SPOT_SRP
        return self.current_preflop_spot_type

    def _betting_round(self, blinds: Dict[str, int], board: List[int], pot_manager: PotManager, start_idx: int, current_highest_bet: int, events: List[Dict]):
        if self._players_who_can_act() <= 1 and all(p.current_bet == current_highest_bet for p in self.players if p.is_active and not p.is_all_in):
            return # No betting round needed if only 1 can act and they match the highest bet

        for p in self.players:
             p.has_acted = False
             
        min_raise = blinds['big']
        preflop_raise_count = 0
        preflop_limp_count = 0
        current_preflop_aggressor_name = ""
        current_preflop_aggressor_position = ""
        current_preflop_aggressor_stack = 0
        idx = start_idx
        num_players = len(self.players)
        
        while True:
            player = self.players[idx]
            
            if player.is_active and not player.is_all_in:
                # Need to act if haven't acted, OR if current bet doesn't match highest bet
                if not player.has_acted or player.current_bet < current_highest_bet:
                    call_amount = current_highest_bet - player.current_bet
                    preflop_spot_type = self._preflop_spot_type(
                        board=board,
                        blinds=blinds,
                        current_highest_bet=current_highest_bet,
                        raise_count=preflop_raise_count,
                        limp_count=preflop_limp_count,
                        call_amount=call_amount,
                        player_stack=player.stack,
                    )
                    
                    state = {
                        "hole_cards": player.hole_cards,
                        "board_cards": board,
                        "_hand_events": events,
                        "pot_size": self._visible_pot_size(),
                        "stack_size": player.stack,
                        "table_stacks": [p.stack for p in self.players],
                        "hero_table_index": idx,
                        "call_amount": call_amount,
                        "min_raise": min_raise,
                        "blinds": blinds,
                        "active_players": self._active_players_count(),
                        "players_left": self.players_left or len(self.players),
                        "starting_field": self.starting_field or len(self.players),
                        "paid_places": self.paid_places,
                        "payouts": dict(self.payouts),
                        "player_id": player.name,
                        "table_id": self.table_id,
                        "hand_id": self.hand_id,
                        "tournament_id": self.tournament_id,
                        "position": self._position_label(idx, num_players),
                        "preflop_spot_type": preflop_spot_type,
                        "table_stats": self.stats_tracker.snapshot(),
                        "opponent_id": current_preflop_aggressor_name if call_amount > 0 else "",
                        "opponent_position": current_preflop_aggressor_position if call_amount > 0 else "",
                        "opponent_stack_size": current_preflop_aggressor_stack if call_amount > 0 else 0,
                        "opponent_stack_bb": (
                            current_preflop_aggressor_stack / float(blinds["big"])
                            if call_amount > 0 and blinds["big"] > 0
                            else 0.0
                        ),
                        "opponent_stats": (
                            self.stats_tracker.player_snapshot(current_preflop_aggressor_name)
                            if call_amount > 0 and current_preflop_aggressor_name
                            else None
                        ),
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

                    street = self._street_label(board)

                    action_position = str(state.get("position", ""))
                    action_pot_size = int(state.get("pot_size", 0) or 0)
                    action_tool_event = state.get("_bot_tool_event")
                    if not isinstance(action_tool_event, dict):
                        action_tool_event = None
                    action_tool_events = state.get("_bot_tool_events")
                    if not isinstance(action_tool_events, list):
                        action_tool_events = None

                    if action == "fold" and call_amount > 0:
                        player.is_active = False
                        event = self._action_event(
                            player.name,
                            "fold",
                            0,
                            street,
                            position=action_position,
                            call_amount=call_amount,
                            pot_size=action_pot_size,
                            tool_event=action_tool_event, tool_events=action_tool_events,
                        )
                        events.append(event)
                        self.stats_tracker.observe_event(event)
                    elif action == "call":
                        actual_bet = player.bet(call_amount)
                        event = self._action_event(
                            player.name,
                            "call",
                            actual_bet,
                            street,
                            position=action_position,
                            call_amount=call_amount,
                            pot_size=action_pot_size,
                            tool_event=action_tool_event, tool_events=action_tool_events,
                        )
                        events.append(event)
                        self.stats_tracker.observe_event(event)
                        if not board and current_highest_bet <= blinds["big"] and actual_bet > 0:
                            preflop_limp_count += 1
                            self.current_preflop_spot_type = PREFLOP_SPOT_LIMPED
                        if not board and player.is_all_in and actual_bet > 0:
                            self.current_preflop_spot_type = PREFLOP_SPOT_ALL_IN_PRESSURE
                    elif action == "raise":
                        if amount is None or amount < min_raise:
                            amount = min_raise
                        total_to_put_in = call_amount + amount
                        actual_bet = player.bet(total_to_put_in)
                        
                        event = self._action_event(
                            player.name,
                            "raise",
                            actual_bet,
                            street,
                            position=action_position,
                            call_amount=call_amount,
                            pot_size=action_pot_size,
                            tool_event=action_tool_event, tool_events=action_tool_events,
                        )
                        events.append(event)
                        self.stats_tracker.observe_event(event)

                        if player.current_bet > current_highest_bet:
                            # Valid raise increases the highest bet
                            raise_size = player.current_bet - current_highest_bet
                            if raise_size > min_raise:
                                min_raise = raise_size
                            current_highest_bet = player.current_bet
                            if not board:
                                preflop_raise_count += 1
                                current_preflop_aggressor_name = player.name
                                current_preflop_aggressor_position = action_position
                                current_preflop_aggressor_stack = player.stack
                                self.current_preflop_spot_type = (
                                    PREFLOP_SPOT_ALL_IN_PRESSURE
                                    if player.is_all_in
                                    else self._preflop_spot_from_raise_count(preflop_raise_count)
                                )
                            
                            # Reset has_acted for others since the bet increased
                            for p in self.players:
                                if p != player:
                                    p.has_acted = False
                    else:
                        # Default check/fold
                        if call_amount > 0:
                            player.is_active = False
                            event = self._action_event(
                                player.name,
                                "fold",
                                0,
                                street,
                                position=action_position,
                                call_amount=call_amount,
                                pot_size=action_pot_size,
                                tool_event=action_tool_event, tool_events=action_tool_events,
                            )
                            events.append(event)
                            self.stats_tracker.observe_event(event)
                        else:
                            event = self._action_event(
                                player.name,
                                "check",
                                0,
                                street,
                                position=action_position,
                                call_amount=call_amount,
                                pot_size=action_pot_size,
                                tool_event=action_tool_event, tool_events=action_tool_events,
                            )
                            events.append(event)
                            self.stats_tracker.observe_event(event)
                            
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
                    events.append({
                        "type": "award_pot",
                        "table_id": self.table_id,
                        "hand_id": self.hand_id,
                        "tournament_id": self.tournament_id,
                        "player": winner.name,
                        "amount": pot.amount,
                        "showdown": False,
                    })
            return
            
        # Draw remaining cards if they went all in early
        while len(board) < 5:
            drawn = self._draw_one(deck)
            board.append(drawn)
            events.append(self._board_event([Card.int_to_str(drawn)], "runout"))

        # Showdown for real
        player_ranks = {p: self.evaluator.evaluate(board, p.hole_cards) for p in active_players}
        
        for p in active_players:
             events.append({
                 "type": "showdown",
                 "table_id": self.table_id,
                 "hand_id": self.hand_id,
                 "tournament_id": self.tournament_id,
                 "player": p.name,
                 "cards": [Card.int_to_str(c) for c in p.hole_cards],
                 "rank": player_ranks[p],
                 "rank_class": self.evaluator.class_to_string(
                     self.evaluator.get_rank_class(player_ranks[p])
                 ),
             })
        
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
                events.append({
                    "type": "award_pot",
                    "table_id": self.table_id,
                    "hand_id": self.hand_id,
                    "tournament_id": self.tournament_id,
                    "player": w.name,
                    "amount": split_amount,
                    "showdown": True,
                })
