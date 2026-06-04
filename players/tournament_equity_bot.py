from __future__ import annotations

from engine.player_interface import Bot
from engine.pokerstove_equity import estimate_equity, pot_odds


_SPOT_TIGHTNESS = {
    "unknown": 0.00,
    "limped": -0.02,
    "srp": 0.00,
    "single_raised": 0.00,
    "three_bet": 0.06,
    "3bet": 0.06,
    "four_bet": 0.11,
    "4bet": 0.11,
    "five_bet_plus": 0.16,
    "five_bet": 0.16,
    "5bet": 0.16,
    "all_in_pressure": 0.18,
    "allin": 0.18,
}


class TournamentEquityBot(Bot):
    """
    Equity bot with explicit MTT risk controls.

    Compared with TightEquityBot, this bot separates call EV from raise pressure,
    tightens marginal calls near payout jumps, and only shoves intentionally when
    the stack is short or the call already commits most of the stack.
    """

    def __init__(self, *, use_preflop_spot_range: bool = True):
        super().__init__("TournamentEquityBot")
        self.use_preflop_spot_range = bool(use_preflop_spot_range)

    def get_action(self, game_state):
        hole_cards = game_state.get("hole_cards", [])
        board_cards = game_state.get("board_cards", [])
        pot_size = int(game_state.get("pot_size", 0) or 0)
        stack_size = int(game_state.get("stack_size", 0) or 0)
        call_amount = int(game_state.get("call_amount", 0) or 0)
        min_raise = int(game_state.get("min_raise", 0) or 0)
        active_players = int(game_state.get("active_players", 2) or 2)
        big_blind = int(game_state.get("blinds", {}).get("big", 1) or 1)
        spot_type = str(game_state.get("preflop_spot_type", "unknown") or "unknown").lower()
        position = str(game_state.get("position", "") or "").upper()

        street = len(board_cards)
        stack_bb = stack_size / float(big_blind)
        max_raise_extra = max(0, stack_size - call_amount)
        can_raise = max_raise_extra >= max(1, min_raise)
        required_equity = pot_odds(call_amount, pot_size)

        if "hero_equity" in game_state:
            equity = float(game_state["hero_equity"])
        else:
            equity = estimate_equity(
                hole_cards=hole_cards,
                board_cards=board_cards,
                active_players=active_players,
                opponent_range_pct=game_state.get("opponent_range_pct"),
                preflop_spot_type=spot_type,
                use_preflop_spot_range=self.use_preflop_spot_range,
            )

        call_margin = self._call_margin(
            street=street,
            active_players=active_players,
            stack_bb=stack_bb,
            game_state=game_state,
            spot_type=spot_type,
            position=position,
        )
        raise_threshold = self._raise_threshold(
            street=street,
            active_players=active_players,
            stack_bb=stack_bb,
            game_state=game_state,
            spot_type=spot_type,
            position=position,
        )
        jam_threshold = self._jam_threshold(
            street=street,
            stack_bb=stack_bb,
            game_state=game_state,
            spot_type=spot_type,
        )

        if call_amount > 0 and equity < required_equity + call_margin:
            return ("fold", 0)

        if can_raise and self._should_jam(
            equity=equity,
            jam_threshold=jam_threshold,
            stack_bb=stack_bb,
            stack_size=stack_size,
            call_amount=call_amount,
        ):
            return ("raise", max_raise_extra)

        if can_raise and equity >= raise_threshold:
            return ("raise", self._raise_size(
                street=street,
                pot_size=pot_size,
                big_blind=big_blind,
                min_raise=min_raise,
                max_raise_extra=max_raise_extra,
                spot_type=spot_type,
                position=position,
            ))

        return ("call", 0) if call_amount == 0 or equity >= required_equity + call_margin else ("fold", 0)

    def _call_margin(self, *, street, active_players, stack_bb, game_state, spot_type, position):
        base = {
            0: 0.025,
            3: 0.035,
            4: 0.040,
            5: 0.045,
        }.get(street, 0.04)
        base += max(0, active_players - 2) * 0.012
        base += _SPOT_TIGHTNESS.get(spot_type, 0.0)
        if street == 0:
            base += self._position_call_adjustment(position=position, spot_type=spot_type)
        base += self._payout_pressure(game_state) * 0.10
        if stack_bb <= 10:
            base -= 0.025
        return max(-0.02, min(0.25, base))

    def _raise_threshold(self, *, street, active_players, stack_bb, game_state, spot_type, position):
        base = {
            0: 0.54,
            3: 0.60,
            4: 0.64,
            5: 0.68,
        }.get(street, 0.62)
        base += max(0, active_players - 2) * 0.015
        base += _SPOT_TIGHTNESS.get(spot_type, 0.0)
        if street == 0:
            base += self._position_raise_adjustment(position=position, spot_type=spot_type)
        base += self._payout_pressure(game_state) * 0.08
        if stack_bb <= 14:
            base -= 0.035
        return max(0.36, min(0.90, base))

    def _jam_threshold(self, *, street, stack_bb, game_state, spot_type):
        base = {
            0: 0.58,
            3: 0.64,
            4: 0.68,
            5: 0.72,
        }.get(street, 0.66)
        base += _SPOT_TIGHTNESS.get(spot_type, 0.0)
        base += self._payout_pressure(game_state) * 0.12
        if stack_bb <= 7:
            base -= 0.08
        elif stack_bb <= 12:
            base -= 0.04
        return max(0.45, min(0.94, base))

    def _should_jam(self, *, equity, jam_threshold, stack_bb, stack_size, call_amount):
        if equity < jam_threshold:
            return False
        if stack_bb <= 10:
            return True
        if stack_size <= 0:
            return False
        committed_fraction = call_amount / float(stack_size)
        return committed_fraction >= 0.70

    def _raise_size(self, *, street, pot_size, big_blind, min_raise, max_raise_extra, spot_type, position):
        if street == 0:
            open_bb = self._preflop_open_bb(position=position, spot_type=spot_type)
            target = max(min_raise, int(open_bb * big_blind))
        else:
            target = max(min_raise, int(max(big_blind, pot_size * 0.60)))
        return min(max_raise_extra, target)

    def _position_call_adjustment(self, *, position, spot_type):
        if spot_type in {"three_bet", "3bet", "four_bet", "4bet", "five_bet_plus", "five_bet", "5bet", "all_in_pressure"}:
            return {
                "BTN": -0.010,
                "CO": -0.006,
                "SB": 0.018,
                "BB": -0.018,
            }.get(position, 0.0)
        return {
            "BTN": -0.018,
            "CO": -0.012,
            "HJ": -0.006,
            "SB": 0.014,
            "BB": -0.014,
            "UTG": 0.018,
            "UTG_1": 0.014,
            "UTG_2": 0.010,
        }.get(position, 0.0)

    def _position_raise_adjustment(self, *, position, spot_type):
        if spot_type in {"three_bet", "3bet", "four_bet", "4bet", "five_bet_plus", "five_bet", "5bet", "all_in_pressure"}:
            return {
                "BTN": -0.012,
                "CO": -0.008,
                "SB": 0.018,
                "BB": -0.010,
            }.get(position, 0.0)
        return {
            "BTN": -0.055,
            "CO": -0.040,
            "HJ": -0.022,
            "SB": -0.025,
            "BB": -0.010,
            "UTG": 0.035,
            "UTG_1": 0.025,
            "UTG_2": 0.015,
        }.get(position, 0.0)

    def _preflop_open_bb(self, *, position, spot_type):
        if spot_type not in {"unknown", "limped"}:
            return 3.0
        if position in {"BTN", "CO", "SB"}:
            return 2.2
        if position in {"UTG", "UTG_1", "UTG_2"}:
            return 2.5
        return 2.3

    def _payout_pressure(self, game_state):
        next_prize_gain = max(0.0, float(game_state.get("next_prize_gain_pct", 0.0) or 0.0))
        players_left = int(game_state.get("players_left", 0) or 0)
        paid_places = int(game_state.get("paid_places", 0) or 0)
        if paid_places <= 0 or players_left <= 0:
            return min(1.0, next_prize_gain * 4.0)
        bubble_pressure = 1.0 if paid_places < players_left <= paid_places + 2 else 0.0
        final_table_pressure = 0.5 if players_left <= 10 else 0.0
        return min(1.0, bubble_pressure + final_table_pressure + next_prize_gain * 4.0)
