from __future__ import annotations

from engine.icm import calculate_exact_icm
from engine.player_interface import Bot
from engine.pokerstove_equity import estimate_equity, pot_odds


_SPOT_TIGHTNESS = {
    "unknown": 0.00,
    "limped": -0.01,
    "srp": 0.02,
    "single_raised": 0.02,
    "three_bet": 0.07,
    "3bet": 0.07,
    "four_bet": 0.12,
    "4bet": 0.12,
    "five_bet_plus": 0.18,
    "five_bet": 0.18,
    "5bet": 0.18,
    "all_in_pressure": 0.20,
    "allin": 0.20,
}


class ICMTightBot(Bot):
    """Conservative equity bot with optional ICM-aware risk pressure."""

    def __init__(
        self,
        *,
        use_icm=True,
        use_preflop_spot_range=True,
        icm_strength=1.0,
        call_margin_shift=0.0,
        raise_threshold_shift=0.0,
    ):
        super().__init__("ICMTightBot")
        self.use_icm = bool(use_icm)
        self.use_preflop_spot_range = bool(use_preflop_spot_range)
        self.icm_strength = max(0.0, float(icm_strength))
        self.call_margin_shift = float(call_margin_shift)
        self.raise_threshold_shift = float(raise_threshold_shift)

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

        street = len(board_cards or [])
        stack_bb = stack_size / float(max(1, big_blind))
        max_raise_extra = max(0, stack_size - call_amount)
        can_raise = max_raise_extra >= max(1, min_raise)

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

        icm_pressure = self._icm_pressure(game_state, stack_size=stack_size)
        call_margin = self._call_margin(
            street=street,
            active_players=active_players,
            stack_bb=stack_bb,
            spot_type=spot_type,
            position=position,
            icm_pressure=icm_pressure,
        )
        raise_threshold = self._raise_threshold(
            street=street,
            active_players=active_players,
            stack_bb=stack_bb,
            spot_type=spot_type,
            position=position,
            icm_pressure=icm_pressure,
        )
        jam_threshold = self._jam_threshold(
            street=street,
            stack_bb=stack_bb,
            spot_type=spot_type,
            icm_pressure=icm_pressure,
        )

        required_equity = pot_odds(call_amount, pot_size)
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

        return ("call", 0)

    def _call_margin(self, *, street, active_players, stack_bb, spot_type, position, icm_pressure):
        base = {
            0: 0.055,
            3: 0.060,
            4: 0.065,
            5: 0.070,
        }.get(street, 0.06)
        base += max(0, active_players - 2) * 0.014
        base += _SPOT_TIGHTNESS.get(spot_type, 0.0)
        if street == 0:
            base += self._position_call_adjustment(position=position, spot_type=spot_type)
        if stack_bb <= 10:
            base -= 0.025
        base += icm_pressure * 0.16
        base += self.call_margin_shift
        return max(-0.03, min(0.35, base))

    def _raise_threshold(self, *, street, active_players, stack_bb, spot_type, position, icm_pressure):
        base = {
            0: 0.62,
            3: 0.68,
            4: 0.72,
            5: 0.76,
        }.get(street, 0.70)
        base += max(0, active_players - 2) * 0.016
        base += _SPOT_TIGHTNESS.get(spot_type, 0.0)
        if street == 0:
            base += self._position_raise_adjustment(position=position, spot_type=spot_type)
        if stack_bb <= 12:
            base -= 0.035
        base += icm_pressure * 0.10
        base += self.raise_threshold_shift
        return max(0.42, min(0.94, base))

    def _jam_threshold(self, *, street, stack_bb, spot_type, icm_pressure):
        base = {
            0: 0.66,
            3: 0.72,
            4: 0.76,
            5: 0.80,
        }.get(street, 0.74)
        base += _SPOT_TIGHTNESS.get(spot_type, 0.0)
        if stack_bb <= 7:
            base -= 0.10
        elif stack_bb <= 12:
            base -= 0.05
        base += icm_pressure * 0.13
        return max(0.52, min(0.96, base))

    def _should_jam(self, *, equity, jam_threshold, stack_bb, stack_size, call_amount):
        if equity < jam_threshold:
            return False
        if stack_bb <= 10:
            return True
        if stack_size <= 0:
            return False
        return call_amount / float(stack_size) >= 0.75

    def _raise_size(self, *, street, pot_size, big_blind, min_raise, max_raise_extra, spot_type, position):
        if street == 0:
            target = max(min_raise, int(self._preflop_open_bb(position=position, spot_type=spot_type) * big_blind))
        else:
            target = max(min_raise, int(max(big_blind, pot_size * 0.55)))
        return min(max_raise_extra, target)

    def _position_call_adjustment(self, *, position, spot_type):
        if spot_type in {"three_bet", "3bet", "four_bet", "4bet", "five_bet_plus", "five_bet", "5bet", "all_in_pressure"}:
            return {
                "BTN": -0.006,
                "CO": -0.004,
                "SB": 0.020,
                "BB": -0.012,
            }.get(position, 0.0)
        return {
            "BTN": -0.014,
            "CO": -0.010,
            "HJ": -0.004,
            "SB": 0.018,
            "BB": -0.012,
            "UTG": 0.024,
            "UTG_1": 0.018,
            "UTG_2": 0.012,
        }.get(position, 0.0)

    def _position_raise_adjustment(self, *, position, spot_type):
        if spot_type in {"three_bet", "3bet", "four_bet", "4bet", "five_bet_plus", "five_bet", "5bet", "all_in_pressure"}:
            return {
                "BTN": -0.006,
                "CO": -0.004,
                "SB": 0.020,
                "BB": -0.004,
            }.get(position, 0.0)
        return {
            "BTN": -0.040,
            "CO": -0.030,
            "HJ": -0.014,
            "SB": -0.010,
            "BB": -0.006,
            "UTG": 0.045,
            "UTG_1": 0.035,
            "UTG_2": 0.024,
        }.get(position, 0.0)

    def _preflop_open_bb(self, *, position, spot_type):
        if spot_type not in {"unknown", "limped"}:
            return 3.0
        if position in {"BTN", "CO", "SB"}:
            return 2.2
        if position in {"UTG", "UTG_1", "UTG_2"}:
            return 2.6
        return 2.4

    def _icm_pressure(self, game_state, *, stack_size):
        if not self.use_icm or self.icm_strength <= 0:
            return 0.0

        exact_pressure = self._exact_icm_pressure(game_state, stack_size=stack_size)
        if exact_pressure is not None:
            return min(1.0, exact_pressure * self.icm_strength)
        return min(1.0, self._fallback_pressure(game_state) * self.icm_strength)

    def _exact_icm_pressure(self, game_state, *, stack_size):
        table_stacks = game_state.get("table_stacks")
        payouts = game_state.get("payouts")
        hero_index = game_state.get("hero_table_index")
        players_left = int(game_state.get("players_left", 0) or 0)
        if not isinstance(table_stacks, list) or not table_stacks:
            return None
        if players_left != len(table_stacks):
            return None
        if hero_index is None:
            return None

        try:
            hero_index = int(hero_index)
            payout_values = self._payout_values(payouts)
            stacks = [float(stack) for stack in table_stacks]
        except (TypeError, ValueError):
            return None
        if hero_index < 0 or hero_index >= len(stacks) or not payout_values:
            return None

        hero_stack = max(0.0, float(stack_size or 0.0))
        total_chips = sum(stacks)
        if hero_stack <= 0 or total_chips <= 0:
            return None

        try:
            current_icm = calculate_exact_icm(stacks, payout_values)[hero_index]
            loss_stacks = list(stacks)
            loss_stacks[hero_index] = 0.0
            loss_icm = calculate_exact_icm(loss_stacks, payout_values)[hero_index]
            gain_stacks = list(stacks)
            gain_stacks[hero_index] += min(hero_stack, max(1.0, total_chips * 0.05))
            gain_icm = calculate_exact_icm(gain_stacks, payout_values)[hero_index]
        except ValueError:
            return None

        downside = max(0.0, current_icm - loss_icm)
        upside = max(0.0, gain_icm - current_icm)
        if downside <= 0:
            return 0.0
        bubble_bonus = self._fallback_pressure(game_state) * 0.35
        asymmetry = downside / max(downside + upside, 1e-9)
        return max(0.0, min(1.0, (asymmetry - 0.50) * 1.6 + bubble_bonus))

    def _payout_values(self, payouts):
        if isinstance(payouts, dict):
            return [float(value) for _place, value in sorted(payouts.items(), key=lambda item: int(item[0]))]
        if isinstance(payouts, list):
            return [float(value) for value in payouts]
        return []

    def _fallback_pressure(self, game_state):
        players_left = int(game_state.get("players_left", 0) or 0)
        paid_places = int(game_state.get("paid_places", 0) or 0)
        itm_distance = max(0.0, min(1.0, float(game_state.get("itm_distance", 1.0) or 0.0)))
        next_prize_gain = max(0.0, float(game_state.get("next_prize_gain_pct", 0.0) or 0.0))

        pressure = 0.0
        if paid_places > 0 and players_left > 0:
            if paid_places < players_left <= paid_places + 3:
                pressure += 0.55
            elif players_left > paid_places:
                pressure += max(0.0, 1.0 - itm_distance) * 0.25
            if players_left <= 10:
                pressure += 0.30
        pressure += min(0.40, next_prize_gain * 5.0)
        return max(0.0, min(1.0, pressure))
