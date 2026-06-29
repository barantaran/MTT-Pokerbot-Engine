from math import sqrt

from engine.player_interface import Bot
from engine.pokerstove_equity import estimate_equity


class EVReactionBot(Bot):
    """Minimal call/fold bot driven only by immediate call chip-EV sign."""

    def __init__(self, use_preflop_spot_range=False):
        super().__init__("EVReactionBot")
        self.use_preflop_spot_range = bool(use_preflop_spot_range)

    def get_action(self, game_state):
        hole_cards = game_state.get("hole_cards", [])
        board_cards = game_state.get("board_cards", [])
        pot_size = float(game_state.get("pot_size", 0) or 0)
        call_amount = float(game_state.get("call_amount", 0) or 0)
        active_players = int(game_state.get("active_players", 2) or 2)

        if "hero_equity" in game_state:
            equity = float(game_state["hero_equity"])
        else:
            equity = estimate_equity(
                hole_cards=hole_cards,
                board_cards=board_cards,
                active_players=active_players,
                opponent_range_pct=game_state.get("opponent_range_pct"),
                preflop_spot_type=game_state.get("preflop_spot_type"),
                use_preflop_spot_range=self.use_preflop_spot_range,
                table_stats=game_state.get("table_stats"),
            )

        call_ev = equity * (pot_size + call_amount) - call_amount
        if call_amount > 0 and call_ev <= 0.0:
            return ("fold", 0)
        return ("call", 0)


class EVInitiativeBot(EVReactionBot):
    """EV reaction bot that raises strong positive-EV spots for initiative."""

    def __init__(
        self,
        use_preflop_spot_range=False,
        *,
        call_margin=0.0,
        sb_call_margin=0.0,
        bb_call_margin=0.0,
        open_threshold_shift=0.0,
        raise_threshold_shift=0.0,
        raise_size_scale=1.0,
    ):
        super().__init__(use_preflop_spot_range=use_preflop_spot_range)
        self.name = "EVInitiativeBot"
        self.call_margin = float(call_margin)
        self.sb_call_margin = float(sb_call_margin)
        self.bb_call_margin = float(bb_call_margin)
        self.open_threshold_shift = float(open_threshold_shift)
        self.raise_threshold_shift = float(raise_threshold_shift)
        self.raise_size_scale = float(raise_size_scale)

    def get_action(self, game_state):
        hole_cards = game_state.get("hole_cards", [])
        board_cards = game_state.get("board_cards", [])
        pot_size = float(game_state.get("pot_size", 0) or 0)
        stack_size = float(game_state.get("stack_size", 0) or 0)
        call_amount = float(game_state.get("call_amount", 0) or 0)
        min_raise = int(game_state.get("min_raise", 0) or 0)
        active_players = int(game_state.get("active_players", 2) or 2)
        big_blind = float(game_state.get("blinds", {}).get("big", 1) or 1)
        position = str(game_state.get("position", "") or "").upper()

        if "hero_equity" in game_state:
            equity = float(game_state["hero_equity"])
        else:
            equity = estimate_equity(
                hole_cards=hole_cards,
                board_cards=board_cards,
                active_players=active_players,
                opponent_range_pct=game_state.get("opponent_range_pct"),
                preflop_spot_type=game_state.get("preflop_spot_type"),
                use_preflop_spot_range=self.use_preflop_spot_range,
                table_stats=game_state.get("table_stats"),
            )

        call_ev = equity * (pot_size + call_amount) - call_amount
        if call_amount > 0 and equity < self._required_call_equity(
            pot_size=pot_size,
            call_amount=call_amount,
            position=position,
            street_cards=len(board_cards or []),
        ):
            return ("fold", 0)

        street_cards = len(board_cards or [])
        stack_bb = stack_size / max(1.0, big_blind)
        max_raise_extra = max(0, int(stack_size - call_amount))
        can_raise = max_raise_extra >= max(1, min_raise)
        if not can_raise:
            if street_cards == 0 and call_amount == 0 and position != "BB":
                return ("fold", 0)
            return ("call", 0)

        raise_threshold = {
            0: 0.56,
            3: 0.61,
            4: 0.64,
            5: 0.67,
        }.get(street_cards, 0.62)
        raise_threshold += self.raise_threshold_shift
        if active_players > 2:
            raise_threshold += min(0.08, (active_players - 2) * 0.015)

        contested_pot = max(1.0, pot_size + call_amount)
        positive_ev_amount = max(0.0, call_ev) / contested_pot

        if street_cards == 0 and call_amount <= big_blind:
            open_threshold = self._preflop_open_threshold(position=position, active_players=active_players)
            if equity >= open_threshold:
                return ("raise", self._raise_size(
                    street_cards=street_cards,
                    pot_size=pot_size,
                    big_blind=big_blind,
                    min_raise=min_raise,
                    max_raise_extra=max_raise_extra,
                    positive_ev_amount=max(0.10, equity - open_threshold),
                ))
            if call_amount == 0 and position != "BB":
                return ("fold", 0)
            return ("call", 0)

        if stack_bb <= 10 and equity >= raise_threshold + 0.04 and positive_ev_amount >= 0.08:
            return ("raise", max_raise_extra)

        if equity >= raise_threshold and positive_ev_amount >= 0.05:
            return ("raise", self._raise_size(
                street_cards=street_cards,
                pot_size=pot_size,
                big_blind=big_blind,
                min_raise=min_raise,
                max_raise_extra=max_raise_extra,
                positive_ev_amount=positive_ev_amount,
            ))

        return ("call", 0)

    def _required_call_equity(self, *, pot_size, call_amount, position, street_cards):
        if call_amount <= 0:
            return 0.0
        required = call_amount / max(1.0, pot_size + call_amount)
        margin = self.call_margin
        if street_cards == 0 and position == "SB":
            margin += self.sb_call_margin
        elif street_cards == 0 and position == "BB":
            margin += self.bb_call_margin
        return max(0.0, min(1.0, required + margin))

    def _preflop_open_threshold(self, *, position, active_players):
        thresholds = {
            "UTG": 0.68,
            "UTG_1": 0.66,
            "UTG_2": 0.64,
            "LJ": 0.61,
            "HJ": 0.58,
            "CO": 0.54,
            "BTN": 0.50,
            "SB": 0.53,
            "BB": 0.55,
        }
        threshold = thresholds.get(position, 0.58)
        if active_players > 2:
            threshold += min(0.06, (active_players - 2) * 0.01)
        return min(0.90, max(0.20, threshold + self.open_threshold_shift))

    def _raise_size(self, *, street_cards, pot_size, big_blind, min_raise, max_raise_extra, positive_ev_amount):
        if positive_ev_amount >= 0.18:
            postflop_pot_fraction = 1.00
            preflop_open_bb = 3.2
        elif positive_ev_amount >= 0.10:
            postflop_pot_fraction = 0.65
            preflop_open_bb = 2.7
        else:
            postflop_pot_fraction = 0.40
            preflop_open_bb = 2.2

        if street_cards == 0:
            target = max(min_raise, int(preflop_open_bb * big_blind * self.raise_size_scale))
        else:
            target = max(min_raise, int(max(big_blind, pot_size * postflop_pot_fraction * self.raise_size_scale)))
        return min(max_raise_extra, target)


class EVFormulaBot(EVReactionBot):
    """Call/fold plus bet/raise EV using simple fold-equity approximation."""

    def __init__(
        self,
        use_preflop_spot_range=False,
        *,
        base_single_fold_prob=0.50,
        max_single_fold_prob=0.80,
        raise_ev_margin_fraction=0.10,
        raise_size_scale=1.0,
        use_tournament_chip_cost=True,
    ):
        super().__init__(use_preflop_spot_range=use_preflop_spot_range)
        self.name = "EVFormulaBot"
        self.base_single_fold_prob = max(0.01, min(0.99, float(base_single_fold_prob)))
        self.max_single_fold_prob = max(
            self.base_single_fold_prob,
            min(0.99, float(max_single_fold_prob)),
        )
        self.raise_ev_margin_fraction = max(0.0, float(raise_ev_margin_fraction))
        self.raise_size_scale = float(raise_size_scale)
        self.use_tournament_chip_cost = bool(use_tournament_chip_cost)

    def get_action(self, game_state):
        hole_cards = game_state.get("hole_cards", [])
        board_cards = game_state.get("board_cards", [])
        pot_size = float(game_state.get("pot_size", 0) or 0)
        stack_size = float(game_state.get("stack_size", 0) or 0)
        call_amount = float(game_state.get("call_amount", 0) or 0)
        min_raise = int(game_state.get("min_raise", 0) or 0)
        active_players = int(game_state.get("active_players", 2) or 2)
        big_blind = float(game_state.get("blinds", {}).get("big", 1) or 1)
        position = str(game_state.get("position", "") or "").upper()

        if "hero_equity" in game_state:
            equity = float(game_state["hero_equity"])
        else:
            equity = estimate_equity(
                hole_cards=hole_cards,
                board_cards=board_cards,
                active_players=active_players,
                opponent_range_pct=game_state.get("opponent_range_pct"),
                preflop_spot_type=game_state.get("preflop_spot_type"),
                use_preflop_spot_range=self.use_preflop_spot_range,
                table_stats=game_state.get("table_stats"),
            )

        call_ev = self._call_ev(
            equity=equity,
            pot_size=pot_size,
            call_amount=call_amount,
            stack_size=stack_size,
            big_blind=big_blind,
            game_state=game_state,
        )
        max_raise_extra = max(0, int(stack_size - call_amount))
        can_raise = max_raise_extra >= max(1, min_raise)

        best_raise_extra = 0
        best_raise_ev = float("-inf")
        if can_raise:
            for raise_extra in self._candidate_raise_extras(
                pot_size=pot_size,
                call_amount=call_amount,
                big_blind=big_blind,
                min_raise=min_raise,
                max_raise_extra=max_raise_extra,
            ):
                raise_ev = self._bet_raise_ev(
                    equity=equity,
                    pot_size=pot_size,
                    call_amount=call_amount,
                    raise_extra=raise_extra,
                    active_players=active_players,
                    stack_size=stack_size,
                    big_blind=big_blind,
                    game_state=game_state,
                )
                if raise_ev > best_raise_ev:
                    best_raise_ev = raise_ev
                    best_raise_extra = raise_extra

        passive_ev = call_ev if call_amount > 0 else 0.0
        raise_margin = self.raise_ev_margin_fraction * max(1.0, pot_size + call_amount)
        if best_raise_ev > max(0.0, passive_ev) + raise_margin:
            return ("raise", best_raise_extra)
        if call_amount > 0 and call_ev <= 0.0:
            return ("fold", 0)
        if len(board_cards or []) == 0 and call_amount == 0 and position != "BB":
            return ("fold", 0)
        return ("call", 0)

    def _fold_prob_all(self, *, active_players, pot_size, call_amount, raise_extra):
        opponents_left = max(1, int(active_players) - 1)
        pressure_ratio = raise_extra / max(1.0, pot_size + call_amount)
        pressure_gain = min(
            self.max_single_fold_prob - self.base_single_fold_prob,
            pressure_ratio * 0.18,
        )
        size_adjusted_single_fold_prob = self.base_single_fold_prob + pressure_gain
        size_adjusted_single_fold_prob = max(0.01, min(0.99, size_adjusted_single_fold_prob))
        return size_adjusted_single_fold_prob ** opponents_left

    def _call_ev(self, *, equity, pot_size, call_amount, stack_size=0.0, big_blind=1.0, game_state=None):
        if call_amount <= 0:
            return 0.0
        reward_multiplier, risk_multiplier = self._chip_value_multipliers(
            stack_size=stack_size,
            big_blind=big_blind,
            invested=call_amount,
            game_state=game_state,
        )
        return equity * pot_size * reward_multiplier - (1.0 - equity) * call_amount * risk_multiplier

    def _bet_raise_ev(
        self,
        *,
        equity,
        pot_size,
        call_amount,
        raise_extra,
        active_players,
        stack_size=0.0,
        big_blind=1.0,
        game_state=None,
    ):
        hero_invested = call_amount + raise_extra
        if hero_invested <= 0:
            return 0.0
        reward_multiplier, risk_multiplier = self._chip_value_multipliers(
            stack_size=stack_size,
            big_blind=big_blind,
            invested=hero_invested,
            game_state=game_state,
        )
        fold_prob_all = self._fold_prob_all(
            active_players=active_players,
            pot_size=pot_size,
            call_amount=call_amount,
            raise_extra=raise_extra,
        )
        final_pot = pot_size + call_amount + 2.0 * raise_extra
        win_profit = final_pot - hero_invested
        called_ev = (
            equity * win_profit * reward_multiplier
            - (1.0 - equity) * hero_invested * risk_multiplier
        )
        fold_win_ev = pot_size * reward_multiplier
        return fold_prob_all * fold_win_ev + (1.0 - fold_prob_all) * called_ev

    def _chip_value_multipliers(self, *, stack_size, big_blind, invested, game_state=None):
        if not self.use_tournament_chip_cost:
            return 1.0, 1.0

        stack_size = float(stack_size or 0.0)
        if stack_size <= 0:
            return 1.0, 1.0

        big_blind = max(1.0, float(big_blind or 1.0))
        stack_bb = stack_size / big_blind
        invested_fraction = max(0.0, min(1.0, float(invested or 0.0) / stack_size))

        reward_multiplier = 1.0 / sqrt(1.0 + stack_bb / 50.0)
        risk_multiplier = 1.0 + 1.5 * invested_fraction

        state = game_state or {}
        players_left = int(state.get("players_left", 0) or 0)
        paid_places = int(state.get("paid_places", 0) or 0)
        next_prize_gain = max(0.0, float(state.get("next_prize_gain_pct", 0.0) or 0.0))

        pressure = 1.0
        if paid_places > 0 and players_left > 0:
            if paid_places < players_left <= paid_places + 9:
                pressure += (paid_places + 9 - players_left) / 18.0
            if players_left <= 10:
                pressure += (10 - players_left) / 9.0
        pressure += min(0.75, next_prize_gain * 4.0)

        return reward_multiplier, risk_multiplier * pressure

    def bet_raise_ev_feature(
        self,
        *,
        equity,
        pot_size,
        call_amount,
        raise_extra,
        active_players,
        stack_size=0.0,
        big_blind=1.0,
        game_state=None,
    ):
        hero_invested = call_amount + raise_extra
        if hero_invested <= 0:
            return 0.0
        normalized = self._bet_raise_ev(
            equity=equity,
            pot_size=pot_size,
            call_amount=call_amount,
            raise_extra=raise_extra,
            active_players=active_players,
            stack_size=stack_size,
            big_blind=big_blind,
            game_state=game_state,
        ) / hero_invested
        return max(-1.0, min(1.0, normalized))

    def _candidate_raise_extras(self, *, pot_size, call_amount, big_blind, min_raise, max_raise_extra):
        del call_amount
        raw_sizes = [
            min_raise,
            int(max(big_blind, pot_size * 0.33) * self.raise_size_scale),
            int(max(big_blind, pot_size * 0.50) * self.raise_size_scale),
            int(max(big_blind, pot_size * 1.00) * self.raise_size_scale),
            int(max(big_blind, pot_size * 1.50) * self.raise_size_scale),
            max_raise_extra,
        ]
        candidates = sorted({
            max(1, min(max_raise_extra, int(size)))
            for size in raw_sizes
            if int(size) >= min_raise
        })
        return candidates or [max(1, min(max_raise_extra, min_raise))]
