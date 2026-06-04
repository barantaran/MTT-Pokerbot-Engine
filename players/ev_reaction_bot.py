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
