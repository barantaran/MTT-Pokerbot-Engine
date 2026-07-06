from engine.player_interface import Bot
from engine.pokerstove_equity import estimate_equity, pot_odds


class StickyCallStationBot(Bot):
    """
    Equity-aware call station built to punish bluff-heavy strategies.
    Never raises. Preflop it folds only true trash against raises; postflop it
    calls with any reasonable piece of the board regardless of price, so bluffs
    burn chips against it while value bets get paid at the same rate as bluffs.
    """

    def __init__(
        self,
        equity_estimator=None,
        *,
        preflop_fold_equity=0.28,
        postflop_fold_equity=0.22,
        big_bet_pot_ratio=1.0,
    ):
        super().__init__("StickyCallStationBot")
        self.equity_estimator = equity_estimator
        self.preflop_fold_equity = float(preflop_fold_equity)
        self.postflop_fold_equity = float(postflop_fold_equity)
        self.big_bet_pot_ratio = float(big_bet_pot_ratio)

    def get_action(self, game_state):
        hole_cards = game_state.get("hole_cards", [])
        board_cards = game_state.get("board_cards", [])
        pot_size = game_state.get("pot_size", 0)
        stack_size = game_state.get("stack_size", 0)
        call_amount = game_state.get("call_amount", 0)
        active_players = game_state.get("active_players", 2)
        big_blind = game_state.get("blinds", {}).get("big", 1) or 1

        if call_amount <= 0:
            return ("call", 0)

        estimator = self.equity_estimator or estimate_equity
        if self.equity_estimator is None and "hero_equity" in game_state:
            equity = float(game_state["hero_equity"])
        else:
            equity = estimator(
                hole_cards=hole_cards,
                board_cards=board_cards,
                active_players=active_players,
                opponent_range_pct=game_state.get("opponent_range_pct"),
                preflop_spot_type=game_state.get("preflop_spot_type"),
                table_stats=game_state.get("table_stats"),
            )

        street = len(board_cards)

        if street == 0:
            # Limps and normal raises: fold only true trash. Blind completes
            # (tiny price) always continue.
            if call_amount <= big_blind:
                return ("call", 0)
            if equity < self.preflop_fold_equity:
                return ("fold", 0)
            return ("call", 0)

        # Postflop: call any bet with a piece. Fold pure air only when the bet
        # is big relative to the pot AND even raw pot odds are hopeless.
        if equity >= self.postflop_fold_equity:
            return ("call", 0)
        if call_amount <= pot_size * self.big_bet_pot_ratio * 0.5:
            return ("call", 0)
        required_equity = pot_odds(call_amount, pot_size)
        if equity >= required_equity:
            return ("call", 0)
        return ("fold", 0)
