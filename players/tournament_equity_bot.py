from __future__ import annotations

from engine.bot_tools import DecisionContext, build_bot_tools
from engine.icm import calculate_exact_icm
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

    def __init__(
        self,
        *,
        use_preflop_spot_range: bool = True,
        range_profile: str = "legacy",
        range_influence=1.0,
        player_range_sampling: bool = True,
        player_range_sample_config=None,
        raise_sizing: str = "legacy",
        pot_size_buckets=(0.33, 0.50, 0.75, 1.00, 1.25),
        pot_bucket_edge_step: float = 0.05,
        pot_bucket_all_in_spr: float = 1.25,
        tools=None,
    ):
        super().__init__("TournamentEquityBot")
        self.use_preflop_spot_range = bool(use_preflop_spot_range)
        self.range_profile = str(range_profile or "legacy")
        self.range_influence = range_influence
        self.player_range_sampling = bool(player_range_sampling)
        self.player_range_sample_config = player_range_sample_config
        self.raise_sizing = str(raise_sizing or "legacy")
        self.pot_size_buckets = tuple(float(bucket) for bucket in pot_size_buckets)
        self.pot_bucket_edge_step = max(0.01, float(pot_bucket_edge_step))
        self.pot_bucket_all_in_spr = max(0.0, float(pot_bucket_all_in_spr))
        self.tools = build_bot_tools(tools)

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
                table_stats=game_state.get("table_stats"),
                opponent_stats=game_state.get("opponent_stats"),
                opponent_position=game_state.get("opponent_position"),
                opponent_stack_bb=game_state.get("opponent_stack_bb"),
                range_profile=self.range_profile,
                position=position,
                stack_bb=stack_bb,
                players_left=game_state.get("players_left"),
                starting_field=game_state.get("starting_field"),
                paid_places=game_state.get("paid_places"),
                itm_distance=game_state.get("itm_distance"),
                range_influence=self.range_influence,
                player_range_sampling=self.player_range_sampling,
                player_range_sample_config=self.player_range_sample_config,
            )

        payout_pressure = self._payout_pressure(game_state)
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
        context = self._apply_tools(
            DecisionContext(
                equity=equity,
                required_equity=required_equity,
                call_margin=call_margin,
                raise_threshold=raise_threshold,
                jam_threshold=jam_threshold,
                street=street,
                active_players=active_players,
                stack_bb=stack_bb,
                spot_type=spot_type,
                position=position,
                call_amount=call_amount,
                stack_size=stack_size,
                min_raise=min_raise,
                big_blind=big_blind,
                payout_pressure=payout_pressure,
                open_size=self._open_size(
                    street=street,
                    big_blind=big_blind,
                    min_raise=min_raise,
                    spot_type=spot_type,
                    position=position,
                ),
                pot_size=pot_size,
                max_raise_extra=max_raise_extra,
            ),
            game_state,
        )
        equity = context.equity
        required_equity = context.required_equity
        call_margin = context.call_margin
        raise_threshold = context.raise_threshold
        jam_threshold = context.jam_threshold

        if context.tool_event is not None:
            game_state["_bot_tool_event"] = dict(context.tool_event)
        if context.tool_events:
            game_state["_bot_tool_events"] = [dict(item) for item in context.tool_events]

        if context.forced_action is not None:
            return context.forced_action

        if can_raise and self._should_jam(
            equity=equity,
            jam_threshold=jam_threshold,
            stack_bb=stack_bb,
            stack_size=stack_size,
            call_amount=call_amount,
        ):
            return ("raise", max_raise_extra)

        if can_raise and self._should_sizing_jam(
            street=street,
            equity=equity,
            jam_threshold=jam_threshold,
            pot_size=pot_size,
            max_raise_extra=max_raise_extra,
        ):
            return ("raise", max_raise_extra)

        if call_amount > 0 and equity < required_equity + call_margin:
            return ("fold", 0)

        if can_raise and equity >= raise_threshold:
            return ("raise", self._raise_size(
                street=street,
                pot_size=pot_size,
                big_blind=big_blind,
                min_raise=min_raise,
                max_raise_extra=max_raise_extra,
                spot_type=spot_type,
                position=position,
                equity=equity,
                raise_threshold=raise_threshold,
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

    def _should_sizing_jam(self, *, street, equity, jam_threshold, pot_size, max_raise_extra):
        if self.raise_sizing not in {"pot_buckets", "postflop_pot_buckets"}:
            return False
        if street == 0 or pot_size <= 0 or max_raise_extra <= 0:
            return False
        if equity < jam_threshold:
            return False
        return max_raise_extra <= pot_size * self.pot_bucket_all_in_spr

    def _raise_size(
        self,
        *,
        street,
        pot_size,
        big_blind,
        min_raise,
        max_raise_extra,
        spot_type,
        position,
        equity=None,
        raise_threshold=None,
    ):
        use_pot_buckets = self.raise_sizing == "pot_buckets" or (
            self.raise_sizing == "postflop_pot_buckets" and street > 0
        )
        if use_pot_buckets:
            target = self._pot_bucket_raise_size(
                pot_size=pot_size,
                big_blind=big_blind,
                min_raise=min_raise,
                equity=equity,
                raise_threshold=raise_threshold,
            )
            return min(max_raise_extra, target)
        if street == 0:
            target = self._open_size(
                street=street,
                big_blind=big_blind,
                min_raise=min_raise,
                spot_type=spot_type,
                position=position,
            )
        else:
            target = max(min_raise, int(max(big_blind, pot_size * 0.60)))
        return min(max_raise_extra, target)

    def _pot_bucket_raise_size(self, *, pot_size, big_blind, min_raise, equity, raise_threshold):
        buckets = self.pot_size_buckets or (0.33, 0.50, 0.75, 1.00, 1.25)
        if raise_threshold is None or equity is None:
            bucket = buckets[0]
        else:
            edge = max(0.0, float(equity) - float(raise_threshold))
            bucket_index = min(len(buckets) - 1, int(edge / self.pot_bucket_edge_step))
            bucket = buckets[bucket_index]
        target = int(max(big_blind, pot_size * float(bucket)))
        return max(int(min_raise or 0), target)

    def _apply_tools(self, context, game_state):
        for tool in self.tools:
            context = tool.apply(context, game_state)
        return context

    def _open_size(self, *, street, big_blind, min_raise, spot_type, position):
        if street != 0:
            return max(0, int(min_raise or 0))
        open_bb = self._preflop_open_bb(position=position, spot_type=spot_type)
        return max(int(min_raise or 0), int(open_bb * big_blind))

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


class TournamentEquityBotV2(TournamentEquityBot):
    """
    TournamentEquityBot with one targeted adjustment: tighter preflop reraises.

    The previous version had reasonable VPIP but elevated 3bet frequency. This
    variant leaves calling, opening, postflop play, and jam rules unchanged, and
    only raises the value threshold when reraising over an existing preflop open.
    """

    def __init__(
        self,
        *,
        use_preflop_spot_range: bool = True,
        range_profile: str = "legacy",
        range_influence=1.0,
        player_range_sampling: bool = True,
        player_range_sample_config=None,
        raise_sizing: str = "legacy",
        pot_size_buckets=(0.33, 0.50, 0.75, 1.00, 1.25),
        pot_bucket_edge_step: float = 0.05,
        pot_bucket_all_in_spr: float = 1.25,
        preflop_reraise_tightness: float = 0.08,
        tools=None,
    ):
        super().__init__(
            use_preflop_spot_range=use_preflop_spot_range,
            range_profile=range_profile,
            range_influence=range_influence,
            player_range_sampling=player_range_sampling,
            player_range_sample_config=player_range_sample_config,
            raise_sizing=raise_sizing,
            pot_size_buckets=pot_size_buckets,
            pot_bucket_edge_step=pot_bucket_edge_step,
            pot_bucket_all_in_spr=pot_bucket_all_in_spr,
            tools=tools,
        )
        self.name = "TournamentEquityBotV2"
        self.preflop_reraise_tightness = float(preflop_reraise_tightness)

    def _raise_threshold(self, *, street, active_players, stack_bb, game_state, spot_type, position):
        threshold = super()._raise_threshold(
            street=street,
            active_players=active_players,
            stack_bb=stack_bb,
            game_state=game_state,
            spot_type=spot_type,
            position=position,
        )
        call_amount = int(game_state.get("call_amount", 0) or 0)
        if street == 0 and call_amount > 0 and spot_type in {"srp", "single_raised"}:
            threshold += self.preflop_reraise_tightness
        return max(0.36, min(0.90, threshold))


class TournamentICMEquityBot(TournamentEquityBot):
    """
    TournamentEquityBot with exact ICM pressure when full-table payout state is visible.

    This keeps TournamentEquityBot's call, raise, jam, position, and sizing logic.
    The only behavioral change is replacing heuristic payout pressure with exact
    ICM pressure once the remaining field is represented by the current table.
    """

    def __init__(
        self,
        *,
        use_preflop_spot_range: bool = True,
        range_profile: str = "legacy",
        range_influence=1.0,
        player_range_sampling: bool = True,
        player_range_sample_config=None,
        raise_sizing: str = "legacy",
        pot_size_buckets=(0.33, 0.50, 0.75, 1.00, 1.25),
        pot_bucket_edge_step: float = 0.05,
        pot_bucket_all_in_spr: float = 1.25,
        use_icm: bool = True,
        icm_strength: float = 1.0,
        tools=None,
    ):
        super().__init__(
            use_preflop_spot_range=use_preflop_spot_range,
            range_profile=range_profile,
            range_influence=range_influence,
            player_range_sampling=player_range_sampling,
            player_range_sample_config=player_range_sample_config,
            raise_sizing=raise_sizing,
            pot_size_buckets=pot_size_buckets,
            pot_bucket_edge_step=pot_bucket_edge_step,
            pot_bucket_all_in_spr=pot_bucket_all_in_spr,
            tools=tools,
        )
        self.name = "TournamentICMEquityBot"
        self.use_icm = bool(use_icm)
        self.icm_strength = max(0.0, float(icm_strength))

    def _payout_pressure(self, game_state):
        if not self.use_icm or self.icm_strength <= 0:
            return super()._payout_pressure(game_state)

        exact_pressure = self._exact_icm_pressure(game_state)
        if exact_pressure is not None:
            return min(1.0, exact_pressure * self.icm_strength)
        return min(1.0, self._fallback_icm_pressure(game_state) * self.icm_strength)

    def _exact_icm_pressure(self, game_state):
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
            stacks = [float(stack) for stack in table_stacks]
            payout_values = self._payout_values(payouts)
            hero_stack = max(0.0, float(game_state.get("stack_size", 0) or 0.0))
        except (TypeError, ValueError):
            return None
        if hero_index < 0 or hero_index >= len(stacks) or hero_stack <= 0 or not payout_values:
            return None

        total_chips = sum(stacks)
        if total_chips <= 0:
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

        bubble_bonus = super()._payout_pressure(game_state) * 0.30
        asymmetry = downside / max(downside + upside, 1e-9)
        return max(0.0, min(1.0, (asymmetry - 0.50) * 1.45 + bubble_bonus))

    def _payout_values(self, payouts):
        if isinstance(payouts, dict):
            return [float(value) for _place, value in sorted(payouts.items(), key=lambda item: int(item[0]))]
        if isinstance(payouts, list):
            return [float(value) for value in payouts]
        return []

    def _fallback_icm_pressure(self, game_state):
        heuristic = super()._payout_pressure(game_state)
        itm_distance = max(0.0, min(1.0, float(game_state.get("itm_distance", 1.0) or 0.0)))
        players_left = int(game_state.get("players_left", 0) or 0)
        paid_places = int(game_state.get("paid_places", 0) or 0)

        pressure = heuristic
        if paid_places > 0 and players_left > paid_places:
            pressure += max(0.0, 1.0 - itm_distance) * 0.20
        return max(0.0, min(1.0, pressure))


class AdaptiveTournamentICMEquityBot(TournamentICMEquityBot):
    """
    ICM tournament bot that adjusts modestly to public table behavior.

    The bot uses aggregate table stats only. Low-sample tables behave like the
    base TournamentICMEquityBot.
    """

    def __init__(
        self,
        *,
        use_preflop_spot_range: bool = True,
        range_profile: str = "legacy",
        range_influence=1.0,
        player_range_sampling: bool = True,
        player_range_sample_config=None,
        raise_sizing: str = "legacy",
        pot_size_buckets=(0.33, 0.50, 0.75, 1.00, 1.25),
        pot_bucket_edge_step: float = 0.05,
        pot_bucket_all_in_spr: float = 1.25,
        use_icm: bool = True,
        icm_strength: float = 1.0,
        tools=None,
    ):
        super().__init__(
            use_preflop_spot_range=use_preflop_spot_range,
            range_profile=range_profile,
            range_influence=range_influence,
            player_range_sampling=player_range_sampling,
            player_range_sample_config=player_range_sample_config,
            raise_sizing=raise_sizing,
            pot_size_buckets=pot_size_buckets,
            pot_bucket_edge_step=pot_bucket_edge_step,
            pot_bucket_all_in_spr=pot_bucket_all_in_spr,
            use_icm=use_icm,
            icm_strength=icm_strength,
            tools=tools,
        )
        self.name = "AdaptiveTournamentICMEquityBot"

    def _call_margin(self, *, street, active_players, stack_bb, game_state, spot_type, position):
        margin = super()._call_margin(
            street=street,
            active_players=active_players,
            stack_bb=stack_bb,
            game_state=game_state,
            spot_type=spot_type,
            position=position,
        )
        stats = self._table_stats(game_state)
        if not stats:
            return margin

        quality = stats["sample_quality"]
        pressure_spot = spot_type in {"srp", "single_raised", "three_bet", "3bet", "four_bet", "4bet", "all_in_pressure"}
        loose_passive = max(0.0, stats["vpip"] - 0.42) * max(0.0, 0.16 - stats["pfr"])
        aggressive = max(0.0, stats["three_bet_rate"] - 0.20)

        margin -= min(0.010, loose_passive * 0.20) * quality
        if pressure_spot:
            margin += min(0.020, aggressive * 0.20) * quality
        if street == 0 and position in {"BTN", "CO", "BB"} and stats["vpip"] >= 0.44 and stats["pfr"] <= 0.14:
            margin -= 0.004 * quality
        return max(-0.02, min(0.25, margin))

    def _raise_threshold(self, *, street, active_players, stack_bb, game_state, spot_type, position):
        threshold = super()._raise_threshold(
            street=street,
            active_players=active_players,
            stack_bb=stack_bb,
            game_state=game_state,
            spot_type=spot_type,
            position=position,
        )
        stats = self._table_stats(game_state)
        if not stats:
            return threshold

        quality = stats["sample_quality"]
        call_amount = int(game_state.get("call_amount", 0) or 0)
        raised_preflop = street == 0 and call_amount > 0
        loose_passive = stats["vpip"] >= 0.44 and stats["pfr"] <= 0.14
        aggressive = stats["three_bet_rate"] >= 0.22 or stats["pfr"] >= 0.32

        if loose_passive:
            threshold -= 0.010 * quality
        if raised_preflop and aggressive:
            threshold += 0.030 * quality
        return max(0.36, min(0.90, threshold))

    def _table_stats(self, game_state):
        stats = game_state.get("table_stats")
        if not isinstance(stats, dict):
            return None
        try:
            quality = max(0.0, min(1.0, float(stats.get("sample_quality", 0.0) or 0.0)))
            if quality < 0.50:
                return None
            return {
                "sample_quality": quality,
                "vpip": max(0.0, min(1.0, float(stats.get("vpip", 0.0) or 0.0))),
                "pfr": max(0.0, min(1.0, float(stats.get("pfr", 0.0) or 0.0))),
                "three_bet_rate": max(0.0, min(1.0, float(stats.get("three_bet_rate", 0.0) or 0.0))),
            }
        except (TypeError, ValueError):
            return None


class ConfiguredTournamentEquityBot(TournamentEquityBot):
    """
    Tournament equity shell assembled from explicit decision tools.

    This class keeps the base equity, position, sizing, and jam mechanics, but
    turns tournament pressure and table exploits into a configurable tool stack.
    It intentionally does not load a hidden default tool set; callers must pass
    every enabled tool explicitly.
    """

    DEFAULT_TOOLS = (
        {"type": "icm_pressure", "priority": 20, "strength": 1.0, "exact_when_available": True},
        {"type": "preflop_reraise_tightness", "priority": 30, "tightness": 0.08},
        {"type": "table_adaptation", "priority": 35, "sample_quality_min": 0.50},
        {
            "type": "button_steal",
            "priority": 40,
            "sample_quality_min": 0.50,
            "max_vpip": 0.30,
            "max_pfr": 0.16,
            "max_three_bet_rate": 0.08,
            "base_discount": 0.018,
            "tightness_multiplier": 0.12,
            "max_discount": 0.040,
            "requires_table_sample": True,
        },
    )

    def __init__(
        self,
        *,
        use_preflop_spot_range: bool = True,
        range_profile: str = "legacy",
        range_influence=1.0,
        player_range_sampling: bool = True,
        player_range_sample_config=None,
        raise_sizing: str = "legacy",
        pot_size_buckets=(0.33, 0.50, 0.75, 1.00, 1.25),
        pot_bucket_edge_step: float = 0.05,
        pot_bucket_all_in_spr: float = 1.25,
        tools=None,
    ):
        super().__init__(
            use_preflop_spot_range=use_preflop_spot_range,
            range_profile=range_profile,
            range_influence=range_influence,
            player_range_sampling=player_range_sampling,
            player_range_sample_config=player_range_sample_config,
            raise_sizing=raise_sizing,
            pot_size_buckets=pot_size_buckets,
            pot_bucket_edge_step=pot_bucket_edge_step,
            pot_bucket_all_in_spr=pot_bucket_all_in_spr,
            tools=[] if tools is None else tools,
        )
        self.name = "ConfiguredTournamentEquityBot"

    def _payout_pressure(self, game_state):
        return 0.0


class ButtonStealTournamentICMEquityBot(TournamentICMEquityBot):
    """
    ICM tournament bot with one exploit: wider unopened button steals.

    This variant keeps the base ICM discipline everywhere else. Its default
    tool stack lowers the preflop raise threshold only when the pot is unopened,
    hero is on the button, the table is tight/passive enough to imply blind
    overfolding, and the open size risks only a small share of the stack.
    """

    DEFAULT_TOOLS = (
        {
            "type": "button_steal",
            "priority": 40,
            "sample_quality_min": 0.50,
            "max_vpip": 0.30,
            "max_pfr": 0.16,
            "max_three_bet_rate": 0.08,
            "base_discount": 0.018,
            "tightness_multiplier": 0.12,
            "max_discount": 0.040,
            "requires_table_sample": True,
        },
    )

    def __init__(
        self,
        *,
        use_preflop_spot_range: bool = True,
        range_profile: str = "legacy",
        range_influence=1.0,
        player_range_sampling: bool = True,
        player_range_sample_config=None,
        raise_sizing: str = "legacy",
        pot_size_buckets=(0.33, 0.50, 0.75, 1.00, 1.25),
        pot_bucket_edge_step: float = 0.05,
        pot_bucket_all_in_spr: float = 1.25,
        use_icm: bool = True,
        icm_strength: float = 1.0,
        tools=None,
    ):
        super().__init__(
            use_preflop_spot_range=use_preflop_spot_range,
            range_profile=range_profile,
            range_influence=range_influence,
            player_range_sampling=player_range_sampling,
            player_range_sample_config=player_range_sample_config,
            raise_sizing=raise_sizing,
            pot_size_buckets=pot_size_buckets,
            pot_bucket_edge_step=pot_bucket_edge_step,
            pot_bucket_all_in_spr=pot_bucket_all_in_spr,
            use_icm=use_icm,
            icm_strength=icm_strength,
            tools=self.DEFAULT_TOOLS if tools is None else tools,
        )
        self.name = "ButtonStealTournamentICMEquityBot"
