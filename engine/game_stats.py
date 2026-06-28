from __future__ import annotations

from typing import Any, Dict, Iterable, Set, Tuple


HandKey = Tuple[int, int, int]
PlayerHandKey = Tuple[int, int, int, str]


class TableStatsTracker:
    """Tracks public aggregate behavior for one table."""

    def __init__(self) -> None:
        self._player_hands: Set[PlayerHandKey] = set()
        self._vpip_hands: Set[PlayerHandKey] = set()
        self._pfr_hands: Set[PlayerHandKey] = set()
        self._preflop_call_hands: Set[PlayerHandKey] = set()
        self._three_bet_opportunity_hands: Set[PlayerHandKey] = set()
        self._three_bet_hands: Set[PlayerHandKey] = set()
        self._preflop_raise_count_by_hand: Dict[HandKey, int] = {}
        self._preflop_opener_by_hand: Dict[HandKey, str] = {}
        self._postflop_raise_count = 0
        self._postflop_call_count = 0
        self._action_total = 0

    def observe_event(self, event: Dict[str, Any]) -> None:
        event_type = str(event.get("type", "") or "")
        if event_type == "hand_start":
            self.observe_hand_start(event)
        elif event_type == "action":
            self.observe_action(event)

    def observe_hand_start(self, event: Dict[str, Any]) -> None:
        hand_key = self._hand_key(event)
        for player in self._players_from_event(event):
            self._player_hands.add((*hand_key, player))

    def observe_action(self, event: Dict[str, Any]) -> None:
        hand_key = self._hand_key(event)
        player = str(event.get("player", "") or "")
        if not player:
            return

        player_hand_key = (*hand_key, player)
        action = str(event.get("action", "unknown") or "unknown")
        amount = int(event.get("amount", 0) or 0)
        if action == "call" and amount == 0:
            action = "check"
        street = str(event.get("street", "") or "")

        self._action_total += 1
        if street == "preflop":
            self._observe_preflop_action(event, hand_key, player, player_hand_key, action, amount)
        elif street in {"flop", "turn", "river"}:
            if action == "raise" and amount > 0:
                self._postflop_raise_count += 1
            elif action == "call" and amount > 0:
                self._postflop_call_count += 1

    def snapshot(self) -> Dict[str, float]:
        player_hands = len(self._player_hands)
        three_bet_opportunities = len(self._three_bet_opportunity_hands)
        postflop_calls = self._postflop_call_count

        return {
            "hands_observed": len({key[:3] for key in self._player_hands}),
            "player_hands_observed": player_hands,
            "action_total": self._action_total,
            "vpip": self._rate(len(self._vpip_hands), player_hands),
            "pfr": self._rate(len(self._pfr_hands), player_hands),
            "preflop_call_rate": self._rate(len(self._preflop_call_hands), player_hands),
            "three_bet_rate": self._rate(len(self._three_bet_hands), three_bet_opportunities),
            "postflop_aggression_factor": (
                float(self._postflop_raise_count) / postflop_calls
                if postflop_calls > 0
                else float(self._postflop_raise_count)
            ),
            "sample_quality": min(1.0, player_hands / 45.0),
        }

    def _observe_preflop_action(
        self,
        event: Dict[str, Any],
        hand_key: HandKey,
        player: str,
        player_hand_key: PlayerHandKey,
        action: str,
        amount: int,
    ) -> None:
        if action in {"call", "raise"} and amount > 0:
            self._vpip_hands.add(player_hand_key)
            if action == "raise":
                self._pfr_hands.add(player_hand_key)
            elif action == "call":
                self._preflop_call_hands.add(player_hand_key)

        raise_count_before = int(self._preflop_raise_count_by_hand.get(hand_key, 0))
        opener = self._preflop_opener_by_hand.get(hand_key)
        call_amount = int(event.get("call_amount", 0) or 0)
        if raise_count_before == 1 and call_amount > 0 and player != opener:
            self._three_bet_opportunity_hands.add(player_hand_key)
        if action == "raise" and amount > 0:
            if raise_count_before == 0:
                self._preflop_opener_by_hand[hand_key] = player
            elif raise_count_before == 1:
                self._three_bet_hands.add(player_hand_key)
            self._preflop_raise_count_by_hand[hand_key] = raise_count_before + 1

    def _hand_key(self, event: Dict[str, Any]) -> HandKey:
        return (
            int(event.get("tournament_id", 0) or 0),
            int(event.get("table_id", 0) or 0),
            int(event.get("hand_id", 0) or 0),
        )

    def _players_from_event(self, event: Dict[str, Any]) -> Iterable[str]:
        for player in event.get("players", []) or []:
            if isinstance(player, dict):
                name = str(player.get("name", "") or "")
            else:
                name = str(player or "")
            if name:
                yield name

    def _rate(self, numerator: int, denominator: int) -> float:
        return float(numerator) / denominator if denominator > 0 else 0.0
