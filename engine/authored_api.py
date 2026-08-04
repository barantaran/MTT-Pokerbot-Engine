"""Card-math capability injected into self-service authored bots.

An authored bot is a bare ``get_action(state, api)`` function that imports
nothing native (see ``docs/BOT_ARCHITECTURE.md``). ``api`` is the one capability
the seat is handed: seeded, deterministic access to pokerstove's 7-card hand
evaluator. One ``AuthoredApi`` is built per seat process (card math is stateless)
and injected through the ``_AuthoredBot`` adapter.

It lives *inside* the seat worker, not across the pipe (``engine/seat_worker.py``):
``equity(..., n=...)`` is uncapped, so answering it in the parent would let a seat
burn the CPU of the very process enforcing its deadline.

Surface (raw primitives plus one convenience):

    api.showdown(hero7, villain7) -> 1.0 / 0.5 / 0.0   fixed 7-card matchup
    api.deal(hole, board, villain_hands, n) -> iterator of (hero7, [villain7,...])
    api.equity(hole, board, villain_hands, n) -> float 0-1   field share averaged

Cards are strings like ``"As"``, ``"Kd"``. ``hole`` is 2 cards, ``board`` is
0-5, ``villain_hands`` is a list whose entries are either a 2-card hand (known)
or ``None`` (unknown -> sampled at random from the remaining deck each runout).
``deal`` / ``equity`` seed their RNG from the call arguments, so the same spot
yields the same estimate in every worker and on every SPOT resume.
"""

import hashlib
import random

from engine.pokerstove_equity import (
    _CARD_SET,
    _FULL_DECK,
    _HOLD_EM_EVALUATOR,
)


def _rank(seven_cards):
    """pokerstove eval score of a 7-card hand (2 hole + 5 board). Higher wins."""
    if len(seven_cards) != 7:
        raise ValueError(f"expected 7 cards, got {len(seven_cards)}: {seven_cards}")
    cards = _CARD_SET("".join(seven_cards))
    return _HOLD_EM_EVALUATOR.evaluateHand(cards, _CARD_SET("")).eval()


def _seed(hole, board, villain_hands, n):
    key = repr((tuple(hole), tuple(board), tuple(map(_as_tuple, villain_hands)), n))
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _as_tuple(hand):
    return None if hand is None else tuple(hand)


class AuthoredApi:
    """Injected card math. Stateless — one instance is reused across tournaments."""

    def showdown(self, hero7, villain7):
        """Heads-up 7-card matchup. 1.0 hero wins, 0.5 tie, 0.0 hero loses."""
        hero = _rank(hero7)
        villain = _rank(villain7)
        if hero > villain:
            return 1.0
        if hero < villain:
            return 0.0
        return 0.5

    def deal(self, hole, board, villain_hands, n):
        """Yield ``n`` seeded runouts as ``(hero7, [villain7, ...])``.

        The board is completed to 5 cards and any ``None`` villain hand is drawn
        at random from the remaining deck, each runout using its own draw. Every
        returned hand is a concrete 7-card list ready for ``showdown``.
        """
        hole = list(hole)
        board = list(board)
        if len(hole) != 2:
            raise ValueError(f"hole must be 2 cards, got {hole}")
        if len(board) > 5:
            raise ValueError(f"board must be 0-5 cards, got {board}")

        rng = random.Random(_seed(hole, board, villain_hands, n))
        missing_board = 5 - len(board)

        for _ in range(n):
            # Dead cards: hero hole, current board, all known villain hands.
            dead = set(hole) | set(board)
            for hand in villain_hands:
                if hand is not None:
                    dead.update(hand)
            deck = [c for c in _FULL_DECK if c not in dead]

            # One draw covers the board runout plus every unknown villain hand.
            unknown = sum(1 for hand in villain_hands if hand is None)
            drawn = rng.sample(deck, missing_board + 2 * unknown)

            board5 = board + drawn[:missing_board]
            cursor = missing_board
            villains7 = []
            for hand in villain_hands:
                if hand is None:
                    vh = drawn[cursor : cursor + 2]
                    cursor += 2
                else:
                    vh = list(hand)
                villains7.append(vh + board5)
            yield (hole + board5, villains7)

    def equity(self, hole, board, villain_hands, n):
        """Hero pot share vs the whole field, averaged over ``n`` seeded runouts.

        A runout counts as a full win for hero only when hero beats every
        villain; ties split the pot among the winners. Convenience for the
        common case — an author wanting a custom estimator uses ``deal`` and
        ``showdown`` directly.
        """
        if not villain_hands:
            return 0.0
        total = 0.0
        runs = 0
        for hero7, villains7 in self.deal(hole, board, villain_hands, n):
            hero = _rank(hero7)
            best_villain = max(_rank(v7) for v7 in villains7)
            if hero > best_villain:
                total += 1.0
            elif hero == best_villain:
                winners = 1 + sum(1 for v7 in villains7 if _rank(v7) == hero)
                total += 1.0 / winners
            runs += 1
        return total / runs if runs else 0.0
