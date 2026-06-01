import importlib
import itertools
import random
import sys
import tempfile
import zipfile
from functools import lru_cache
from pathlib import Path

from treys import Card


_FULL_DECK = tuple(
    f"{rank}{suit}"
    for rank in "23456789TJQKA"
    for suit in "cdhs"
)


def _load_pokerstove_module():
    try:
        module = importlib.import_module("pokerstove")
        if hasattr(module, "CardSet") and hasattr(module, "PokerHandEvaluator"):
            return module
    except Exception:
        pass

    project_root = Path(__file__).resolve().parents[2]
    dist_dir = project_root / "pokerstove" / "dist"
    wheel_paths = sorted(dist_dir.glob("pokerstove-*.whl"), reverse=True)

    for wheel_path in wheel_paths:
        extract_dir = Path(tempfile.gettempdir()) / f"pokerstove-wheel-{wheel_path.stat().st_mtime_ns}"
        if not (extract_dir / "pokerstove.py").exists():
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(wheel_path) as wheel:
                wheel.extractall(extract_dir)

        sys.modules.pop("pokerstove", None)
        sys.modules.pop("_pokerstove", None)
        if str(extract_dir) not in sys.path:
            sys.path.insert(0, str(extract_dir))

        module = importlib.import_module("pokerstove")
        if hasattr(module, "CardSet") and hasattr(module, "PokerHandEvaluator"):
            return module

    raise ImportError(
        "Unable to load pokerstove. Install the wheel or keep a built wheel in pokerstove/dist/."
    )


_POKERSTOVE = _load_pokerstove_module()
_CARD_SET = _POKERSTOVE.CardSet
_HOLD_EM_EVALUATOR = _POKERSTOVE.PokerHandEvaluator.alloc("h")

_RANK_VALUE = {rank: index for index, rank in enumerate("23456789TJQKA", start=2)}
_PREFLOP_SPOT_RANGE_PCT = {
    "unknown": 1.00,
    "limped": 0.55,
    "srp": 0.35,
    "single_raised": 0.35,
    "three_bet": 0.16,
    "3bet": 0.16,
    "four_bet": 0.08,
    "4bet": 0.08,
    "five_bet": 0.05,
    "5bet": 0.05,
    "all_in_pressure": 0.05,
    "allin": 0.05,
}


def _to_card_strings(cards):
    return [Card.int_to_str(card) for card in cards]


def normalize_range_pct(opponent_range_pct=None, preflop_spot_type=None, use_preflop_spot_range=False):
    if opponent_range_pct is not None:
        value = float(opponent_range_pct)
        if value > 1.0:
            value /= 100.0
        return min(1.0, max(0.01, value))

    if not use_preflop_spot_range:
        return 1.0

    spot_type = str(preflop_spot_type or "unknown").lower()
    return _PREFLOP_SPOT_RANGE_PCT.get(spot_type, 1.0)


def _starting_hand_score(first, second):
    first_rank, first_suit = first[0], first[1]
    second_rank, second_suit = second[0], second[1]
    high = max(_RANK_VALUE[first_rank], _RANK_VALUE[second_rank])
    low = min(_RANK_VALUE[first_rank], _RANK_VALUE[second_rank])
    gap = high - low

    if high == low:
        return 1000 + (high * 20)

    score = high * 24 + low * 2
    if first_suit == second_suit:
        score += 18
    if gap == 1:
        score += 14
    elif gap == 2:
        score += 8
    elif gap == 3:
        score += 3
    if low >= 10:
        score += 10
    return score


@lru_cache(maxsize=20000)
def _range_filtered_hands_cached(available_cards, range_pct):
    hands = list(itertools.combinations(available_cards, 2))
    if range_pct >= 0.999:
        return tuple(hands)

    hands.sort(key=lambda hand: _starting_hand_score(hand[0], hand[1]), reverse=True)
    hand_count = max(1, int(round(len(hands) * range_pct)))
    return tuple(hands[:hand_count])


def _range_filtered_hands(available_cards, range_pct):
    return _range_filtered_hands_cached(tuple(sorted(available_cards)), round(float(range_pct), 4))


def _draw_opponent_hands(rng, range_hands, opponents):
    used_cards = set()
    opponent_hands = []
    attempt_limit = max(20, len(range_hands) * 2)
    attempts = 0
    while len(opponent_hands) < opponents and attempts < attempt_limit:
        attempts += 1
        first, second = rng.choice(range_hands)
        if first in used_cards or second in used_cards:
            continue
        opponent_hands.append((first, second))
        used_cards.add(first)
        used_cards.add(second)

    if len(opponent_hands) != opponents:
        return None, used_cards
    return opponent_hands, used_cards


@lru_cache(maxsize=200000)
def _estimate_equity_cached(hero_cards, board, active_players, iterations, range_pct):
    opponents = max(1, int(active_players) - 1)
    missing_board_cards = 5 - len(board)

    if iterations is None:
        base_iterations = {
            0: 220,
            3: 180,
            4: 150,
            5: 120,
        }.get(len(board), 140)
        iterations = max(50, base_iterations // opponents)

    dead_cards = set(hero_cards + board)
    available_cards = [card for card in _FULL_DECK if card not in dead_cards]
    range_hands = None if range_pct >= 0.999 else _range_filtered_hands(available_cards, range_pct)
    hero_hand = _CARD_SET("".join(hero_cards))

    rng = random.Random()
    hero_equity = 0.0

    for _ in range(iterations):
        if range_hands is None:
            drawn = rng.sample(available_cards, missing_board_cards + (2 * opponents))
            opponent_hands = []
            for opponent_index in range(opponents):
                start = missing_board_cards + (opponent_index * 2)
                opponent_hands.append(tuple(drawn[start : start + 2]))
            board_runout = board + tuple(drawn[:missing_board_cards])
        else:
            opponent_hands, used_cards = _draw_opponent_hands(rng, range_hands, opponents)
            if opponent_hands is None:
                drawn = rng.sample(available_cards, missing_board_cards + (2 * opponents))
                opponent_hands = []
                for opponent_index in range(opponents):
                    start = missing_board_cards + (opponent_index * 2)
                    opponent_hands.append(tuple(drawn[start : start + 2]))
                board_runout = board + tuple(drawn[:missing_board_cards])
            else:
                board_pool = [card for card in available_cards if card not in used_cards]
                board_runout = board + tuple(rng.sample(board_pool, missing_board_cards))
        board_set = _CARD_SET("".join(board_runout))

        evaluations = [
            _HOLD_EM_EVALUATOR.evaluateHand(hero_hand, board_set).eval()
        ]

        for opponent_hand in opponent_hands:
            villain_hand = _CARD_SET("".join(opponent_hand))
            evaluations.append(_HOLD_EM_EVALUATOR.evaluateHand(villain_hand, board_set).eval())

        best_eval = evaluations[0]
        for current_eval in evaluations[1:]:
            if current_eval > best_eval:
                best_eval = current_eval

        winners = sum(1 for evaluation in evaluations if evaluation == best_eval)
        if evaluations[0] == best_eval:
            hero_equity += 1.0 / winners

    return hero_equity / iterations


def estimate_equity(
    hole_cards,
    board_cards,
    active_players,
    iterations=None,
    opponent_range_pct=None,
    preflop_spot_type=None,
    use_preflop_spot_range=False,
):
    """
    Estimate hero equity against opponent ranges in Texas Hold'em.
    Returns a value between 0.0 and 1.0.
    """
    if len(hole_cards) != 2:
        return 0.0

    hero_cards = tuple(_to_card_strings(hole_cards))
    board = tuple(_to_card_strings(board_cards))
    range_pct = normalize_range_pct(opponent_range_pct, preflop_spot_type, use_preflop_spot_range)
    return _estimate_equity_cached(hero_cards, board, int(active_players), iterations, range_pct)


def equity_cache_info():
    return {
        "equity": _estimate_equity_cached.cache_info()._asdict(),
        "range_hands": _range_filtered_hands_cached.cache_info()._asdict(),
    }


def clear_equity_cache():
    _estimate_equity_cached.cache_clear()
    _range_filtered_hands_cached.cache_clear()


def pot_odds(call_amount, pot_size):
    if call_amount <= 0:
        return 0.0
    return call_amount / float(pot_size + call_amount)
