import importlib
import random
import sys
import tempfile
import zipfile
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


def _to_card_strings(cards):
    return [Card.int_to_str(card) for card in cards]


def estimate_equity(hole_cards, board_cards, active_players, iterations=None):
    """
    Estimate hero equity against random opponent ranges in Texas Hold'em.
    Returns a value between 0.0 and 1.0.
    """
    if len(hole_cards) != 2:
        return 0.0

    hero_cards = _to_card_strings(hole_cards)
    board = _to_card_strings(board_cards)
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
    hero_hand = _CARD_SET("".join(hero_cards))

    rng = random.Random()
    hero_equity = 0.0

    for _ in range(iterations):
        drawn = rng.sample(available_cards, missing_board_cards + (2 * opponents))
        board_runout = board + drawn[:missing_board_cards]
        board_set = _CARD_SET("".join(board_runout))

        evaluations = [
            _HOLD_EM_EVALUATOR.evaluateHand(hero_hand, board_set).eval()
        ]

        offset = missing_board_cards
        for opponent_index in range(opponents):
            start = offset + (opponent_index * 2)
            villain_hand = _CARD_SET("".join(drawn[start:start + 2]))
            evaluations.append(_HOLD_EM_EVALUATOR.evaluateHand(villain_hand, board_set).eval())

        best_eval = evaluations[0]
        for current_eval in evaluations[1:]:
            if current_eval > best_eval:
                best_eval = current_eval

        winners = sum(1 for evaluation in evaluations if evaluation == best_eval)
        if evaluations[0] == best_eval:
            hero_equity += 1.0 / winners

    return hero_equity / iterations


def pot_odds(call_amount, pot_size):
    if call_amount <= 0:
        return 0.0
    return call_amount / float(pot_size + call_amount)
