import hashlib
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
    "five_bet_plus": 0.05,
    "five_bet": 0.05,
    "5bet": 0.05,
    "all_in_pressure": 0.05,
    "allin": 0.05,
}

_ADAPTIVE_PREFLOP_SPOT_RANGE_PCT = {
    "unknown": 1.00,
    "limped": 0.70,
    "srp": 0.50,
    "single_raised": 0.50,
    "three_bet": 0.25,
    "3bet": 0.25,
    "four_bet": 0.12,
    "4bet": 0.12,
    "five_bet_plus": 0.08,
    "five_bet": 0.08,
    "5bet": 0.08,
    "all_in_pressure": 0.10,
    "allin": 0.10,
}


def _to_card_strings(cards):
    return [Card.int_to_str(card) for card in cards]


def normalize_range_pct(
    opponent_range_pct=None,
    preflop_spot_type=None,
    use_preflop_spot_range=False,
    table_stats=None,
    opponent_stats=None,
    opponent_position=None,
    opponent_stack_bb=None,
    range_profile="legacy",
    position=None,
    stack_bb=None,
    players_left=None,
    starting_field=None,
    paid_places=None,
    itm_distance=None,
    range_influence=1.0,
    player_range_sampling=True,
    player_range_sample_config=None,
):
    if opponent_range_pct is not None:
        value = float(opponent_range_pct)
        if value > 1.0:
            value /= 100.0
        return min(1.0, max(0.01, value))

    if not use_preflop_spot_range:
        return 1.0

    spot_type = str(preflop_spot_type or "unknown").lower()
    profile = str(range_profile or "legacy").lower()
    adaptive_profile = profile in {"adaptive", "player", "player_adaptive"}
    range_map = _ADAPTIVE_PREFLOP_SPOT_RANGE_PCT if adaptive_profile else _PREFLOP_SPOT_RANGE_PCT
    range_pct = range_map.get(spot_type, 1.0)
    if adaptive_profile:
        range_pct = _adjust_range_pct_for_mtt_context(
            range_pct,
            spot_type,
            position=position,
            stack_bb=stack_bb,
            players_left=players_left,
            starting_field=starting_field,
            paid_places=paid_places,
            itm_distance=itm_distance,
        )
    if profile in {"player", "player_adaptive"}:
        range_pct = _adjust_range_pct_for_player_profile(
            range_pct,
            spot_type,
            opponent_stats,
            opponent_position=opponent_position,
            use_sample_quality=player_range_sampling,
            sample_config=player_range_sample_config,
            players_left=players_left,
            starting_field=starting_field,
            paid_places=paid_places,
            itm_distance=itm_distance,
        )
        range_pct = _adjust_range_pct_for_opponent_stack_pressure(
            range_pct,
            spot_type,
            opponent_stack_bb=opponent_stack_bb,
            hero_stack_bb=stack_bb,
            players_left=players_left,
            starting_field=starting_field,
            paid_places=paid_places,
            itm_distance=itm_distance,
        )
    range_pct = _adjust_range_pct_for_table_stats(range_pct, spot_type, table_stats, profile=profile)
    influence = _range_influence_for_spot(range_influence, spot_type)
    if influence >= 0.999:
        return range_pct
    return _clamp(1.0 - ((1.0 - range_pct) * influence), 0.01, 1.0)


def _range_influence_for_spot(range_influence, spot_type):
    if isinstance(range_influence, dict):
        raw_value = range_influence.get(spot_type)
        if raw_value is None:
            raw_value = range_influence.get("default", 1.0)
    else:
        raw_value = range_influence
    return _clamp(_safe_float(raw_value, 1.0), 0.0, 1.0)


def player_range(vpip, pfr, three_bet, position=None):
    """
    Return opponent range percent by preflop context from public action rates.

    Values are fractions of all starting hands. The map describes the opponent's
    likely range for the action sequence, not a single permanent player type.
    """
    vpip = min(0.70, max(0.04, _safe_float(vpip, 0.25)))
    pfr = min(min(vpip, 0.55), max(0.01, _safe_float(pfr, 0.16)))
    three_bet = min(0.25, max(0.005, _safe_float(three_bet, 0.08)))
    passive_gap = max(0.0, vpip - pfr)

    limped = _clamp(vpip + passive_gap * 0.35, 0.10, 0.75)
    srp = _clamp(pfr * 1.15 + passive_gap * 0.20, 0.06, 0.55)
    three_bet_range = _clamp(three_bet * 2.2 + pfr * 0.10, 0.025, 0.35)
    four_bet = _clamp(three_bet * 1.15 + pfr * 0.03, 0.015, 0.18)
    five_bet_plus = _clamp(three_bet * 0.75, 0.01, 0.12)
    all_in_pressure = _clamp(three_bet * 0.9 + pfr * 0.05, 0.015, 0.16)
    open_multiplier = _position_range_multiplier(position, "open")
    pressure_multiplier = _position_range_multiplier(position, "pressure")

    srp = _clamp(srp * open_multiplier, 0.04, 0.65)
    three_bet_range = _clamp(three_bet_range * pressure_multiplier, 0.02, 0.42)
    four_bet = _clamp(four_bet * pressure_multiplier, 0.012, 0.22)
    five_bet_plus = _clamp(five_bet_plus * pressure_multiplier, 0.008, 0.14)
    all_in_pressure = _clamp(all_in_pressure * pressure_multiplier, 0.012, 0.20)

    return {
        "unknown": 1.00,
        "limped": limped,
        "srp": srp,
        "single_raised": srp,
        "three_bet": three_bet_range,
        "3bet": three_bet_range,
        "four_bet": four_bet,
        "4bet": four_bet,
        "five_bet_plus": five_bet_plus,
        "five_bet": five_bet_plus,
        "5bet": five_bet_plus,
        "all_in_pressure": all_in_pressure,
        "allin": all_in_pressure,
    }


def _position_range_multiplier(position, action_class):
    return 1.0


def _adjust_range_pct_for_mtt_context(
    range_pct,
    spot_type,
    *,
    position=None,
    stack_bb=None,
    players_left=None,
    starting_field=None,
    paid_places=None,
    itm_distance=None,
):
    multiplier = 1.0
    position = str(position or "").upper()
    pressure_spot = spot_type in {
        "srp",
        "single_raised",
        "three_bet",
        "3bet",
        "four_bet",
        "4bet",
        "five_bet_plus",
        "five_bet",
        "5bet",
        "all_in_pressure",
        "allin",
    }

    if position in {"BTN", "CO"} and spot_type in {"unknown", "limped", "srp", "single_raised"}:
        multiplier *= 1.18
    elif position in {"UTG", "EP"} and pressure_spot:
        multiplier *= 0.88

    bb = _optional_float(stack_bb)
    if bb is not None:
        if bb <= 8 and spot_type in {"all_in_pressure", "allin", "three_bet", "3bet"}:
            multiplier *= 1.45
        elif bb <= 16 and pressure_spot:
            multiplier *= 1.22
        elif bb >= 60 and spot_type in {"four_bet", "4bet", "five_bet_plus", "five_bet", "5bet"}:
            multiplier *= 0.88

    stage = _mtt_stage_pressure(players_left, starting_field, paid_places, itm_distance)
    if stage >= 0.80 and pressure_spot:
        multiplier *= 0.82
    elif stage >= 0.50 and pressure_spot:
        multiplier *= 0.92

    return min(1.0, max(0.01, range_pct * multiplier))


def _adjust_range_pct_for_opponent_stack_pressure(
    range_pct,
    spot_type,
    *,
    opponent_stack_bb=None,
    hero_stack_bb=None,
    players_left=None,
    starting_field=None,
    paid_places=None,
    itm_distance=None,
):
    pressure_spot = spot_type in {
        "srp",
        "single_raised",
        "three_bet",
        "3bet",
        "four_bet",
        "4bet",
        "five_bet_plus",
        "five_bet",
        "5bet",
        "all_in_pressure",
        "allin",
    }
    if not pressure_spot:
        return range_pct

    villain_bb = _optional_float(opponent_stack_bb)
    if villain_bb is None or villain_bb <= 0:
        return range_pct

    hero_bb = _optional_float(hero_stack_bb)
    multiplier = 1.0
    if villain_bb <= 8:
        multiplier *= 1.45
    elif villain_bb <= 16:
        multiplier *= 1.25
    elif villain_bb <= 25:
        multiplier *= 1.12

    if hero_bb is not None and hero_bb > 0 and villain_bb >= hero_bb * 1.8:
        multiplier *= 1.12

    stage = _mtt_stage_pressure(players_left, starting_field, paid_places, itm_distance)
    if stage >= 0.80:
        multiplier *= 1.16
    elif stage >= 0.50:
        multiplier *= 1.08

    return min(1.0, max(0.01, range_pct * multiplier))


def _adjust_range_pct_for_player_profile(
    range_pct,
    spot_type,
    table_stats,
    *,
    opponent_position=None,
    use_sample_quality=True,
    sample_config=None,
    players_left=None,
    starting_field=None,
    paid_places=None,
    itm_distance=None,
):
    if not isinstance(table_stats, dict):
        return range_pct

    quality = _sample_quality(table_stats)
    if quality <= 0.0:
        return range_pct
    if not use_sample_quality:
        quality = 1.0

    ranges = player_range(
        table_stats.get("vpip", 0.25),
        table_stats.get("pfr", 0.16),
        table_stats.get("three_bet_rate", 0.08),
        position=opponent_position,
    )
    player_pct = ranges.get(spot_type, range_pct)
    quality = _player_range_profile_quality(
        quality,
        range_pct,
        player_pct,
        spot_type,
        table_stats,
        sample_config,
        players_left=players_left,
        starting_field=starting_field,
        paid_places=paid_places,
        itm_distance=itm_distance,
    )
    return _clamp((range_pct * (1.0 - quality)) + (player_pct * quality), 0.01, 1.0)


def _player_range_profile_quality(
    default_quality,
    base_pct,
    player_pct,
    spot_type,
    stats,
    sample_config,
    *,
    players_left=None,
    starting_field=None,
    paid_places=None,
    itm_distance=None,
):
    if not isinstance(sample_config, dict):
        return default_quality

    hands = _optional_float(stats.get("player_hands_observed"))
    if hands is None:
        hands = _optional_float(stats.get("hands_observed"))
    if hands is None:
        return default_quality

    pressure_spot = spot_type in {
        "three_bet",
        "3bet",
        "four_bet",
        "4bet",
        "five_bet_plus",
        "five_bet",
        "5bet",
        "all_in_pressure",
        "allin",
    }
    is_widening = player_pct > base_pct
    if pressure_spot:
        start = _safe_float(sample_config.get("pressure_start_hands"), 40.0)
        full = _safe_float(sample_config.get("pressure_full_hands"), 120.0)
    elif is_widening:
        start = _safe_float(sample_config.get("loose_start_hands"), 25.0)
        full = _safe_float(sample_config.get("loose_full_hands"), 75.0)
    else:
        start = _safe_float(sample_config.get("tight_start_hands"), 10.0)
        full = _safe_float(sample_config.get("tight_full_hands"), 35.0)

    if full <= start:
        hand_quality = 1.0 if hands >= full else 0.0
    else:
        hand_quality = _clamp((hands - start) / (full - start), 0.0, 1.0)
    quality = default_quality * hand_quality

    if is_widening:
        quality *= _safe_float(sample_config.get("loose_multiplier"), 0.80)
        stage = _mtt_stage_pressure(players_left, starting_field, paid_places, itm_distance)
        if stage >= 0.80:
            quality *= _safe_float(sample_config.get("bubble_loose_multiplier"), 0.45)
        elif stage >= 0.50:
            quality *= _safe_float(sample_config.get("late_loose_multiplier"), 0.65)
    else:
        quality *= _safe_float(sample_config.get("tight_multiplier"), 1.10)

    return _clamp(quality, 0.0, 1.0)


def _mtt_stage_pressure(players_left, starting_field, paid_places, itm_distance):
    distance = _optional_float(itm_distance)
    if distance is not None:
        if distance <= 0.05:
            return 1.0
        if distance <= 0.15:
            return 0.70

    left = _optional_float(players_left)
    paid = _optional_float(paid_places)
    field = _optional_float(starting_field)
    if left is None or left <= 0:
        return 0.0
    if paid is not None and paid > 0:
        if paid < left <= paid + 2:
            return 1.0
        if paid < left <= paid * 1.15:
            return 0.70
    if field is not None and field > 0 and left <= min(10.0, field * 0.10):
        return 0.55
    return 0.0


def _adjust_range_pct_for_table_stats(range_pct, spot_type, table_stats, *, profile="legacy"):
    if not isinstance(table_stats, dict):
        return range_pct

    quality = _sample_quality(table_stats)
    if quality <= 0.0:
        return range_pct

    if str(profile or "legacy").lower() in {"player", "player_adaptive"}:
        return range_pct

    vpip = _clamped_rate(table_stats, "vpip", 0.25)
    pfr = _clamped_rate(table_stats, "pfr", 0.16)
    three_bet = _clamped_rate(table_stats, "three_bet_rate", 0.08)

    if str(profile or "legacy").lower() == "adaptive":
        if spot_type in {"three_bet", "3bet", "four_bet", "4bet", "five_bet_plus", "five_bet", "5bet", "all_in_pressure", "allin"}:
            tendency = ((three_bet - 0.08) * 3.0) + ((pfr - 0.16) * 1.0)
        elif spot_type in {"srp", "single_raised"}:
            tendency = ((pfr - 0.16) * 1.7) + ((vpip - 0.25) * 0.7)
        elif spot_type == "limped":
            tendency = (vpip - 0.25) * 1.2
        else:
            tendency = ((vpip - 0.25) * 0.7) + ((pfr - 0.16) * 0.5)
        multiplier = 1.0 + max(-0.55, min(1.10, tendency)) * quality
        return min(1.0, max(0.01, range_pct * multiplier))

    if spot_type in {"three_bet", "3bet", "four_bet", "4bet", "five_bet_plus", "five_bet", "5bet", "all_in_pressure", "allin"}:
        tendency = ((three_bet - 0.08) * 2.2) + ((pfr - 0.16) * 0.8)
    elif spot_type in {"srp", "single_raised"}:
        tendency = ((pfr - 0.16) * 1.2) + ((vpip - 0.25) * 0.5)
    elif spot_type == "limped":
        tendency = (vpip - 0.25) * 1.0
    else:
        tendency = ((vpip - 0.25) * 0.6) + ((pfr - 0.16) * 0.4)

    multiplier = 1.0 + max(-0.45, min(0.85, tendency)) * quality
    return min(1.0, max(0.01, range_pct * multiplier))


def _optional_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value, default):
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value, lower, upper):
    return min(upper, max(lower, float(value)))


def _sample_quality(stats):
    return _clamp(stats.get("sample_quality", 0.0) or 0.0, 0.0, 1.0)


def _clamped_rate(stats, key, default):
    value = stats.get(key, default)
    if value is None:
        value = default
    return min(1.0, max(0.0, float(value)))


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

    # Seed deterministically from the call args so the same spot yields the same
    # estimate in every worker and on every SPOT resume (README promises "same
    # seed -> identical result"). builtin hash() is salted per process, so digest
    # a stable repr instead.
    seed_key = repr((hero_cards, board, active_players, iterations, range_pct)).encode()
    seed = int.from_bytes(hashlib.sha256(seed_key).digest()[:8], "big")
    rng = random.Random(seed)
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
    table_stats=None,
    opponent_stats=None,
    opponent_position=None,
    opponent_stack_bb=None,
    range_profile="legacy",
    position=None,
    stack_bb=None,
    players_left=None,
    starting_field=None,
    paid_places=None,
    itm_distance=None,
    range_influence=1.0,
    player_range_sampling=True,
    player_range_sample_config=None,
):
    """
    Estimate hero equity against opponent ranges in Texas Hold'em.
    Returns a value between 0.0 and 1.0.
    """
    if len(hole_cards) != 2:
        return 0.0

    hero_cards = tuple(_to_card_strings(hole_cards))
    board = tuple(_to_card_strings(board_cards))
    range_pct = normalize_range_pct(
        opponent_range_pct,
        preflop_spot_type,
        use_preflop_spot_range,
        table_stats,
        opponent_stats=opponent_stats,
        opponent_position=opponent_position,
        opponent_stack_bb=opponent_stack_bb,
        range_profile=range_profile,
        position=position,
        stack_bb=stack_bb,
        players_left=players_left,
        starting_field=starting_field,
        paid_places=paid_places,
        itm_distance=itm_distance,
        range_influence=range_influence,
        player_range_sampling=player_range_sampling,
        player_range_sample_config=player_range_sample_config,
    )
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
