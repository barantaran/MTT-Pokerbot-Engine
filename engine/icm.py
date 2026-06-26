from __future__ import annotations

from typing import Sequence


MAX_EXACT_PLAYERS = 30


def calculate_exact_icm(chip_stacks: Sequence[float], payouts: Sequence[float]) -> list[float]:
    """Calculate exact ICM equities with the Malmuth-Harville equation."""
    if not chip_stacks or not payouts:
        return []

    player_count = len(chip_stacks)
    if player_count > MAX_EXACT_PLAYERS:
        raise ValueError(f"calculate_exact_icm supports at most {MAX_EXACT_PLAYERS} chip stacks")

    stacks = [float(stack) for stack in chip_stacks]
    if any(stack < 0 for stack in stacks):
        raise ValueError("chip stacks must be non-negative")

    total_chips = sum(stacks)
    if total_chips <= 0:
        return [0.0 for _ in stacks]

    sorted_payouts = sorted((float(payout) for payout in payouts), reverse=True)[:player_count]
    equities = [0.0 for _ in stacks]

    reach_probabilities = {0: 1.0}
    finished_chips = {0: 0.0}

    for place, payout in enumerate(sorted_payouts):
        is_last_paid_place = place == len(sorted_payouts) - 1
        next_reach_probabilities = {}
        next_finished_chips = {}

        for finished_mask, reach_probability in reach_probabilities.items():
            chips_finished = finished_chips[finished_mask]
            remaining_chips = total_chips - chips_finished
            if remaining_chips <= 0:
                continue

            for index, stack in enumerate(stacks):
                bit = 1 << index
                if finished_mask & bit:
                    continue
                if stack <= 0:
                    continue

                placement_probability = reach_probability * (stack / remaining_chips)
                equities[index] += placement_probability * payout

                if not is_last_paid_place:
                    next_mask = finished_mask | bit
                    next_reach_probabilities[next_mask] = (
                        next_reach_probabilities.get(next_mask, 0.0) + placement_probability
                    )
                    next_finished_chips.setdefault(next_mask, chips_finished + stack)

        reach_probabilities = next_reach_probabilities
        finished_chips = next_finished_chips

    return equities
