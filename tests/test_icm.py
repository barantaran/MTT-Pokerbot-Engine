import pytest

from engine.icm import MAX_EXACT_PLAYERS, calculate_exact_icm


def test_calculate_exact_icm_returns_empty_for_missing_inputs():
    assert calculate_exact_icm([], [1.0]) == []
    assert calculate_exact_icm([100.0], []) == []


def test_calculate_exact_icm_matches_readme_example():
    equities = calculate_exact_icm([150000, 98750, 45500, 13250], [425, 280, 130, 75])

    assert equities == pytest.approx([323.2, 278.12, 195.79, 112.89], abs=0.01)


def test_calculate_exact_icm_sorts_and_trims_payouts():
    sorted_trimmed = calculate_exact_icm([100, 100], [10, 50, 30])
    already_trimmed = calculate_exact_icm([100, 100], [50, 30])

    assert sorted_trimmed == pytest.approx(already_trimmed)
    assert sorted_trimmed == pytest.approx([40.0, 40.0])


def test_calculate_exact_icm_equal_stacks_have_equal_equity():
    equities = calculate_exact_icm([100, 100, 100], [0.5, 0.3])

    assert equities == pytest.approx([0.2666666667, 0.2666666667, 0.2666666667])


def test_calculate_exact_icm_zero_total_chips_returns_zero_equity():
    assert calculate_exact_icm([0, 0], [1.0]) == [0.0, 0.0]


def test_calculate_exact_icm_raises_above_exact_limit():
    with pytest.raises(ValueError, match=str(MAX_EXACT_PLAYERS)):
        calculate_exact_icm([1] * (MAX_EXACT_PLAYERS + 1), [1.0])
