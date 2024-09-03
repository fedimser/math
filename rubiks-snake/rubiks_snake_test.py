import pytest

from rubiks_snake import RubiksSnakeCounter
from rubiks_snake_slow import enumerate_valid_formulas_slow


def test_count_shapes():
    counts = RubiksSnakeCounter.count_all_shapes(16)
    for i in range(1, 16):
        assert counts[i - 1] == RubiksSnakeCounter.S[i]
    for i in range(1, 8):
        assert counts[i - 1] == len(enumerate_valid_formulas_slow(i))


def test_palindrome_shapes():
    assert RubiksSnakeCounter.count_palindrome_shapes(8) == 221
    assert RubiksSnakeCounter.count_palindrome_shapes(9) == 185
    assert RubiksSnakeCounter.count_palindrome_shapes(10) == 802


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8])
def test_palindromes_slow(n):
    valid_shapes = enumerate_valid_formulas_slow(n)
    num_shapes = len(valid_shapes)
    num_palindromes = sum(1 for s in valid_shapes if s[::-1] == s)
    num_shapes_up_to_reverse = len(set(min(s, s[::-1]) for s in valid_shapes))
    assert num_shapes == RubiksSnakeCounter.S[n]
    assert num_palindromes == RubiksSnakeCounter.count_palindrome_shapes(n)
    assert 2 * num_shapes_up_to_reverse == num_palindromes + num_shapes
