import pytest

from rubiks_snake import RubiksSnakeCounter, FACE_IDS_TO_WEDGE_ID, decode_formula, \
    encode_formula_as_int, reverse_encoded_formula, min_cyclic_shift
from rubiks_snake_slow import enumerate_valid_formulas_slow


def test_count_shapes():
    counts = RubiksSnakeCounter.count_all_shapes(16)
    for i in range(1, 16):
        assert counts[i - 1] == RubiksSnakeCounter.S[i]
    for i in range(1, 8):
        assert counts[i - 1] == len(enumerate_valid_formulas_slow(i))


def test_palindrome_shapes():
    expected = [None, 1, 4, 4, 16, 13, 60, 52, 221, 185, 802, 700, 2957, 2483, 10820, 9199, 39608]
    for n in range(1, 17):
        assert RubiksSnakeCounter.count_palindrome_shapes(n) == expected[n]

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


def test_wedges_facing_up():
    wedge_ids_facing_up = [wedge_id for faces, wedge_id in FACE_IDS_TO_WEDGE_ID.items() if
                           faces[1] == 5]
    assert set(wedge_ids_facing_up) == {25, 26, 27, 28}


def test_formula_encoding():
    for n in range(1, 7):
        for code in range(4 ** n):
            formula = decode_formula(code, n)
            assert len(formula) == n
            assert encode_formula_as_int(formula) == code
            assert decode_formula(reverse_encoded_formula(code, n), n) == formula[::-1]
            min_shift = decode_formula(min_cyclic_shift(code, n), n)
            expected_min_shift = min(formula[i:] + formula[:i] for i in range(n))
            assert min_shift == expected_min_shift
