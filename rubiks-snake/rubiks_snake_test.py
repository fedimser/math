from rubiks_snake import RubiksSnakeCounter


def test_count_shapes():
    counts = RubiksSnakeCounter.count_all_shapes(16)
    for i in range(1, 16):
        assert counts[i - 1] == RubiksSnakeCounter.S[i]


def test_palindrome_shapes():
    assert RubiksSnakeCounter.count_palindrome_shapes(8) == 221
    assert RubiksSnakeCounter.count_palindrome_shapes(9) == 185
    assert RubiksSnakeCounter.count_palindrome_shapes(10) == 802
