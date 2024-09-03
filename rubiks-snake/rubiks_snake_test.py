from rubiks_snake import RubiksSnakeCounter


def test_count_shapes():
    counts = RubiksSnakeCounter.count_shapes_all(16)
    for i in range(1, 16):
        assert counts[i - 1] == RubiksSnakeCounter.S[i]
