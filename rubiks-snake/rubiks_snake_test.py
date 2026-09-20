from fractions import Fraction

import numpy as np
import pytest

from rubiks_snake import (
    RubiksSnakeCounter,
    FACE_IDS_TO_WEDGE_ID,
    decode_formula,
    encode_formula,
    reverse_encoded_formula,
    min_cyclic_shift,
    DELTAS,
    DX,
    DY,
    DZ,
    INIT_WEDGE,
    WEDGE_ID_TO_FACE_IDS,
    WEDGE_ID_TO_NEXT_DELTA,
    _get_next_wedge_id,
    _increment_slab_count,
    _window_cyclic_core,
    irreducible_slab_counts,
    renewal_lower_bound,
    renewal_lower_prefactor,
    renewal_polynomial_value,
    renewal_values,
    slab_counts,
    verify_vector_bound,
    window_graph,
    window_upper_bound,
)
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


def test_palindrome_loops():
    expected = [None, 0, 0, 0, 1, 0, 2, 0, 4, 0, 10, 0, 29, 0, 90, 0, 226, 0, 862, 0, 2610]
    for n in range(1, 21):
        assert RubiksSnakeCounter.count_palindrome_loops(n) == expected[n]


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
    wedge_ids_facing_up = [wedge_id for faces, wedge_id in FACE_IDS_TO_WEDGE_ID.items() if faces[1] == 5]
    assert set(wedge_ids_facing_up) == {25, 26, 27, 28}


def test_formula_encoding():
    for n in range(1, 7):
        for code in range(4**n):
            formula = decode_formula(code, n)
            assert len(formula) == n
            assert encode_formula(formula) == code
            assert decode_formula(reverse_encoded_formula(code, n), n) == formula[::-1]
            min_shift = decode_formula(min_cyclic_shift(code, n), n)
            expected_min_shift = min(formula[i:] + formula[:i] for i in range(n))
            assert min_shift == expected_min_shift


def test_enumerate_shapes():
    for i in range(1, 9):
        assert len(RubiksSnakeCounter.enumerate_shapes(i)[0]) == RubiksSnakeCounter.S[i]


def test_list_all_loops():
    assert len(RubiksSnakeCounter.list_all_loops(10)) == 280


def test_direction_convention():
    directions = {
        DX: np.array([1, 0, 0]),
        -DX: np.array([-1, 0, 0]),
        DY: np.array([0, 1, 0]),
        -DY: np.array([0, -1, 0]),
        DZ: np.array([0, 0, 1]),
        -DZ: np.array([0, 0, -1]),
    }
    for wedge, (entrance, _) in WEDGE_ID_TO_FACE_IDS.items():
        incoming = directions[DELTAS[entrance]]
        outgoing = directions[WEDGE_ID_TO_NEXT_DELTA[wedge]]
        for rotation in range(4):
            expected = (incoming, np.cross(outgoing, incoming), -incoming, -np.cross(outgoing, incoming))[rotation]
            next_id = _get_next_wedge_id(wedge, rotation)
            assert np.array_equal(directions[WEDGE_ID_TO_NEXT_DELTA[next_id]], expected)


def _geometry_slab_counts(width, limit):
    deltas = (DX, -DX, DY, -DY, DZ, -DZ)
    moves = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    counter = RubiksSnakeCounter()
    counts = [0] * (limit + 1)

    def next_rotation(wedge, direction):
        matches = [
            (str(r), _get_next_wedge_id(wedge, r))
            for r in range(4)
            if WEDGE_ID_TO_NEXT_DELTA[_get_next_wedge_id(wedge, r)] == deltas[direction]
        ]
        assert len(matches) == 1
        return matches[0]

    def search(x, incoming, word, wedge):
        length = len(word)
        if x == width and incoming // 2 != 0:
            rotation, _ = next_rotation(wedge, 0)
            if counter.is_formula_valid(word + rotation + "0"):
                counts[length] += 1
        if length == limit:
            return
        for direction, move in enumerate(moves):
            if direction // 2 == incoming // 2 or not 0 <= x + move[0] <= width:
                continue
            rotation, next_wedge = next_rotation(wedge, direction)
            search(x + move[0], direction, word + rotation, next_wedge)

    search(0, 0, "", INIT_WEDGE)
    return counts


@pytest.mark.parametrize("width,limit", [(0, 9), (1, 8), (2, 7)])
def test_slab_geometry(width, limit):
    assert slab_counts(width, limit) == _geometry_slab_counts(width, limit)


def test_plane_complementary_revisits():
    assert slab_counts(0, 9) == [0, 4, 8, 16, 24, 40, 72, 136, 224, 392]
    # The vertex-avoiding subfamily has only 104 paths at internal length 7.
    assert slab_counts(0, 7)[7] > 104
    assert slab_counts(3, 6) == [0] * 7


@pytest.fixture
def published_slab_tables():
    return {
        1: [
            0,
            4,
            8,
            16,
            24,
            40,
            72,
            136,
            224,
            392,
            712,
            1272,
            2168,
            3840,
            6832,
            12112,
            20904,
            36856,
            65192,
            115096,
            199368,
            350696,
            618032,
            1087696,
            1887888,
            3314376,
            5825784,
            10230736,
            17775440,
        ],
        2: [
            0,
            0,
            0,
            16,
            64,
            192,
            448,
            1096,
            2960,
            8688,
            25264,
            71768,
            199984,
            553568,
            1536880,
            4276240,
            11894352,
            33015408,
            91581712,
            253615768,
            702030784,
        ],
        3: [
            0,
            0,
            0,
            0,
            0,
            64,
            384,
            1536,
            4736,
            13760,
            41088,
            129536,
            412160,
            1293608,
            4005936,
            12395208,
            38595792,
            120780664,
            378267416,
        ],
    }


def test_irreducible_reconstruction(published_slab_tables):
    raw = {d: row[:10] for d, row in published_slab_tables.items()}
    previous = [0] * 10
    irreducibles = {}
    for d in raw:
        total = irreducible_slab_counts({a: raw[a] for a in range(1, d + 1)})
        irreducibles[d] = [a - b for a, b in zip(total, previous)]
        previous = total
        for length in range(10):
            reconstructed = irreducibles[d][length] + sum(
                irreducibles[a][b] * raw[d - a][length - b - 1] for a in range(1, d) for b in range(length)
            )
            assert reconstructed == raw[d][length]
    assert previous == [0, 4, 8, 16, 24, 40, 72, 272, 1200, 4984]


def test_published_lower_certificate(published_slab_tables):
    counts = irreducible_slab_counts(published_slab_tables)
    q = renewal_lower_bound(counts)
    assert q == Fraction(3400034903, 10**9)
    assert renewal_polynomial_value(counts, q.numerator, q.denominator) < 0
    assert renewal_polynomial_value(counts, 3400034904, 10**9) >= 0
    assert renewal_lower_prefactor(counts, q) >= Fraction(41, 250)
    values = renewal_values(counts, 26)
    assert [values[k] for k in (2, 8, 12, 26)] == [4, 3024, 399568, 11951167017736]
    assert all(values[n - 2] <= RubiksSnakeCounter.S[n] for n in range(2, 29))


def test_renewal_exact_root_and_periodicity():
    assert renewal_lower_bound([0, 4], denominator=10) == Fraction(19, 10)
    assert renewal_polynomial_value([0, 4], 2, 1) == 0
    assert renewal_values([0, 4], 5) == [1, 0, 4, 0, 16, 0]
    with pytest.raises(ValueError, match="zero"):
        renewal_lower_prefactor([0, 4], Fraction(19, 10))
    with pytest.raises(ValueError, match="exceeds"):
        renewal_lower_prefactor([0, 4], Fraction(3))


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {2: [0, 0, 0, 16]},
        {1: [0, 4], 3: [0]},
        {1: [0, 4], 2: [0, 0, 0, 16]},
        {1: [1]},
        {1: [0, -1]},
        {1: [0, 4, 8, 16], 2: [0, 0, 0, 0]},
    ],
)
def test_invalid_irreducible_tables(raw):
    with pytest.raises(ValueError):
        irreducible_slab_counts(raw)


def test_slab_overflow_and_inputs():
    counts = np.array([np.iinfo(np.int64).max], dtype=np.int64)
    with pytest.raises(OverflowError):
        _increment_slab_count(counts, 0)
    assert counts[0] == np.iinfo(np.int64).max
    for width, length in [(-1, 5), (0, -1)]:
        with pytest.raises(ValueError):
            slab_counts(width, length)
    with pytest.raises(ValueError):
        renewal_lower_bound([0, 0])
    with pytest.raises(ValueError):
        renewal_values([0, 4], -1)


def test_window_graph():
    source, target, size = window_graph(3)
    assert size == 64 and len(source) == 241
    assert np.array_equal(source, RubiksSnakeCounter.enumerate_shapes(5)[0] >> 2)
    assert np.array_equal(target, RubiksSnakeCounter.enumerate_shapes(5)[0] & 63)
    vector = np.ones(size, dtype=object)
    for n in range(4, 9):
        assert sum(vector) >= RubiksSnakeCounter.S[n]
        image = np.zeros(size, dtype=object)
        np.add.at(image, target, vector[source])
        vector = image
    with pytest.raises(ValueError):
        window_graph(19)


def test_cyclic_core_retains_singleton_loops():
    source = np.array([0, 1, 2, 2, 3])
    target = np.array([1, 2, 1, 3, 3])
    new_source, new_target, labels, size = _window_cyclic_core(source, target, 4)
    assert size == 3 and len(new_source) == 3
    assert np.all(labels[new_source] == labels[new_target])
    assert np.count_nonzero(new_source == new_target) == 1


def test_overflow_free_vector_certificate():
    vectors = [np.array([1, 10**12, 10**12 - 17], dtype=np.uint64), np.array([2**64 - 1, 2**63, 11], dtype=np.uint64)]
    for vector in vectors:
        for q in (Fraction(3685468366, 10**9), Fraction(7, 11), Fraction(1, 10**20)):
            image = np.array([min(2**64 - 1, q.numerator * int(v) // q.denominator) for v in vector], dtype=np.uint64)
            assert verify_vector_bound(image, vector, q)
            for i in range(len(image)):
                if int(image[i]) < 2**64 - 1:
                    bad = image.copy()
                    bad[i] += np.uint64(1)
                    expected = all(q.denominator * int(a) <= q.numerator * int(v) for a, v in zip(bad, vector))
                    assert verify_vector_bound(bad, vector, q) == expected
    with pytest.raises(ValueError):
        verify_vector_bound(np.array([1]), np.array([0]), Fraction(4))


def test_window_upper_certificates():
    coarse = window_upper_bound(3, iterations=100)
    fine = window_upper_bound(5, iterations=150)
    assert Fraction(381, 100) < coarse["bound"] < Fraction(3811, 1000)
    assert Fraction(372, 100) < fine["bound"] < Fraction(3721, 1000)
    full = window_upper_bound(3, iterations=100, full_graph=True)
    for n in range(4, 9):
        assert RubiksSnakeCounter.S[n] <= full["pointwise_factor"] * full["bound"] ** (n - 4)
    assert coarse["pointwise_factor"] is None


def test_published_full_graph_certificate():
    result = window_upper_bound(9, full_graph=True)
    assert result["bound"] == Fraction(3685468366, 10**9)
    assert result["states"] == 172226 and result["edges"] == 633138
    assert result["vector_min"] == 1
    assert result["vector_sum"] == 136209763558000711
    assert result["pointwise_factor"] / result["bound"] ** 10 < 294632981756
