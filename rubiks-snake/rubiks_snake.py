import numpy as np
import numba

# ================   GEOMETRY   ===================
# Prepare the grid.
MAX_N = 32
BOX_SIZE = 2 * (MAX_N // 2) + 1
DX, DY, DZ = 1, BOX_SIZE, BOX_SIZE**2
CENTER_COORD = (MAX_N // 2) * (DX + DY + DZ)

# Pre-calculate geometry.
CUBE = [[1, 3, 4, 2], [0, 2, 5, 3], [0, 4, 5, 1], [0, 1, 5, 4], [0, 3, 5, 2], [1, 2, 4, 3]]
DELTAS = np.array([DY, DZ, DX, -DX, -DZ, -DY])  # "+y","+z","+x","-x","-z","-y"
WEDGE_ID_TO_FACE_IDS = dict()
FACE_IDS_TO_WEDGE_ID = dict()
WEDGE_ID_TO_NEXT_DELTA = np.zeros(36, dtype=np.int64)
ROT_AND_WEDGE_ID_TO_NEXT_WEDGE_ID = np.zeros(36 * 4, dtype=np.int64)


def _register_wedge(f1, f2, wedge_id):
    WEDGE_ID_TO_FACE_IDS[wedge_id] = (f1, f2)
    FACE_IDS_TO_WEDGE_ID[(f1, f2)] = wedge_id


def _init_globals():
    for i, (f1, f2) in enumerate([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3)]):
        _register_wedge(f1, f2, i + 1)
        _register_wedge(f2, f1, i + 1 + 16)
        _register_wedge(5 - f1, 5 - f2, 13 - (i + 1))
        _register_wedge(5 - f2, 5 - f1, 13 - (i + 1) + 16)

    for f1 in range(6):
        for f2 in CUBE[f1]:
            wedge_id = FACE_IDS_TO_WEDGE_ID[(f1, f2)]
            f1p = 5 - f2
            f2p = [5 - f1, 0, f1, 0]
            f2p[1] = CUBE[f1][(CUBE[f1].index(f2) + 1) % 4]
            f2p[3] = 5 - f2p[1]
            WEDGE_ID_TO_NEXT_DELTA[wedge_id] = DELTAS[f1p]
            for rot in range(4):
                ROT_AND_WEDGE_ID_TO_NEXT_WEDGE_ID[wedge_id + rot * 36] = FACE_IDS_TO_WEDGE_ID[(f1p, f2p[rot])]


_init_globals()
INIT_WEDGE = FACE_IDS_TO_WEDGE_ID[(0, 3)]


@numba.jit("i8(i8,i8)", inline="always")
def _encode_wedge(coord, wedge_id):
    return (coord << 6) + wedge_id


@numba.jit("i8(i8,i8)", inline="always")
def _get_next_wedge_coord(last_wedge_id, last_wedge_coord):
    return last_wedge_coord + WEDGE_ID_TO_NEXT_DELTA[last_wedge_id]


@numba.jit("i8(i8,i8)", inline="always")
def _get_next_wedge_id(last_wedge_id, rot):
    return ROT_AND_WEDGE_ID_TO_NEXT_WEDGE_ID[last_wedge_id + 36 * rot]


# ================   FORMULA ENCODING   ===================
def encode_formula(s):
    assert all(48 <= ord(c) <= 51 for c in s)
    n = len(s)
    return sum((ord(s[i]) - 48) << (2 * (n - 1 - i)) for i in range(len(s)))


def decode_formula(code, length):
    return "".join(str((code >> (2 * i)) % 4) for i in range(length))[::-1]


@numba.jit("i8(i8,i8)", inline="always")
def reverse_encoded_formula(code, length):
    ans = 0
    for i in range(length):
        ans = (ans << 2) + (code >> (2 * i)) % 4
    return ans


@numba.jit("i8(i8,i8)", inline="always")
def min_cyclic_shift(code, length):
    ans = code
    l = 2 * (length - 1)
    for i in range(length - 1):
        code = (code >> 2) + ((code & 3) << l)
        if code < ans:
            ans = code
    return ans


@numba.jit("i8(i8,i8,i8)", inline="always")
def concat_encoded_formulas(code1, code2, length2):
    return (code1 << (2 * length2)) + code2


# ================   ARENA   ===================
@numba.jit("(i8,i8,i8[:],i8[:])", inline="always")
def _push_wedge(wedge_coord, wedge_id, wedges, cubes):
    wedges[0] -= 1
    wedges[wedges[0]] = _encode_wedge(wedge_coord, wedge_id)
    cubes[wedge_coord] += wedge_id & 15


@numba.jit("(i8[:],i8[:])", inline="always")
def _pop_wedge(wedges, cubes):
    last_wedge = wedges[wedges[0]]
    cubes[last_wedge >> 6] -= last_wedge & 15
    wedges[0] += 1


@numba.jit("(i8,i8[:],i8[:])", inline="always")
def _pop_n_wedges(n, wedges, cubes):
    for _ in range(n):
        _pop_wedge(wedges, cubes)


@numba.jit("(i8[:],i8[:])", inline="always")
def _pop_all_but_one(wedges, cubes):
    head_pos = len(wedges) - 1
    while wedges[0] != head_pos:
        _pop_wedge(wedges, cubes)


@numba.jit("i8(i8,i8[:],i8[:])", inline="always")
def _push_next_wedge_if_can(rot, wedges, cubes):
    last_wedge = wedges[wedges[0]]
    last_wedge_coord, last_wedge_id = last_wedge >> 6, last_wedge & 63
    next_wedge_coord = _get_next_wedge_coord(last_wedge_id, last_wedge_coord)
    next_wedge_id = _get_next_wedge_id(last_wedge_id, rot)
    next_wedge_occ_type = next_wedge_id & 15
    next_cube_occ_type = cubes[next_wedge_coord]
    can_push = next_cube_occ_type == 0 or (next_cube_occ_type + next_wedge_occ_type == 13)
    if can_push:
        _push_wedge(next_wedge_coord, next_wedge_id, wedges, cubes)
        return 1
    else:
        return 0


@numba.jit("i8(i8,i8,i8[:],i8[:])", inline="always")
def _add_wedges_from_formula_while_can(formula_code, formula_length, wedges, cubes) -> int:
    """Tries to add wedges to tail, instructed by rotations in formula.

    Formula has given length(>0) and encoded by formula encoding convention.
    Returns number of added wedges. If result == n, means all added successfully. If result <n,
    only this much were added and then got spacial conflict.
    Needs to be undone by _pop_n_wedges.
    """
    k = 2 * (formula_length - 1)
    for i in range(formula_length):
        rot = (formula_code >> k) & 3
        k -= 2
        if not _push_next_wedge_if_can(rot, wedges, cubes):
            return i
    return formula_length


@numba.jit("UniTuple(i8[:],2)(i8,i8)")
def _prepare_arena(n, init_wedge_id):
    wedges = np.zeros(n + 1, dtype=np.int64)
    wedges[0] = n + 1
    cubes = np.zeros(BOX_SIZE**3, dtype=np.int64)
    _push_wedge(CENTER_COORD, init_wedge_id, wedges, cubes)
    return wedges, cubes


# Whether next wedge, while impossible to physically add, would exactly coincide with the head.
# Useful for checking if loop-formula describes a loop.
@numba.jit("i8(i8,i8[:])", inline="always")
def _next_wedge_would_match_head(rot, wedges):
    last_wedge = wedges[wedges[0]]
    last_wedge_coord, last_wedge_id = last_wedge >> 6, last_wedge & 63
    next_wedge_coord = _get_next_wedge_coord(last_wedge_id, last_wedge_coord)
    next_wedge_id = _get_next_wedge_id(last_wedge_id, rot)
    return _encode_wedge(next_wedge_coord, next_wedge_id) == wedges[-1]


# ================   COUNTING   ===================
@numba.jit("(i8[:],i8[:],i8[:])")
def _count_shapes_rec(wedges, cubes, total_count):
    last_wedge_index = wedges[0]
    total_count[last_wedge_index] += 1
    if last_wedge_index == 1:
        return  # Full length shape, stop recusrion.
    last_wedge = wedges[wedges[0]]
    last_wedge_id = last_wedge & 63
    last_wedge_coord = last_wedge >> 6
    next_wedge_coord = _get_next_wedge_coord(last_wedge_id, last_wedge_coord)
    next_cube_occupancy_type = cubes[next_wedge_coord]

    if next_cube_occupancy_type == 0 and last_wedge_index == 2:
        total_count[1] += 4
        return
    if next_cube_occupancy_type == 0 and last_wedge_index == 3:
        c = next_wedge_coord
        s = cubes[c - DX] + cubes[c + DX] + cubes[c - DY] + cubes[c + DY] + cubes[c - DZ] + cubes[c + DZ]
        if s == cubes[last_wedge_coord]:
            total_count[2] += 4
            total_count[1] += 16
            return

    for rot in range(4):
        next_wedge_id = _get_next_wedge_id(last_wedge_id, rot)
        next_wedge_occupancy_type = next_wedge_id & 15
        can_push = next_cube_occupancy_type == 0 or (next_cube_occupancy_type + next_wedge_occupancy_type == 13)
        if can_push:
            _push_wedge(next_wedge_coord, next_wedge_id, wedges, cubes)
            _count_shapes_rec(wedges, cubes, total_count)
            cubes[next_wedge_coord] -= next_wedge_occupancy_type  # pop
            wedges[0] += 1  # pop


@numba.jit("i8(i8[:],i8[:],i8[:])")
def _is_shape_valid_rec(formula, wedges, cubes):
    if len(formula) == 0:
        return 1
    if _push_next_wedge_if_can(formula[0], wedges, cubes):
        ans = _is_shape_valid_rec(formula[1:], wedges, cubes)
        _pop_wedge(wedges, cubes)
        return ans
    else:
        return 0


@numba.jit("i8(i8,i8[:],i8[:])")
def _count_palindrome_shapes(n, wedges, cubes):
    ans = 0
    rots = np.zeros(n - 1, dtype=np.int64)
    for i in range(4 ** (n // 2)):
        for j in range(n // 2):
            rots[j] = (i >> (2 * j)) & 3
            rots[n - 2 - j] = rots[j]
        if _is_shape_valid_rec(rots, wedges, cubes):
            ans += 1
    return ans


@numba.jit("i8(i8,i8,i8[:],i8[:])")
def _is_loop(formula_code, formula_length, wedges, cubes):
    """Checks whether given formula describes a loop.

    There are 2 kinds of formulas:
        * Shape-formula. String of n-1 characters describing shape of n-wedge Snake by listing all
            rotations at joints between wedges. Loop always have shape-formula of odd length.
        * Loop-formula. String of n-characters describing a loop of n-wedge Snake, which is a shape
            formula plus one extra rotation, as if there was a joint between head and tail. It is
            useful for describing loops, because all its cyclic shifts describe the same loop (in a
            sense). Loops always have loop-formula of even length.
    Both kinds are supported. That is, this function returns true if formula can be interpreted as
        shape-formula or loop-formula describing a loop.
    """
    ans = 0
    n = _add_wedges_from_formula_while_can(formula_code, formula_length, wedges, cubes)
    if n == formula_length and formula_length % 2 == 1:
        # This can be a shape-formula, iff the last wedge is below center and is facing up.
        last_wedge = wedges[wedges[0]]
        last_wedge_coord = last_wedge >> 6
        last_wedge_id = last_wedge & 63
        if (last_wedge_coord == CENTER_COORD - DY) and 25 <= last_wedge_id <= 28:
            ans = 1
    elif n == formula_length - 1 and formula_length % 2 == 0:
        # This can be a loop-formula, iff the potential tail coincided with head.
        if _next_wedge_would_match_head(formula_code % 4, wedges):
            ans = 1
    _pop_all_but_one(wedges, cubes)
    return ans


@numba.jit("i8(i8)")
def _count_palindrome_loops(n):
    if n % 2 == 1:
        return 0
    n2 = n // 2
    wedges, cubes = _prepare_arena(n + 1, INIT_WEDGE)
    ans = 0
    for i in range(4**n2):
        formula = concat_encoded_formulas(i, reverse_encoded_formula(i >> 2, n2 - 1), n2 - 1)
        if _is_loop(formula, n - 1, wedges, cubes):
            ans += 1
    return ans


# Enumerates all shapes.
@numba.jit("(i8[:],i8[:],i8,i8[:],i8[:])")
def _enumerate_shapes_rec(wedges, cubes, cur_formula, formulas, last_wedges):
    last_wedge = wedges[wedges[0]]
    if wedges[0] == 1:
        formulas[formulas[0]] = cur_formula
        last_wedges[formulas[0]] = last_wedge
        formulas[0] += 1
        return
    for rot in range(4):
        if _push_next_wedge_if_can(rot, wedges, cubes):
            _enumerate_shapes_rec(wedges, cubes, (cur_formula << 2) + rot, formulas, last_wedges)
            _pop_wedge(wedges, cubes)


class RubiksSnakeCounter:
    # Number of formulas of length n-1 describing a valid shape of n-wedge snake.
    # Pre-computed up to n=28.
    S = [
        None,
        1,
        4,
        16,
        64,
        241,
        920,
        3384,
        12585,
        46471,
        172226,
        633138,
        2333757,
        8561679,
        31462176,
        115247629,
        422677188,
        1546186675,
        5661378449,
        20689242550,
        75663420126,
        276279455583,
        1009416896015,
        3683274847187,
        13446591920995,
        49037278586475,
        178904588083788,
        652111697384508,
        2377810831870022,
    ]

    # Number of formulas of length n-1 describing a loop of n-wedge snake.
    # Equivalent: number of loop-formulas of length n describing a loop of n-wedge snake.
    # Pre-computed up to n=25.
    L1 = [
        None,
        0,
        0,
        0,
        1,
        0,
        8,
        0,
        16,
        0,
        280,
        0,
        2229,
        0,
        20720,
        0,
        226000,
        0,
        2293422,
        0,
        24965960,
        0,
        275633094,
        0,
        3069890660,
        0,
    ]

    @staticmethod
    def count_all_shapes(n):
        total_count = np.zeros(n + 1, dtype=np.int64)
        wedges, cubes = _prepare_arena(n, INIT_WEDGE)
        _count_shapes_rec(wedges, cubes, total_count)
        return total_count[1:][::-1]

    @staticmethod
    def count_palindrome_shapes(n):
        wedges, cubes = _prepare_arena(n + 1, INIT_WEDGE)
        return _count_palindrome_shapes(n, wedges, cubes)

    @staticmethod
    def count_palindrome_loops(n):
        """Count formulas of length n-1 that are palindromes and describe loops."""
        return _count_palindrome_loops(n)

    @staticmethod
    def enumerate_shapes(n, first_wedge_faces=(0, 3)):
        """Enumerates shapes of length n, their formulas have length n-1."""
        assert 1 <= n <= 20
        wedges, cubes = _prepare_arena(n, FACE_IDS_TO_WEDGE_ID[first_wedge_faces])

        num_shapes = RubiksSnakeCounter.S[n]
        formulas = np.zeros(num_shapes + 1, dtype=np.int64)
        last_wedges = np.zeros_like(formulas)
        formulas[0] = 1

        _enumerate_shapes_rec(wedges, cubes, 0, formulas, last_wedges)

        assert formulas[0] == 1 + num_shapes
        return formulas[1:], last_wedges[1:]

    @staticmethod
    def list_all_loops(n):
        """All loops for n-wedge Snake, represented by loop-formulas."""
        if n % 2 == 1:
            return []
        wedges, cubes = _prepare_arena(n + 1, INIT_WEDGE)
        return [i for i in range(4**n) if _is_loop(i, n, wedges, cubes)]

    def __init__(self):
        self.wedges, self.cubes = _prepare_arena(MAX_N, INIT_WEDGE)

    def is_formula_valid(self, formula: str) -> bool:
        n = len(formula)
        enc = encode_formula(formula)
        ans = _add_wedges_from_formula_while_can(enc, n, self.wedges, self.cubes) == n
        _pop_all_but_one(self.wedges, self.cubes)
        return ans


# ================   CERTIFIED ASYMPTOTIC BOUNDS   ===================
from collections.abc import Mapping, Sequence
from fractions import Fraction
from operator import index as _index


def _bound_integer(value: int, name: str, minimum: int = 0) -> int:
    value = _index(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


_SLAB_DIRECTIONS = np.array(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int64
)
_SLAB_CORNERS = np.zeros((6, 6), dtype=np.uint8)
_SLAB_COMPLEMENTS = np.zeros(37, dtype=np.uint8)
for _incoming in range(6):
    for _outgoing in range(6):
        _a, _b = sorted((_incoming ^ 1, _outgoing))
        _SLAB_CORNERS[_incoming, _outgoing] = 6 * _a + _b + 1
        _ca, _cb = sorted((_a ^ 1, _b ^ 1))
        _SLAB_COMPLEMENTS[6 * _a + _b + 1] = 6 * _ca + _cb + 1


@numba.njit(inline="always")
def _increment_slab_count(counts, length):
    if counts[length] == 9223372036854775807:
        raise OverflowError("slab count exceeds int64; use an arbitrary-precision enumerator")
    counts[length] += 1


@numba.njit
def _slab_search(width, limit, x, y, z, incoming, length, occupancy, counts):
    side = 2 * limit + 1
    position = (x * side + y + limit) * side + z + limit
    previous = occupancy[position]
    if x == width and incoming // 2 != 0:
        corner = _SLAB_CORNERS[incoming, 0]
        if previous == 0 or (previous != 255 and corner == _SLAB_COMPLEMENTS[previous]):
            _increment_slab_count(counts, length)
    if length == limit:
        return
    for outgoing in range(6):
        if outgoing // 2 == incoming // 2:
            continue
        nx = x + _SLAB_DIRECTIONS[outgoing, 0]
        if nx < 0 or nx > width:
            continue
        corner = _SLAB_CORNERS[incoming, outgoing]
        if previous == 0:
            occupancy[position] = corner
        elif previous != 255 and corner == _SLAB_COMPLEMENTS[previous]:
            occupancy[position] = 255
        else:
            continue
        _slab_search(
            width, limit, nx, y + _SLAB_DIRECTIONS[outgoing, 1],
            z + _SLAB_DIRECTIONS[outgoing, 2], outgoing, length + 1, occupancy, counts
        )
        occupancy[position] = previous


def slab_counts(width: int, max_length: int) -> list[int]:
    """Count exact slab blocks by internal edge length, including both boundary wedges.

    The incoming and outgoing directions are +x; internal vertices stay in
    0 <= x <= width. The returned row at index l is B[width+1, l+1].
    Memory is O((width+1)*max_length**2); search time is exponential.
    Counts use checked int64 increments, then convert to Python integers.
    """
    width = _bound_integer(width, "width")
    max_length = _bound_integer(max_length, "max_length")
    if max_length < 2 * width + 1:
        return [0] * (max_length + 1)
    occupancy = np.zeros((width + 1) * (2 * max_length + 1) ** 2, dtype=np.uint8)
    counts = np.zeros(max_length + 1, dtype=np.int64)
    _slab_search(width, max_length, 0, 0, 0, 0, 0, occupancy, counts)
    return counts.tolist()


def _renewal_counts(counts: Sequence[int]) -> list[int]:
    result = [_bound_integer(c, "block count") for c in counts]
    if not result or not any(result):
        raise ValueError("at least one block count must be positive")
    return result


def irreducible_slab_counts(raw: Mapping[int, Sequence[int]]) -> list[int]:
    """Extract known irreducibles from B=1/(1-I), using exact Python integers.

    Keys are consecutive progress values 1,...,D. Rows are indexed by internal
    length l, not total block length l+1. Row lengths must be nonincreasing;
    this ensures that every coefficient needed by the convolution is known.
    Unknown irreducible tails are omitted only after coefficient extraction.
    """
    if not raw or sorted(raw) != list(range(1, len(raw) + 1)):
        raise ValueError("raw tables must have consecutive progress keys 1,...,D")
    rows = {d: [_bound_integer(c, "block count") for c in row] for d, row in raw.items()}
    sizes = [len(rows[d]) for d in range(1, len(rows) + 1)]
    if not all(sizes) or sizes != sorted(sizes, reverse=True):
        raise ValueError("raw row lengths must be positive and nonincreasing in progress")
    for d, row in rows.items():
        if any(row[:min(2 * d - 1, len(row))]):
            raise ValueError("a block of progress d requires at least 2*d-1 internal edges")
    irreducibles: dict[int, list[int]] = {}
    total = [0] * sizes[0]
    for d in range(1, len(rows) + 1):
        row = rows[d].copy()
        for length in range(len(row)):
            for left_d in range(1, d):
                for left_length in range(length):
                    row[length] -= (
                        irreducibles[left_d][left_length]
                        * rows[d - left_d][length - left_length - 1]
                    )
            if row[length] < 0:
                raise ValueError("raw tables give a negative irreducible coefficient")
            total[length] += row[length]
        irreducibles[d] = row
    return total


def renewal_polynomial_value(counts: Sequence[int], numerator: int, denominator: int) -> int:
    """Return denominator**L * P(numerator/denominator), exactly.

    L=len(counts), P(x)=x**L-sum(counts[l]*x**(L-l-1)).
    Counts are indexed by internal length; a block uses l+1 letters.
    """
    counts = _renewal_counts(counts)
    numerator = _bound_integer(numerator, "numerator")
    denominator = _bound_integer(denominator, "denominator", 1)
    value = 1
    power = 1
    for count in counts:
        power *= denominator
        value = numerator * value - count * power
    return value


def renewal_lower_bound(counts: Sequence[int], denominator: int = 10**9) -> Fraction:
    """Largest positive grid point p/denominator strictly below the renewal root."""
    counts = _renewal_counts(counts)
    denominator = _bound_integer(denominator, "denominator", 1)
    low, high = 0, denominator
    while renewal_polynomial_value(counts, high, denominator) < 0:
        high *= 2
    while high - low > 1:
        middle = (low + high) // 2
        if renewal_polynomial_value(counts, middle, denominator) < 0:
            low = middle
        else:
            high = middle
    if low == 0:
        raise ValueError("denominator is too small for a positive strict lower bound")
    return Fraction(low, denominator)


def renewal_values(counts: Sequence[int], max_length: int) -> list[int]:
    """Return c[0],...,c[max_length]; c[k] <= S[k+2]."""
    counts = _renewal_counts(counts)
    max_length = _bound_integer(max_length, "max_length")
    values = [1] + [0] * max_length
    for k in range(1, max_length + 1):
        values[k] = sum(counts[l] * values[k - l - 1] for l in range(min(k, len(counts))))
    return values


def renewal_lower_prefactor(counts: Sequence[int], bound: Fraction, start: int = 2) -> Fraction:
    """Certify c[k] >= a*bound**k for k>=start by a finite induction base."""
    counts = _renewal_counts(counts)
    start = _bound_integer(start, "start")
    if not isinstance(bound, Fraction) or bound <= 0:
        raise ValueError("bound must be a positive Fraction")
    if renewal_polynomial_value(counts, bound.numerator, bound.denominator) > 0:
        raise ValueError("bound exceeds the renewal root")
    values = renewal_values(counts, start + len(counts) - 1)
    factor = min(Fraction(values[k], 1) / bound**k for k in range(start, len(values)))
    if factor <= 0:
        raise ValueError("no positive prefactor: the induction base contains a zero")
    return factor


def window_graph(m: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Edges of the full valid-window graph, A[target,source]=1.

    State words have m symbols. Enumeration is supported through m+2=20
    wedges by the existing enumerator, although memory becomes prohibitive
    well before that limit. No state or edge is silently omitted.
    """
    m = _bound_integer(m, "m", 1)
    if m > 18:
        raise ValueError("the existing word enumerator supports m <= 18")
    states, _ = RubiksSnakeCounter.enumerate_shapes(m + 1)
    windows, _ = RubiksSnakeCounter.enumerate_shapes(m + 2)
    states.sort()
    prefix, suffix = windows >> 2, windows & ((1 << (2 * m)) - 1)
    source, target = np.searchsorted(states, prefix), np.searchsorted(states, suffix)
    if np.any(source >= len(states)) or np.any(target >= len(states)):
        raise RuntimeError("a window endpoint is absent from the state table")
    if not np.array_equal(states[source], prefix) or not np.array_equal(states[target], suffix):
        raise RuntimeError("a window endpoint is absent from the state table")
    return source, target, len(states)


def _window_cyclic_core(source, target, size):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    adjacency = csr_matrix((np.ones(len(source), dtype=np.uint8), (source, target)), shape=(size, size))
    component_count, labels = connected_components(adjacency, directed=True, connection="strong")
    cyclic = np.bincount(labels, minlength=component_count) > 1
    cyclic[labels[source[source == target]]] = True
    active = cyclic[labels]
    renumber = np.full(size, -1, dtype=np.int64)
    renumber[active] = np.arange(np.count_nonzero(active))
    _, active_labels = np.unique(labels[active], return_inverse=True)
    internal = active[source] & active[target] & (labels[source] == labels[target])
    return renumber[source[internal]], renumber[target[internal]], active_labels, int(active.sum())


def _integer_vector(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0 or values.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty integer vector")
    if values.dtype.kind == "i" and np.any(values < 0):
        raise ValueError(f"{name} must be nonnegative")
    return values.astype(np.uint64, copy=False)


def verify_vector_bound(image: np.ndarray, vector: np.ndarray, bound: Fraction) -> bool:
    """Check image <= bound*vector with exact, overflow-free arithmetic.

    Split v at the denominator to compute floor(bound*v) without forming
    products such as 10**9 * 10**12. Unusual scales use Python integers.
    """
    image, vector = _integer_vector(image, "image"), _integer_vector(vector, "vector")
    if image.shape != vector.shape or np.any(vector == 0):
        raise ValueError("image and positive vector must have the same shape")
    if not isinstance(bound, Fraction) or bound < 0:
        raise ValueError("bound must be a nonnegative Fraction")
    p, d = bound.numerator, bound.denominator
    maximum = int(vector.max())
    limit = int(np.iinfo(np.uint64).max)
    whole, remainder = divmod(p, d)
    if d <= limit and p * maximum // d <= limit and remainder * (d - 1) <= limit:
        for start in range(0, len(vector), 1_000_000):
            v = vector[start:start + 1_000_000]
            ceiling = whole * v + remainder * (v // d) + (remainder * (v % d)) // d
            if np.any(image[start:start + len(v)] > ceiling):
                return False
        return True
    return all(d * int(a) <= p * int(v) for a, v in zip(image, vector))


def _exact_window_image(source, target, vector, size):
    degree = np.bincount(target, minlength=size)
    if int(degree.max()) * int(vector.max()) > int(np.iinfo(np.uint64).max):
        raise OverflowError("integer matrix-vector product would overflow uint64; reduce scale")
    image = np.zeros(size, dtype=np.uint64)
    np.add.at(image, target, vector[source])
    return image


def window_upper_bound(
    m: int, iterations: int = 500, scale: int = 10**12,
    denominator: int = 10**9, full_graph: bool = False
) -> dict:
    """Return a rigorous Fraction upper bound and certificate statistics.

    Default: independently normalized cyclic SCCs certify spectral radius.
    full_graph=True: unshifted power iteration on the entire graph also
    provides a pointwise prefactor H=sum(v)/min(v), starting at n=m+1.
    A poor iteration gives a weaker bound, never an unverified estimate.
    """
    iterations = _bound_integer(iterations, "iterations", 1)
    scale = _bound_integer(scale, "scale", 1)
    denominator = _bound_integer(denominator, "denominator", 1)
    if scale > 2**53 or denominator > 2**53:
        raise ValueError("scale and denominator must not exceed 2**53")
    source, target, size = window_graph(m)
    full_states, full_edges = size, len(source)
    labels = np.empty(0, dtype=np.int64)
    if not full_graph:
        source, target, labels, size = _window_cyclic_core(source, target, size)
    if not size:
        raise RuntimeError("the window graph contains no directed cycle")
    x = np.ones(size, dtype=np.float64)
    component_count = int(labels.max()) + 1 if labels.size else 0
    for _ in range(iterations):
        y = np.bincount(target, weights=x[source], minlength=size)
        if full_graph:
            x = y / y.max() + np.finfo(np.float64).tiny
        else:
            y += x
            norms = np.zeros(component_count, dtype=np.float64)
            np.maximum.at(norms, labels, y)
            x = y / norms[labels]
    if not np.all(np.isfinite(x)):
        raise ArithmeticError("nonfinite candidate vector")
    vector = np.maximum(1, np.rint(x * scale)).astype(np.uint64)
    image = _exact_window_image(source, target, vector, size)
    numerator = int(np.ceil(float(np.max(image / vector)) * denominator))
    bound = Fraction(numerator, denominator)
    while not verify_vector_bound(image, vector, bound):
        numerator += 1
        bound = Fraction(numerator, denominator)
    # Python's sum must be used: the full vector sum can overflow uint64.
    vector_sum = sum(map(int, vector))
    vector_min = int(vector.min())
    return {
        "bound": bound, "states": size, "edges": len(source),
        "full_states": full_states, "full_edges": full_edges,
        "vector_min": vector_min, "vector_sum": vector_sum,
        "iterations": iterations, "full_graph": full_graph,
        "pointwise_factor": Fraction(vector_sum, vector_min) if full_graph else None,
    }
