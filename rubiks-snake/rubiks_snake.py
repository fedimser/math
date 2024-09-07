import numpy as np
import numba

# ================   GEOMETRY   ===================
# Prepare the grid.
MAX_N = 26
BOX_SIZE = 2 * (MAX_N // 2) + 1
DX, DY, DZ = 1, BOX_SIZE, BOX_SIZE ** 2
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
                ROT_AND_WEDGE_ID_TO_NEXT_WEDGE_ID[wedge_id + rot * 36] = FACE_IDS_TO_WEDGE_ID[
                    (f1p, f2p[rot])]


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
    return ''.join(str((code >> (2 * i)) % 4) for i in range(length))[::-1]


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
        if code < ans: ans = code
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
    cubes = np.zeros(BOX_SIZE ** 3, dtype=np.int64)
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
    if last_wedge_index == 1: return  # Full length shape, stop recusrion.
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
        s = cubes[c - DX] + cubes[c + DX] + cubes[c - DY] + cubes[c + DY] + cubes[c - DZ] + cubes[
            c + DZ]
        if s == cubes[last_wedge_coord]:
            total_count[2] += 4
            total_count[1] += 16
            return

    for rot in range(4):
        next_wedge_id = _get_next_wedge_id(last_wedge_id, rot)
        next_wedge_occupancy_type = next_wedge_id & 15
        can_push = next_cube_occupancy_type == 0 or (
                next_cube_occupancy_type + next_wedge_occupancy_type == 13)
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
    for i in range(4 ** n2):
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
    # Pre-computed up to n=26.
    S = [None, 1, 4, 16, 64, 241, 920, 3384, 12585, 46471, 172226, 633138, 2333757, 8561679,
         31462176, 115247629, 422677188, 1546186675, 5661378449, 20689242550, 75663420126,
         276279455583, 1009416896015, 3683274847187, 13446591920995, 49037278586475,
         178904588083788]

    # Number of formulas of length n-1 describing a loop of n-wedge snake.
    # Equivalent: number of loop-formulas of length n describing a loop of n-wedge snake.
    # Pre-computed up to n=25.
    L1 = [None, 0, 0, 0, 1, 0, 8, 0, 16, 0, 280, 0, 2229, 0, 20720, 0, 226000, 0, 2293422, 0,
          24965960, 0, 275633094, 0, 3069890660, 0]

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
        return [i for i in range(4 ** n) if _is_loop(i, n, wedges, cubes)]

    def __init__(self):
        self.wedges, self.cubes = _prepare_arena(MAX_N, INIT_WEDGE)

    def is_formula_valid(self, formula: str) -> bool:
        n = len(formula)
        enc = encode_formula(formula)
        ans = (_add_wedges_from_formula_while_can(enc, n, self.wedges, self.cubes) == n)
        _pop_all_but_one(self.wedges, self.cubes)
        return ans
