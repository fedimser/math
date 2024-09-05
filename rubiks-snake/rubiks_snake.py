import numpy as np
import numba

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


@numba.jit("i8(i8,i8)", inline="always")
def _get_next_wedge_coord(last_wedge_id, last_wedge_coord):
    return last_wedge_coord + WEDGE_ID_TO_NEXT_DELTA[last_wedge_id]


@numba.jit("i8(i8,i8)", inline="always")
def _get_next_wedge_id(last_wedge_id, rot):
    return ROT_AND_WEDGE_ID_TO_NEXT_WEDGE_ID[last_wedge_id + 36 * rot]


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


@numba.jit("UniTuple(i8[:],2)(i8,i8)")
def _prepare_arena(n, init_wedge_id):
    wedges = np.zeros(n + 1, dtype=np.int64)
    wedges[0] = n + 1
    cubes = np.zeros(BOX_SIZE ** 3, dtype=np.int64)
    _push_wedge(CENTER_COORD, init_wedge_id, wedges, cubes)
    return wedges, cubes


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


#
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


class RubiksSnakeCounter:
    S = [None, 1, 4, 16, 64, 241, 920, 3384, 12585, 46471, 172226, 633138, 2333757, 8561679,
         31462176, 115247629, 422677188, 1546186675, 5661378449, 20689242550, 75663420126,
         276279455583, 1009416896015, 3683274847187, 13446591920995, 49037278586475,
         178904588083788]

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


if __name__ == "__main__":
    import time

    t0 = time.time()
    print(RubiksSnakeCounter.count_palindrome_shapes(22))
    print("time", time.time() - t0)
