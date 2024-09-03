import numpy as np
import numba

# Prepare the grid.
MAX_N = 26
K = 2 * (MAX_N // 2)
box_size = 2 * K + 1
dx, dy, dz = 1, box_size, box_size ** 2
CENTER_COORD = K * (dx + dy + dz)

# Pre-calculate geometry.
CUBE = [[1, 3, 4, 2], [0, 2, 5, 3], [0, 4, 5, 1], [0, 1, 5, 4], [0, 3, 5, 2], [1, 2, 4, 3]]
DELTAS = np.array([dy, dz, dx, -dx, -dz, -dy])  # "+y","+z","+x","-x","-z","-y"
WEDGE_ID_TO_FACE_IDS = dict()
FACE_IDS_TO_WEDGE_ID = dict()


def register_wedge(f1, f2, wedge_id):
    WEDGE_ID_TO_FACE_IDS[wedge_id] = (f1, f2)
    FACE_IDS_TO_WEDGE_ID[(f1, f2)] = wedge_id


for i, (f1, f2) in enumerate([(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3)]):
    register_wedge(f1, f2, i + 1)
    register_wedge(f2, f1, i + 1 + 16)
    register_wedge(5 - f1, 5 - f2, 13 - (i + 1))
    register_wedge(5 - f2, 5 - f1, 13 - (i + 1) + 16)

WEDGE_ID_TO_NEXT_DELTA = np.zeros(36, dtype=np.int64)
ROT_AND_WEDGE_ID_TO_NEXT_WEDGE_ID = np.zeros(36 * 4, dtype=np.int64)
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


@numba.jit("i8(i8,i8)", inline="always")
def encode_wedge(coord, wedge_id):
    return (coord << 6) + wedge_id


@numba.jit("(i8,i8,i8[:],i8[:])", inline="always")
def push_wedge(wedge_coord, wedge_id, wedges, cubes):
    wedges[0] -= 1
    wedges[wedges[0]] = encode_wedge(wedge_coord, wedge_id)
    cubes[wedge_coord] += wedge_id & 15


@numba.jit("i8(i8,i8)", inline="always")
def get_next_wedge_coord(last_wedge_id, last_wedge_coord):
    return last_wedge_coord + WEDGE_ID_TO_NEXT_DELTA[last_wedge_id]


@numba.jit("i8(i8,i8)", inline="always")
def get_next_wedge_id(last_wedge_id, rot):
    return ROT_AND_WEDGE_ID_TO_NEXT_WEDGE_ID[last_wedge_id + 36 * rot]


@numba.jit("(i8[:],i8[:],i8[:])")
def _count_shapes_rec(wedges, cubes, total_count):
    last_wedge_index = wedges[0]
    total_count[last_wedge_index] += 1
    if last_wedge_index == 1: return  # Full length shape, stop recusrion.
    last_wedge = wedges[wedges[0]]
    last_wedge_id = last_wedge & 63
    last_wedge_coord = last_wedge >> 6
    next_wedge_coord = get_next_wedge_coord(last_wedge_id, last_wedge_coord)
    next_cube_occupancy_type = cubes[next_wedge_coord]

    if next_cube_occupancy_type == 0 and last_wedge_index == 2:
        total_count[1] += 4
        return
    if next_cube_occupancy_type == 0 and last_wedge_index == 3:
        c = next_wedge_coord
        s = cubes[c - dx] + cubes[c + dx] + cubes[c - dy] + cubes[c + dy] + cubes[c - dz] + cubes[
            c + dz]
        if s == cubes[last_wedge_coord]:
            total_count[2] += 4
            total_count[1] += 16
            return

    for rot in range(4):
        next_wedge_id = get_next_wedge_id(last_wedge_id, rot)
        next_wedge_occupancy_type = next_wedge_id & 15
        can_push = next_cube_occupancy_type == 0 or (
                next_cube_occupancy_type + next_wedge_occupancy_type == 13)
        if can_push:
            push_wedge(next_wedge_coord, next_wedge_id, wedges, cubes)
            _count_shapes_rec(wedges, cubes, total_count)
            cubes[next_wedge_coord] -= next_wedge_occupancy_type  # pop
            wedges[0] += 1  # pop


class RubiksSnakeCounter:
    S = [None, 1, 4, 16, 64, 241, 920, 3384, 12585, 46471, 172226, 633138, 2333757, 8561679,
         31462176, 115247629, 422677188, 1546186675, 5661378449, 20689242550, 75663420126,
         276279455583, 1009416896015, 3683274847187, 13446591920995, 49037278586475]

    @staticmethod
    def count_shapes_all(n):
        total_count = np.zeros(n + 1, dtype=np.int64)
        wedges = np.zeros(n + 1, dtype=np.int64)
        wedges[0] = n + 1  # wedges[0] indicates last wedge index
        cubes = np.zeros(box_size ** 3, dtype=np.int64)
        push_wedge(CENTER_COORD, FACE_IDS_TO_WEDGE_ID[(0, 3)], wedges, cubes)  # Initial wedge.
        _count_shapes_rec(wedges, cubes, total_count)
        return total_count[1:][::-1]

print(RubiksSnakeCounter.count_shapes_all(10))