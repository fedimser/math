from rubiks_snake import *


class Wedge:
    @staticmethod
    def from_code(wedge_code):
        return Wedge(wedge_code >> 6, wedge_code & 63)

    def __init__(self, coord, wedge_id):
        self.coord = coord
        self.wedge_id = wedge_id

    def __repr__(self):
        x, y, z = self.get_coord_relative_to_center()
        f1, f2 = WEDGE_ID_TO_FACE_IDS[self.wedge_id]
        return f"({x},{y},{z};{f1}->{f2})"

    def get_next(self, rot):
        return Wedge(
            self.coord + WEDGE_ID_TO_NEXT_DELTA[self.wedge_id],
            ROT_AND_WEDGE_ID_TO_NEXT_WEDGE_ID[self.wedge_id + 36 * rot],
        )

    def occ_type(self):
        return self.wedge_id & 15

    def get_coord_relative_to_center(self):
        x = self.coord % BOX_SIZE
        y = (self.coord // BOX_SIZE) % BOX_SIZE
        z = (self.coord // BOX_SIZE) // BOX_SIZE
        c = BOX_SIZE // 2
        return x - c, y - c, z - c

    def get_face_id_2(self):
        return WEDGE_ID_TO_FACE_IDS[self.wedge_id][1]


def construct_formula(formula: list[int]):
    """Returns wedges (if formula is valid), or None if it's invalid."""
    wedges = [Wedge(CENTER_COORD, FACE_IDS_TO_WEDGE_ID[(0, 3)])]
    cubes = {wedges[0].coord: wedges[0].occ_type()}
    assert len(formula) + 1 <= MAX_N
    for rot in formula:
        next_wedge = wedges[-1].get_next(rot)
        if next_wedge.coord not in cubes:
            cubes[next_wedge.coord] = next_wedge.occ_type()
        else:
            assert cubes[next_wedge.coord] > 0
            can_push = cubes[next_wedge.coord] + next_wedge.occ_type() == 13
            if not can_push:
                return None
            cubes[next_wedge.coord] += next_wedge.occ_type()
            assert cubes[next_wedge.coord] == 13
        wedges.append(next_wedge)
    return wedges


def is_formula_valid(formula):
    if type(formula) is str:
        formula = list(map(int, formula))
    return construct_formula(formula) is not None


def enumerate_valid_formulas_slow(n):
    ans = []
    rots = np.zeros(n - 1, dtype=np.int64)
    for i in range(4 ** (n - 1)):
        for j in range(n - 1):
            rots[j] = (i >> (2 * j)) & 3
        if is_formula_valid(rots):
            ans.append("".join(map(str, rots)))
    return ans
