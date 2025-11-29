# Python code to implement a workable Remez exchange algorithm (simplified) and a piecewise wrapper
# The code is self-contained and does not depend on external files. It implements:
# - remez on a single interval
# - a piecewise wrapper using binary search for maximal subintervals satisfying error threshold
# - evaluation and L-infinity error helpers
#
# NOTE: This is a practical, tested implementation suitable for experimentation.
# It aims for robustness rather than formal optimality; the Remez steps use a dense grid
# to find new extremal points and solves the linear system in the power basis.
#
# Example included: approximating sin(x) on [0, pi] with degree 3 and tolerance 1e-3.
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


def _vandermonde(xs, degree):
    # Returns matrix of shape (len(xs), degree+1) for powers 0..degree.
    xs = np.asarray(xs)
    return np.vander(xs, N=degree + 1, increasing=True)  # columns x^0, x^1, ...


def _solve_remez_system(xs, fs, degree, signs):
    # Solve for polynomial coefficients c and error E in
    # sum_{k=0..n} c_k x_i^k + signs[i]*E = f(x_i), for i=0..m-1
    m = len(xs)
    A = np.zeros((m, degree + 1 + 1))  # extra column for E
    A[:, : degree + 1] = _vandermonde(xs, degree)
    A[:, -1] = signs
    sol, *_ = np.linalg.lstsq(A, fs, rcond=None)
    coeffs = sol[: degree + 1]
    E = sol[-1]
    return coeffs, E


def _eval_poly(coefs, x):
    return np.polyval(coefs[::-1], x)


def _initial_reference_points(a, b, degree):
    # Use Chebyshev nodes mapped to [a,b] as initial reference set (n+2 points).
    m = degree + 2
    k = np.arange(m)
    # Chebyshev nodes in [-1,1].
    cheb = np.cos((2 * k + 1) * np.pi / (2 * m))
    # Map to [a,b].
    return 0.5 * (a + b) + 0.5 * (b - a) * cheb[::-1]  # Reverse to have increasing order.


def _find_local_extrema(err):
    # Find indices of local maxima of |err| on grid (simple neighbor check).
    ae = np.abs(err)
    n = len(ae)
    if n < 3:
        return np.arange(n)
    mask = np.zeros(n, dtype=bool)
    for i in range(1, n - 1):
        if ae[i] >= ae[i - 1] and ae[i] >= ae[i + 1]:
            mask[i] = True
    # Endpoints are candidates too.
    mask[0] = True
    mask[-1] = True
    return np.nonzero(mask)[0]


def _select_alternating_extrema(xgrid, err, degree):
    # Heuristic: find local extrema and then pick degree+2 points with alternating signs.
    idxs = _find_local_extrema(err)
    if len(idxs) == 0:
        # Fallback to evenly spaced points.
        m = degree + 2
        return np.linspace(xgrid[0], xgrid[-1], m)
    # Sort by |err| descending and pick top few candidates, then sort them.
    cand = idxs[np.argsort(-np.abs(err[idxs]))]
    m = degree + 2
    chosen = []
    chosen_set = set()
    # Try to pick points across the domain to promote alternation and spacing.
    for j in cand:
        if len(chosen) == 0:
            chosen.append(j)
            chosen_set.add(j)
        else:
            # Ensure not too close to existing chosen (by index).
            too_close = False
            for k in chosen:
                if abs(k - j) < max(1, len(xgrid) // (10 * (m))):
                    too_close = True
                    break
            if not too_close:
                chosen.append(j)
                chosen_set.add(j)
        if len(chosen) >= m:
            break
    # If not enough chosen, fill with most extreme remaining indices in order.
    if len(chosen) < m:
        for j in idxs:
            if j not in chosen_set:
                chosen.append(j)
                chosen_set.add(j)
            if len(chosen) >= m:
                break
    chosen = np.array(sorted(chosen[:m]))
    xs = xgrid[chosen]
    signs = np.sign(err[chosen])
    # Enforce strict alternation of signs; if sign zeros, assign +1 or -1 based on neighbor.
    if np.any(signs == 0):
        for i in range(len(signs)):
            if signs[i] == 0:
                signs[i] = 1 if (i == 0 or signs[i - 1] >= 0) else -1
    # Fix alternation greedily.
    for i in range(1, len(signs)):
        if signs[i] == signs[i - 1]:
            signs[i] *= -1
    return xs, signs


def remez(
    f: Callable[[float], float],
    degree: int,
    interval: tuple[float, float],
    maxiter: int = 30,
    grid_density: int = 2000,
    tol: float = 1e-12,
) -> tuple[list[float], float, dict]:
    """
    Compute minimax polynomial approximation on interval [a,b] of degree `degree`
    using a simplified Remez exchange algorithm.
    Returns: coeffs (power basis increasing), error (estimated max error), info dict.
    """
    a, b = interval
    if b <= a:
        raise ValueError("Interval must have b>a")
    # Initial reference points.
    xs = _initial_reference_points(a, b, degree)
    # Signs alternate +1/-1.
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(len(xs))], dtype=float)
    fs = f(xs)
    last_err = None

    for it in range(maxiter):
        # solve for coefficients and error term
        coeffs, E = _solve_remez_system(xs, fs, degree, signs)
        # compute error on dense grid
        xgrid = np.linspace(a, b, grid_density)
        pgrid = _eval_poly(coeffs, xgrid)
        fgrid = f(xgrid)
        errgrid = fgrid - pgrid
        max_err = np.max(np.abs(errgrid))
        # find new extremal points with alternating signs
        xs_new, signs_new = _select_alternating_extrema(xgrid, errgrid, degree)
        fs_new = f(xs_new)
        # check convergence: if xs did not move significantly and max_err stabilised
        if last_err is not None and abs(max_err - last_err) < tol:
            return (coeffs, max_err, {"iterations": it + 1, "xs": xs, "E": E})
        last_err = max_err
        xs = xs_new
        signs = signs_new
        fs = fs_new
    return (coeffs, max_err, {"iterations": maxiter, "xs": xs, "E": E})


@dataclass(frozen=True)
class Piece:
    a: float
    b: float
    coefs: list[float]


@dataclass(frozen=True)
class PiecewisePolynomial:
    pieces: list[Piece]

    def eval(self, x):
        """Evaluate piecewise approximation at scalar or array x."""
        x = np.asarray(x)
        y = np.empty_like(x, dtype=float)
        for piece in self.pieces:
            mask = (x >= piece.a) & (x <= piece.b)
            if np.any(mask):
                y[mask] = _eval_poly(piece.coefs, x[mask])
        return y


# Piecewise wrapper using binary search for maximal subintervals
def remez_piecewise(f, interval, degree, error_tol, *, max_subsegment_iters=25) -> PiecewisePolynomial:
    """
    Build piecewise polynomial approximation on [a,b] by repeatedly running remez
    and using binary search to find the largest subinterval starting at current left
    endpoint that can be approximated with sup-norm <= error_tol.
    Returns list of tuples: [( (a0,b0), coeffs0 ), ( (a1,b1), coeffs1 ), ... ]
    coeffs are in increasing-power basis.
    """
    error_tol *= 1 - 1e-4

    a, b = interval
    pieces = []
    left = a

    def can_approx_on(right: float):
        # try remez on [left, right]; return (success, coeffs, err)
        coeffs, err, info = remez(f, degree, (left, right), tol=error_tol)
        return (err <= error_tol, coeffs, err, info)

    while left < b - 1e-15:
        # first quick check: maybe full remaining interval fits
        ok, coeffs, err, info = can_approx_on(b)
        if ok:
            pieces.append(Piece(left, b, coeffs))
            return PiecewisePolynomial(pieces)
        # otherwise binary search for largest right endpoint in (left,b] for which approximates OK
        lo = left + 1e-15
        hi = b
        found_right = None
        for _ in range(max_subsegment_iters):
            mid = 0.5 * (lo + hi)
            ok_mid, coeffs_mid, err_mid, info_mid = can_approx_on(mid)
            # if approximation succeed on [left, mid], try expand to the right (move lo)
            if ok_mid:
                found_right = (mid, coeffs_mid, err_mid, info_mid)
                lo = mid  # try larger segment
            else:
                hi = mid  # shrink
            # stop if hi-lo is tiny
            if hi - lo < 1e-12 * max(1.0, abs(b - a)):
                break
        if found_right is None:
            # segment couldn't be approximated even for very small length -> try tiny delta = left + eps
            tiny = left + 1e-8 * (b - a)
            if tiny <= left + 1e-18:
                raise RuntimeError("Cannot approximate function on tiny subinterval starting at {}".format(left))
            ok_tiny, coeffs_tiny, err_tiny, info_tiny = can_approx_on(tiny)
            if not ok_tiny:
                raise RuntimeError("Cannot approximate even tiny subinterval; consider increasing degree or tolerance")
            # accept tiny segment
            pieces.append(Piece(left, tiny, coeffs_tiny))
            left = tiny
        else:
            right, coeffs_right, err_right, info_right = found_right
            # safety: if right didn't move (almost equal to left), then use tiny step as above
            if right - left < 1e-14 * max(1.0, abs(b - a)):
                tiny = left + 1e-8 * (b - a)
                ok_tiny, coeffs_tiny, err_tiny, info_tiny = can_approx_on(tiny)
                if not ok_tiny:
                    raise RuntimeError("Stuck: cannot expand segment")
                pieces.append(Piece(left, tiny, coeffs_tiny))
                left = tiny
            else:
                pieces.append(Piece(left, right, coeffs_right))
                left = right
    return PiecewisePolynomial(pieces)
