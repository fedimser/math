# References.
# [1] Hager, William W., and Hongchao Zhang.
#      "A new conjugate gradient method with guaranteed descent and an efficient line search."
#      SIAM Journal on optimization 16.1 (2005): 170-192.
#      https://www.math.lsu.edu/~hozhang/papers/cg_descent.pdf
# [2] Hager, William W., and Hongchao Zhang.
#       "Algorithm 851: CG_DESCENT, a conjugate gradient method with guaranteed descent."
#       ACM Transactions on Mathematical Software (TOMS) 32.1 (2006): 113-137.
#       http://users.clas.ufl.edu/hager/papers/CG/cg_compare.pdf
# [3] Nocedal, Jorge, and Stephen Wright.
#       Numerical optimization.
#       Springer Science & Business Media, 2006.

import numpy as np


class EvaluatingDict:
    def __init__(self):
        self.d = dict()

    def __setitem__(self, key, val):
        self.d[key] = val

    def eval(self, key, func):
        if key not in self.d:
            self.d[key] = func(key)
        return self.d[key]


# Represents sclar function, defined as:
#   phi(x) := f(x0 + dir * x)
# Analytically calculates derivative for it.
# Caches calls to f and fprime.
class CachedScalarFunction:
    def __init__(self, f, fprime, dir, x0, f_0=None, fprime_0=None):
        self.f = f
        self.fprime = fprime
        self.dir = dir
        self.x0 = x0
        self.vals = EvaluatingDict()
        self.grads = EvaluatingDict()
        self.vals[0.0] = f(x0) if f_0 is None else f_0
        self.grads[0.0] = fprime(x0) if fprime_0 is None else fprime_0

    def val(self, x, must_be_cached=False):
        return self.vals.eval(x, lambda y: self.f(self.x0 + y * self.dir))

    def full_gradient(self, x, must_be_cached=False):
        return self.grads.eval(
            x, lambda y: self.fprime(
                self.x0 + y * self.dir))

    def der(self, x):
        return np.dot(self.dir, self.full_gradient(x))

# Finds mnimum of the function f using .
# Arguments:
#   f - callable for function f.
#   x0 - initial point.
#   fprime - callable for gradient of f.
# Stopping conditions:
#   maxiter - maximal number of iterations to make.
#   gtol - gradient tolerance. Terminates if current gradient notm is below this
#          value.
# Method parameters:
#   delta - range (0, .5), sufficient decrease parameter,
#                          used in the Wolfe conditions (2.2) and (4.1).
#   sigma - range [delta, 1), curature parameter,
#                          used in the Wolfe conditions (2.3) and (4.1).
#   eps - range [0,+inf), used in the approximate Wolfe termination (T2)
#   theta - range (0, 1), used in the update rules when the potential intervals
#       [a, c] or [c, b] violate the opposite slope condition contained in (4.4)
#   gamma - range (0, 1), determines when a bisection step is performed (L2).
#   eta - range (0, +inf), used in the lower bound for beta_k in (1.6).
#   rho - factor by which c_j grows in each step of the bracket routine
# psi_0, psi_1, psi_2 - used for selecting initial step length for line
# search.


def minimize_hz(f, x0, fprime, maxiter=1000, gtol=1e-4,
                delta=0.1,
                sigma=0.9,
                eps=1e-6,
                theta=0.5,
                gamma=0.66,
                eta=0.01,
                rho=5.0,
                psi_0=0.01,
                psi_1=0.1,
                psi_2=2.0,
                quad_step=True):
    x_k = x0
    f_k = f(x_k)
    g_k = fprime(x_k)
    f_km1 = f_k + np.linalg.norm(g_k) / 2
    d_k = -g_k
    k = 0

    while True:
        if np.linalg.norm(g_k) < gtol or k >= maxiter:
            break

        # Preparing to the line search.
        phi = CachedScalarFunction(f, fprime, d_k, x_k, f_0=f_k, fprime_0=g_k)

        # Calculating inital guess for line seacrh.
        c = None
        if k == 0:
            if np.max(x_k) != 0:
                c = psi_0 * np.max(x_k) / np.max(g_k)
            elif f_k != 0:
                c = psi_0 * np.abs(f_k) / np.dot(g_k, g_k)
            else:
                c = 1.0
        else:
            if quad_step:
                # Quadratic interpolation. (See Nocedal's book).
                c = None
                phi_0 = f_k
                derphi_0 = phi.der(0)
                a0 = psi_0 * a_km1
                phi_a0 = phi.val(a0)
                q_koef = phi_a0 - phi_0 - a0 * derphi_0
                if phi_a0 <= phi_0 and q_koef > 0:
                    c = - 0.5 * (derphi_0 * a0**2) / q_koef
            if c is None:
                c = psi_2 * a_km1

        # Line search.
        a_k = hager_zhang_line_search(phi,
                                      initial_guess=c,
                                      delta=delta,
                                      sigma=sigma,
                                      eps=eps,
                                      theta=theta,
                                      gamma=gamma,
                                      rho=rho)

        # Evalutaing new direction.
        x_kp1 = x_k + a_k * d_k
        f_kp1 = phi.val(a_k, must_be_cached=True)
        g_kp1 = phi.full_gradient(a_k, must_be_cached=True)
        y_k = g_kp1 - g_k
        y_k_n2 = np.dot(y_k, y_k)
        b1 = np.dot(y_k - 2 * d_k * y_k_n2 / np.dot(d_k, y_k), g_kp1)
        b2 = np.dot(d_k, y_k)
        b_k = b1 / b2
        eta_k = -1.0 / (np.linalg.norm(d_k) * min(eta, np.linalg.norm(g_k)))
        b_k = max(b_k, eta_k)
        d_kp1 = -g_kp1 + b_k * d_k

        # Moving to next step.
        k += 1
        f_km1 = f_k
        a_km1 = a_k
        x_k = x_kp1
        g_k = g_kp1
        d_k = d_kp1
        f_k = f_kp1

    #print('Done %d iterations.' % k)
    return x_k


def hager_zhang_line_search(phi,
                            initial_guess=1.0,
                            delta=0.1,
                            sigma=0.9,
                            eps=1e-6,
                            theta=0.5,
                            gamma=0.66,
                            rho=5.0):
    phi_0 = phi.val(0)
    derphi_0 = phi.der(0)
    eps_k = eps * abs(phi_0)

    # Step U3 of update procedure from [2].
    def interval_update_u3(a, b):
        for iter in range(100):
            d = (1 - theta) * a + theta * b
            if phi.der(d) >= 0:
                return a, d
            else:
                if phi.val(d) <= phi_0 + eps_k:
                    a = d
                else:
                    b = d
        print('Warning. Iterations exceeded in interval_update_u3.')
        return a, b

    # Given inital guess [0, c], reduces it to [a,b], for which opposite slope
    # condition is satisifed.
    def bracket(c):
        c_j = c
        c_i = 0
        for iter in range(100):
            if phi.der(c_j) >= 0:
                return c_i, c_j
            else:
                if phi.val(c_j) > phi_0 + eps_k:
                    return interval_update_u3(0, c_j)
            if phi.val(c_j) <= phi_0 + eps_k:
                c_i = c_j
            c_j = rho * c_j
        print('Warning. Iterations exceeded in bracket.')
        return c_i, c_j

    def interval_update(a, b, c):
        assert (a < b)
        if c <= a or c >= b:
            return a, b
        if phi.der(c) >= 0:
            return a, c
        else:
            if phi.val(c) <= phi_0 + eps_k:
                return c, b
            else:
                return interval_update_u3(a, c)

    def secant(a, b):
        if a == b:
            return (a + b) / 2
        derphi_a = phi.der(a)
        derphi_b = phi.der(b)
        if abs(derphi_a - derphi_b) < eps_k:
            return (a + b) / 2
        return (a * derphi_b - b * derphi_a) / (derphi_b - derphi_a)

    def double_secant(a, b):
        c = secant(a, b)
        A, B = interval_update(a, b, c)
        if c == B:
            c1 = secant(b, B)
        if c == A:
            c1 = secant(a, A)
        if c == A or c == B:
            return interval_update(A, B, c1)
        return A, B

    a_k, b_k = bracket(initial_guess)
    for iter in range(100):
        phi_a = phi.val(a_k)
        derphi_a = phi.der(a_k)

        # L0. Check condittion T1 (Wolfe conditions) for a.
        if a_k * delta * derphi_0 >= phi_a - phi_0 and derphi_a >= sigma * derphi_0:
            break

        # L0. Check T2 (approx Wolfe):
        if (2 * delta - 1) * derphi_0 >= derphi_a and derphi_a >= sigma * \
                derphi_0 and phi_a <= phi_0 + eps_k:
            break

        # L1.
        a, b = double_secant(a_k, b_k)

        # L2.
        if b - a > gamma * (b_k - a_k):
            c = 0.5 * (a + b)
            a, b = interval_update(a, b, c)

        # L3.
        a_k, b_k = a, b
    return a_k
