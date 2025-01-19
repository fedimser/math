import numpy as np
import scipy
import scipy.stats


def _normalize(x):
    l = np.linalg.norm(x)
    if l > 1e-9:
        x /= l
    return x


class OptimizationTestCase:
    def __init__(self,
                 name='',
                 start_point=[0],
                 func=lambda x: x,
                 grad=None,
                 expected_argmin=None,
                 gtol=1e-4):
        self.name = name
        self.start_point = np.array(start_point, dtype=np.float64)
        self.dim = self.start_point.shape[0]
        self.func = func
        self.grad = grad
        self.gtol = gtol
        if expected_argmin is None:
            self.expected_argmin = np.zeros(self.dim)
        else:
            self.expected_argmin = np.array(expected_argmin, dtype=np.float64)

    def prepare(self):
        self.func_call_counter = 0
        self.grad_call_counter = 0

    def get_func_value(self, x):
        if x.shape[0] != self.dim:
            raise ValueError(
                'Wrong dimension: %d!=%d' %
                (s.shape[0], self.dim))
        self.func_call_counter += 1
        return self.func(x)

    def _get_numerical_grad_value(self, x):
        eps = 1e-9
        f_x = self.func(x)

        def e(i):
            ans = np.zeros(self.dim, dtype=np.float64)
            ans[i] = 1.0
            return ans
        return np.array([self.func(x + eps * e(i)) -
                         f_x for i in range(self.dim)]) / eps

    def get_grad_value(self, x):
        if x.shape[0] != self.dim:
            raise ValueError(
                'Wrong dimension: %d!=%d' %
                (s.shape[0], self.dim))
        self.grad_call_counter += 1
        gr_num = self._get_numerical_grad_value(x)
        if self.grad is None:
            return gr_num
        else:
            gr_exact = self.grad(x)
            diff = gr_exact - gr_num
            if not np.allclose(gr_exact, gr_num, atol=1e-1, rtol=1e-1):
                raise ValueError(
                    'Analytic formula for gradient is wrong! '
                    'x=%s, grad_exact=%s, grad_num=%s, diff=%s' %
                    (x, gr_exact, gr_num, diff))
            return gr_exact

    # Runs given optimization algorithm on this testcase.
    # `algo` must be a function which takes f, fprime and start_point, and
    # returns found arg_min.
    def check_algorithm(self, algo):
        # print("Running testcase " + self.name)
        self.prepare()
        if self.grad is None:
            print(
                'Warning! Gradient is not defined for %s, will approximate' %
                self.name)
        x_found = algo(self.get_func_value,
                       self.start_point,
                       self.get_grad_value,
                       gtol=self.gtol)
        correct = self.near(x_found, self.expected_argmin)
        if not correct:
            print(
                "Wrong answer. Expected %s, found %s. Grad norm=%s" %
                (str(
                    self.expected_argmin), str(x_found), np.linalg.norm(
                    self.get_grad_value(x_found))))
        return {
            'test_case': self.name,
            'func_calls': self.func_call_counter,
            'grad_calls': self.grad_call_counter,
            'cost': self.func_call_counter + 3 * self.grad_call_counter,
            'correct': correct,
            'x': x_found,
            'f': self.func(x_found),
            'g': self.get_grad_value(x_found)
        }

    def compare(self, algo1, algo2, verbose=False):
        print("Running comparison for " + self.name)
        self.prepare()
        prev_point = self.start_point
        for maxiter in range(100):
            x1 = algo1(self.get_func_value,
                       self.start_point,
                       self.get_grad_value,
                       gtol=self.gtol,
                       maxiter=maxiter)
            x2 = algo2(self.get_func_value,
                       self.start_point,
                       self.get_grad_value, gtol=self.gtol,
                       maxiter=maxiter)
            dir1 = _normalize(x1 - prev_point)
            dir2 = _normalize(x2 - prev_point)
            if not self.near(x1, x2):
                print(
                    'Diverged at step %d: %s vs %s' %
                    (maxiter, str(x1), str(x2)))
                if self.near(dir1, dir2):
                    print('Directions were the same, but step size different')
                else:
                    print('Directions were different')
                return False
            else:
                if verbose:
                    print('Step %d: %s' % (maxiter, str(x1)))
                prev_point = x1
            if self.near(x1, self.expected_argmin):
                return True
        return True

    def near(self, a, b):
        return np.allclose(a, b, rtol=1e-2, atol=1e-2)


TEST_CASES = []

# 1D functions.
TEST_CASES.append(OptimizationTestCase(
    name='Parabola 1D',
    start_point=[100.0],
    func=lambda x: (x[0] - 20)**2,
    grad=lambda x: 2.0 * (x[0] - 20),
    expected_argmin=[20.0]))


############ Paraboloids. ##########################

def paraboloid_random_tc(n, seed, name):
    np.random.seed(43)
    A = np.random.uniform(size=(n, n))
    A = A @ A.T  # Must be positive semidefinite.
    b = np.random.uniform(size=(n,))
    return OptimizationTestCase(
        name=name,
        start_point=np.random.uniform(size=(n,)),
        func=lambda x: 0.5 * (x @ A @ x) + (b@ x),
        grad=lambda x: (A @ x) + b,
        expected_argmin=- np.linalg.inv(A) @ b)


TEST_CASES.append(OptimizationTestCase(
    name='Paraboloid simplest',
    start_point=[1, 1],
    func=lambda x: np.sum(np.square(x)),
    grad=lambda x: 2 * x,
    expected_argmin=[0, 0]))
TEST_CASES.append(paraboloid_random_tc(2, 123, 'Paraboloid 2D v1'))
TEST_CASES.append(paraboloid_random_tc(2, 42, 'Paraboloid 3D v1'))
TEST_CASES.append(paraboloid_random_tc(5, 123, 'Paraboloid 5D v1'))
TEST_CASES.append(paraboloid_random_tc(10, 123, 'Paraboloid 10D v1'))

TEST_CASES.append(OptimizationTestCase(
    name='Paraboloid of 4th order',
    start_point=[1, 2, 3, 4, 5],
    func=lambda x: np.sum(x**4),
    grad=lambda x: 4 * x**3,
    expected_argmin=[0, 0, 0, 0, 0],
    gtol=1e-10))

############ Rosenbrock function. ###################
# https://en.wikipedia.org/wiki/Rosenbrock_function


def rosenbrock_tc(start_point, name):
    return OptimizationTestCase(
        name=name,
        start_point=start_point,
        func=lambda x: (1.0 - x[0])**2 + 100 * (x[1] - x[0]**2)**2,
        grad=lambda x: np.array([
            - 2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
            200 * (x[1] - x[0]**2)]),
        expected_argmin=[1.0, 1.0])


TEST_CASES.append(rosenbrock_tc([-1.2, 2], 'Rosenbrock 2D v1'))
TEST_CASES.append(rosenbrock_tc([7, -12], 'Rosenbrock 2D v1'))
TEST_CASES.append(OptimizationTestCase(
    name='Rosenbrock 7D',
    start_point=np.zeros(7),
    func=lambda x: 100 *
    np.sum(np.square(x[1:] - np.square(x[:-1]))) +
    np.sum(np.square(1 - x[:-1])),
    grad=lambda x: np.concatenate([[0], 200 * (x[1:] - np.square(x[:-1]))])
    + np.concatenate([- 400 * (x[1:] - np.square(x[:-1])) * x[:-1]
                      - 2 * (1 - x[:-1]), [0]]),
    expected_argmin=np.ones(7)))


########### Himmelblau function. #######################

def himmelblau_tc(start_point, end_point, name):
    return OptimizationTestCase(
        name=name,
        start_point=start_point,
        func=lambda x: (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2,
        grad=lambda x: np.array([
            4 * (x[0]**2 + x[1] - 11) * x[0] + 2 * (x[0] + x[1]**2 - 7),
            2 * (x[0]**2 + x[1] - 11) + 4 * (x[0] + x[1]**2 - 7) * x[1]
        ]),
        expected_argmin=end_point)


TEST_CASES.append(himmelblau_tc([-50, -50], [3.0, 2.0], 'Himmelblau v1'))
TEST_CASES.append(himmelblau_tc(
    [50, -50], [-2.805118, 3.131312], 'Himmelblau v2'))
TEST_CASES.append(himmelblau_tc(
    [50, 50], [-3.779310, -3.283186], 'Himmelblau v3'))
TEST_CASES.append(himmelblau_tc(
    [-50, 50], [3.584428, -1.848126], 'Himmelblau v4'))

########### McCormick function. #######################
TEST_CASES.append(OptimizationTestCase(
    name='McCormick',
    start_point=[0, 0],
    func=lambda x: np.sin(x[0] + x[1]) + np.square(x[0] -
                                                   x[1]) - 1.5 * x[0] + 2.5 * x[1] + 1,
    grad=lambda x: np.array([
        np.cos(x[0] + x[1]) + 2 * (x[0] - x[1]) - 1.5,
        np.cos(x[0] + x[1]) - 2 * (x[0] - x[1]) + 2.5,
    ]),
    expected_argmin=[-0.54719, -1.54719]))

########### Beale function. #######################
# https://en.wikipedia.org/wiki/Rastrigin_function
TEST_CASES.append(OptimizationTestCase(
    name='Beale',
    start_point=[-1.0, -1.0],
    func=lambda x: (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] +
                                                    x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0] * x[1]**3)**2,
    grad=lambda x: np.array([
        2 * (1.5 - x[0] + x[0] * x[1]) * (-1 + x[1]) +
        2 * (2.25 - x[0] + x[0] * x[1]**2) * (-1 + x[1]**2) +
        2 * (2.625 - x[0] + x[0] * x[1]**3) * (-1 + x[1]**3),
        2 * x[0] * (1.5 - x[0] + x[0] * x[1]) +
        4 * x[0] * x[1] * (2.25 - x[0] + x[0] * x[1]**2) +
        6 * x[0] * x[1]**2 * (2.625 - x[0] + x[0] * x[1]**3)
    ]),
    expected_argmin=[3.0, 0.5], gtol=1e-8))
