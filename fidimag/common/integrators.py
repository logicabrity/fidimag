"""
Implement the time integrators used in the simulation of magnetisation dynamics.

"""
import numpy as np
from scipy.integrate import ode
from fidimag.extensions.cvode import CvodeSolver as SundialsIntegrator

EPSILON = 1e-16


class BaseIntegrator(object):
    def __init__(self, spins, rhs_fun):
        self.y = spins
        self.ydot = np.zeros(spins.shape)
        self.t = 0
        self.rhs = rhs_fun
        self.rhs_evals_nb = 0

    def run_until(self, t):
        pass

    def set_initial_value(self, spins, t):
        pass

    def rhs_evals(self):
        return self.rhs_evals_nb


class ScikitsSundialsIntegrator(object):
    def __init__(self, spins, rhs_fun):
        super(ScikitsSundialsIntegrator, self).__init__(spins, rhs_fun)

        # wrapping RHS function to count evals, until we find a way
        # to access the stats or patch upstream (to bmcage/odes)
        def rhs_wrap(t, y, ydot):
            self.rhs_evals_nb += 1
            rhs_fun(t, y, ydot)
        self.rhs = rhs_wrap

        from scikits.odes import ode
        self.solver = ode('cvode', self.rhs, old_api=False,
                                             max_steps=1e6,
                                             rtol=1e-8,
                                             atol=1e-8,
                                             order=2,
                                             maxl=300)
        # TODO: make options accessible, add preconditioner/jacobian

    def run_until(self, t):
        self.solver.step(t, y_retn=self.y)

    def set_initial_value(self, spins, t):
        self.solver.init_step(t, spins)


class StepIntegrator(BaseIntegrator):
    def __init__(self, spins, rhs_fun, step="euler", stepsize=1e-15):
        super(StepIntegrator, self).__init__(spins, rhs_fun)
        step_choices = {'euler': euler_step, 'rk4': runge_kutta_step}
        if step not in step_choices:
            raise NotImplemented("step must be euler or rk4")
        self.step = step_choices[step]
        self.stepsize = stepsize

    def run_until(self, t):
        while abs(self.t - t) > EPSILON:
            self.t, self.y, evals = self.step(self.t, self.y, self.ydot, self.stepsize, self.rhs)
            self.rhs_evals_nb += evals
            if self.t > t:
                break
        return 0


class ScipyIntegrator(BaseIntegrator):
    def __init__(self, spins, rhs_fun):
        super(ScipyIntegrator, self).__init__(spins, rhs_fun)
        self.integrator_created = False
        self.internal_timesteps = [0]

        def rhs_wrap(y, t):
            self.rhs_evals_nb += 1
            rhs_fun(t, y, self.ydot)
            return self.ydot
        self.rhs = rhs_wrap  # overwriting rhs to count evals

    def solout(self, t, y):
        self.internal_timesteps.append(t)
        return 0  # all ok signal for scipy

    def set_tols(self, rtol, atol):
        self.rtol = rtol
        self.atol = atol

    def _create_integrator(self):
        self.ode = ode(self.rhs).set_integrator("dopri5", rtol=self.rtol, atol=self.atol)
        self.ode.set_solout(self.solout)  # needs to be before set_initial_value for scipy < 0.17.0
        self.ode.set_initial_value(self.y, self.t)
        self.integrator_created = True

    def run_until(self, t):
        if not self.integrator_created:
            # as late as possible so the user had a chance to set options
            self._create_integrator()

        r = self.ode.integrate(t)
        if not self.ode.successful():
            raise RuntimeError("integration with ode unsuccessful")
        self.y[:] = r
        self.t = t
        return 0


def euler_step(t, y, ydot, h, f):
    """
    Numerical integration using the Euler method.

    Given the initial value problem y'(t) = f(t, y(t)), y(t_0) = y_0 one step
    of size h is y_{n+1} = y_n + h * f(t_n, y_n).

    """
    tp = t + h
    f(t, y, ydot)
    yp = y + h * ydot
    evals = 1
    return tp, yp, evals


def runge_kutta_step(t, y, ydot, h, f):
    """
    Numerical integration using the classical Runge-Kutta method (RK4).

    Given the initial value problem y'(t) = f(t, y(t)), y(t_0) = y_0 one step
    of size h is y_{n+1} = y_n + h/6 * (k_1 + 2k_2 + 2k_3 + k4), where the
    weights are:
        k_1 = f(t_n,       y_n)
        k_2 = f(t_n + h/2, y_n + h/2 * k_1)
        k_3 = f(t_n + h/2, y_n + h/2 * k_2)
        k_4 = f(t_n + h,   y_n + h   * k_3).

    """
    yp = y.copy()

    f(t, y, ydot)  # ydot = k1
    yp += h / 6.0 * ydot

    f(t + h / 2.0, y + h * ydot / 2.0, ydot)  # ydot = k2
    yp += h / 3.0 * ydot

    f(t + h / 2.0, y + h * ydot / 2.0, ydot)  # ydot = k3
    yp += h / 3.0 * ydot 

    f(t + h, y + h * ydot, ydot)  # ydot = k4
    yp += h / 6.0 * ydot

    tp = t + h
    evals = 4
    return tp, yp, evals
