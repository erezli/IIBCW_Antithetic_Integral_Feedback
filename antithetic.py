import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.optimize import fsolve


class Antithetic:
    def __init__(self, omega=0, rho=1, theta1=1, theta2=1, k=1, degrade_p=1, mu=100, eta=100):
        self.omega = omega
        self.rho = rho
        self.theta1 = theta1
        self.theta2 = theta2
        self.k = k
        self.degrade_p = degrade_p
        self.mu = mu
        self.eta = eta
        self.A = np.array([])

    def system(self, t, x, pbar, state):
        # progress bar feature edited from https://gist.github.com/thomaslima/d8e795c908f334931354da95acb97e54
        # x = [x1, x2, z1, z2]
        # cannot use matrix form because non-linearity
        x1_dot = self.theta1 * x[2] / (self.rho + x[2]) - self.degrade_p * x[0]
        x2_dot = self.k * x[0] - self.degrade_p * x[1] + self.omega
        z1_dot = self.mu - self.eta * x[2] * x[3]
        z2_dot = self.theta2 * x[1] - self.eta * x[2] * x[3]

        if pbar:
            last_t, dt = state
            n = int((t - last_t) / dt)
            pbar.update(n)

            # we need this to take into account that n is a rounded number.
            state[0] = last_t + dt * n
        return np.array([x1_dot, x2_dot, z1_dot, z2_dot])

    def get_response(self, init, length, methods='RK45', show_progress=True, plot=0, label=""):
        t = (0.0, float(length))
        if show_progress:
            with tqdm(total=1000, unit="‰") as pbar:
                timeseries = solve_ivp(
                    self.system,
                    t,
                    init,
                    method=methods,
                    args=[pbar, [0.0, (length - 0.0) / 1000]],
                )
        else:
            timeseries = solve_ivp(
                self.system,
                t,
                init,
                method=methods,
                args=[None, [0.0, (length - 0.0) / 1000]],
            )
        # if plot and isinstance(plot, int):
        #     plt.plot(time_series.t, time_series.y[plot+1])
        #     plt.xlabel("time")
        #     plt.ylabel("{}".format(label))
        return timeseries

    def responses_at_theta1(self, init, length, theta1, methods='RK45'):
        for t in theta1:
            self.theta1 = t
            timeseries = self.get_response(init, length, methods, show_progress=False)
            yield timeseries, t

    def linearisation(self, theta1):
        for t in theta1:
            self.theta1 = t
            x2_ss = self.mu / self.theta2
            x1_ss = (self.degrade_p * x2_ss - self.omega) / self.k
            z1_ss = (self.degrade_p * x1_ss * self.rho) / (self.theta1 - self.degrade_p * x1_ss)
            z2_ss = self.mu / (self.eta * z1_ss)
            self.A = np.array([[-self.degrade_p, 0, self.theta1 * self.rho / (self.rho + z1_ss) ** 2, 0],
                               [self.k, -self.degrade_p, 0, 0],
                               [0, 0, -self.eta * z2_ss, -self.eta * z1_ss],
                               [0, self.theta2, -self.eta * z2_ss, -self.eta * z1_ss]])
            w, v = np.linalg.eig(self.A)
            yield w

    def stable_threshold_rho(self, rho, init):
        x2_ss = self.mu / self.theta2
        x1_ss = (self.degrade_p * x2_ss - self.omega) / self.k
        func = lambda t: (self.degrade_p ** 3) * 2 * self.eta * \
                         (rho + (self.degrade_p * x1_ss * rho) / (t - self.degrade_p * x1_ss)) ** 2 - \
                         t * rho * self.eta * self.k * self.theta2
        theta_initial_guess = init
        stable = fsolve(func, theta_initial_guess)
        # print("The solution is theta = %f" % stable)
        # print("at which the value of the expression is %f" % func(stable))
        return stable

    def stable_threshold_omega(self, omega, init):
        x2_ss = self.mu / self.theta2
        x1_ss = (self.degrade_p * x2_ss - omega) / self.k
        func = lambda t: self.degrade_p ** 3 * 2 * self.eta * \
                         (self.rho + (self.degrade_p * x1_ss * self.rho) / (t - self.degrade_p * x1_ss)) ** 2 \
                         - t * self.rho * self.eta * self.k * self.theta2
        theta_initial_guess = init
        stable = fsolve(func, theta_initial_guess)
        return stable
