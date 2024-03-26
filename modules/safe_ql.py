#!/usr/bin/env python3
"""Safe Q learning for linear time invariant systems."""
# import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from scipy.integrate import ode

from modules.bases import QuadraticBasis
from modules.proj import proj
from modules.systems import MIMO
from modules.utils import Decomposer


class SafeQL:
    """Class implementing the Safe Q learning algorithm."""

    def __init__(self, A, B, barrier, cost, idealK, idealQ):
        """Safe Q learning class."""
        # Problem specific environments
        self.system = MIMO(A, B, cost.R)
        self.barrier = barrier
        self.cost = cost
        # Dimensions
        self.state_dim, self.action_dim = self.system.dimensions()
        self.basis_dim = (
            (self.state_dim + self.action_dim)
            * (self.state_dim + self.action_dim + 1)
            // 2
        )
        # State decomposition object
        self.dec = Decomposer(
            {
                "x": (self.state_dim, 1),
                "cost": (1, 1),
            }
        )
        self.idealK = idealK
        self.idealQ = idealQ
        self.idealWc = 0.5 * self.vech(idealQ)

        self.internal_basis = QuadraticBasis(self.state_dim + self.action_dim)

        num_freq = 4 * self.basis_dim
        freq_table = 10 * np.random.randn(self.action_dim, num_freq)
        self.freq_table = np.power(freq_table, 2)

        self.actor_gain = 0.2
        self.critic_gain = 50
        self.lamb = 0

    def vech(self, A):
        """Implement vectorization operator.

        :A: A numpy 2D array which is assumed to be a symmetric square matrix
        :returns: A numpy 2D column vector with diagonal elements as it is
        and the upper-triangular elements multiplied by 2
        """
        assert A.shape[0] == A.shape[1]
        assert len(A.shape) == 2

        ans = []
        for i in range(A.shape[0]):
            for j in range(i, A.shape[1]):
                if i == j:
                    ans.append(A[i, j])
                else:
                    ans.append(2 * A[i, j])
        return np.array(ans).reshape((-1, 1))

    def inv_vech(self, vector):
        """Implement inverse of vectorization operator."""
        assert len(vector.shape) == 2, "Please provide a l x 1 vector"
        assert vector.shape[1] == 1, "Please provide a l x 1 vector"

        vec_len = vector.shape[0]
        n = int((np.sqrt(1 + 8 * vec_len) - 1) // 2)

        A = np.zeros((n, n))
        counter = 0
        for i in range(A.shape[0]):
            for j in range(i, A.shape[1]):
                if i == j:
                    A[i, j] = vector[counter]
                else:
                    A[i, j] = 0.5 * vector[counter]
                    A[j, i] = 0.5 * vector[counter]
                counter += 1
        return A

    def basis(self, x, u):
        """
        Compute the basis for the Q function.

        Arguments:
        x: State vector of dimension n
        u: Action vector of dimension m

        Returns:
        basis_vector: vector of dimensions (n+m)(n+m+1)/2
        """
        # Concatenate state and u
        U = np.concatenate((x, u), axis=0)
        return self.internal_basis(U)

    def pe_controller(self, x, time):
        """Generate the pe enforcing controller."""
        if time < 10:
            ans = 2 * np.sum(np.sin(self.freq_table * time), axis=1)
            return ans.reshape(-1, 1)
        else:
            return np.zeros((self.action_dim, 1))

    def controller(self, x, wa_hat, time):
        """Generate the control action."""
        pe_u = self.pe_controller(x, time)
        grad_bf = self.barrier(x, der=True)

        barrier_u = -np.matmul(self.system.rinvg(x), grad_bf)
        return np.matmul(wa_hat.T, x) + pe_u + self.lamb * barrier_u

    def closed_loop(self, time, state, wc_hat, wa_hat):
        """
        Close loop of the sys.

        Arguments:
        time: float
        state: vector of dimension n+1
        wc_hat: vector of dimension
        wa_hat: matrix of dimension

        Returns:
        state_derivative: vector of dimension n+1
        """
        self.dec.computeDecomposition(state)
        x = self.dec.get("x")
        cost = self.dec.get("cost")
        self.dec.clear()

        u = self.controller(x, wa_hat, time)

        x_dot = self.system(x, u)
        cost_dot = self.cost(x, u)

        # Combination of the states
        self.dec.initCompositor()
        # Plant States
        self.dec.add("x", x_dot)
        self.dec.add("cost", cost_dot)
        # self.dec.add("pe", pe_dot)

        return self.dec.combine()

    def set_init_condition(self, **kwargs):
        """Initialize the initial condition for closed loop state vector."""
        self.dec.initCompositor()
        for i in self.dec.dimensions.keys():
            self.dec.add(i, np.random.randn(*self.dec.dimensions[i]))
            if i == "gamma":
                self.dec.add(i, np.eye(self.dec.dimensions[i][0]))

        for i in kwargs.keys():
            self.dec.add(i, kwargs[i])

        self.dec.add("cost", np.zeros((1, 1)))
        self.init_con = self.dec.combine()

    def simulate(
        self,
        wa_hat,
        wc_hat,
        final_time=100,
        dt=0.01,
    ):
        """
        Hybrid simulation loop.

        This is a hybrid simulation loop where the state is updated
        in continuous time domain, while the actor and critic parameters
        are updated in discrete time domain.
        """
        r = ode(self.closed_loop)

        closed_state = self.init_con
        r.set_integrator("dop853", rtol=1e-8, atol=1e-12)
        r.set_initial_value(closed_state)

        T = []
        sol = []
        u_hist = []
        wa_hat_hist = []
        wc_hat_hist = []

        while r.successful() and r.t < final_time:
            r.set_f_params(wc_hat, wa_hat)
            # Fill up the history with past values
            T.append(r.t)
            sol.append(closed_state)
            # Compute decomposition of the previous closed loop state
            self.dec.computeDecomposition(closed_state)
            # Get the previous state and cost
            prev_x = self.dec.get("x")
            prev_cost = self.dec.get("cost")
            # Clear the decomposition
            self.dec.clear()
            # Compute the previous control action
            prev_u = self.controller(prev_x, wa_hat, r.t)

            # Append to the history
            u_hist.append(prev_u.reshape(-1))
            wc_hat_hist.append(wc_hat.copy().reshape(-1))
            wa_hat_hist.append(wa_hat.copy().reshape(-1))

            # Integrate the system for dt time
            closed_state = r.integrate(r.t + dt)

            # Again compute the state decomposition
            self.dec.computeDecomposition(closed_state)
            next_x = self.dec.get("x")
            next_cost = self.dec.get("cost")
            self.dec.clear()

            next_u = self.controller(next_x, wa_hat, r.t + dt)

            integral = next_cost - prev_cost

            prev_basis = self.basis(prev_x, prev_u)
            next_basis = self.basis(next_x, next_u)
            sigma = next_basis - prev_basis
            # Error in critic
            e = integral + np.matmul(wc_hat.T, sigma)

            # Error in actor
            # Extract the components Qxx, Qxu, Quu
            # from wc_hat
            n = self.state_dim
            reconstructed_Q_mat = self.inv_vech(wc_hat)
            # Qxx = reconstructed_Q_mat[:n, :n]
            Qxu = reconstructed_Q_mat[:n, n:]
            Quu = self.cost.R
            Quu_inv = np.linalg.inv(Quu)
            desiredK = -np.matmul(Quu_inv, Qxu.T)
            Ktilde = wa_hat - desiredK.T
            ea = np.matmul(Ktilde.T, prev_x)

            # Actor critic update laws
            normalizing_term = np.power(1 + np.matmul(sigma.T, sigma), 2)
            wc_hat_dot = -self.critic_gain * e * sigma / normalizing_term

            wa_hat_dot = -self.actor_gain * np.matmul(prev_x, ea.T)

            # For practical purposes you don't need the projection operator
            # wa_hat_dot = proj(wa_hat, wa_hat_dot, np.eye(self.action_dim), 100)

            wc_hat += wc_hat_dot * dt
            wa_hat += wa_hat_dot * dt
            # wa_hat = wa_hat - 0.1 * Ktilde

        return {
            "T": np.array(T),
            "sol": np.array(sol),
            "u": np.array(u_hist),
            "wc_hat": np.array(wc_hat_hist),
            "wa_hat": np.array(wa_hat_hist),
        }
