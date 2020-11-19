"""
This section contains the classes and functions necessary to run calculations to determine call and put option pricing,
as well as computation of the Greeks. There are four functions: explicit_method, implicit_method, crank_n_method, and
calc_greeks.
"""

import numpy as np
from scipy.stats import norm
from math import exp, floor, log, sqrt


class Option:
    def __init__(self, S, K, T, r, q, sigma):
        self.S = float(S)  # Stock price in $
        self.K = float(K)  # Exercise price in $
        self.T = float(T) / 365  # Time to maturity in days
        self.r = float(r) / 100  # Risk-free interest rate in %
        self.q = float(q) / 100  # Dividend rate in %
        self.sigma = float(sigma) / 100  # Implied volatility in %


class EuropeanOption(Option):

    def explicit_method(self, N, M):
        S_max = self.K * 2  # Maximum price of stock, assumed to be 2K
        dt = self.T / N  # Equal time periods from t=0 to maturity
        ds = S_max / M  # Equal price levels of stock from 0 to S_max

        # Next steps
        mat_call = np.zeros([M + 1, N + 1])  # Initialise the f_N matrix, with (M + 1) rows and (N + 1) columns
        mat_put = np.zeros([M + 1, N + 1])
        f_N_Array = np.arange(M + 1)  # Initialise the initial values column N
        mat_call[:, 0] = [max(x * ds - self.K, 0) for x in f_N_Array]  # Call value matrix
        mat_put[:, 0] = [max(self.K - x * ds, 0) for x in f_N_Array]  # Put value matrix

        mat_A = np.zeros([M + 1, M + 1])  # Initialise the matrix A
        for j in range(1, M):  # j from 1 to M - 1
            mat_A[j][j - 1] = dt / 2 * ((self.sigma ** 2) * (j ** 2) - (self.r - self.q) * j)
            mat_A[j][j] = 1 - dt * ((self.sigma ** 2) * (j ** 2) + self.r)
            mat_A[j][j + 1] = dt / 2 * ((self.sigma ** 2) * (j ** 2) + (self.r - self.q) * j)
        mat_A[0][0] = 1
        mat_A[M][M] = 1

        for i in range(0, N):
            vec_call = np.dot(mat_A, mat_call[:, i])  # f(i) = A.f(i+1)
            vec_call[-1] = S_max - self.K * exp(-self.r * (i + 1) * dt)
            mat_call[:, i + 1] = vec_call  # f(N) to f(0) columns
            vec_put = np.dot(mat_A, mat_put[:, i])  # f(i) = A.f(i+1)
            vec_put[0] = self.K * exp(-self.r * (i + 1) * dt)
            mat_put[:, i + 1] = vec_put  # f(N) to f(0) columns

        k = floor(self.S / ds)
        V_call = mat_call[k, -1] + (mat_call[k + 1, -1] - mat_call[k, -1]) / ds * (self.S - k * ds)
        V_put = mat_put[k, -1] + (mat_put[k + 1, -1] - mat_put[k, -1]) / ds * (self.S - k * ds)

        return {'call': V_call, 'put': V_put}

    def implicit_method(self, N, M):
        S_max = self.K * 2
        dt = self.T / N
        ds = S_max / M

        mat_call = np.zeros([M + 1, N + 1])  # Initialise the f_N matrix
        mat_put = np.zeros([M + 1, N + 1])
        f_N_Array = np.arange(M + 1)  # Initialise the initial values column N
        mat_call[:, 0] = np.maximum(f_N_Array * ds - self.K, 0)  # Call value matrix
        mat_put[:, 0] = np.maximum(self.K - f_N_Array * ds, 0)  # Put value matrix
        mat_call[-1, 0] = S_max - self.K * exp(-self.r * 1 * dt)
        mat_put[0, 0] = self.K * exp(-self.r * 1 * dt)

        mat_A = np.zeros([M + 1, M + 1])  # Initialise the matrix A
        for j in range(1, M):  # j from 1 to M - 1
            mat_A[j][j] = 1 + dt * (self.sigma ** 2 * (j ** 2) + self.r)
            mat_A[j][j - 1] = dt / 2 * ((self.r - self.q) * j - self.sigma ** 2 * (j ** 2))
            mat_A[j][j + 1] = -dt / 2 * (self.sigma ** 2 * (j ** 2) + (self.r - self.q) * j)
        mat_A[0][0] = 1
        mat_A[M][M] = 1

        mat_A_inv = np.linalg.inv(mat_A)  # Compute the A inverse matrix

        for i in range(0, N):
            vec_call = np.dot(mat_A_inv, mat_call[:, i])  # f(i) = A^-1.f(i+1)
            vec_call[-1] = S_max - self.K * exp(-self.r * (i + 1) * dt)
            mat_call[:, i + 1] = vec_call  # f(N) to f(0) columns
            vec_put = np.dot(mat_A_inv, mat_put[:, i])  # f(i) = A^-1.f(i+1)
            vec_put[0] = self.K * exp(-self.r * (i + 1) * dt)
            mat_put[:, i + 1] = vec_put  # f(N) to f(0) columns

        k = floor(self.S / ds)
        V_call = mat_call[k, -1] + (mat_call[k + 1, -1] - mat_call[k, -1]) / ds * (self.S - k * ds)
        V_put = mat_put[k, -1] + (mat_put[k + 1, -1] - mat_put[k, -1]) / ds * (self.S - k * ds)

        return {'call': V_call, 'put': V_put}

    def crank_n_method(self, N, M):
        S_max = self.K * 2
        dt = self.T / N
        ds = S_max / M

        # Initialise the f_N matrix # R*C
        mat_call = np.zeros([M + 1, N + 1])  # Initialise the f_N matrix
        mat_put = np.zeros([M + 1, N + 1])
        f_N_Array = np.arange(M + 1)  # Initialise the initial values column N
        mat_call[:, 0] = list(map(lambda x: max(x * ds - self.K, 0), f_N_Array))  # Call value matrix
        mat_put[:, 0] = list(map(lambda x: max(self.K - x * ds, 0), f_N_Array))  # Put value matrix

        # Initialise the M2 Matrix
        mat_M2 = np.zeros([M + 1, M + 1])  # M+1 by M+1
        for j in range(1, M):
            mat_M2[j][j] = 1 + (-dt / 2) * (self.sigma ** 2 * (j ** 2) + self.r)  #
            mat_M2[j][j - 1] = (dt / 4) * (self.sigma ** 2 * (j ** 2) - (self.r - self.q) * j)  # Alpha
            mat_M2[j][j + 1] = (dt / 4) * (self.sigma ** 2 * (j ** 2) + (self.r - self.q) * j)  # Gamma
            mat_M2[0][0] = 1
            mat_M2[M][M] = 1

        # Initialise the M1 Matrix
        mat_M1 = np.zeros([M + 1, M + 1])  # M+1 by M+1
        for j in range(1, M):
            mat_M1[j][j] = 1 - (-dt / 2) * (self.sigma ** 2 * (j ** 2) + self.r)  # 1 - Beta
            mat_M1[j][j - 1] = -(dt / 4) * (self.sigma ** 2 * (j ** 2) - (self.r - self.q) * j)  # Alpha
            mat_M1[j][j + 1] = -(dt / 4) * (self.sigma ** 2 * (j ** 2) + (self.r - self.q) * j)  # Gamma
            mat_M1[0][0] = 1
            mat_M1[M][M] = 1

        mat_M1_inv = np.linalg.inv(mat_M1)  # Compute the A inverse matrix

        for i in range(0, N):
            b_call = np.dot(mat_M2, mat_call[:, i])
            b_call[-1] = S_max - self.K * exp(-self.r * (i + 1) * dt)
            vec_call = np.dot(mat_M1_inv, b_call)  # f(i) = M1^-1 . b
            mat_call[:, i + 1] = vec_call  # f(N) to f(0) columns

            b_put = np.dot(mat_M2, mat_put[:, i])
            b_put[0] = self.K * exp(-self.r * (i + 1) * dt)
            vec_put = np.dot(mat_M1_inv, b_put)  # f(i) = M1^-1 . b
            mat_put[:, i + 1] = vec_put  # f(N) to f(0) columns

        k = floor(self.S / ds)
        V_call = mat_call[k, -1] + (mat_call[k + 1, -1] - mat_call[k, -1]) / ds * (self.S - k * ds)
        V_put = mat_put[k, -1] + (mat_put[k + 1, -1] - mat_put[k, -1]) / ds * (self.S - k * ds)

        return {'call': V_call, 'put': V_put}

    def calc_greeks(self):
        S = self.S
        K = self.K
        T = self.T
        r = self.r
        q = self.q
        sigma = self.sigma

        d1 = (log(S / K) + ((r - q) + sigma ** 2 / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        # Greeks for call option
        delta_call = exp(-q * T) * norm.cdf(d1)
        gamma_call = exp(-q * T) * norm.pdf(d1) / (S * sigma * sqrt(T))
        vega_call = exp(-q * T) * S * norm.pdf(d1) * sqrt(T) / 100
        theta_call = (-exp(-q * T) * (S * norm.pdf(d1) * sigma / (2 * sqrt(T))) - (r * K * exp(-r * T) * norm.cdf(d2))
                      + (q * S * exp(-q * T) * norm.cdf(d1))) / 365
        rho_call = K * T * exp(-q * T) * norm.cdf(d2) / 100

        # Greeks for put option
        delta_put = -exp(-q * T) * norm.cdf(-d1)
        gamma_put = gamma_call
        vega_put = vega_call
        theta_put = (-exp(-q * T) * (S * norm.pdf(d1) * sigma / (2 * sqrt(T))) + (r * K * exp(-r * T) * norm.cdf(-d2))
                     - (q * S * exp(-q * T) * norm.cdf(-d1))) / 365
        rho_put = -K * T * exp(-q * T) * norm.cdf(-d2) / 100

        return {"delta_c": delta_call, "gamma_c": gamma_call, "vega_c": vega_call, "theta_c": theta_call,
                "rho_c": rho_call, "delta_p": delta_put, "gamma_p": gamma_put, "vega_p": vega_put,
                "theta_p": theta_put, "rho_p": rho_put}
