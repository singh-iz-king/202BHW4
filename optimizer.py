import pandas as pd
import numpy as np


def beta_cd1(Y, X, Beta, Omega, lmbda, n_outer=10):
    n, q = X.shape
    p = Y.shape[1]
    x_norm_sq = np.sum(X * X, axis=0)

    for _ in range(n_outer):
        XB = X @ Beta
        R = Y - XB

        for j in range(p):
            omega_jj = max(Omega[j, j], 1e-8)

            weights = Omega[:, j].copy()
            weights[j] = 0.0

            # restore column j so the pseudo-response uses only k != j residuals
            R[:, j] = Y[:, j] - X @ Beta[:, j]
            new_y = Y[:, j] + (R @ weights) / omega_jj

            beta_old = Beta[:, j].copy()
            beta_j = beta_old.copy()
            r = new_y - X @ beta_j

            for m in range(q):
                if x_norm_sq[m] <= 1e-12:
                    beta_j[m] = 0.0
                    continue

                r += X[:, m] * beta_j[m]
                rho = X[:, m] @ r
                beta_j[m] = soft_threshold(rho, lmbda / omega_jj) / x_norm_sq[m]
                r -= X[:, m] * beta_j[m]

            Beta[:, j] = beta_j
            R[:, j] = Y[:, j] - X @ beta_j

    return Beta


def beta_cd(Y, X, Beta, Omega, lmbda):

    q, p = Beta.shape

    for _ in range(10):
        for j in range(p):
            R = Y - X @ Beta
            weights = Omega[:, j].copy()
            weights[j] = 0.0
            omega_jj = max(Omega[j, j], 1e-6)
            new_y = Y[:, j] + (R @ weights) / omega_jj

            beta_j = Beta[:, j].copy()

            r = new_y - X @ beta_j

            x_norm_sq = np.sum(X * X, axis=0)

            for m in range(X.shape[1]):
                if x_norm_sq[m] == 0:
                    beta_j[m] = 0.0
                    continue

                r = r + X[:, m] * beta_j[m]

                rho = X[:, m] @ r
                beta_j[m] = soft_threshold(rho, lmbda / omega_jj) / x_norm_sq[m]

                r = r - X[:, m] * beta_j[m]

            R[:, j] += X @ (Beta[:, j] - beta_j)
            Beta[:, j] = beta_j

    return Beta


def neighberhood_selection_cd(Y, X, Beta, lmbda):

    q, p = Beta.shape
    R = Y - X @ Beta
    mask = ~np.eye(p, dtype=bool)

    Omega = np.zeros((p, p))

    for j in range(p):

        X_j = R[:, mask[j, :]]
        y = R[:, j]

        x_norm_sq = np.sum(X_j * X_j, axis=0)

        beta_j = np.zeros(p - 1)

        for s in range(10):
            for m in range(X_j.shape[1]):
                if x_norm_sq[m] == 0:
                    beta_j[m] = 0.0
                    continue

                r_partial = y - X_j @ beta_j + X_j[:, m] * beta_j[m]
                rho = X_j[:, m] @ r_partial
                beta_j[m] = soft_threshold(rho / x_norm_sq[m], lmbda / x_norm_sq[m])

        r = y - X_j @ beta_j
        sigma2 = max((r @ r) / len(r), 1e-6)
        Omega[j, mask[j, :]] = -beta_j * (1 / sigma2)
        Omega[j, j] = 1 / sigma2

    return 0.5 * (Omega + Omega.T)


def ADMM(Y, X, Omega, Gamma, U, Beta, rho, lmbda, nsweeps):

    R = Y - X @ Beta
    S = R.T @ R / Y.shape[0]

    mask = ~np.eye(Omega.shape[0], dtype=bool)

    for k in range(1000):
        Gamma_old = Gamma.copy()

        # Omega update
        M = rho * (Gamma - U) - S
        eigvals, eigvecs = np.linalg.eigh(M)
        new_vals = (eigvals + np.sqrt(eigvals**2 + 4 * rho)) / (2 * rho)
        Omega = eigvecs @ np.diag(new_vals) @ eigvecs.T

        # Gamma update
        Gamma = Omega + U
        Gamma[mask] = soft_threshold(Gamma[mask], lmbda / rho)
        np.fill_diagonal(Gamma, np.diag(Omega + U))

        # Dual update
        U = U + Omega - Gamma

        # Convergence check
        r_norm = np.linalg.norm(Omega - Gamma, ord="fro")
        s_norm = rho * np.linalg.norm(Gamma - Gamma_old, ord="fro")

        if r_norm < 1e-3 and s_norm < 1e-3:
            break

    return Omega, Gamma, U


def soft_threshold(z, lmbda):
    return np.sign(z) * np.maximum(np.abs(z) - lmbda, 0.0)
