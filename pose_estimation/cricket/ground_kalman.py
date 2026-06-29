"""Singer-acceleration Kalman filter on the 2D ground plane (P4a global tracking).

State is ``[x, y, vx, vy, ax, ay]`` in world metres. The Singer model gives
role-aware manoeuvrability (a bowler turns harder than a stationary umpire).
Filter state is exposed as public KF-style attributes (``x``, ``P``, ``F`` …)
so the track manager never reaches into private fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_discrete_lyapunov

# 95th percentile of chi-squared with 2 DOF — the default ground-plane gate.
CHI2_95_2DOF = 5.991


@dataclass(frozen=True)
class RoleParams:
    alpha: float              # manoeuvre frequency (1/s); higher = more agile
    sigma_a: float            # acceleration-noise std (m/s^2)
    measurement_noise: float  # position-measurement std (m)


# Canonical defaults; P4Config may override per deployment.
ROLE_PARAMS: dict[str, RoleParams] = {
    "bowler":       RoleParams(alpha=2.0, sigma_a=3.0, measurement_noise=0.3),
    "striker":      RoleParams(alpha=1.5, sigma_a=2.5, measurement_noise=0.3),
    "non_striker":  RoleParams(alpha=0.5, sigma_a=1.0, measurement_noise=0.3),
    "wicketkeeper": RoleParams(alpha=0.3, sigma_a=0.5, measurement_noise=0.2),
    "umpire":       RoleParams(alpha=0.2, sigma_a=0.3, measurement_noise=0.2),
    "fielder":      RoleParams(alpha=1.0, sigma_a=2.0, measurement_noise=0.4),
    "unknown":      RoleParams(alpha=1.0, sigma_a=2.0, measurement_noise=0.4),
}


def _singer_dynamics(alpha: float, sigma_a: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Discrete ``(F_d, Q_d)`` for the Singer model. State ``[x, y, vx, vy, ax, ay]``."""

    n = 6
    Fc = np.zeros((n, n))
    Fc[0, 2] = 1.0; Fc[1, 3] = 1.0  # pos <- vel
    Fc[2, 4] = 1.0; Fc[3, 5] = 1.0  # vel <- acc
    Fc[4, 4] = -alpha; Fc[5, 5] = -alpha

    F_d = expm(Fc * dt)

    # Van Loan method for the discrete process-noise covariance Q_d.
    G = np.zeros((n, 2))
    G[4, 0] = 1.0; G[5, 1] = 1.0
    Q_c = np.eye(2) * (sigma_a ** 2)
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -Fc
    M[:n, n:] = G @ Q_c @ G.T
    M[n:, n:] = Fc.T
    expM = expm(M * dt)
    Q_d = expM[n:, n:].T @ expM[:n, n:]
    Q_d = 0.5 * (Q_d + Q_d.T)  # symmetrize
    return F_d, Q_d


class SingerGroundKalman:
    """Singer-model Kalman filter on the 2D ground plane with Mahalanobis gating."""

    def __init__(
        self,
        pos_world_xy: np.ndarray,
        role: str = "unknown",
        dt: float = 1.0,
        *,
        role_params: dict[str, RoleParams] | None = None,
        initial_pos_var: float = 4.0,
        initial_vel_var: float = 4.0,
        initial_acc_var: float = 2.0,
    ) -> None:
        self.dt = dt
        self.role = role
        self._role_params = role_params if role_params is not None else ROLE_PARAMS

        self.H = np.zeros((2, 6))
        self.H[0, 0] = 1.0; self.H[1, 1] = 1.0  # observe x, y

        self.x = np.zeros(6)
        self.x[:2] = np.asarray(pos_world_xy, float)
        # High initial uncertainty; velocity and acceleration are unknown.
        self.P = np.diag([
            initial_pos_var, initial_pos_var,
            initial_vel_var, initial_vel_var,
            initial_acc_var, initial_acc_var,
        ]).astype(float)

        params = self._role_params[role]
        self.F, self.Q = _singer_dynamics(params.alpha, params.sigma_a, dt)
        self.R = np.eye(2) * (params.measurement_noise ** 2)

    def predict(self) -> None:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z_world_xy: np.ndarray) -> None:
        z = np.asarray(z_world_xy, float)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(2))
        self.x = self.x + K @ (z - self.H @ self.x)
        I_KH = np.eye(6) - K @ self.H
        # Joseph form for numerical stability.
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    def mahalanobis_sq(self, z_world_xy: np.ndarray) -> float:
        """Squared Mahalanobis distance of a measurement (compare to a chi^2 gate)."""

        z = np.asarray(z_world_xy, float)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            return float(innovation @ np.linalg.solve(S, innovation))
        except np.linalg.LinAlgError:
            return float("inf")

    def switch_role(self, new_role: str) -> None:
        """Swap the motion model, inflating P (Lyapunov steady state) to avoid overconfidence."""

        params = self._role_params[new_role]
        new_F, new_Q = _singer_dynamics(params.alpha, params.sigma_a, self.dt)
        try:
            P_ss = solve_discrete_lyapunov(new_F, new_Q)
            self.P = self.P + 2.0 * P_ss
        except Exception:
            self.P = self.P * 4.0  # fallback inflation
        self.F, self.Q = new_F, new_Q
        self.R = np.eye(2) * (params.measurement_noise ** 2)
        self.role = new_role

    def cap_covariance(self, max_pos_var: float = 25.0) -> None:
        """Prevent covariance blow-up during long Lost windows."""

        for i in range(2):
            if self.P[i, i] > max_pos_var:
                self.P = self.P * (max_pos_var / self.P[i, i])
                break

    def propagate_state(self, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x_pred, P_pred)`` after ``n_frames`` of prediction without mutating state."""

        x = self.x.copy()
        P = self.P.copy()
        for _ in range(n_frames):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
        return x, P

    @property
    def pos_world_xy(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity_xy(self) -> np.ndarray:
        return self.x[2:4].copy()
