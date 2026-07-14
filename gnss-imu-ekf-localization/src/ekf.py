"""
Extended Kalman Filter for 2D vehicle localization.

State vector:
    x = [x, y, yaw, v]^T
        x, y   : position in local ENU frame (meters)
        yaw    : heading angle (radians)
        v      : forward velocity (m/s)

Control input (from IMU):
    u = [a, w]^T
        a : forward acceleration (m/s^2)
        w : yaw rate (rad/s)

Measurement (from GNSS):
    z = [x, y]^T   (position only)
"""

import numpy as np


class EKF:
    def __init__(self, x0, P0=None, Q=None, R=None):
        """
        x0 : initial state, shape (4,)  -> [x, y, yaw, v]
        P0 : initial covariance, shape (4,4)
        Q  : process noise covariance, shape (4,4)  (how much we trust IMU)
        R  : measurement noise covariance, shape (2,2)  (how much we trust GNSS)
        """
        self.x = np.array(x0, dtype=float).reshape(4,)
        self.P = P0 if P0 is not None else np.eye(4) * 1.0

        # Default process noise: trust IMU short-term, but allow some drift
        self.Q = Q if Q is not None else np.diag([0.05, 0.05, np.deg2rad(0.5), 0.1]) ** 2

        # Default measurement noise: typical consumer-grade GNSS ~ 3m std
        self.R = R if R is not None else np.diag([3.0, 3.0]) ** 2

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

    @staticmethod
    def _wrap_angle(angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def predict(self, u, dt):
        """
        Prediction step using IMU input.
        u  : [a, w] - forward acceleration, yaw rate
        dt : time step (s)
        """
        x, y, yaw, v = self.x
        a, w = u

        # Nonlinear motion model (bicycle-like kinematics)
        x_new = x + v * np.cos(yaw) * dt
        y_new = y + v * np.sin(yaw) * dt
        yaw_new = self._wrap_angle(yaw + w * dt)
        v_new = v + a * dt

        self.x = np.array([x_new, y_new, yaw_new, v_new])

        # Jacobian of motion model w.r.t. state
        F = np.array([
            [1, 0, -v * np.sin(yaw) * dt, np.cos(yaw) * dt],
            [0, 1,  v * np.cos(yaw) * dt, np.sin(yaw) * dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(self, z):
        """
        Update step using GNSS position measurement.
        z : [x_meas, y_meas]
        """
        z = np.array(z, dtype=float).reshape(2,)

        y_residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_residual
        self.x[2] = self._wrap_angle(self.x[2])

        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()
