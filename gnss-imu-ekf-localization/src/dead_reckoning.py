"""
Dead Reckoning: position estimation using IMU (acceleration, yaw rate) only.
No external correction -> error accumulates over time.
"""

import numpy as np


def dead_reckoning(accel, yaw_rate, dt_arr, x0=0.0, y0=0.0, yaw0=0.0, v0=0.0):
    """
    accel    : array of forward acceleration (m/s^2), shape (N,)
    yaw_rate : array of yaw rate (rad/s), shape (N,)
    dt_arr   : array of time steps (s), shape (N,)
    x0,y0,yaw0,v0 : initial state

    Returns:
        traj : array of shape (N+1, 4) -> [x, y, yaw, v] at each step (including t=0)
    """
    n = len(accel)
    traj = np.zeros((n + 1, 4))
    traj[0] = [x0, y0, yaw0, v0]

    x, y, yaw, v = x0, y0, yaw0, v0
    for i in range(n):
        dt = dt_arr[i]
        x = x + v * np.cos(yaw) * dt
        y = y + v * np.sin(yaw) * dt
        yaw = yaw + yaw_rate[i] * dt
        v = v + accel[i] * dt
        traj[i + 1] = [x, y, yaw, v]

    return traj
