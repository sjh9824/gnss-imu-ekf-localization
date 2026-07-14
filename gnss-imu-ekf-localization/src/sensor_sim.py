"""
Sensor noise simulation utilities.

1. add_gnss_noise: given a ground-truth trajectory, generate noisy GNSS
   position measurements to emulate a consumer-grade receiver. Used both
   for KITTI (since KITTI OXTS itself is already GPS/IMU-fused and has
   no separate raw noisy GNSS stream) and for the synthetic dataset.

2. generate_synthetic_trajectory: builds a fully synthetic ground-truth
   trajectory + noisy IMU + noisy GNSS, so the whole pipeline can be
   tested/demoed without downloading KITTI.
"""

import numpy as np


def add_gnss_noise(x, y, std=3.0, dropout_prob=0.0, seed=42):
    """
    x, y        : ground-truth positions (m), shape (N,)
    std         : GNSS position noise std (m) - ~3m is typical for consumer GPS
    dropout_prob: probability of a sample being a large outlier (multipath/urban canyon)
    seed        : RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    noise = rng.normal(0, std, size=(n, 2))

    # occasional large outliers to emulate multipath / signal blockage
    outlier_mask = rng.random(n) < dropout_prob
    noise[outlier_mask] *= 8.0

    x_noisy = x + noise[:, 0]
    y_noisy = y + noise[:, 1]
    return x_noisy, y_noisy


def add_imu_noise(accel, yaw_rate, accel_std=0.1, gyro_std=np.deg2rad(0.5), seed=1):
    """
    Adds Gaussian noise + slowly drifting bias to IMU channels, which is
    what causes dead-reckoning error to accumulate over time.
    """
    rng = np.random.default_rng(seed)
    n = len(accel)

    accel_bias = rng.normal(0, accel_std * 0.3)
    gyro_bias = rng.normal(0, gyro_std * 0.3)

    accel_noisy = accel + accel_bias + rng.normal(0, accel_std, size=n)
    gyro_noisy = yaw_rate + gyro_bias + rng.normal(0, gyro_std, size=n)
    return accel_noisy, gyro_noisy


def generate_synthetic_trajectory(duration=60.0, dt=0.1, seed=0):
    """
    Generates a synthetic S-curve vehicle trajectory as ground truth,
    then derives noisy IMU (accel, yaw rate) and noisy GNSS (x, y) from it.

    Returns a dict with the same keys as kitti_loader.load_oxts_sequence,
    so it is a drop-in replacement for testing without KITTI data.
    """
    n = int(duration / dt)
    t = np.arange(n) * dt

    # Ground-truth control: gentle sinusoidal steering + mild acceleration profile
    true_accel = 0.5 * np.sin(2 * np.pi * t / 30.0)
    true_yaw_rate = 0.15 * np.sin(2 * np.pi * t / 20.0)

    x = np.zeros(n)
    y = np.zeros(n)
    yaw = np.zeros(n)
    v = np.zeros(n)
    v[0] = 5.0  # start at 5 m/s

    for i in range(1, n):
        x[i] = x[i - 1] + v[i - 1] * np.cos(yaw[i - 1]) * dt
        y[i] = y[i - 1] + v[i - 1] * np.sin(yaw[i - 1]) * dt
        yaw[i] = yaw[i - 1] + true_yaw_rate[i - 1] * dt
        v[i] = v[i - 1] + true_accel[i - 1] * dt

    af_noisy, wu_noisy = add_imu_noise(true_accel, true_yaw_rate, seed=seed)

    return {
        "t": t,
        "dt": np.diff(t),
        "x": x,
        "y": y,
        "yaw": yaw,
        "vf": v,
        "af": af_noisy,
        "wu": wu_noisy,
    }
