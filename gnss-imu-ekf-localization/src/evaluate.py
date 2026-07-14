"""
Evaluation metrics and plotting utilities.
"""

import numpy as np
import matplotlib.pyplot as plt


def rmse(estimated_xy, gt_xy):
    """
    estimated_xy, gt_xy : arrays of shape (N, 2)
    Returns scalar RMSE (m) over the full trajectory.
    """
    diff = estimated_xy - gt_xy
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def plot_trajectories(gt_xy, imu_xy, gnss_xy, ekf_xy, save_path=None, title="Trajectory Comparison"):
    """
    Plots ground truth, IMU-only (dead reckoning), GNSS-only, and EKF fusion
    trajectories on a single 2D plot.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.plot(gt_xy[:, 0], gt_xy[:, 1], color="black", linewidth=2.5, label="Ground Truth", zorder=5)
    ax.plot(imu_xy[:, 0], imu_xy[:, 1], color="tab:red", linestyle="--", linewidth=1.5, label="IMU only (Dead Reckoning)")
    ax.scatter(gnss_xy[:, 0], gnss_xy[:, 1], color="tab:orange", s=8, alpha=0.5, label="GNSS only")
    ax.plot(ekf_xy[:, 0], ekf_xy[:, 1], color="tab:blue", linewidth=2, label="EKF Fusion (GNSS+IMU)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.axis("equal")
    ax.grid(alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved trajectory plot -> {save_path}")
    return fig


def print_rmse_table(rmse_dict):
    print("\n=== RMSE Comparison (lower is better) ===")
    print(f"{'Method':<30}{'RMSE (m)':>12}")
    print("-" * 42)
    for name, val in rmse_dict.items():
        print(f"{name:<30}{val:>12.3f}")
