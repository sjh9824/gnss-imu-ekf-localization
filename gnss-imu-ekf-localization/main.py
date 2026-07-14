"""
GNSS + IMU based vehicle localization using an Extended Kalman Filter.

Usage:
    # Run with synthetic data (no KITTI download needed, good for a quick demo/sanity check)
    python main.py --mode synthetic

    # Run with a real KITTI raw sequence
    python main.py --mode kitti --seq_dir /path/to/2011_09_26_drive_0005_sync
"""

import argparse
import os
import numpy as np

from src.ekf import EKF
from src.dead_reckoning import dead_reckoning
from src.sensor_sim import add_gnss_noise, generate_synthetic_trajectory
from src.kitti_loader import load_oxts_sequence
from src.evaluate import rmse, plot_trajectories, print_rmse_table


def run_pipeline(data, gnss_std=3.0, out_dir="outputs"):
    os.makedirs(out_dir, exist_ok=True)

    gt_xy = np.stack([data["x"], data["y"]], axis=1)
    n = len(data["af"])
    dt_arr = data["dt"]

    # --- 1. Simulated noisy GNSS measurements (see sensor_sim.add_gnss_noise) ---
    gnss_x, gnss_y = add_gnss_noise(data["x"], data["y"], std=gnss_std)
    gnss_xy = np.stack([gnss_x, gnss_y], axis=1)

    # --- 2. IMU-only Dead Reckoning ---
    dr_traj = dead_reckoning(
        accel=data["af"][:-1] if len(data["af"]) == n else data["af"],
        yaw_rate=data["wu"][:-1] if len(data["wu"]) == n else data["wu"],
        dt_arr=dt_arr,
        x0=data["x"][0], y0=data["y"][0], yaw0=data["yaw"][0], v0=data["vf"][0],
    )
    dr_xy = dr_traj[:, :2]

    # --- 3. EKF fusion (predict every step with IMU, update every step with GNSS) ---
    ekf = EKF(x0=[data["x"][0], data["y"][0], data["yaw"][0], data["vf"][0]])
    ekf_xy = np.zeros((n + 1, 2))
    ekf_xy[0] = [data["x"][0], data["y"][0]]

    for i in range(n - 1):
        ekf.predict(u=[data["af"][i], data["wu"][i]], dt=dt_arr[i])
        ekf.update(z=[gnss_xy[i + 1, 0], gnss_xy[i + 1, 1]])
        ekf_xy[i + 1] = ekf.x[:2]
    ekf_xy = ekf_xy[: n]

    # --- 4. Evaluation ---
    gt_eval = gt_xy[: len(dr_xy)]
    results = {
        "IMU only (Dead Reckoning)": rmse(dr_xy, gt_eval),
        "GNSS only": rmse(gnss_xy[: len(dr_xy)], gt_eval),
        "EKF Fusion (GNSS+IMU)": rmse(ekf_xy[: len(dr_xy)], gt_eval),
    }
    print_rmse_table(results)

    with open(os.path.join(out_dir, "rmse_table.txt"), "w") as f:
        for name, val in results.items():
            f.write(f"{name}: {val:.3f} m\n")

    plot_trajectories(
        gt_eval, dr_xy, gnss_xy[: len(dr_xy)], ekf_xy[: len(dr_xy)],
        save_path=os.path.join(out_dir, "trajectory_comparison.png"),
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="GNSS+IMU EKF localization")
    parser.add_argument("--mode", choices=["synthetic", "kitti"], default="synthetic")
    parser.add_argument("--seq_dir", type=str, default=None, help="Path to KITTI raw sequence (required if --mode kitti)")
    parser.add_argument("--gnss_std", type=float, default=3.0, help="Simulated GNSS noise std (m)")
    parser.add_argument("--out_dir", type=str, default="outputs")
    args = parser.parse_args()

    if args.mode == "synthetic":
        print("Running with synthetic trajectory (no KITTI data required)...")
        data = generate_synthetic_trajectory()
    else:
        if not args.seq_dir:
            raise ValueError("--seq_dir is required when --mode kitti")
        print(f"Loading KITTI OXTS sequence from {args.seq_dir} ...")
        data = load_oxts_sequence(args.seq_dir)

    run_pipeline(data, gnss_std=args.gnss_std, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
