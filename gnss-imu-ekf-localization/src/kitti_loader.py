"""
Loader for KITTI raw dataset OXTS (GPS/IMU) data.

Expected directory structure (standard KITTI raw sequence layout):

    <sequence_dir>/
        oxts/
            data/
                0000000000.txt
                0000000001.txt
                ...
            timestamps.txt

Each oxts/data/*.txt line contains (space-separated, KITTI dataformat.txt spec):
    lat lon alt roll pitch yaw vn ve vf vl vu ax ay az af al au
    wx wy wz wf wl wu pos_accuracy vel_accuracy navstat numsats posmode velmode orimode

Only a subset of these fields is used here:
    lat, lon      -> converted to local ENU (x, y) in meters
    yaw           -> heading (rad)
    vf            -> forward velocity (m/s)
    af            -> forward acceleration (m/s^2)
    wu            -> yaw rate (rad/s, "up" angular rate = vehicle yaw rate)

NOTE: The KITTI OXTS stream is itself a GPS/IMU-fused, RTK-corrected output,
so it is treated here as ground truth. Since KITTI does not distribute a
separate raw/noisy GNSS-only stream, a synthetic GNSS noise model is applied
on top of this ground-truth position to emulate a consumer-grade GNSS
receiver (see `sensor_sim.add_gnss_noise`). The IMU channels (af, wu) used
for prediction are the dataset's real IMU measurements, not synthetic.
"""

import os
import numpy as np

EARTH_RADIUS = 6378137.0  # WGS-84 equatorial radius (m)


def _latlon_to_mercator(lat, lon, lat0):
    """
    Convert lat/lon (degrees) to local x/y (meters) using the same
    Mercator projection approach as the official KITTI devkit, so that
    (0, 0) corresponds to the first frame.
    """
    scale = np.cos(lat0 * np.pi / 180.0)
    x = scale * EARTH_RADIUS * (lon * np.pi / 180.0)
    y = scale * EARTH_RADIUS * np.log(np.tan(np.pi / 4.0 + (lat * np.pi / 180.0) / 2.0))
    return x, y


def load_oxts_sequence(seq_dir):
    """
    Parse a KITTI raw sequence's oxts folder.

    Returns a dict with keys:
        t        : timestamps (s), shape (N,), starts at 0
        dt       : time deltas between consecutive samples (s), shape (N-1,)
        x, y     : ground-truth local position (m), shape (N,)
        yaw      : ground-truth heading (rad), shape (N,)
        vf       : forward velocity (m/s), shape (N,)
        af       : forward acceleration (m/s^2), shape (N,)
        wu       : yaw rate (rad/s), shape (N,)
    """
    data_dir = os.path.join(seq_dir, "oxts", "data")
    ts_path = os.path.join(seq_dir, "oxts", "timestamps.txt")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"OXTS data folder not found at {data_dir}. "
            "Expected KITTI raw layout: <seq_dir>/oxts/data/*.txt"
        )

    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".txt"))
    rows = []
    for fname in files:
        with open(os.path.join(data_dir, fname)) as f:
            vals = [float(v) for v in f.read().split()]
            rows.append(vals)
    rows = np.array(rows)

    lat, lon = rows[:, 0], rows[:, 1]
    yaw = rows[:, 5]
    vf = rows[:, 8]
    af = rows[:, 14]
    wu = rows[:, 21]

    x, y = _latlon_to_mercator(lat, lon, lat[0])
    # shift so trajectory starts at origin
    x = x - x[0]
    y = y - y[0]

    if os.path.isfile(ts_path):
        with open(ts_path) as f:
            lines = [l.strip() for l in f.readlines()]
        # KITTI timestamp format: "2011-09-26 13:21:35.134391445"
        import datetime
        t0 = None
        t = []
        for line in lines:
            dt_obj = datetime.datetime.strptime(line[:26], "%Y-%m-%d %H:%M:%S.%f")
            if t0 is None:
                t0 = dt_obj
            t.append((dt_obj - t0).total_seconds())
        t = np.array(t)
    else:
        # fall back to fixed 10 Hz sampling if timestamps.txt missing
        t = np.arange(len(rows)) * 0.1

    dt = np.diff(t)

    return {
        "t": t,
        "dt": dt,
        "x": x,
        "y": y,
        "yaw": yaw,
        "vf": vf,
        "af": af,
        "wu": wu,
    }
