#!/usr/bin/env python3
"""
calibrate_intrinsics.py  —  TRACK 1b   (run once)

Checkerboard intrinsics for the overhead camera. Produces the yaml that
global_cam_bridge_node loads and that ray-casting depends on.

Capture grabs from the ROS topic the bridge already publishes -- NOT from the
UDP socket directly (the bridge owns port 5000, and the stream is a custom
length-prefixed JPEG protocol that VideoCapture cannot parse anyway).

    # 1. bridge must be running
    # 2. capture ~20 views:  SPACE = save, q = quit
    python calibrate_intrinsics.py --capture

    # 3. solve
    python calibrate_intrinsics.py --images ./calib_shots \
        --rows 6 --cols 9 --square 0.025 \
        --out config/global_cam_intrinsics.yaml

CRITICAL: calibrate at the SAME resolution you stream at (640x480 by default).
Intrinsics are resolution-dependent -- fx, fy, cx, cy are in pixels. Calibrating
at one resolution and streaming at another silently corrupts every ray you cast.
"""

import argparse
import glob
import os

import cv2
import numpy as np
import yaml


def calibrate(image_paths, rows, cols, square, out):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

    obj_points, img_points = [], []
    shape = None

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        shape = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not found:
            print(f"  no board: {os.path.basename(path)}")
            continue

        corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        obj_points.append(objp)
        img_points.append(corners)
        print(f"  ok: {os.path.basename(path)}")

    if len(obj_points) < 8:
        raise SystemExit(f"only {len(obj_points)} usable views -- want 15+")

    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, shape, None, None)

    print(f"\nRMS reprojection error: {rms:.4f} px   ({len(obj_points)} views)")
    print("  < 0.5 excellent | < 1.0 fine | > 1.0 recapture")
    print(f"K =\n{K}\ndist = {dist.ravel()}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        yaml.safe_dump({
            "image_width": int(shape[0]),
            "image_height": int(shape[1]),
            "camera_matrix": K.tolist(),
            "distortion_model": "plumb_bob",
            "distortion_coefficients": dist.ravel().tolist(),
            "rms_reprojection_error_px": float(rms),
        }, f)
    print(f"\nsaved -> {out}")


def capture(topic, outdir, rows, cols):
    """Grab frames from the bridge's ROS topic. SPACE saves, q quits."""
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge

    os.makedirs(outdir, exist_ok=True)
    rclpy.init()
    node = Node("calib_capture")
    bridge = CvBridge()
    state = {"frame": None, "n": len(glob.glob(f"{outdir}/*.png"))}

    def cb(msg):
        state["frame"] = bridge.imgmsg_to_cv2(msg, "bgr8")

    node.create_subscription(Image, topic, cb, 10)
    print(f"subscribing to {topic} -- SPACE saves, q quits")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if state["frame"] is None:
                continue
            disp = state["frame"].copy()

            # live feedback: is the board even detectable in this pose?
            gray = cv2.cvtColor(disp, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                flags=cv2.CALIB_CB_FAST_CHECK)
            if found:
                cv2.drawChessboardCorners(disp, (cols, rows), corners, found)

            cv2.putText(disp,
                        f"saved:{state['n']}  board:{'YES' if found else 'no'}"
                        f"  SPACE=save q=quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if found else (0, 165, 255), 2)
            cv2.imshow("calib capture", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                path = f"{outdir}/shot_{state['n']:03d}.png"
                cv2.imwrite(path, state["frame"])
                print(f"saved {path}")
                state["n"] += 1
            elif key == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--topic", default="/global_cam/image_raw")
    ap.add_argument("--images", default="./calib_shots")
    ap.add_argument("--rows", type=int, default=6, help="INNER corners per column")
    ap.add_argument("--cols", type=int, default=9, help="INNER corners per row")
    ap.add_argument("--square", type=float, default=0.025, help="square edge (m)")
    ap.add_argument("--out", default="config/global_cam_intrinsics.yaml")
    args = ap.parse_args()

    if args.capture:
        capture(args.topic, args.images, args.rows, args.cols)
    else:
        paths = sorted(glob.glob(f"{args.images}/*.png") + glob.glob(f"{args.images}/*.jpg"))
        if not paths:
            raise SystemExit(f"no images in {args.images}")
        calibrate(paths, args.rows, args.cols, args.square, args.out)
