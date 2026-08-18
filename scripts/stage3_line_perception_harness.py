"""
Stage 3 perception harness: painted-line -> (offset, angle).

Runs a STATIONARY G1 in a project-local MuJoCo scene:
  configs/line_follow/stage3_line_scene.xml
which includes the real menagerie g1.xml unchanged and paints a green
polyline on a dark checker floor.  This harness:

  1. loads the robot at the 'stand' keyframe (parked, not walking),
  2. renders the headcam view with an offscreen renderer,
  3. detects the green line via pure-numpy color segmentation,
  4. fits a robust line through the per-row centroids,
  5. computes the standard two line-following features,
       offset    in [-1, 1]  (normalized image-center deviation)
       angle_deg in [-90, 90] (normalized-coordinate line pitch)
  6. validates them against a ground-truth computed with an independent
     pinhole projection of the actual painted-line geometry into the
     same camera, reduced with the EXACT same per-row-centroid + robust
     fit code as the detector.

No locomotion / env / checkpoint file is touched.

Usage (PowerShell, from repo root):
    .\venv\Scripts\python.exe scripts\stage3_line_perception_harness.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "configs" / "line_follow" / "stage3_line_scene.xml"
CAMERA_NAME = "headcam"
MODEL_PATH = str(SCENE)


# ----------------------------------------------------------------------
# Shared reduction: per-row centroids + robust fit -> (offset, angle)
# ----------------------------------------------------------------------

def _fit_reduce(xs: np.ndarray, ys: np.ndarray, width: int, height: int):
    """
    xs  : column (u) pixels, top-left origin
    ys  : row (v) pixels, top-left origin
    Returns (offset, angle_deg) using the identical convention for both
    the detector and the geometric ground truth.
    """
    if len(xs) < 5:
        return 0.0, 0.0

    cov = np.polyfit(ys, xs, 1)
    res = xs - np.polyval(cov, ys)
    mad = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    thresh = float(np.median(np.abs(res))) + 3.0 * mad + 1e-9
    inl = np.abs(res) <= thresh
    if int(inl.sum()) >= 5:
        cov = np.polyfit(ys[inl], xs[inl], 1)

    slope_px = float(cov[0])
    center_col = float(np.mean(xs[inl])) if int(inl.sum()) >= 5 else float(np.mean(xs))

    offset = 2.0 * (center_col / width - 0.5)

    # Normalized-coordinate line pitch, +v points down the image.
    dx_n = slope_px * (1.0 / width)
    dy_n = 1.0 / height
    angle_deg = float(np.degrees(np.arctan2(dy_n, dx_n)))
    angle_deg = float(np.clip(angle_deg, -89.0, 89.0))

    return offset, angle_deg


def _row_centroids(mask: np.ndarray, margin: float, width: int, height: int):
    """
    Reduce a boolean green mask to per-row centroid (x, y) samples inside
    the scan band [margin, 1-margin] of the image height.
    """
    y0 = int(height * margin)
    y1 = int(height * (1.0 - margin))
    sub = mask[y0:y1, :]

    counts = sub.sum(axis=1)
    good_rows = np.where(counts >= 3)[0]
    if len(good_rows) >= 5:
        cols = np.arange(width)
        xs = np.empty(len(good_rows), dtype=np.float64)
        for i, rr in enumerate(good_rows):
            xs[i] = float(cols[sub[rr]].mean())
        ys = (y0 + good_rows).astype(np.float64)
    else:
        row_ids, col_ids = np.nonzero(sub)
        ys = (y0 + row_ids).astype(np.float64)
        xs = col_ids.astype(np.float64)
        if len(xs) < 5:
            xs = np.array([])
            ys = np.array([])
    return xs, ys


# ----------------------------------------------------------------------
# Detector (pure numpy color segmentation)
# ----------------------------------------------------------------------

def detect_line(img: np.ndarray, row_margin: float = 0.25):
    """
    Pure-numpy green-line detection.

    Returns:
        offset      : normalized lateral deviation in [-1, 1]
        angle_deg   : normalized-coordinate line pitch in [-90, 90]
        coverage    : fraction of scan rows that found green pixels
        n_green     : total green pixel count
    """
    h, w = img.shape[:2]
    r = img[..., 0].astype(np.int16)
    g = img[..., 1].astype(np.int16)
    b = img[..., 2].astype(np.int16)

    mask = (g > 120) & (g > r + 30) & (g > b + 30)

    y0 = int(h * row_margin)
    y1 = int(h * (1.0 - row_margin))
    counts = mask[y0:y1, :].sum(axis=1)
    n_green = int(mask.sum())
    covered = int((counts > 0).sum())
    coverage = covered / max(y1 - y0, 1)

    if n_green < 50:
        return 0.0, 0.0, coverage, 0

    xs, ys = _row_centroids(mask, row_margin, w, h)
    offset, angle_deg = _fit_reduce(xs, ys, w, h)
    return offset, angle_deg, coverage, n_green


# ----------------------------------------------------------------------
# Ground truth: independent pinhole projection of the painted line
# ----------------------------------------------------------------------

def _sample_line_points(model: mujoco.MjModel) -> np.ndarray:
    """
    Sample 3D points along the painted line geoms 'line_seg_*' using the
    model's own geometry definitions (positions, orientations, half-length).
    """
    pts = []
    for gid in range(model.ngeom):
        name = bytes(model.geom_names[gid]).decode("utf-8", "replace")
        if not name.startswith("line_seg_"):
            continue
        quat = np.asarray(model.geom_quat[gid], dtype=np.float64)
        mat = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(mat, quat)          # w,x,y,z quat -> rotation
        x_axis = mat[:3]                         # local +x in world coords
        center = np.asarray(model.geom_pos[gid], dtype=np.float64)
        for t in (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45):
            pts.append(center + t * x_axis)
    return np.asarray(pts, dtype=np.float64)


def _project_samples(model, data, cam_id, pts, img_w, img_h):
    """
    Pinhole-project world points into the headcam image (top-left pixel
    convention) using ONLY the sim camera pose -- independent of the
    renderer, so it is true ground truth.
    """
    cam_pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
    cam_mat = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
    fovy = float(model.cam_fovy[cam_id])
    fy_px = (img_h / 2.0) / np.tan(np.radians(fovy) / 2.0)
    fx_px = fy_px * (img_w / img_h)

    us, vs = [], []
    for p in pts:
        rel = cam_mat.T @ (p - cam_pos)
        fwd = float(rel[2])
        if fwd <= 1e-6:
            continue
        u = img_w / 2.0 + fx_px * (rel[0] / fwd)
        v = img_h / 2.0 - fy_px * (rel[1] / fwd)
        if 0.0 <= u < img_w and 0.0 <= v < img_h:
            us.append(u)
            vs.append(v)
    return np.asarray(us, dtype=np.float64), np.asarray(vs, dtype=np.float64)


def ground_truth_detect(model, data, cam_id, img_w, img_h, row_margin):
    """
    Ideal-renderer ground truth: project the true line geometry, then run
    through the SAME per-row centroid + robust-fit reduction as the real
    detector.  Any remaining difference is therefore renderer/segmentation
    error, not convention error.
    """
    pts = _sample_line_points(model)
    us, vs = _project_samples(model, data, cam_id, pts, img_w, img_h)
    if len(us) < 5:
        return 0.0, 0.0, 0.0

    y0 = int(img_h * row_margin)
    y1 = int(img_h * (1.0 - row_margin))
    keep = (vs >= y0) & (vs <= y1)
    us, vs = us[keep], vs[keep]
    if len(us) < 5:
        return 0.0, 0.0, 0.0

    # Per-row centroids (same rule as the detector: >=3 px per row).
    rows = np.unique(vs)
    xs = np.empty(len(rows), dtype=np.float64)
    for i, rr in enumerate(rows):
        sel = vs == rr
        xs[i] = float(us[sel].mean())

    offset, angle_deg = _fit_reduce(xs, rows.astype(np.float64), img_w, img_h)
    return offset, angle_deg, len(rows)


# ----------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------

def make_model():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if kid < 0:
        raise RuntimeError("no 'stand' keyframe in g1.xml")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    mujoco.mj_forward(model, data)
    return model, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seg-margin", type=float, default=0.30,
                        help="fraction of rows cropped at top/bottom")
    parser.add_argument("--save-preview", type=str,
                        default=str(ROOT / "results" / "stage3_line_preview.png"))
    parser.add_argument("--offset-tol", type=float, default=0.07)
    parser.add_argument("--angle-tol", type=float, default=5.0)
    parser.add_argument("--min-coverage", type=float, default=0.35)
    args = parser.parse_args()

    model, data = make_model()
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
    if cam_id < 0:
        sys.exit(f"camera not found: {CAMERA_NAME}")

    # Aim the chase camera at the line ahead of the robot.
    focus_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cam_focus")
    if focus_id < 0:
        sys.exit("cam_focus body not found")
    fj = model.jnt_qposadr[model.body_jntadr[focus_id]]
    data.qpos[fj:fj + 3] = [2.0, 0.2, 0.10]
    data.qpos[fj + 3:fj + 7] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    try:
        poses = [
            ("aligned", [0.0, 0.0]),
            ("shifted", [0.0, 0.15]),
        ]

        rows = []
        for name, shift in poses:
            data.qpos[0:3] = [shift[0], shift[1], data.qpos[2]]
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            mujoco.mj_forward(model, data)

            renderer.update_scene(data, camera=cam_id)
            img = renderer.render()

            offset, angle_deg, coverage, n_green = detect_line(
                img, row_margin=args.seg_margin
            )
            off_gt, ang_gt, n_gt_rows = ground_truth_detect(
                model, data, cam_id, img.shape[1], img.shape[0], args.seg_margin
            )

            d_off = abs(offset - off_gt)
            d_ang = abs(angle_deg - ang_gt)
            ok = (
                d_off <= args.offset_tol
                and d_ang <= args.angle_tol
                and coverage >= args.min_coverage
            )

            rows.append((name, offset, angle_deg, off_gt, ang_gt,
                         d_off, d_ang, coverage, n_green, ok, img))

            print(
                f"[{name:8s}] det(off={offset:+.3f}, ang={angle_deg:+6.2f}) "
                f"gt(off={off_gt:+.3f}, ang={ang_gt:+6.2f}) "
                f"err={d_off:.3f}/{d_ang:.2f} cov={coverage:.2f} "
                f"green={n_green:5d} gtRows={n_gt_rows:2d} "
                f"{'PASS' if ok else 'FAIL'}"
            )

        all_ok = all(r[9] for r in rows)
        print(f"\nSTAGE 3 RESULT: {'PASS' if all_ok else 'FAIL'}")
        if not all_ok:
            print("One or more validation poses exceeded tolerance.")
            sys.exit(1)

        if args.save_preview:
            from matplotlib import pyplot as plt

            fig, axes = plt.subplots(1, len(rows), figsize=(12, 4.5))
            if len(rows) == 1:
                axes = [axes]
            for ax, (name, offset, angle_deg, off_gt, ang_gt, _,
                     _, coverage, n_green, _, img) in enumerate(rows):
                ax.imshow(img)
                ax.set_title(
                    f"{name}\ndet({offset:+.2f},{angle_deg:+.0f}) "
                    f"gt({off_gt:+.2f},{ang_gt:+.0f}) cov={coverage:.2f}"
                )
                ax.axis("off")
            Path(args.save_preview).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.save_preview, dpi=120, bbox_inches="tight")
            print("Preview saved:", args.save_preview)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
