#!/usr/bin/env python3
"""Unaligned MPJPE and depth / metric-scale errors of the three models on 3DPW, against the real
camera-frame ground truth. Fills the S0 metrics the project plan lists that the PA-MPJPE / PVE
evaluation hides (Procrustes alignment removes rotation, translation and scale).

Per record (GT box, every 20th frame; same selection as the earlier 3DPW evaluations):
  pa_mpjpe_mm        Procrustes-aligned MPJPE over 24 joints. A sanity check: it must reproduce the
                     earlier numbers (HMR2.0b 58.6, TokenHMR 51.1, CameraHMR 45.7 mm).
  mpjpe_root_mm      MPJPE after aligning the pelvis only (rotation and scale are NOT removed): pose,
                     shape, global orientation and body size all count.
  mpjpe_asis_mm      MPJPE with the model's own camera, nothing aligned. Dominated by the focal length
                     each model assumes: HMR2.0b / TokenHMR use a fixed 5000 px per 256 px of the longer
                     image side (37,500 px on a 1080x1920 photo; the true focal length is about 1,960 px),
                     CameraHMR predicts one from the image.
  mpjpe_gtk_mm       the same, but the predicted body is moved to where it would stand under the TRUE
                     camera intrinsics while keeping its image position and apparent size (depth rescaled
                     by f_true / f_model). This separates "the model assumed the wrong focal length" from
                     "the model put the body at the wrong distance".
  root_err_gtk_mm    3-D pelvis position error under the true intrinsics, and its X / Y / Z parts
                     (dz signed: positive = too far); rel_dz_pct = |dz| / true depth.
GT in the camera frame = R * joints_world + t with the dataset's cam_poses (checked below by projecting
the GT joints with the true intrinsics onto the annotated 2-D keypoints).

Caveats: 3DPW test = 5 men, outdoors; the GT SMPL joints come from the dataset's own fits; the models'
principal point is the image centre (the true one is within a few pixels of it); depth here is the
pelvis depth, not the depth of a body surface.

Usage (self-test, numpy): python eval/eval_3dpw_global.py --self-test

Usage (server, env camerahmr; needs torch + smplx):
    python eval/eval_3dpw_global.py --anthro_root ~/SMPL-Anthropometry \\
        --model HMR2.0b=results/3dpw_full_hmr2.npz --model CameraHMR=results/3dpw_full_camerahmr.npz \\
        --model TokenHMR=results/3dpw_full_tokenhmr.npz --out results/3dpw_global.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_ssp3d_pose_cam import mpjpe_root_mm, pa_mpjpe_mm, project  # noqa: E402

# SMPL joint index -> OpenPose-18 index (the 12 limb joints), for checking the GT camera
SMPL_TO_OPENPOSE = {16: 5, 17: 2, 18: 6, 19: 3, 20: 7, 21: 4, 1: 11, 2: 8, 4: 12, 5: 9, 7: 13, 8: 10}
METRICS = ["pa_mpjpe_mm", "mpjpe_root_mm", "mpjpe_asis_mm", "mpjpe_gtk_mm", "root_err_gtk_mm", "abs_dx_mm",
           "abs_dy_mm", "abs_dz_mm", "dz_mm", "rel_dz_pct"]


def world_to_cam(joints_world: np.ndarray, cam_pose: np.ndarray) -> np.ndarray:
    return joints_world @ cam_pose[:3, :3].T + cam_pose[:3, 3]


def retarget_to_gt_camera(points_model: np.ndarray, f_model: float, center_model: tuple, K: np.ndarray) -> np.ndarray:
    """Move a body from the model's camera to the true one: same image position and apparent size of the
    root (joint 0), shape and orientation untouched. points_model (J,3) in the model's camera frame."""
    root = points_model[0]
    u = f_model * root[0] / root[2] + center_model[0]
    v = f_model * root[1] / root[2] + center_model[1]
    z = root[2] * K[0, 0] / f_model
    root_new = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
    return points_model - root + root_new


def gt_reprojection_px(gt_cam: np.ndarray, K: np.ndarray, poses2d: np.ndarray) -> float:
    """Mean pixel error of the GT camera-frame joints, projected with the true K, against the annotated
    OpenPose keypoints (poses2d (3,18): x, y, confidence). Validates the world -> camera convention."""
    p = gt_cam[:, :2] / gt_cam[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    errs = [np.linalg.norm(p[s] - poses2d[:2, o]) for s, o in SMPL_TO_OPENPOSE.items() if poses2d[2, o] > 0.3]
    return float(np.mean(errs)) if errs else float("nan")


def score_record(pred_joints_model, cam_t, f_model, wh, joints_world, cam_pose, K) -> dict:
    gt = world_to_cam(joints_world, cam_pose)
    P = pred_joints_model + cam_t
    center = (wh[0] / 2.0, wh[1] / 2.0)
    Pk = retarget_to_gt_camera(P, f_model, center, K)
    d = Pk[0] - gt[0]
    return {"pa_mpjpe_mm": pa_mpjpe_mm(P, gt), "mpjpe_root_mm": mpjpe_root_mm(P, gt),
            "mpjpe_asis_mm": float(np.linalg.norm(P - gt, axis=1).mean() * 1000),
            "mpjpe_gtk_mm": float(np.linalg.norm(Pk - gt, axis=1).mean() * 1000),
            "root_err_gtk_mm": float(np.linalg.norm(d) * 1000),
            "abs_dx_mm": abs(d[0]) * 1000, "abs_dy_mm": abs(d[1]) * 1000, "abs_dz_mm": abs(d[2]) * 1000,
            "dz_mm": d[2] * 1000, "rel_dz_pct": abs(d[2]) / gt[0, 2] * 100,
            "gt_depth_m": float(gt[0, 2]), "pred_depth_gtk_m": float(Pk[0, 2])}


def summarize(rows: list[dict], methods: list[str]) -> list[dict]:
    out = []
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        if not sel:
            continue
        row = {"method": m, "n": len(sel)}
        for k in METRICS:
            row[k] = float(np.mean([r[k] for r in sel]))
        row["abs_dz_median_mm"] = float(np.median([r["abs_dz_mm"] for r in sel]))
        row["rel_dz_median_pct"] = float(np.median([r["rel_dz_pct"] for r in sel]))
        row["focal_model_px"] = float(np.mean([r["focal_px"] for r in sel]))
        ratio = np.array([r["focal_ratio"] for r in sel])
        row["asis_median_mm"] = float(np.median([r["mpjpe_asis_mm"] for r in sel]))
        row["focal_ratio_p10"], row["focal_ratio_p50"], row["focal_ratio_p90"] = (float(np.percentile(ratio, q)) for q in (10, 50, 90))
        row["share_depth_within_10pct"] = float(np.mean([r["rel_dz_pct"] < 10 for r in sel]))
        row["share_focal_within_10pct"] = float(np.mean(np.abs(ratio - 1) < 0.10))
        out.append(row)
    return out


def print_table(summ: list[dict], gt_focal: float, gt_depth: float, gt_reproj: float) -> None:
    print(f"\nGT camera: focal {gt_focal:.0f} px (mean), pelvis depth {gt_depth:.2f} m (mean); "
          f"GT camera-frame joints reproject onto the annotated 2-D keypoints with {gt_reproj:.1f} px mean error")
    print(f"\n{'':11s}{'n':>6s}{'PA-MPJPE':>10s}{'MPJPE-rt':>10s}{'as-is':>10s}{'true-K':>9s}{'root err':>10s}"
          f"{'|dx|':>7s}{'|dy|':>7s}{'|dz|':>7s}{'dz bias':>9s}{'rel dz%':>9s}{'focal px':>10s}")
    for r in summ:
        print(f"{r['method']:11s}{r['n']:6d}{r['pa_mpjpe_mm']:10.1f}{r['mpjpe_root_mm']:10.1f}{r['mpjpe_asis_mm']:10.0f}"
              f"{r['mpjpe_gtk_mm']:9.1f}{r['root_err_gtk_mm']:10.1f}{r['abs_dx_mm']:7.0f}{r['abs_dy_mm']:7.0f}"
              f"{r['abs_dz_mm']:7.0f}{r['dz_mm']:+9.0f}{r['rel_dz_pct']:9.1f}{r['focal_model_px']:10.0f}")
    print("\n(all mm except 'rel dz%' and 'focal px'; dz bias > 0 = predicted body too far; "
          "'as-is' uses each model's own camera)")
    print(f"\n{'':11s}{'as-is median':>14s}{'focal/true p10':>16s}{'p50':>7s}{'p90':>7s}{'focal<10% off':>15s}{'depth<10% off':>15s}")
    for r in summ:
        print(f"{r['method']:11s}{r['asis_median_mm']:14.0f}{r['focal_ratio_p10']:16.2f}{r['focal_ratio_p50']:7.2f}"
              f"{r['focal_ratio_p90']:7.2f}{r['share_focal_within_10pct']:15.2f}{r['share_depth_within_10pct']:15.2f}")


def write_summary(summ: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "n"] + METRICS + ["abs_dz_median_mm", "rel_dz_median_pct", "focal_model_px", "asis_median_mm",
                                          "focal_ratio_p10", "focal_ratio_p50", "focal_ratio_p90",
                                          "share_depth_within_10pct", "share_focal_within_10pct"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summ)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anthro_root", required=True, help="folder with data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=NPZ")
    ap.add_argument("--out", default="results/3dpw_global.csv")
    args = ap.parse_args()

    import smplx
    import torch
    specs = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model expects NAME=NPZ, got {spec!r}")
        specs[name] = np.load(Path(path).expanduser().resolve())
    out_path = Path(args.out).expanduser().resolve()
    layer = smplx.SMPLLayer(model_path=str(Path(args.anthro_root).expanduser().resolve() / "data" / "smpl" / "SMPL_NEUTRAL.pkl"),
                            num_betas=10).eval()
    first = next(iter(specs.values()))
    names = [str(x) for x in first["names"]]
    for n, z in specs.items():
        if [str(x) for x in z["names"]] != names:
            raise SystemExit(f"{n}: records differ from the first file -- rerun with the same --stride")

    def forward(betas, body, orient):
        with torch.no_grad():
            o = layer(betas=torch.tensor(betas, dtype=torch.float32)[None],
                      body_pose=torch.tensor(body, dtype=torch.float32)[None],
                      global_orient=torch.tensor(orient, dtype=torch.float32).reshape(1, 1, 3, 3), pose2rot=False)
        return o.joints[0, :24].numpy().astype(np.float64)

    # the GT camera check, once (the GT arrays are the same in every file)
    gt_px, gt_depth, gt_focal = [], [], []
    for i in range(len(names)):
        g = world_to_cam(first["joints_world"][i], first["cam_pose"][i])
        gt_px.append(gt_reprojection_px(g, first["K"][i], first["poses2d"][i]))
        gt_depth.append(g[0, 2])
        gt_focal.append(first["K"][i][0, 0])
    gt_reproj = float(np.nanmean(gt_px))
    print(f"{len(names)} record(s); GT reprojection check {gt_reproj:.1f} px; "
          f"{100 * np.mean(np.array(gt_depth) > 0):.0f}% of GT pelvis depths are in front of the camera", flush=True)

    rows = []
    for name, z in specs.items():
        for i in range(len(names)):
            if not z["ok"][i]:
                continue
            J = forward(z["betas"][i], z["body_pose"][i], z["global_orient"][i])
            r = score_record(J, z["cam_t"][i], float(z["focal"][i]), z["img_wh"][i], first["joints_world"][i],
                             first["cam_pose"][i], first["K"][i])
            rows.append({"method": name, "focal_px": float(z["focal"][i]), "focal_ratio": float(z["focal"][i]) / first["K"][i][0, 0], **r})
        print(f"  scored {name}", flush=True)
    methods = list(specs)
    summ = summarize(rows, methods)
    print_table(summ, float(np.mean(gt_focal)), float(np.mean(gt_depth)), gt_reproj)
    write_summary(summ, out_path)


def self_test() -> None:
    rng = np.random.default_rng(0)
    # a camera: rotation about y by 30 degrees, translation; world joints are the inverse image of camera joints
    th = np.radians(30)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    E = np.eye(4)
    E[:3, :3], E[:3, 3] = R, [0.3, -0.2, 0.5]
    gt_cam = rng.normal(0, 0.4, (24, 3)) + np.array([0.2, 0.1, 4.0])
    joints_world = (gt_cam - E[:3, 3]) @ R                           # inverse of R x + t
    assert np.allclose(world_to_cam(joints_world, E), gt_cam)
    K = np.array([[1962.0, 0, 540], [0, 1969.0, 960], [0, 0, 1]])
    W, H = 1080, 1920

    # GT camera check: annotated keypoints generated from the GT joints -> ~0 px
    p2d = np.zeros((3, 18))
    p = gt_cam[:, :2] / gt_cam[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    for s, o in SMPL_TO_OPENPOSE.items():
        p2d[:, o] = [p[s, 0], p[s, 1], 0.9]
    assert gt_reprojection_px(gt_cam, K, p2d) < 1e-9
    p2d[0] += 10
    assert abs(gt_reprojection_px(gt_cam, K, p2d) - 10) < 1e-9

    # a model that sees the right image but assumes focal 5000: its body stands 5000/1962 times too far
    f_m = 5000.0
    u = K[0, 0] * gt_cam[0, 0] / gt_cam[0, 2] + K[0, 2]
    v = K[1, 1] * gt_cam[0, 1] / gt_cam[0, 2] + K[1, 2]
    z_m = gt_cam[0, 2] * f_m / K[0, 0]
    root_m = np.array([(u - W / 2) * z_m / f_m, (v - H / 2) * z_m / f_m, z_m])
    pred_rel = gt_cam - gt_cam[0]
    cam_t = root_m                                                   # pred joints are root-relative here
    r = score_record(pred_rel + 0.0, cam_t - pred_rel[0], f_m, (W, H), joints_world, E, K)
    assert r["pa_mpjpe_mm"] < 1e-6 and r["mpjpe_root_mm"] < 1e-6
    assert r["mpjpe_asis_mm"] > 1000, r                              # wrong focal -> metres of error as-is
    assert r["mpjpe_gtk_mm"] < 0.01 and r["root_err_gtk_mm"] < 0.01 and r["rel_dz_pct"] < 1e-6, r

    # a model that really puts the body 20% too far (right focal): dz bias +20% of the depth
    f_ok = K[0, 0]
    z_far = gt_cam[0, 2] * 1.2
    root_far = np.array([(u - W / 2) * z_far / f_ok, (v - H / 2) * z_far / f_ok, z_far])
    r2 = score_record(pred_rel, root_far, f_ok, (W, H), joints_world, E, K)
    assert abs(r2["rel_dz_pct"] - 20.0) < 0.5 and r2["dz_mm"] > 0, r2
    assert r2["mpjpe_root_mm"] < 1e-6, "a translation-only error leaves the root-aligned MPJPE at zero"

    # a body 10% too large (pose right): root-aligned MPJPE sees it, PA-MPJPE does not
    r3 = score_record(pred_rel * 1.1, root_m - pred_rel[0] * 1.1, f_m, (W, H), joints_world, E, K)
    assert r3["pa_mpjpe_mm"] < 1e-6 and r3["mpjpe_root_mm"] > 10, r3

    rows = [{"method": "A", "focal_px": 5000.0, "focal_ratio": 5000 / 1962, **r},
            {"method": "A", "focal_px": 5000.0, "focal_ratio": 5000 / 1962, **r2}]
    s = summarize(rows, ["A"])
    assert s[0]["n"] == 2 and abs(s[0]["rel_dz_pct"] - (r["rel_dz_pct"] + r2["rel_dz_pct"]) / 2) < 1e-9
    print_table(s, 1962.0, 4.0, 0.0)
    print("\n[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
