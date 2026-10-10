#!/usr/bin/env python3
"""Pose (theta), silhouette and camera accuracy of the three models on SSP-3D -- the S0 metrics the
project plan lists that the 3DPW evaluation did not cover:

  pose            PA-MPJPE over the 24 SMPL joints, root-aligned MPJPE, and the rotation error
                  (geodesic angle, degrees) of the global orientation and of the 23 body joints
                  (legs / arms / torso reported separately), against SSP-3D's pose labels.
  silhouette IoU  the predicted mesh projected with the model's own camera, filled, vs the
                  annotated silhouette -- SSP-3D's official "mIOU". It measures shape, pose AND
                  camera together.
  reprojection    the 12 limb joints (shoulders, elbows, wrists, hips, knees, ankles) projected with
                  the model's camera vs the annotated 2-D keypoints: mean error as a percentage of
                  the person's box size, and the share of joints within 5% / 10% of it (PCK).

A "GT mesh (floor)" row projects the dataset's own pseudo-GT body with its own camera (focal 5000,
principal point at the image centre, as in SSP-3D's visualisation.py): the best the labels themselves
can score on these metrics, so model numbers are read against it.

Inputs: results/ssp3d_full_<model>.npz written by `predict_ssp3d.py --dump-npz`, plus SSP-3D's
labels.npz and silhouettes, and the SMPL neutral/male/female .pkl in <anthro_root>/data/smpl/.

Caveats: SSP-3D's pose, shape, camera and 2-D keypoints are themselves optimised pseudo-GT (the
labels were fitted to the same photos), the camera of the three models is defined by each model's
own demo code, the COCO hip/shoulder joints are not at exactly the SMPL joint positions (a small
constant offset, the same for every model), and the global orientation is compared in each model's
crop-camera frame (a small frame difference for a 512 px photo). All photos are tight sports
clothes: no information about loose clothing.

Usage (self-test, numpy + cv2): python eval/eval_ssp3d_pose_cam.py --self-test

Usage (server, env camerahmr; needs torch + smplx):
    python eval/eval_ssp3d_pose_cam.py --ssp3d ~/datasets/SSP-3D --anthro_root ~/SMPL-Anthropometry \\
        --model HMR2.0b=results/ssp3d_full_hmr2.npz --model CameraHMR=results/ssp3d_full_camerahmr.npz \\
        --model TokenHMR=results/ssp3d_full_tokenhmr.npz --out results/ssp3d_pose_cam.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

GT_FOCAL = 5000.0
CONF_MIN = 0.3
# SMPL joint index -> COCO-17 index, for the 12 limb joints (the face points have no SMPL joint)
SMPL_TO_COCO = {16: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10, 1: 11, 2: 12, 4: 13, 5: 14, 7: 15, 8: 16}
LEGS = [1, 2, 4, 5, 7, 8, 10, 11]                 # body-joint indices (SMPL joint number, 1..23)
ARMS = [13, 14, 16, 17, 18, 19, 20, 21, 22, 23]
TORSO = [3, 6, 9, 12, 15]
METRICS = ["pa_mpjpe_mm", "mpjpe_root_mm", "rot_global_deg", "rot_body_deg", "rot_legs_deg", "rot_arms_deg",
           "rot_torso_deg", "iou", "reproj_pct", "pck05", "pck10"]


# ---------------------------------------------------------------- geometry
def procrustes_align(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity transform (scale, rotation, translation) of pred onto gt, least squares (Umeyama)."""
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    P, G = pred - mu_p, gt - mu_g
    U, S, Vt = np.linalg.svd(P.T @ G)
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    scale = (S * np.diag(D)).sum() / (P ** 2).sum()
    return scale * P @ R + mu_g


def pa_mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm(procrustes_align(pred, gt) - gt, axis=1).mean() * 1000)


def mpjpe_root_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.linalg.norm((pred - pred[0]) - (gt - gt[0]), axis=1).mean() * 1000)


def aa_to_rotmat(aa: np.ndarray) -> np.ndarray:
    """(..., 3) axis-angle -> (..., 3, 3) (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    k = aa / np.maximum(theta, 1e-12)
    K = np.zeros(aa.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    t = theta[..., None]
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> np.ndarray:
    """Angle (degrees) of the rotation R1^T R2, per leading index."""
    tr = np.einsum("...ji,...ji->...", R1, R2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def project(points: np.ndarray, focal: float, w: int, h: int) -> np.ndarray:
    return points[:, :2] / points[:, 2:3] * focal + np.array([w / 2.0, h / 2.0])


def silhouette_mask(verts_cam: np.ndarray, faces: np.ndarray, focal: float, w: int, h: int) -> np.ndarray:
    import cv2
    mask = np.zeros((h, w), np.uint8)
    pts = project(verts_cam, focal, w, h)
    ok = (verts_cam[faces][:, :, 2] > 0.05).all(axis=1)
    tri = np.round(pts[faces[ok]]).astype(np.int32)
    tri = np.clip(tri, -10_000, 10_000)
    if len(tri):
        cv2.fillPoly(mask, list(tri), 1)
    return mask


def iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    union = (a | b).sum()
    return float((a & b).sum() / union) if union else float("nan")


def reproj_errors(joints24_cam: np.ndarray, focal: float, w: int, h: int, joints2d: np.ndarray,
                  box_size: float) -> np.ndarray:
    """Normalised (fraction of box size) 2-D error of the 12 limb joints with confidence >= CONF_MIN."""
    p = project(joints24_cam, focal, w, h)
    errs = []
    for smpl_i, coco_i in SMPL_TO_COCO.items():
        if joints2d[coco_i, 2] >= CONF_MIN:
            errs.append(np.linalg.norm(p[smpl_i] - joints2d[coco_i, :2]) / box_size)
    return np.array(errs)


def rotation_errors(pred_global: np.ndarray, pred_body: np.ndarray, gt_aa: np.ndarray) -> dict:
    """pred_global (3,3), pred_body (23,3,3), gt_aa (72,) -> angle errors in degrees."""
    gt_g = aa_to_rotmat(gt_aa[:3])
    gt_b = aa_to_rotmat(gt_aa[3:].reshape(23, 3))
    body = geodesic_deg(pred_body, gt_b)                       # (23,), joint i+1 at index i
    sel = lambda idx: float(body[[i - 1 for i in idx]].mean())  # noqa: E731
    return {"rot_global_deg": float(geodesic_deg(pred_global, gt_g)), "rot_body_deg": float(body.mean()),
            "rot_legs_deg": sel(LEGS), "rot_arms_deg": sel(ARMS), "rot_torso_deg": sel(TORSO)}


def score_image(pred_joints_cam, pred_verts_cam, pred_focal, pred_gl, pred_body, gt_joints, gt_aa,
                gt_sil, joints2d, box_size, faces, w, h) -> dict:
    out = {"pa_mpjpe_mm": pa_mpjpe_mm(pred_joints_cam[:24], gt_joints[:24]),
           "mpjpe_root_mm": mpjpe_root_mm(pred_joints_cam[:24], gt_joints[:24])}
    out.update(rotation_errors(pred_gl, pred_body, gt_aa))
    out["iou"] = iou(silhouette_mask(pred_verts_cam, faces, pred_focal, w, h), gt_sil)
    e = reproj_errors(pred_joints_cam, pred_focal, w, h, joints2d, box_size)
    out["reproj_pct"] = float(e.mean() * 100) if len(e) else float("nan")
    out["pck05"] = float((e < 0.05).mean()) if len(e) else float("nan")
    out["pck10"] = float((e < 0.10).mean()) if len(e) else float("nan")
    return out


def summarize(rows: list[dict], methods: list[str]) -> list[dict]:
    out = []
    for group in ("all", "m", "f"):
        for m in methods:
            sel = [r for r in rows if r["method"] == m and (group == "all" or r["gender"] == group)]
            if not sel:
                continue
            row = {"method": m, "group": group, "n": len(sel)}
            for k in METRICS:
                vals = [r[k] for r in sel if k in r and not np.isnan(r[k])]
                row[k] = float(np.mean(vals)) if vals else float("nan")
            if all("iou" in r for r in sel):
                row["iou_median"] = float(np.nanmedian([r["iou"] for r in sel]))
            out.append(row)
    return out


def print_tables(summ: list[dict]) -> None:
    for group, title in (("all", "all photos"), ("m", "men"), ("f", "women")):
        sel = [r for r in summ if r["group"] == group]
        if not sel:
            continue
        print(f"\n=== {title} ({sel[0]['n']} photos) ===")
        print(f"{'':16s}{'PA-MPJPE':>9s}{'MPJPE-rt':>9s}{'rotGlob':>8s}{'rotBody':>8s}{'legs':>7s}{'arms':>7s}{'torso':>7s}"
              f"{'IoU':>7s}{'IoU med':>8s}{'reproj%':>8s}{'PCK5':>7s}{'PCK10':>7s}")
        for r in sel:
            f = lambda k, p=1: "      -" if np.isnan(r.get(k, float("nan"))) else f"{r[k]:{p}.{1}f}"  # noqa: E731
            print(f"{r['method']:16s}{f('pa_mpjpe_mm', 9)}{f('mpjpe_root_mm', 9)}{f('rot_global_deg', 8)}{f('rot_body_deg', 8)}"
                  f"{f('rot_legs_deg', 7)}{f('rot_arms_deg', 7)}{f('rot_torso_deg', 7)}"
                  f"{'      -' if np.isnan(r['iou']) else format(r['iou'], '7.3f')}"
                  f"{'       -' if np.isnan(r.get('iou_median', float('nan'))) else format(r['iou_median'], '8.3f')}"
                  f"{f('reproj_pct', 8)}"
                  f"{'      -' if np.isnan(r['pck05']) else format(r['pck05'], '7.2f')}"
                  f"{'      -' if np.isnan(r['pck10']) else format(r['pck10'], '7.2f')}")


def write_summary(summ: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "group", "n"] + METRICS + ["iou_median"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(summ)
    print(f"wrote {path}")


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ssp3d", required=True)
    ap.add_argument("--anthro_root", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=NPZ")
    ap.add_argument("--out", default="results/ssp3d_pose_cam.csv")
    args = ap.parse_args()

    import cv2
    import torch
    import smplx

    specs = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model expects NAME=NPZ, got {spec!r}")
        specs[name] = np.load(Path(path).expanduser().resolve())
    out_path = Path(args.out).expanduser().resolve()
    root = Path(args.ssp3d).expanduser().resolve() / "ssp_3d"
    smpl_dir = Path(args.anthro_root).expanduser().resolve() / "data" / "smpl"
    lab = np.load(root / "labels.npz", allow_pickle=True)
    fnames = [str(f) for f in lab["fnames"]]
    for n, z in specs.items():
        if [str(f) for f in z["fnames"]] != fnames:
            raise SystemExit(f"{n}: photo order differs from labels.npz -- rerun predict_ssp3d.py without --stride")

    layers = {g: smplx.SMPLLayer(model_path=str(smpl_dir / f"SMPL_{g}.pkl"), num_betas=10).eval()
              for g in ("NEUTRAL", "MALE", "FEMALE")}
    faces = np.asarray(layers["NEUTRAL"].faces, dtype=np.int64)

    def forward(layer, betas, body_rot, global_rot):
        with torch.no_grad():
            o = layer(betas=torch.tensor(betas, dtype=torch.float32)[None],
                      body_pose=torch.tensor(body_rot, dtype=torch.float32)[None],
                      global_orient=torch.tensor(global_rot, dtype=torch.float32)[None].reshape(1, 1, 3, 3),
                      pose2rot=False)
        return o.joints[0, :24].numpy().astype(np.float64), o.vertices[0].numpy().astype(np.float64)

    rows, floor_rows, vdiff = [], [], {n: [] for n in specs}
    for i, fname in enumerate(fnames):
        gender = str(lab["genders"][i]).lower()
        gt_aa = lab["poses"][i].astype(np.float64)
        gt_j, gt_v = forward(layers["MALE" if gender == "m" else "FEMALE"], lab["shapes"][i],
                             aa_to_rotmat(gt_aa[3:].reshape(23, 3)), aa_to_rotmat(gt_aa[:3]))
        cam = lab["cam_trans"][i].astype(np.float64)
        sil = cv2.imread(str(root / "silhouettes" / fname), 0)
        h, w = sil.shape
        sil = sil > 127
        j2d, box = lab["joints2D"][i], float(lab["bbox_whs"][i])
        base = {"gender": gender}
        # floor: the dataset's own body with its own camera
        e = reproj_errors(gt_j + cam, GT_FOCAL, w, h, j2d, box)
        floor_rows.append({**base, "method": "GT mesh (floor)",
                           "iou": iou(silhouette_mask(gt_v + cam, faces, GT_FOCAL, w, h), sil),
                           "reproj_pct": float(e.mean() * 100) if len(e) else float("nan"),
                           "pck05": float((e < 0.05).mean()) if len(e) else float("nan"),
                           "pck10": float((e < 0.10).mean()) if len(e) else float("nan")})
        for name, z in specs.items():
            if not z["ok"][i]:
                continue
            pj, pv = forward(layers["NEUTRAL"], z["betas"][i], z["body_pose"][i], z["global_orient"][i])
            vdiff[name].append(np.abs(pv - z["vertices"][i]).max())
            cam_t = z["cam_t"][i].astype(np.float64)
            r = score_image(pj + cam_t, z["vertices"][i].astype(np.float64) + cam_t, float(z["focal"][i]),
                            z["global_orient"][i], z["body_pose"][i], gt_j, gt_aa, sil, j2d, box, faces, w, h)
            rows.append({**base, "method": name, **r})
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(fnames)}", flush=True)

    print("\nsanity: max |recomputed - saved| vertex difference (m), median over photos: "
          + ", ".join(f"{n} {np.median(v):.2e}" for n, v in vdiff.items()))
    methods = list(specs)
    summ = summarize(rows, methods)
    # the floor row only has the silhouette / reprojection metrics
    for group in ("all", "m", "f"):
        sel = [r for r in floor_rows if group == "all" or r["gender"] == group]
        row = {"method": "GT mesh (floor)", "group": group, "n": len(sel)}
        for k in METRICS:
            vals = [r[k] for r in sel if k in r and not np.isnan(r[k])]
            row[k] = float(np.mean(vals)) if vals else float("nan")
        row["iou_median"] = float(np.nanmedian([r["iou"] for r in sel]))
        summ.append(row)
    summ.sort(key=lambda r: ("all", "m", "f").index(r["group"]))
    print_tables(summ)
    write_summary(summ, out_path)


def self_test() -> None:
    import cv2  # noqa: F401
    rng = np.random.default_rng(0)

    # Procrustes: a similarity-transformed copy aligns exactly
    gt = rng.normal(size=(24, 3))
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    moved = 1.8 * gt @ R.T + np.array([0.3, -0.2, 1.0])
    assert pa_mpjpe_mm(moved, gt) < 1e-6
    assert pa_mpjpe_mm(gt + rng.normal(0, 0.01, gt.shape), gt) > 1.0
    assert mpjpe_root_mm(gt + np.array([5.0, 5.0, 5.0]), gt) < 1e-6      # translation does not count
    assert abs(mpjpe_root_mm(gt * 1.1, gt) - 100 * np.linalg.norm(gt - gt[0], axis=1).mean()) < 1e-6

    # rotations
    assert np.allclose(aa_to_rotmat(np.zeros(3)), np.eye(3))
    assert np.allclose(aa_to_rotmat(np.array([0, 0, np.pi / 2])), [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
    assert abs(geodesic_deg(np.eye(3), aa_to_rotmat(np.array([0, 0, np.pi / 2]))) - 90) < 1e-6
    gt_aa = np.zeros(72)
    gt_aa[3 + 3 * 3 + 2] = 0.5                                          # body joint 4 (L knee): z rotation 0.5 rad
    pred_body = np.tile(np.eye(3), (23, 1, 1))
    err = rotation_errors(np.eye(3), pred_body, gt_aa)
    assert abs(err["rot_legs_deg"] - np.degrees(0.5) / len(LEGS)) < 1e-6 and err["rot_arms_deg"] == 0.0
    assert abs(err["rot_body_deg"] - np.degrees(0.5) / 23) < 1e-6 and err["rot_global_deg"] == 0.0

    # projection and silhouettes
    assert np.allclose(project(np.array([[0, 0, 5.0], [1, 0, 5.0]]), 500, 512, 512), [[256, 256], [356, 256]])
    quad = np.array([[-0.5, -1, 5], [0.5, -1, 5], [0.5, 1, 5], [-0.5, 1, 5]], float)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    m1 = silhouette_mask(quad, faces, 500, 512, 512)
    assert m1.sum() > 0 and abs(m1.sum() - 100 * 200) < 800                # 1 m x 2 m at 5 m, f=500: 100 x 200 px
    assert iou(m1, m1) == 1.0
    m2 = silhouette_mask(quad + np.array([0.5, 0, 0]), faces, 500, 512, 512)
    assert 0.2 < iou(m1, m2) < 0.5                                         # half-overlap shift: ~1/3
    assert silhouette_mask(quad * np.array([1, 1, -1]), faces, 500, 512, 512).sum() == 0   # behind the camera

    # reprojection: perfect keypoints -> 0 error; an offset of 10 px on a 100 px box -> 10%
    joints = np.zeros((24, 3))
    joints[:, 2] = 5.0
    j2d = np.zeros((17, 3))
    for s, c in SMPL_TO_COCO.items():
        joints[s, 0] = 0.01 * s
        j2d[c] = [256 + 500 * 0.01 * s / 5.0, 256, 1.0]
    e0 = reproj_errors(joints, 500, 512, 512, j2d, 100.0)
    assert len(e0) == 12 and e0.max() < 1e-9
    j2d[:, 0] += 10
    assert abs(reproj_errors(joints, 500, 512, 512, j2d, 100.0).mean() - 0.10) < 1e-9
    j2d[SMPL_TO_COCO[16], 2] = 0.1                                          # low confidence -> dropped
    assert len(reproj_errors(joints, 500, 512, 512, j2d, 100.0)) == 11

    # score_image + summarize
    gt_sil = m1.astype(bool)
    full = score_image(joints, quad, 500.0, np.eye(3), pred_body, joints[:24], gt_aa, gt_sil, j2d, 100.0, faces, 512, 512)
    assert full["pa_mpjpe_mm"] < 1e-6 and full["iou"] == 1.0 and abs(full["reproj_pct"] - 10.0) < 1e-6
    rows = [{**full, "method": "A", "gender": "m"}, {**full, "method": "A", "gender": "f", "iou": 0.5}]
    s = {(r["method"], r["group"]): r for r in summarize(rows, ["A"])}
    assert s[("A", "all")]["n"] == 2 and abs(s[("A", "all")]["iou"] - 0.75) < 1e-9 and s[("A", "f")]["iou"] == 0.5
    print_tables(summarize(rows, ["A"]))
    print("\n[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
