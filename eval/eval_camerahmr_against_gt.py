#!/usr/bin/env python3
"""Quantitative 3DPW evaluation for CameraHMR, directly comparable to
eval_against_gt.py's HMR2.0b numbers (results/eval_3dpw_fixed.csv:
pa_mpjpe_mm mean 58.68/median 54.48, beta_l2 mean 2.76).

Motivation: the whole point of trying CameraHMR for this project is that
its paper targets exactly the problem found in HMR2 -- betas collapsing
toward the average body shape regardless of the real person (see
verify_neutral_gender_bias.py / check_beta_collapse.py / this project's
report). This script runs the SAME 3DPW test set through CameraHMR and
scores it with the SAME metrics (PA-MPJPE, beta_l2) so the two numbers can
be compared head-to-head, not eyeballed from demo images.

Design choices, and why:
  - Feeds CameraHMR the GT box directly (from 3DPW poses2d, same
    bbox_from_keypoints_2d() used everywhere else in this project),
    bypassing its own Detectron2 detector entirely. compare_crop_modes.py
    already established detector-vs-GT-box crop quality is a negligible
    factor for HMR2 (-0.36mm) -- using GT boxes here isolates CameraHMR's
    *regression* quality (the thing actually being tested) from a second,
    independent detector's miss rate, and avoids having to write separate
    GT-matching logic for a detector this project doesn't otherwise use.
  - Reuses eval_against_gt.py's PoseRecord / load_3dpw_gt / pa_mpjpe /
    beta_error / bbox_from_keypoints_2d as-is (already self-tested and
    validated against a real 3DPW file in that module) instead of
    reimplementing them. GT joints come straight from 3DPW's own
    jointPositions (load_3dpw_gt already puts these on
    PoseRecord.joints_3d), so no SMPL forward pass or hmr2 package import
    is needed for the GT side at all.
  - CameraHMR's SMPL output format (global_orient (1,3,3), body_pose
    (23,3,3) rotation matrices, betas (10,)) is -- confirmed by reading
    core/camerahmr_model.py -- identical in shape/convention to HMR2's, so
    the two sides' betas and SMPL-forward joints are directly comparable
    without any extra conversion.
  - Subclasses mesh_estimator.HumanMeshEstimator and overrides
    init_detector() to skip loading Detectron2 (~2.6GB checkpoint, and
    minutes of load time) since every box used here comes from GT, not the
    detector -- everything else (model, HumanFoV cam model, SMPL body
    model, get_cam_intrinsics(), get_output_mesh()) is reused unchanged
    from the real inference path demo.py uses, so this isn't a
    reimplementation that could drift from how the repo actually runs.

Must be run with CameraHMR's own conda env active (needs its `core`
package, `mesh_estimator`, detectron2, etc.) AND from (or pointed at) the
CameraHMR repo root, because core/constants.py's checkpoint paths
('data/pretrained-models/...') are relative.

Status: --self-test validates the bbox->center/scale conversion and the
CSV-writing/summary bookkeeping against synthetic data. Not yet run
against the real model/3DPW data.

Usage (self-test, no GPU/model/data needed):
    python eval/eval_camerahmr_against_gt.py --self-test

Usage (GPU server, camerahmr conda env, from the CameraHMR repo root):
    python /path/to/hmr2.0/eval/eval_camerahmr_against_gt.py \\
        --camerahmr_root ~/workspace/dresson/CameraHMR \\
        --img_root /home/intern/datasets/3DPW/imageFiles \\
        --gt_dir /home/intern/datasets/3DPW/sequenceFiles/test \\
        --stride 20 --out results/camerahmr_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for eval_against_gt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def bbox_to_center_scale(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[x1,y1,x2,y2] -> (center (2,), scale (2,)), matching exactly what
    mesh_estimator.py's process_image computes from its Detectron2 boxes:
    bbox_scale = (x2-x1, y2-y1) / 200.0, bbox_center = midpoint. The /200
    divisor and the later `scale*200` in Dataset.__getitem__ are CameraHMR's
    own (inherited-from-SPIN/HMR) convention, not something introduced here."""
    x1, y1, x2, y2 = bbox
    center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    scale = np.array([(x2 - x1) / 200.0, (y2 - y1) / 200.0], dtype=np.float32)
    return center, scale


def summarize(rows: list[dict]) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"\n{len(rows)} GT record(s), {len(ok_rows)} scored "
          f"({len(rows) - len(ok_rows)} skipped -- see 'status' column)")
    if not ok_rows:
        return
    print("\n=== CameraHMR on 3DPW-TEST (GT box), vs HMR2.0b reference "
          "(eval_3dpw_fixed.csv: pa_mpjpe 58.68/54.48mm, beta_l2 2.76) ===")
    for key in ("pa_mpjpe_mm", "beta_l2", "pred_beta_norm", "gt_beta_norm"):
        values = np.array([r[key] for r in ok_rows])
        print(f"  {key}: mean={values.mean():.2f}  median={np.median(values):.2f}")


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camerahmr_root", type=str, required=True,
                     help="path to the CameraHMR repo clone (so `core`/`mesh_estimator` "
                          "are importable, and so relative checkpoint paths resolve)")
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--stride", type=int, default=20, help="use every Nth frame per sequence")
    ap.add_argument("--pad", type=float, default=0.15, help="GT box padding around 2D keypoints")
    ap.add_argument("--out", type=str, default="results/camerahmr_eval.csv")
    args = ap.parse_args()

    camerahmr_root = Path(args.camerahmr_root).expanduser().resolve()
    sys.path.insert(0, str(camerahmr_root))
    os.chdir(camerahmr_root)  # core/constants.py paths are relative to cwd

    import cv2
    import torch

    from eval_against_gt import beta_error, load_3dpw_gt, pa_mpjpe  # noqa: E402
    from mesh_estimator import HumanMeshEstimator  # noqa: E402

    class GTBoxEstimator(HumanMeshEstimator):
        """Identical to HumanMeshEstimator, except it never loads the
        Detectron2 detector -- every box used by this script comes from
        3DPW's GT, so the detector is both unused and (at 2.6GB + load
        time) not worth paying for."""

        def init_detector(self, threshold):
            return None

    est = GTBoxEstimator(model_type="smpl")

    rows: list[dict] = []
    for pkl_path in sorted(Path(args.gt_dir).glob("*.pkl")):
        seq = pkl_path.stem
        n_before = len(rows)
        for rec in load_3dpw_gt(pkl_path, bbox_pad=args.pad):
            frame = int(rec.image_id.rsplit("_", 1)[1])
            if frame % args.stride != 0 or rec.bbox is None or rec.joints_3d is None:
                continue

            img_path = Path(args.img_root) / seq / f"image_{frame:05d}.jpg"
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                rows.append({"image_id": rec.image_id, "person_id": rec.person_id,
                             "status": "no_image"})
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            center, scale = bbox_to_center_scale(rec.bbox)
            cam_int = est.get_cam_intrinsics(img_rgb)

            from core.datasets.dataset import Dataset
            dataset = Dataset(img_rgb, center[None], scale[None], cam_int, False, img_path)
            batch = torch.utils.data.dataloader.default_collate([dataset[0]])
            from core.utils import recursive_to
            batch = recursive_to(batch, est.device)

            with torch.no_grad():
                out_smpl_params, out_cam, _ = est.model(batch)
            pred_vertices, pred_joints, _ = est.get_output_mesh(out_smpl_params, out_cam, batch)

            pred_betas = out_smpl_params["betas"][0].detach().cpu().numpy()
            pred_joints_24 = pred_joints[0, :24].detach().cpu().numpy()

            err = beta_error(pred_betas, rec.betas)
            rows.append({
                "image_id": rec.image_id, "person_id": rec.person_id, "status": "ok",
                "pa_mpjpe_mm": pa_mpjpe(pred_joints_24, rec.joints_3d),
                "beta_mae": err["beta_mae"], "beta_l2": err["beta_l2"],
                "pred_beta_norm": float(np.linalg.norm(pred_betas)),
                "gt_beta_norm": float(np.linalg.norm(rec.betas)),
            })
        print(f"{seq}: {len(rows) - n_before} frame(s) added, {len(rows)} total", flush=True)

    write_csv(rows, Path(args.out))
    summarize(rows)


def self_test() -> None:
    """No model/GPU/data needed: validates bbox->center/scale conversion
    and the CSV/summary bookkeeping against synthetic data."""
    center, scale = bbox_to_center_scale(np.array([100.0, 50.0, 300.0, 450.0]))
    assert np.allclose(center, [200.0, 250.0]), f"unexpected center {center}"
    assert np.allclose(scale, [1.0, 2.0]), f"unexpected scale {scale}"

    rows = [
        {"image_id": "a", "person_id": 0, "status": "ok", "pa_mpjpe_mm": 50.0,
         "beta_mae": 0.1, "beta_l2": 2.0, "pred_beta_norm": 0.5, "gt_beta_norm": 2.5},
        {"image_id": "b", "person_id": 0, "status": "ok", "pa_mpjpe_mm": 60.0,
         "beta_mae": 0.2, "beta_l2": 3.0, "pred_beta_norm": 0.4, "gt_beta_norm": 3.1},
        {"image_id": "c", "person_id": 0, "status": "no_image"},
    ]
    out_path = Path("results") / "camerahmr_eval_self_test.csv"
    write_csv(rows, out_path)
    summarize(rows)
    assert out_path.exists(), "self-test CSV was not written"
    print("\n[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
