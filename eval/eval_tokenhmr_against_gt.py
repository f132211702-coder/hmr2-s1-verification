#!/usr/bin/env python3
"""Quantitative 3DPW evaluation for TokenHMR, on the same metrics/protocol
as eval_against_gt.py (HMR2.0b) and eval_camerahmr_against_gt.py
(CameraHMR), so all three can be compared head-to-head: pa_mpjpe_mm,
pve_mm, beta_l2/beta_mae, pred_beta_norm/gt_beta_norm.

Design choices (same rationale as eval_camerahmr_against_gt.py, not
repeated in full here -- read that module's docstring first):
  - Feeds TokenHMR the GT box directly (from 3DPW poses2d via
    eval_against_gt.bbox_from_keypoints_2d/load_3dpw_gt), bypassing its
    own Detectron2 detector, to isolate regression quality from a second
    detector's miss rate.
  - Reuses eval_against_gt.py's PoseRecord / load_3dpw_gt / pa_mpjpe / pve
    / beta_error as-is. GT joints come straight from 3DPW's own
    jointPositions; GT vertices need an SMPL forward pass (3DPW doesn't
    ship vertices), done through TokenHMR's OWN (neutral) SMPL instance
    (model.smpl) so both sides of the PVE comparison share one body model
    -- same reasoning as eval_camerahmr_against_gt.py's gt_vertices_neutral.
  - TokenHMR's SMPL output format (global_orient (1,3,3), body_pose
    (23,3,3) rotation matrices, betas (10,)) -- confirmed by reading
    lib/models/tokenhmr.py's forward_step -- is the same shape/convention
    as HMR2's and CameraHMR's, so no extra conversion is needed.
  - Uses `lib.datasets.vitdet_dataset.ViTDetDataset` directly with a GT
    box array in place of the detector's boxes -- it already computes
    center/scale from a plain [x1,y1,x2,y2] box itself (no separate
    bbox_to_center_scale helper needed here, unlike the CameraHMR script).

Must be run with TokenHMR's own conda env active (needs its `lib` package,
detectron2 isn't actually needed here since the detector is skipped) AND
with the TokenHMR repo's `tokenhmr/` subfolder importable, because its
code does `import lib...` expecting that directory on sys.path (see
tokenhmr/demo.py, which is run as `python tokenhmr/demo.py` from the repo
root for the same reason).

Status (updated 2026-10-04): a first real run (stride=20, 1787 samples)
gave pve_mm mean 58.71 (reasonable, in line with HMR2.0b's 64.32 and
CameraHMR's 51.91) but pa_mpjpe_mm mean 433.75 -- wildly inconsistent with
a pve_mm that low on the same predictions. Root cause: model.smpl is
lib.models.smpl_wrapper.SMPL, a smplx.SMPLLayer subclass whose forward()
overwrites .joints with an OpenPose-25 remap before returning (see
lib/models/smpl_wrapper.py), so out['pred_keypoints_3d'] was never in the
standard SMPL 24-joint order 3DPW's jointPositions uses. Fixed by
raw_smpl_joints_24(), which calls smplx.SMPLLayer.forward() directly
(bypassing the subclass override) on model.smpl's own predicted
betas/pose. Not yet re-run with this fix.

Usage (self-test, no GPU/model/data needed):
    python eval/eval_tokenhmr_against_gt.py --self-test

Usage (GPU server, tokenhmr conda env, from the TokenHMR repo root):
    python /path/to/hmr2.0/eval/eval_tokenhmr_against_gt.py \\
        --tokenhmr_root ~/workspace/dresson/TokenHMR \\
        --checkpoint data/checkpoints/tokenhmr_model_latest.ckpt \\
        --model_config data/checkpoints/model_config.yaml \\
        --img_root /home/intern/datasets/3DPW/imageFiles \\
        --gt_dir /home/intern/datasets/3DPW/sequenceFiles/test \\
        --stride 20 --out results/tokenhmr_eval.csv
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


def summarize(rows: list[dict]) -> None:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"\n{len(rows)} GT record(s), {len(ok_rows)} scored "
          f"({len(rows) - len(ok_rows)} skipped -- see 'status' column)")
    if not ok_rows:
        return
    print("\n=== TokenHMR on 3DPW-TEST (GT box) ===")
    print("Compare against eval_against_gt.py (--smpl-gender neutral, its default) and "
          "eval_camerahmr_against_gt.py on the same 3DPW-TEST split -- all three use a "
          "neutral SMPL body model for GT vertex reconstruction, so pve_mm is comparable "
          "across all three. beta_l2 is this project's own shape-collapse probe (see "
          "verify_neutral_gender_bias.py), not a standard metric any of these papers report.")
    for key in ("pa_mpjpe_mm", "pve_mm", "beta_l2", "pred_beta_norm", "gt_beta_norm"):
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
    ap.add_argument("--tokenhmr_root", type=str, required=True,
                     help="path to the TokenHMR repo clone (so `lib` is importable, and "
                          "relative checkpoint/config paths resolve)")
    ap.add_argument("--checkpoint", type=str, default="data/checkpoints/tokenhmr_model_latest.ckpt")
    ap.add_argument("--model_config", type=str, default="data/checkpoints/model_config.yaml")
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--stride", type=int, default=20, help="use every Nth frame per sequence")
    ap.add_argument("--pad", type=float, default=0.15, help="GT box padding around 2D keypoints")
    ap.add_argument("--out", type=str, default="results/tokenhmr_eval.csv")
    args = ap.parse_args()

    tokenhmr_root = Path(args.tokenhmr_root).expanduser().resolve()
    # demo.py is run as `python tokenhmr/demo.py` FROM THE REPO ROOT -- Python puts the
    # script's own directory (tokenhmr/) on sys.path[0] automatically, which is how its
    # `import lib...` resolves without any extra sys.path setup on the user's part. We
    # replicate that here instead of chdir-ing into tokenhmr/, so --checkpoint/
    # --model_config stay relative to the repo root, matching the README's own usage.
    sys.path.insert(0, str(tokenhmr_root / "tokenhmr"))
    os.chdir(tokenhmr_root)  # fetch_demo_data.sh's paths (data/checkpoints/...) are relative to here

    import cv2
    import torch

    from eval_against_gt import beta_error, load_3dpw_gt, pa_mpjpe, pve  # noqa: E402
    from lib.models import load_tokenhmr  # noqa: E402
    from lib.datasets.vitdet_dataset import ViTDetDataset  # noqa: E402
    from lib.utils import recursive_to  # noqa: E402
    import smplx  # noqa: E402

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_tokenhmr(checkpoint_path=args.checkpoint,
                                      model_cfg=args.model_config,
                                      is_train_state=False, is_demo=True)
    model = model.to(device).eval()

    def raw_smpl_joints_24(pred_smpl_params: dict) -> np.ndarray:
        """model.smpl is lib.models.smpl_wrapper.SMPL, a smplx.SMPLLayer
        SUBCLASS whose forward() overwrites .joints with an OpenPose-25
        remap (joints = smpl_output.joints[:, self.joint_map, :], see
        lib/models/smpl_wrapper.py) before returning -- so out['pred_keypoints_3d']
        is in OpenPose order, NOT the standard SMPL 24-joint order 3DPW's
        jointPositions uses (this was confirmed the hard way: scoring
        against it gave pa_mpjpe_mm ~434mm, wildly inconsistent with a
        pve_mm of ~59mm on the same predictions). Calling the PARENT
        class's forward directly bypasses that override while still using
        model.smpl's own loaded weights/template -- no separate SMPL file
        or config key to track down."""
        b = pred_smpl_params["betas"].reshape(1, -1)
        bp = pred_smpl_params["body_pose"].reshape(1, -1, 3, 3)
        go = pred_smpl_params["global_orient"].reshape(1, -1, 3, 3)
        with torch.no_grad():
            raw_out = smplx.SMPLLayer.forward(model.smpl, betas=b, body_pose=bp,
                                              global_orient=go, pose2rot=False)
        return raw_out.joints[0, :24].detach().cpu().numpy()

    def gt_vertices_neutral(rec) -> np.ndarray:
        """GT mesh vertices via TokenHMR's OWN (neutral) SMPL instance,
        the same model its predictions use -- see module docstring for why."""
        from scipy.spatial.transform import Rotation
        body_pose_rotmat = Rotation.from_rotvec(rec.body_pose_aa).as_matrix()
        global_orient_rotmat = Rotation.from_rotvec(rec.global_orient_aa[None]).as_matrix()
        betas_t = torch.tensor(rec.betas, dtype=torch.float32, device=device)[None]
        body_pose_t = torch.tensor(body_pose_rotmat, dtype=torch.float32, device=device)[None]
        global_orient_t = torch.tensor(global_orient_rotmat, dtype=torch.float32, device=device)[None]
        with torch.no_grad():
            # body_pose_t/global_orient_t are already rotation matrices (via scipy
            # above), matching how forward_step() calls this same self.smpl with
            # pose2rot=False on its own (also-rotmat) predictions.
            out = model.smpl(betas=betas_t, body_pose=body_pose_t,
                              global_orient=global_orient_t, pose2rot=False)
        return out.vertices[0].detach().cpu().numpy()

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

            dataset = ViTDetDataset(model_cfg, img_bgr, rec.bbox[None])
            batch = torch.utils.data.dataloader.default_collate([dataset[0]])
            batch = recursive_to(batch, device)

            with torch.no_grad():
                out = model(batch)

            pred_betas = out["pred_smpl_params"]["betas"].reshape(-1).detach().cpu().numpy()
            pred_joints_24 = raw_smpl_joints_24(out["pred_smpl_params"])
            pred_vertices = out["pred_vertices"][0].detach().cpu().numpy()

            err = beta_error(pred_betas, rec.betas)
            rows.append({
                "image_id": rec.image_id, "person_id": rec.person_id, "status": "ok",
                "pa_mpjpe_mm": pa_mpjpe(pred_joints_24, rec.joints_3d),
                "pve_mm": pve(pred_vertices, gt_vertices_neutral(rec)),
                "beta_mae": err["beta_mae"], "beta_l2": err["beta_l2"],
                "pred_beta_norm": float(np.linalg.norm(pred_betas)),
                "gt_beta_norm": float(np.linalg.norm(rec.betas)),
            })
        print(f"{seq}: {len(rows) - n_before} frame(s) added, {len(rows)} total", flush=True)

    write_csv(rows, Path(args.out))
    summarize(rows)


def self_test() -> None:
    """No model/GPU/data needed: validates the CSV/summary bookkeeping
    against synthetic data."""
    rows = [
        {"image_id": "a", "person_id": 0, "status": "ok", "pa_mpjpe_mm": 48.0, "pve_mm": 55.0,
         "beta_mae": 0.15, "beta_l2": 2.2, "pred_beta_norm": 1.8, "gt_beta_norm": 2.6},
        {"image_id": "b", "person_id": 0, "status": "ok", "pa_mpjpe_mm": 52.0, "pve_mm": 60.0,
         "beta_mae": 0.25, "beta_l2": 2.8, "pred_beta_norm": 1.6, "gt_beta_norm": 3.0},
        {"image_id": "c", "person_id": 0, "status": "no_image"},
    ]
    out_path = Path("results") / "tokenhmr_eval_self_test.csv"
    write_csv(rows, out_path)
    summarize(rows)
    assert out_path.exists(), "self-test CSV was not written"
    print("\n[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
