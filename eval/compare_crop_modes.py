#!/usr/bin/env python3
"""How much of the PA-MPJPE gap to HMR2's published 3DPW number comes from
crop quality and joint-set choice, rather than from the regressor itself?

The official evaluation (eval/official_eval.py -> ImageDataset) feeds HMR2
pre-computed crop centers/scales from a shipped .npz and scores a 14-joint
subset; our end-to-end pipeline runs a person detector first and scores all
24 SMPL joints. This script runs BOTH crop sources on the SAME frames and
scores BOTH joint sets, so each factor can be read off separately:

    crop source   "detector": s1_infer's normal path, matched to the subject
                              by IoU with the GT box (as eval_against_gt.py)
                  "gt_box":   the GT box (from 3DPW poses2d) handed straight
                              to HMR2, detector skipped
    joint set     "24": all SMPL body joints (what eval_against_gt.py scores)
                  "14": limbs + neck + head -- SMPL joints [ankles, knees,
                        hips, wrists, elbows, shoulders, neck, head], a
                        PROXY for the official 14-keypoint list

Caveats, so the numbers aren't over-read:
  - The official 14 keypoints are regressed from mesh vertices with an extra
    joint regressor (HMR2's J_regressor_extra); the proxy here picks SMPL
    skeleton joints at the same anatomical places, which is close but not
    identical.
  - The GT box comes from OpenPose 2D keypoints padded by --pad, not from
    the official .npz. How tight/loose a box is changes HMR2's crop scale,
    so try more than one --pad before concluding anything about "GT boxes".
  - Frames are subsampled (--stride); metrics are computed only on
    subject/frame pairs where the detector also found the subject, so the
    two crop sources are compared on identical samples. The miss rate is
    reported separately.

Status: pure bookkeeping is validated by --self-test; the full run has not
been executed against real data yet.

Usage (GPU server):
    python eval/compare_crop_modes.py \\
        --img_root /home/intern/datasets/3DPW/imageFiles \\
        --gt_dir /home/intern/datasets/3DPW/sequenceFiles/test \\
        --stride 20 --pad 0.15 --out results/crop_modes.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# SMPL 24-joint indices: 0 pelvis, 1/2 L/R hip, 4/5 L/R knee, 7/8 L/R ankle,
# 12 neck, 15 head, 16/17 L/R shoulder, 18/19 L/R elbow, 20/21 L/R wrist.
SUBSET_14 = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20, 12, 15]


def summarize(rows: list[dict], n_missed: int, n_gt: int) -> None:
    n = len(rows)
    print(f"\n{n_gt} GT subject/frame pair(s) sampled; detector missed the subject in {n_missed} "
          f"({100 * n_missed / max(n_gt, 1):.1f}%); {n} compared on identical samples")
    if n == 0:
        return

    def col(name):
        return np.array([r[name] for r in rows])

    print("\n=== PA-MPJPE (mm), same frames, mean / median ===")
    print(f"{'':28s}{'24 joints':>18s}{'14-joint proxy':>18s}")
    for label, k24, k14 in (("detector crop", "pa24_det", "pa14_det"),
                             ("GT-box crop", "pa24_gt", "pa14_gt")):
        a, b = col(k24), col(k14)
        print(f"{label:28s}{a.mean():9.2f} /{np.median(a):7.2f}{b.mean():9.2f} /{np.median(b):7.2f}")

    d24 = col("pa24_det").mean() - col("pa24_gt").mean()
    d14 = col("pa14_det").mean() - col("pa14_gt").mean()
    j = col("pa24_det").mean() - col("pa14_det").mean()
    print(f"\ncrop effect (detector - GT box), mean: 24 joints {d24:+.2f} mm, 14 joints {d14:+.2f} mm")
    print(f"joint-set effect (24 - 14 joints), with detector crops: {j:+.2f} mm")
    print(f"mean IoU, detector box vs GT box: {col('iou_det_gt').mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--stride", type=int, default=20, help="use every Nth frame per sequence")
    ap.add_argument("--pad", type=float, default=0.15, help="GT box padding around 2D keypoints")
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--out", type=str, default="results/crop_modes.csv")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    import cv2
    import torch
    from eval_against_gt import (PoseRecord, bbox_iou, build_smpl_layer, get_joints,
                                 load_3dpw_gt, pa_mpjpe, rotmat_to_aa)
    from s1_infer import HMR2Estimator

    est = HMR2Estimator(gender="neutral", device=args.device)
    device = est.device
    smpl_layer = build_smpl_layer(device=device, gender="neutral")

    def pred_joints(person: dict) -> np.ndarray:
        rec = PoseRecord(
            image_id="", person_id=0, betas=person["betas"],
            body_pose_aa=rotmat_to_aa(person["body_pose"]),
            global_orient_aa=rotmat_to_aa(person["global_orient"])[0],
        )
        return get_joints(smpl_layer, rec, device)

    rows: list[dict] = []
    n_gt = n_missed = 0
    for pkl_path in sorted(Path(args.gt_dir).glob("*.pkl")):
        seq = pkl_path.stem
        by_frame: dict[int, list] = {}
        for rec in load_3dpw_gt(pkl_path, bbox_pad=args.pad):
            frame = int(rec.image_id.rsplit("_", 1)[1])
            if frame % args.stride == 0 and rec.bbox is not None and rec.joints_3d is not None:
                by_frame.setdefault(frame, []).append(rec)

        for frame, gts in sorted(by_frame.items()):
            img = cv2.imread(str(Path(args.img_root) / seq / f"image_{frame:05d}.jpg"))
            if img is None:
                continue
            detections = est.estimate(img)
            gt_people = est.estimate(img, boxes=np.stack([g.bbox for g in gts]))
            gt_by_idx = {p["person_id"]: p for p in gt_people}

            for idx, gt in enumerate(gts):
                n_gt += 1
                best, best_iou = None, args.min_iou
                for p in detections:
                    iou = bbox_iou(p["bbox"], gt.bbox)
                    if iou > best_iou:
                        best, best_iou = p, iou
                if best is None or idx not in gt_by_idx:
                    n_missed += 1
                    continue

                gt_j = gt.joints_3d
                det_j, box_j = pred_joints(best), pred_joints(gt_by_idx[idx])
                rows.append({
                    "sequence": seq, "frame": frame, "person_id": gt.person_id,
                    "iou_det_gt": best_iou,
                    "pa24_det": pa_mpjpe(det_j, gt_j), "pa24_gt": pa_mpjpe(box_j, gt_j),
                    "pa14_det": pa_mpjpe(det_j[SUBSET_14], gt_j[SUBSET_14]),
                    "pa14_gt": pa_mpjpe(box_j[SUBSET_14], gt_j[SUBSET_14]),
                })
        print(f"{seq}: {len(rows)} compared so far", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_path}")
    summarize(rows, n_missed, n_gt)


def self_test() -> None:
    """Bookkeeping only; no model, images, or dataset."""
    assert len(SUBSET_14) == 14 and len(set(SUBSET_14)) == 14, "14 distinct joints"
    assert all(0 <= i < 24 for i in SUBSET_14), "indices must be valid SMPL-24 joints"

    rows = [
        {"pa24_det": 60.0, "pa24_gt": 50.0, "pa14_det": 45.0, "pa14_gt": 38.0, "iou_det_gt": 0.8},
        {"pa24_det": 70.0, "pa24_gt": 60.0, "pa14_det": 55.0, "pa14_gt": 48.0, "iou_det_gt": 0.7},
    ]
    summarize(rows, n_missed=1, n_gt=3)
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
