#!/usr/bin/env python3
"""Compare s1_infer.py's gendered-mesh modes on real photos: manual
(--gender male/female), auto (--gender auto, DeepFace) and neutral.

The reference is the mesh computed through the subject's TRUE-gender SMPL
model (3DPW ships each person's gender), using the same predicted
betas/pose in every mode -- so the only difference between modes is which
skeleton got picked, and "error" is how far a mode's mesh is from the
correct-gender one (mean per-vertex L2, mm):
    manual  0 by construction when the user picks the right gender (it IS
            the reference); this script reports it so the table is complete.
    neutral the cost of not picking at all.
    auto    0 when the classifier is right, the neutral gap when it falls
            back (no confident face), and the full male<->female gap when
            it is confidently wrong.
It also times the classifier and reports how often it actually produced a
label, split by person size in the image, since it needs a visible face.

What this does NOT measure: whether the gendered mesh is closer to the
person's real body (there is no GT for that here), only which mode picks
the skeleton the ground truth says it should. The earlier
`eval_against_gt.py --smpl-gender male` run found the pose-accuracy effect
of the skeleton choice to be ~1mm.

Persons are picked by matching detections to the GT box built from 3DPW's
`poses2d` (see eval_against_gt.load_3dpw_gt), so bystanders in street
scenes are excluded -- their gender is unknown. Status: unverified against
real data until run; the 3DPW test split is all male, so wrong-gender
behavior of the classifier (male photographed as 'female') is measured but
female subjects are not.

Usage (on the GPU server):
    python eval/compare_gender_modes.py \\
        --img_root /home/intern/datasets/3DPW/imageFiles \\
        --gt_dir /home/intern/datasets/3DPW/sequenceFiles/test \\
        --stride 60 --out results/gender_modes.csv
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

GENDER_NAMES = {"m": "male", "f": "female"}


def mesh_error_mm(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-vertex L2 distance between two (V,3) meshes in meters -> mm."""
    return float(np.linalg.norm(a - b, axis=1).mean() * 1000)


def classify_outcome(auto_label: str | None, true_label: str) -> str:
    """'correct' / 'wrong' / 'fallback' (classifier produced no label)."""
    if auto_label is None:
        return "fallback"
    return "correct" if auto_label == true_label else "wrong"


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n{n} subject detection(s) matched to a GT person")
    if n == 0:
        return

    outcomes = [r["auto_outcome"] for r in rows]
    print("\n=== auto (DeepFace): what it did ===")
    for name in ("correct", "fallback", "wrong"):
        c = outcomes.count(name)
        print(f"  {name:9s} {c:5d}  ({100 * c / n:.1f}%)")
    print("  (fallback = no confident face -> neutral mesh; wrong = confidently the other gender)")

    heights = np.array([r["bbox_h_px"] for r in rows])
    med = float(np.median(heights))
    for name in ("correct", "fallback", "wrong"):
        sel = heights[[o == name for o in outcomes]]
        if len(sel):
            print(f"  median person height in image, {name:9s}: {np.median(sel):.0f} px")
    print(f"  (all subjects: median {med:.0f} px)")

    print("\n=== mesh error vs the correct-gender mesh (mean per-vertex L2, mm) ===")
    print(f"  manual (right gender picked): {0.0:6.1f}   (the reference itself, by construction)")
    print(f"  neutral (no picking)        : {np.mean([r['err_neutral_mm'] for r in rows]):6.1f}")
    print(f"  auto                        : {np.mean([r['err_auto_mm'] for r in rows]):6.1f}")

    ms = [r["classify_ms"] for r in rows]
    print(f"\nauto classifier cost: mean {np.mean(ms):.0f} ms, median {np.median(ms):.0f} ms per person (CPU)")
    print("manual cost: one SMPL forward pass, effectively free")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", type=str, required=True, help="3DPW imageFiles/ (contains <sequence>/image_XXXXX.jpg)")
    ap.add_argument("--gt_dir", type=str, required=True, help="3DPW sequenceFiles/test")
    ap.add_argument("--stride", type=int, default=60, help="use every Nth frame per sequence")
    ap.add_argument("--min-iou", type=float, default=0.3)
    ap.add_argument("--out", type=str, default="results/gender_modes.csv")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--gender-min-confidence", type=float, default=60.0)
    args = ap.parse_args()

    import cv2
    from eval_against_gt import bbox_iou, load_3dpw_gt
    from s1_infer import GenderClassifier, HMR2Estimator, crop_person

    est = HMR2Estimator(gender="neutral", device=args.device)
    classifier = GenderClassifier(args.gender_min_confidence)

    rows: list[dict] = []
    for pkl_path in sorted(Path(args.gt_dir).glob("*.pkl")):
        seq = pkl_path.stem
        with open(pkl_path, "rb") as f:
            genders = pickle.load(f, encoding="latin1").get("genders")
        if genders is None:
            print(f"[warn] {seq}: no `genders` in pkl, skipping")
            continue

        by_frame: dict[int, list] = {}
        for rec in load_3dpw_gt(pkl_path):
            frame = int(rec.image_id.rsplit("_", 1)[1])
            if frame % args.stride == 0 and rec.bbox is not None:
                by_frame.setdefault(frame, []).append(rec)

        for frame, gts in sorted(by_frame.items()):
            img_path = Path(args.img_root) / seq / f"image_{frame:05d}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[warn] cannot read {img_path}")
                continue
            people = est.estimate(img)

            for gt in gts:
                best, best_iou = None, args.min_iou
                for p in people:
                    iou = bbox_iou(p["bbox"], gt.bbox)
                    if iou > best_iou:
                        best, best_iou = p, iou
                if best is None:
                    continue

                true_label = GENDER_NAMES[str(genders[gt.person_id])]
                ref = est.gendered_vertices(true_label, best)
                neutral = best["pred_vertices"]

                crop = crop_person(img, best["bbox"])
                t0 = time.perf_counter()
                auto_label = classifier(crop) if crop is not None else None
                classify_ms = (time.perf_counter() - t0) * 1000
                auto_vertices = est.gendered_vertices(auto_label, best) if auto_label else neutral

                rows.append({
                    "sequence": seq, "frame": frame, "person_id": gt.person_id,
                    "true_gender": true_label, "auto_label": auto_label or "none",
                    "auto_outcome": classify_outcome(auto_label, true_label),
                    "bbox_h_px": float(best["bbox"][3] - best["bbox"][1]),
                    "err_neutral_mm": mesh_error_mm(neutral, ref),
                    "err_auto_mm": mesh_error_mm(auto_vertices, ref),
                    "classify_ms": classify_ms,
                })
        print(f"{seq}: {len(rows)} matched so far")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out_path}")
    summarize(rows)


def self_test() -> None:
    """Pure-numpy checks of the bookkeeping; no model, images, or dataset."""
    a = np.zeros((10, 3))
    b = np.zeros((10, 3))
    b[:, 0] = 0.03  # every vertex 30 mm away
    assert abs(mesh_error_mm(a, b) - 30.0) < 1e-6
    assert mesh_error_mm(a, a) == 0.0

    assert classify_outcome("male", "male") == "correct"
    assert classify_outcome("female", "male") == "wrong"
    assert classify_outcome(None, "male") == "fallback"

    rows = [
        {"auto_outcome": "correct", "bbox_h_px": 400.0, "err_neutral_mm": 30.0, "err_auto_mm": 0.0, "classify_ms": 200.0},
        {"auto_outcome": "fallback", "bbox_h_px": 80.0, "err_neutral_mm": 30.0, "err_auto_mm": 30.0, "classify_ms": 150.0},
        {"auto_outcome": "wrong", "bbox_h_px": 300.0, "err_neutral_mm": 30.0, "err_auto_mm": 60.0, "classify_ms": 250.0},
    ]
    summarize(rows)
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
