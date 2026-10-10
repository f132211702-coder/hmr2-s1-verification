#!/usr/bin/env python3
"""Run HMR2.0b, CameraHMR or TokenHMR on the HBW validation photos (box from the
OpenPose keypoints that ship with the data, no detector) and save each photo's
predicted betas for eval_hbw_shape.py.

Same three models, same loaders and one-env-per-model arrangement as
predict_ssp3d.py (its make_hmr2 / make_camerahmr / make_tokenhmr are reused).
Output: results/hbw_<model>.csv (format: see hbw_common.py).

Usage (self-test, no model/data): python eval/predict_hbw.py --self-test

Usage (server, repo root; HBW val subset in ~/datasets/HBW):
    conda activate 4D-humans
    python eval/predict_hbw.py --model hmr2 --hbw ~/datasets/HBW --out results/hbw_hmr2.csv
    conda activate camerahmr
    python eval/predict_hbw.py --model camerahmr --camerahmr_root ~/workspace/dresson/CameraHMR \\
        --hbw ~/datasets/HBW --out ~/workspace/dresson/Tims_dir/results/hbw_camerahmr.csv
    conda activate tokenhmr
    python eval/predict_hbw.py --model tokenhmr --tokenhmr_root ~/workspace/dresson/TokenHMR \\
        --hbw ~/datasets/HBW --out ~/workspace/dresson/Tims_dir/results/hbw_tokenhmr.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from hbw_common import N_BETAS, bbox_from_openpose, hbw_records, write_csv  # noqa: E402


def run(predict, records, read_image) -> list[dict]:
    rows = []
    for n, rec in enumerate(records):
        base = {"image_id": rec.image_id, "subject": rec.subject, "kind": rec.kind}
        img = read_image(rec.img_path)
        if img is None:
            rows.append({**base, "status": "no_image"})
            continue
        bbox = bbox_from_openpose(rec.kp_path)
        if bbox is None:
            rows.append({**base, "status": "no_person"})
            continue
        betas = np.asarray(predict(img, bbox), dtype=np.float64).reshape(-1)
        rows.append({**base, "status": "ok", **{f"pred_beta_{i}": float(b) for i, b in enumerate(betas[:N_BETAS])}})
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(records)}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["hmr2", "camerahmr", "tokenhmr"])
    ap.add_argument("--hbw", required=True, help="folder with images/, keypoints/, smplx/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--camerahmr_root")
    ap.add_argument("--tokenhmr_root")
    ap.add_argument("--checkpoint", default="data/checkpoints/tokenhmr_model_latest.ckpt")
    ap.add_argument("--model_config", default="data/checkpoints/model_config.yaml")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()           # resolve before any loader chdirs
    records = hbw_records(Path(args.hbw).expanduser().resolve(), stride=args.stride)
    print(f"{len(records)} photo(s), {len({r.subject for r in records})} subject(s)")

    from predict_ssp3d import make_camerahmr, make_hmr2, make_tokenhmr
    if args.model == "hmr2":
        predict = make_hmr2()
    elif args.model == "camerahmr":
        if not args.camerahmr_root:
            raise SystemExit("--camerahmr_root is required")
        predict = make_camerahmr(Path(args.camerahmr_root).expanduser().resolve())
    else:
        if not args.tokenhmr_root:
            raise SystemExit("--tokenhmr_root is required")
        predict = make_tokenhmr(Path(args.tokenhmr_root).expanduser().resolve(), args.checkpoint, args.model_config)

    import cv2
    rows = run(predict, records, lambda p: cv2.imread(str(p)))
    write_csv(rows, out)
    from collections import Counter
    print("status counts:", dict(Counter(r["status"] for r in rows)))


def self_test() -> None:
    import json
    import tempfile
    from hbw_common import load_prediction_csv
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        sub = d / "images/val_small_resolution/012_2_0/Photos_Lab"
        ksub = d / "keypoints/val_small_resolution/012_2_0/Photos_Lab"
        sub.mkdir(parents=True)
        ksub.mkdir(parents=True)
        good = [100, 50, 0.9, 120, 80, 0.9, 110, 200, 0.8, 90, 260, 0.9]
        for i in range(3):
            (sub / f"0000{i}.png").write_bytes(b"x")
        (ksub / "00000.json").write_text(json.dumps({"people": [{"pose_keypoints_2d": good}]}))
        (ksub / "00001.json").write_text(json.dumps({"people": []}))      # no person -> skipped, not scored
        (ksub / "00002.json").write_text(json.dumps({"people": [{"pose_keypoints_2d": good}]}))
        recs = hbw_records(d)
        seen = []

        def fake_predict(img, bbox):
            seen.append(bbox.copy())
            return np.arange(N_BETAS) * float(img)

        rows = run(fake_predict, recs, lambda p: None if p.name == "00002.png" else 2)
        assert [r["status"] for r in rows] == ["ok", "no_person", "no_image"] and len(seen) == 1
        write_csv(rows, d / "o.csv")
        back = load_prediction_csv(d / "o.csv")
        assert set(back) == {"012_2_0/Photos_Lab/00000.png"}
        assert np.allclose(next(iter(back.values()))["pred"], np.arange(N_BETAS) * 2.0)
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
