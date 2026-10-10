#!/usr/bin/env python3
"""Run HMR2.0b, CameraHMR or TokenHMR on the 3DPW test split (GT box, every Nth frame, the same
record selection as eval_against_gt.py / eval_camerahmr_against_gt.py / eval_tokenhmr_against_gt.py)
and save, per record, the model's full prediction (betas, pose rotation matrices, full-image camera
translation and focal length, vertices) next to the GT needed for the camera-frame evaluation:
the GT joints in WORLD coordinates, the GT camera extrinsics (world -> camera, 4x4), the GT intrinsics
(3x3) and the 2-D keypoints. eval_3dpw_global.py turns this into unaligned MPJPE and depth errors.

Same three models, loaders and one-env-per-model arrangement as predict_ssp3d.py (its make_hmr2 /
make_camerahmr / make_tokenhmr are reused, in full-output mode).

Usage (self-test, no model/data): python eval/predict_3dpw_full.py --self-test

Usage (server, repo root):
    conda activate 4D-humans
    python eval/predict_3dpw_full.py --model hmr2 --img_root /home/intern/datasets/3DPW/imageFiles \\
        --gt_dir /home/intern/datasets/3DPW/sequenceFiles/test --out results/3dpw_full_hmr2.npz
    (camerahmr / tokenhmr: same, in their envs, with --camerahmr_root / --tokenhmr_root)
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))


def gt_records(gt_dir: Path, stride: int, pad: float, bbox_fn):
    """Yield one dict per (sequence, person, frame): the selection of eval_against_gt.load_3dpw_gt plus
    `frame % stride == 0`, with the camera-frame GT ingredients added."""
    for pkl in sorted(Path(gt_dir).glob("*.pkl")):
        with open(pkl, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        seq = pkl.stem
        K = np.asarray(d["cam_intrinsics"], dtype=np.float64)
        n_people = len(d["poses"])
        for pid in range(n_people):
            joints = d["jointPositions"][pid]
            valid = d["campose_valid"][pid]
            p2d = d["poses2d"][pid]
            cams = d["cam_poses"]
            for frame in range(d["poses"][pid].shape[0]):
                if valid is not None and not valid[frame]:
                    continue
                if frame % stride != 0:
                    continue
                bbox = bbox_fn(p2d[frame], pad=pad)
                if bbox is None:
                    continue
                yield {"seq": seq, "person": pid, "frame": frame,
                       "joints_world": np.asarray(joints[frame], dtype=np.float64).reshape(24, 3),
                       "cam_pose": np.asarray(cams[frame], dtype=np.float64).reshape(4, 4),
                       "K": K, "poses2d": np.asarray(p2d[frame], dtype=np.float64).reshape(3, 18),
                       "bbox": bbox}


GT_KEYS = ("joints_world", "cam_pose", "K", "poses2d", "bbox")


def run(predict, records, read_image, img_root: Path) -> dict:
    """Stack the GT of every record and the full prediction of every record that could be predicted
    (zeros + ok=False otherwise). The vertices are not saved: the evaluation recomputes joints from the
    parameters and checks them against nothing else it needs."""
    names, ok, gts, preds = [], [], [], []
    for n, r in enumerate(records):
        names.append(f"{r['seq']}__image_{r['frame']:05d}__p{r['person']}")
        img = read_image(Path(img_root) / r["seq"] / f"image_{r['frame']:05d}.jpg")
        out = predict(img, r["bbox"]) if img is not None else None
        ok.append(out is not None)
        gts.append({k: np.asarray(r[k], dtype=np.float64) for k in GT_KEYS})
        preds.append(out)
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(records)}", flush=True)
    ref = next((o for o in preds if o is not None), None)
    if ref is None:
        raise SystemExit("no record was predicted")
    arrays = {"names": np.array(names), "ok": np.array(ok)}
    for k in GT_KEYS:
        arrays[k] = np.stack([g[k] for g in gts])
    for k, v in ref.items():
        if k == "vertices":
            continue
        arr = np.zeros((len(records),) + np.asarray(v).shape, dtype=np.float64)
        for i, o in enumerate(preds):
            if o is not None:
                arr[i] = np.asarray(o[k])
        arrays[k] = arr
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["hmr2", "camerahmr", "tokenhmr"])
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--camerahmr_root")
    ap.add_argument("--tokenhmr_root")
    ap.add_argument("--checkpoint", default="data/checkpoints/tokenhmr_model_latest.ckpt")
    ap.add_argument("--model_config", default="data/checkpoints/model_config.yaml")
    args = ap.parse_args()

    out = Path(args.out).expanduser().resolve()           # resolve before any loader chdirs
    img_root = Path(args.img_root).expanduser().resolve()
    gt_dir = Path(args.gt_dir).expanduser().resolve()
    from eval_against_gt import bbox_from_keypoints_2d
    records = list(gt_records(gt_dir, args.stride, args.pad, bbox_from_keypoints_2d))
    print(f"{len(records)} record(s)")

    from predict_ssp3d import make_camerahmr, make_hmr2, make_tokenhmr
    if args.model == "hmr2":
        predict = make_hmr2(True)
    elif args.model == "camerahmr":
        if not args.camerahmr_root:
            raise SystemExit("--camerahmr_root is required")
        predict = make_camerahmr(Path(args.camerahmr_root).expanduser().resolve(), True)
    else:
        if not args.tokenhmr_root:
            raise SystemExit("--tokenhmr_root is required")
        predict = make_tokenhmr(Path(args.tokenhmr_root).expanduser().resolve(), args.checkpoint,
                                args.model_config, True)
    import cv2
    arrays = run(predict, records, lambda p: cv2.imread(str(p)), img_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays)
    print(f"wrote {out} ({int(arrays['ok'].sum())}/{len(arrays['ok'])} records)")


def self_test() -> None:
    import pickle as _pk
    import tempfile
    rng = np.random.default_rng(0)
    n = 45
    data = {"cam_intrinsics": np.array([[1962.0, 0, 540], [0, 1969.0, 960], [0, 0, 1]]),
            "poses": [rng.normal(size=(n, 72))], "jointPositions": [rng.normal(size=(n, 72))],
            "campose_valid": [np.array([True] * 30 + [False] * 15)], "poses2d": [rng.uniform(100, 900, (n, 3, 18))],
            "cam_poses": np.tile(np.eye(4), (n, 1, 1))}
    with tempfile.TemporaryDirectory() as d2:
        d2 = Path(d2)
        with open(d2 / "seq_a.pkl", "wb") as f:
            _pk.dump(data, f)
        recs = list(gt_records(d2, 20, 0.15, lambda k, pad: np.array([0, 0, 10, 10], np.float32)))
        assert [r["frame"] for r in recs] == [0, 20], [r["frame"] for r in recs]    # 40 is invalid
        assert recs[0]["joints_world"].shape == (24, 3) and recs[0]["poses2d"].shape == (3, 18)
        none_recs = list(gt_records(d2, 20, 0.15, lambda k, pad: None))
        assert none_recs == [], "records without a usable box are skipped"

        def fake_predict(img, bbox):
            return {"betas": np.zeros(10), "global_orient": np.eye(3), "body_pose": np.tile(np.eye(3), (23, 1, 1)),
                    "cam_t": np.array([0, 0, 5.0]), "focal": 5000.0, "vertices": np.zeros((6890, 3)),
                    "img_wh": np.array([1080, 1920])}

        arrays = run(fake_predict, recs, lambda p: 1, d2)
        assert arrays["ok"].tolist() == [True, True] and arrays["cam_t"].shape == (2, 3)
        assert arrays["joints_world"].shape == (2, 24, 3) and arrays["K"].shape == (2, 3, 3)
        assert "vertices" not in arrays
        arrays2 = run(lambda img, bbox: None if bbox[2] == 10 and False else fake_predict(img, bbox),
                      recs, lambda p: None if "00020" in str(p) else 1, d2)
        assert arrays2["ok"].tolist() == [True, False] and arrays2["names"][1].endswith("__p0")
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
