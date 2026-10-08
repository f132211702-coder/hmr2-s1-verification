#!/usr/bin/env python3
"""Run HMR2.0b, CameraHMR or TokenHMR on the SSP-3D photos (ground-truth boxes,
no detector) and save each photo's predicted betas next to the pseudo-GT betas
and gender, for eval_ssp3d_shape.py.

One script, three models, because each model lives in its own conda env and
only differs in how it is loaded and called. Each run writes
results/ssp3d_<model>.csv (format: see ssp3d_common.py).

    HMR2.0b    env 4D-humans, from the repo root (uses s1_infer.HMR2Estimator)
    CameraHMR  env camerahmr, --camerahmr_root (same loading as
               eval_camerahmr_against_gt.py)
    TokenHMR   env tokenhmr, --tokenhmr_root (same loading as
               eval_tokenhmr_against_gt.py)

Usage (self-test, no model/data): python eval/predict_ssp3d.py --self-test

Usage (server, repo root = ~/workspace/dresson/Tims_dir):
    conda activate 4D-humans
    python eval/predict_ssp3d.py --model hmr2 --ssp3d ~/datasets/SSP-3D --out results/ssp3d_hmr2.csv
    conda activate camerahmr
    python eval/predict_ssp3d.py --model camerahmr --camerahmr_root ~/workspace/dresson/CameraHMR \\
        --ssp3d ~/datasets/SSP-3D --out ~/workspace/dresson/Tims_dir/results/ssp3d_camerahmr.csv
    conda activate tokenhmr
    python eval/predict_ssp3d.py --model tokenhmr --tokenhmr_root ~/workspace/dresson/TokenHMR \\
        --ssp3d ~/datasets/SSP-3D --out ~/workspace/dresson/Tims_dir/results/ssp3d_tokenhmr.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from ssp3d_common import (beta_columns, load_ssp3d, write_prediction_csv)  # noqa: E402


def make_hmr2():
    from s1_infer import HMR2Estimator
    est = HMR2Estimator()

    def predict(img_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        people = est.estimate(img_bgr, boxes=bbox[None])
        return np.asarray(people[0]["betas"], dtype=np.float64).reshape(-1)

    return predict


def make_camerahmr(root: Path):
    sys.path.insert(0, str(root))
    os.chdir(root)  # core/constants.py checkpoint paths are relative
    import cv2
    import torch
    from core.datasets.dataset import Dataset
    from core.utils import recursive_to
    from mesh_estimator import HumanMeshEstimator

    class GTBoxEstimator(HumanMeshEstimator):
        def init_detector(self, threshold):  # boxes come from the labels
            return None

    est = GTBoxEstimator(model_type="smpl")

    def predict(img_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
        scale = np.array([(x2 - x1) / 200.0, (y2 - y1) / 200.0], dtype=np.float32)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cam_int = est.get_cam_intrinsics(img_rgb)
        dataset = Dataset(img_rgb, center[None], scale[None], cam_int, False, "ssp3d")
        batch = recursive_to(torch.utils.data.dataloader.default_collate([dataset[0]]), est.device)
        with torch.no_grad():
            smpl_params, _, _ = est.model(batch)
        return smpl_params["betas"][0].detach().cpu().numpy().astype(np.float64)

    return predict


def make_tokenhmr(root: Path, checkpoint: str, model_config: str):
    sys.path.insert(0, str(root / "tokenhmr"))
    os.chdir(root)
    import torch
    from lib.datasets.vitdet_dataset import ViTDetDataset
    from lib.models import load_tokenhmr
    from lib.utils import recursive_to

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_tokenhmr(checkpoint_path=checkpoint, model_cfg=model_config,
                                     is_train_state=False, is_demo=True)
    model = model.to(device).eval()

    def predict(img_bgr: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        dataset = ViTDetDataset(model_cfg, img_bgr, bbox[None])
        batch = recursive_to(torch.utils.data.dataloader.default_collate([dataset[0]]), device)
        with torch.no_grad():
            out = model(batch)
        return out["pred_smpl_params"]["betas"].reshape(-1).detach().cpu().numpy().astype(np.float64)

    return predict


def run(predict, records, read_image) -> list[dict]:
    rows = []
    for n, rec in enumerate(records):
        img = read_image(rec.img_path)
        base = {"image_id": rec.fname, "person_id": rec.person_id, "gender": rec.gender}
        if img is None:
            rows.append({**base, "status": "no_image"})
            continue
        betas = predict(img, rec.bbox)
        rows.append({**base, "status": "ok", **beta_columns("pred_beta", betas),
                     **beta_columns("gt_beta", rec.gt_betas)})
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(records)}", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["hmr2", "camerahmr", "tokenhmr"])
    ap.add_argument("--ssp3d", required=True, help="SSP-3D clone (contains ssp_3d/labels.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--camerahmr_root")
    ap.add_argument("--tokenhmr_root")
    ap.add_argument("--checkpoint", default="data/checkpoints/tokenhmr_model_latest.ckpt")
    ap.add_argument("--model_config", default="data/checkpoints/model_config.yaml")
    args = ap.parse_args()

    # resolve user paths before any model loader chdirs
    out = Path(args.out).expanduser().resolve()
    records = load_ssp3d(Path(args.ssp3d).expanduser().resolve(), stride=args.stride)
    print(f"{len(records)} photo(s), {len({r.person_id for r in records})} people")

    if args.model == "hmr2":
        predict = make_hmr2()
    elif args.model == "camerahmr":
        if not args.camerahmr_root:
            raise SystemExit("--camerahmr_root is required")
        predict = make_camerahmr(Path(args.camerahmr_root).expanduser().resolve())
    else:
        if not args.tokenhmr_root:
            raise SystemExit("--tokenhmr_root is required")
        predict = make_tokenhmr(Path(args.tokenhmr_root).expanduser().resolve(),
                                args.checkpoint, args.model_config)

    import cv2
    rows = run(predict, records, lambda p: cv2.imread(str(p)))
    write_prediction_csv(rows, out)
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"{ok}/{len(rows)} photos predicted")


def self_test() -> None:
    import tempfile
    from ssp3d_common import SSP3DRecord, load_prediction_csv
    recs = [SSP3DRecord(f"{i}.png", Path(f"{i}.png"), np.array([0, 0, 10, 10], np.float32),
                        np.full(10, 0.5), "MALE", "p00") for i in range(3)]
    seen = []

    def fake_predict(img, bbox):
        seen.append(bbox.copy())
        return np.arange(10, dtype=float) * float(img)

    rows = run(fake_predict, recs, lambda p: None if p.name == "1.png" else 2)
    assert [r["status"] for r in rows] == ["ok", "no_image", "ok"] and len(seen) == 2
    with tempfile.TemporaryDirectory() as d:
        write_prediction_csv(rows, Path(d) / "o.csv")
        back = load_prediction_csv(Path(d) / "o.csv")
    assert set(back) == {"0.png", "2.png"}
    assert np.allclose(back["0.png"]["pred"], np.arange(10) * 2.0)
    assert np.allclose(back["0.png"]["gt"], 0.5) and back["0.png"]["gender"] == "MALE"
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
