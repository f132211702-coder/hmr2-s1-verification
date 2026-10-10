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

--dump-npz PATH additionally saves the full prediction (pose, camera, vertices) for
eval_ssp3d_pose_cam.py: betas (N,10), global_orient (N,3,3), body_pose (N,23,3,3) as rotation
matrices, cam_t (N,3) and focal (N,) = full-image perspective camera (principal point = image
centre), vertices (N,6890,3) in the model's crop-camera frame (add cam_t to place them), img_wh,
fnames and an ok mask. Each model's own demo code defines the camera (see the make_* docstrings).

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


def make_hmr2(full: bool = False):
    """full=True: predict returns a dict (see --dump-npz) instead of just the betas. Camera = the
    one s1_infer.HMR2Estimator already converts to full-image (cam_crop_to_full)."""
    from s1_infer import HMR2Estimator
    est = HMR2Estimator()

    def predict(img_bgr: np.ndarray, bbox: np.ndarray):
        p = est.estimate(img_bgr, boxes=bbox[None])[0]
        if not full:
            return np.asarray(p["betas"], dtype=np.float64).reshape(-1)
        return {"betas": np.asarray(p["betas"], dtype=np.float32).reshape(-1),
                "global_orient": np.asarray(p["global_orient"], dtype=np.float32).reshape(3, 3),
                "body_pose": np.asarray(p["body_pose"], dtype=np.float32).reshape(23, 3, 3),
                "cam_t": np.asarray(p["cam_t"], dtype=np.float32).reshape(3),
                "focal": float(p["scaled_focal_length"]),
                "vertices": np.asarray(p["pred_vertices"], dtype=np.float32),
                "img_wh": np.array([img_bgr.shape[1], img_bgr.shape[0]])}

    return predict


def make_camerahmr(root: Path, full: bool = False):
    """full=True: camera = what CameraHMR's own demo renders with: est.get_output_mesh's full-image
    translation and the focal length the model returns (its third output)."""
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

    def predict(img_bgr: np.ndarray, bbox: np.ndarray):
        x1, y1, x2, y2 = bbox
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
        scale = np.array([(x2 - x1) / 200.0, (y2 - y1) / 200.0], dtype=np.float32)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        cam_int = est.get_cam_intrinsics(img_rgb)
        dataset = Dataset(img_rgb, center[None], scale[None], cam_int, False, "ssp3d")
        batch = recursive_to(torch.utils.data.dataloader.default_collate([dataset[0]]), est.device)
        with torch.no_grad():
            smpl_params, out_cam, focal_ = est.model(batch)
        if not full:
            return smpl_params["betas"][0].detach().cpu().numpy().astype(np.float64)
        verts, _, cam_trans = est.get_output_mesh(smpl_params, out_cam, batch)
        f = focal_[0] if hasattr(focal_, "__len__") else focal_
        return {"betas": smpl_params["betas"][0].detach().cpu().numpy().astype(np.float32),
                "global_orient": smpl_params["global_orient"][0].detach().cpu().numpy().astype(np.float32).reshape(3, 3),
                "body_pose": smpl_params["body_pose"][0].detach().cpu().numpy().astype(np.float32).reshape(23, 3, 3),
                "cam_t": cam_trans[0].detach().cpu().numpy().astype(np.float32),
                "focal": float(f),
                "vertices": verts[0].detach().cpu().numpy().astype(np.float32),
                "img_wh": np.array([img_bgr.shape[1], img_bgr.shape[0]])}

    return predict


def cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal_length):
    """Crop camera (s, tx, ty) -> full-image translation; same formula as 4D-Humans' / TokenHMR's
    lib.utils.renderer.cam_crop_to_full (re-implemented to avoid importing the OpenGL renderer)."""
    import torch
    w_2, h_2 = img_size[:, 0] / 2.0, img_size[:, 1] / 2.0
    bs = box_size * pred_cam[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = 2 * (box_center[:, 0] - w_2) / bs + pred_cam[:, 1]
    ty = 2 * (box_center[:, 1] - h_2) / bs + pred_cam[:, 2]
    return torch.stack([tx, ty, tz], dim=-1)


def make_tokenhmr(root: Path, checkpoint: str, model_config: str, full: bool = False):
    """full=True: camera as in TokenHMR's demo.py: cam_crop_to_full with focal =
    model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * max(image size)."""
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

    def predict(img_bgr: np.ndarray, bbox: np.ndarray):
        dataset = ViTDetDataset(model_cfg, img_bgr, bbox[None])
        batch = recursive_to(torch.utils.data.dataloader.default_collate([dataset[0]]), device)
        with torch.no_grad():
            out = model(batch)
        betas = out["pred_smpl_params"]["betas"].reshape(-1).detach().cpu().numpy()
        if not full:
            return betas.astype(np.float64)
        img_size = batch["img_size"].float()
        focal = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_t = cam_crop_to_full(out["pred_cam"], batch["box_center"].float(), batch["box_size"].float(),
                                 img_size, focal)
        sp = out["pred_smpl_params"]
        return {"betas": betas.astype(np.float32),
                "global_orient": sp["global_orient"].reshape(3, 3).detach().cpu().numpy().astype(np.float32),
                "body_pose": sp["body_pose"].reshape(23, 3, 3).detach().cpu().numpy().astype(np.float32),
                "cam_t": cam_t[0].detach().cpu().numpy().astype(np.float32),
                "focal": float(focal),
                "vertices": out["pred_vertices"][0].detach().cpu().numpy().astype(np.float32),
                "img_wh": np.array([img_bgr.shape[1], img_bgr.shape[0]])}

    return predict


def run(predict, records, read_image, collect: list | None = None) -> list[dict]:
    """collect (optional): list that receives one entry per record, the model's full-output dict (or
    None when the photo was skipped), in record order -- for --dump-npz."""
    rows = []
    for n, rec in enumerate(records):
        img = read_image(rec.img_path)
        base = {"image_id": rec.fname, "person_id": rec.person_id, "gender": rec.gender}
        if img is None:
            rows.append({**base, "status": "no_image"})
            if collect is not None:
                collect.append(None)
            continue
        out = predict(img, rec.bbox)
        betas = out["betas"] if isinstance(out, dict) else out
        if collect is not None:
            collect.append(out if isinstance(out, dict) else None)
        rows.append({**base, "status": "ok", **beta_columns("pred_beta", betas),
                     **beta_columns("gt_beta", rec.gt_betas)})
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(records)}", flush=True)
    return rows


def save_full_npz(path: Path, fnames: list[str], collected: list) -> None:
    """Stack the per-photo dicts into arrays (zeros + ok=False for skipped photos)."""
    ref = next((c for c in collected if c is not None), None)
    if ref is None:
        raise SystemExit("no photo was predicted; nothing to save")
    n = len(fnames)
    arrays = {k: np.zeros((n,) + np.asarray(v).shape, dtype=np.float32 if k != "img_wh" else np.int64)
              for k, v in ref.items()}
    ok = np.zeros(n, dtype=bool)
    for i, c in enumerate(collected):
        if c is None:
            continue
        ok[i] = True
        for k in arrays:
            arrays[k][i] = c[k]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, fnames=np.array(fnames), ok=ok, **arrays)
    print(f"wrote {path} ({int(ok.sum())}/{n} photos)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["hmr2", "camerahmr", "tokenhmr"])
    ap.add_argument("--ssp3d", required=True, help="SSP-3D clone (contains ssp_3d/labels.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--dump-npz", default=None, help="also save the full prediction (pose, camera, vertices)")
    ap.add_argument("--camerahmr_root")
    ap.add_argument("--tokenhmr_root")
    ap.add_argument("--checkpoint", default="data/checkpoints/tokenhmr_model_latest.ckpt")
    ap.add_argument("--model_config", default="data/checkpoints/model_config.yaml")
    args = ap.parse_args()

    # resolve user paths before any model loader chdirs
    out = Path(args.out).expanduser().resolve()
    dump = Path(args.dump_npz).expanduser().resolve() if args.dump_npz else None
    full = dump is not None
    records = load_ssp3d(Path(args.ssp3d).expanduser().resolve(), stride=args.stride)
    print(f"{len(records)} photo(s), {len({r.person_id for r in records})} people")

    if args.model == "hmr2":
        predict = make_hmr2(full)
    elif args.model == "camerahmr":
        if not args.camerahmr_root:
            raise SystemExit("--camerahmr_root is required")
        predict = make_camerahmr(Path(args.camerahmr_root).expanduser().resolve(), full)
    else:
        if not args.tokenhmr_root:
            raise SystemExit("--tokenhmr_root is required")
        predict = make_tokenhmr(Path(args.tokenhmr_root).expanduser().resolve(),
                                args.checkpoint, args.model_config, full)

    import cv2
    collected: list | None = [] if full else None
    rows = run(predict, records, lambda p: cv2.imread(str(p)), collected)
    write_prediction_csv(rows, out)
    if dump is not None:
        save_full_npz(dump, [r.fname for r in records], collected)
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
    # full-output mode: dicts are collected in record order (None for skipped photos) and stacked
    def fake_full(img, bbox):
        return {"betas": np.arange(10, dtype=np.float32), "global_orient": np.eye(3, dtype=np.float32),
                "body_pose": np.tile(np.eye(3, dtype=np.float32), (23, 1, 1)), "cam_t": np.array([0, 0, 5], np.float32),
                "focal": 5000.0, "vertices": np.zeros((6890, 3), np.float32), "img_wh": np.array([512, 512])}
    coll: list = []
    run(fake_full, recs, lambda p: None if p.name == "1.png" else 2, coll)
    assert [c is None for c in coll] == [False, True, False]
    import tempfile as _tf
    with _tf.TemporaryDirectory() as _d:
        save_full_npz(Path(_d) / "f.npz", [r.fname for r in recs], coll)
        z = np.load(Path(_d) / "f.npz")
        assert z["ok"].tolist() == [True, False, True] and z["vertices"].shape == (3, 6890, 3)
        assert z["body_pose"].shape == (3, 23, 3, 3) and np.allclose(z["cam_t"][2], [0, 0, 5]) and z["focal"][0] == 5000.0
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
