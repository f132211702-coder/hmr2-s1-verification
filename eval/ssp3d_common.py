#!/usr/bin/env python3
"""Shared pieces for the SSP-3D shape evaluation (predict_ssp3d.py makes the
per-model prediction CSVs, eval_ssp3d_shape.py scores them).

SSP-3D (Sengupta et al., github.com/akashsengupta1997/SSP-3D, MIT): 311 photos
of 62 sportspeople in tight clothing. Layout after unzipping ssp_3d.zip:
    <root>/ssp_3d/images/<fname>          photos
    <root>/ssp_3d/labels.npz              fnames, shapes (311,10), genders ('m'/'f'),
                                          bbox_centres (311,2) [x,y], bbox_whs (311,)
                                          (square box side), poses, cam_trans, joints2D
(verified against a real checkout with data_prep/inspect_ssp3d.py.)

The shapes are pseudo-ground-truth: SMPL fits to the person (multi-frame
optimisation against silhouettes/keypoints), expressed in the GENDERED SMPL
model matching the person's label -- not tape measurements.

Prediction CSV format (one row per photo; the same beta columns as the 3DPW
eval CSVs, plus gender): image_id, person_id, status, gender, pred_beta_0..9,
gt_beta_0..9. person_id is a stable id per distinct GT shape vector (= per
person).

Usage (self-test, numpy only):
    python eval/ssp3d_common.py --self-test
"""
from __future__ import annotations

import csv
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_BETAS = 10
GENDER_NAMES = {"m": "MALE", "f": "FEMALE"}


@dataclass
class SSP3DRecord:
    fname: str
    img_path: Path
    bbox: np.ndarray        # [x1, y1, x2, y2], full-image pixels
    gt_betas: np.ndarray    # (10,), gendered SMPL
    gender: str             # "MALE" / "FEMALE"
    person_id: str          # one id per distinct GT shape vector


def bbox_from_center_wh(center: np.ndarray, wh: float) -> np.ndarray:
    """SSP-3D stores a square box as (centre x, centre y) and a side length."""
    cx, cy = float(center[0]), float(center[1])
    h = float(wh) / 2.0
    return np.array([cx - h, cy - h, cx + h, cy + h], dtype=np.float32)


def load_ssp3d(root: Path, stride: int = 1) -> list[SSP3DRecord]:
    root = Path(root).expanduser()
    labels = root / "ssp_3d" / "labels.npz"
    if not labels.exists():
        raise SystemExit(f"{labels} not found -- unzip ssp_3d.zip inside the SSP-3D clone.")
    with np.load(labels, allow_pickle=True) as d:
        fnames, shapes, genders = d["fnames"], d["shapes"], d["genders"]
        centres, whs = d["bbox_centres"], d["bbox_whs"]
    ids: dict[tuple, str] = {}
    out = []
    for i in range(0, len(fnames), stride):
        key = tuple(np.round(shapes[i].astype(np.float64), 4))
        pid = ids.setdefault(key, f"p{len(ids):02d}")
        g = GENDER_NAMES[str(genders[i]).lower()]
        out.append(SSP3DRecord(str(fnames[i]), root / "ssp_3d" / "images" / str(fnames[i]),
                               bbox_from_center_wh(centres[i], whs[i]),
                               shapes[i].astype(np.float64), g, pid))
    return out


def scale_and_translation_transform(P: np.ndarray, T: np.ndarray) -> np.ndarray:
    """SSP-3D metrics.py: move/scale mesh P so its centroid and RMS distance
    to the centroid match mesh T."""
    Pc = P - P.mean(axis=0)
    Tc = T - T.mean(axis=0)
    p_rms = np.sqrt((Pc ** 2).sum() / len(P))
    t_rms = np.sqrt((Tc ** 2).sum() / len(T))
    return Pc / p_rms * t_rms + T.mean(axis=0)


def pve_t_sc(v_pred: np.ndarray, v_gt: np.ndarray) -> float:
    """SSP-3D's PVE-T-SC: mean per-vertex distance between two T-pose meshes after
    the scale-and-translation correction above. Same units as the input."""
    return float(np.linalg.norm(scale_and_translation_transform(v_pred, v_gt) - v_gt, axis=1).mean())


def beta_columns(prefix: str, betas: np.ndarray) -> dict:
    return {f"{prefix}_{i}": float(b) for i, b in enumerate(np.asarray(betas).reshape(-1)[:N_BETAS])}


def write_prediction_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def load_prediction_csv(path: Path) -> dict[str, dict]:
    """fname -> {pred, gt, gender, person_id} for rows with status ok."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        need = [f"{p}_beta_{i}" for p in ("pred", "gt") for i in range(N_BETAS)] + ["gender"]
        missing = [c for c in need if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path} has no {missing[0]} column -- not a predict_ssp3d.py CSV.")
        for r in reader:
            if r.get("status") != "ok":
                continue
            out[r["image_id"]] = {
                "pred": np.array([float(r[f"pred_beta_{i}"]) for i in range(N_BETAS)]),
                "gt": np.array([float(r[f"gt_beta_{i}"]) for i in range(N_BETAS)]),
                "gender": r["gender"], "person_id": r["person_id"]}
    return out


def self_test() -> None:
    assert np.allclose(bbox_from_center_wh(np.array([250.0, 260.0]), 100), [200, 210, 300, 310])

    rng = np.random.default_rng(0)
    T = rng.normal(size=(50, 3))
    assert pve_t_sc(T, T) < 1e-12
    assert pve_t_sc(T * 3.0 + 7.0, T) < 1e-12, "scale and translation must not count as error"
    noisy = T + rng.normal(scale=0.1, size=T.shape)
    assert pve_t_sc(noisy, T) > 0.01

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "ssp_3d").mkdir()
        shapes = np.repeat(np.eye(10, dtype=np.float32)[:2], [2, 1], axis=0)
        np.savez(d / "ssp_3d" / "labels.npz", fnames=np.array(["a.png", "b.png", "c.png"]),
                 shapes=shapes, genders=np.array(["f", "f", "m"]),
                 bbox_centres=np.full((3, 2), 256.0), bbox_whs=np.array([200, 200, 300]))
        recs = load_ssp3d(d)
        assert [r.person_id for r in recs] == ["p00", "p00", "p01"], [r.person_id for r in recs]
        assert [r.gender for r in recs] == ["FEMALE", "FEMALE", "MALE"]
        assert np.allclose(recs[2].bbox, [106, 106, 406, 406])
        assert len(load_ssp3d(d, stride=2)) == 2

        rows = [{"image_id": r.fname, "person_id": r.person_id, "status": "ok",
                 "gender": r.gender, **beta_columns("pred_beta", np.zeros(10)),
                 **beta_columns("gt_beta", r.gt_betas)} for r in recs]
        rows.append({"image_id": "x.png", "person_id": "p09", "status": "no_image"})
        write_prediction_csv(rows, d / "pred.csv")
        back = load_prediction_csv(d / "pred.csv")
        assert set(back) == {"a.png", "b.png", "c.png"}
        assert back["c.png"]["gender"] == "MALE" and np.allclose(back["c.png"]["gt"], shapes[2])
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        raise SystemExit("library module; run with --self-test")
