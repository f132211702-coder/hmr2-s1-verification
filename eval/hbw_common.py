#!/usr/bin/env python3
"""Shared pieces for the HBW (Human Bodies in the Wild, SHAPY, CVPR 2022) evaluation.

Licence: research / education only, no redistribution -- keep the data on the
server, never in git (see the LICENSE file shipped with the data).

Layout of the validation subset (verified on the real download, "low resolution"
version, photos about 200x300 px):
    images/val_small_resolution/<sid>_<nLab>_<nWild>/{Photos_Lab,Pictures_in_the_Wild}/NNNNN.png
    keypoints/val_small_resolution/<same>/<same>/NNNNN.json   OpenPose BODY_25 in image pixels
    smplx/val/<sid>.npy   (10475, 3) scan-aligned SMPL-X vertices, metres, T-pose, y up
    smplx/val/<sid>.obj   the same mesh with faces
The subset has no gender, height or weight files: the height comes from the scan.
macOS tar leaves "._*" AppleDouble files behind; every walker here skips them.

Prediction CSV (predict_hbw.py): image_id (path under images/), subject, kind
(lab / wild), status, pred_beta_0..9.

Usage (self-test, numpy only): python eval/hbw_common.py --self-test
"""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_BETAS = 10
KINDS = {"Photos_Lab": "lab", "Pictures_in_the_Wild": "wild"}


@dataclass
class HBWRecord:
    image_id: str       # relative path under images/
    subject: str        # "012"
    kind: str           # "lab" | "wild"
    img_path: Path
    kp_path: Path


def _visible(p: Path) -> bool:
    return not p.name.startswith("._") and not p.name.startswith(".")


def hbw_records(root: Path, split: str = "val", stride: int = 1) -> list[HBWRecord]:
    root = Path(root).expanduser()
    img_root = root / "images" / f"{split}_small_resolution"
    kp_root = root / "keypoints" / f"{split}_small_resolution"
    if not img_root.is_dir():
        raise SystemExit(f"{img_root} not found")
    out = []
    for sub in sorted(p for p in img_root.iterdir() if p.is_dir() and _visible(p)):
        sid = sub.name.split("_")[0]
        for kdir in sorted(p for p in sub.iterdir() if p.is_dir() and _visible(p)):
            kind = KINDS.get(kdir.name)
            if kind is None:
                continue
            for img in sorted(p for p in kdir.glob("*.png") if _visible(p)):
                out.append(HBWRecord(f"{sub.name}/{kdir.name}/{img.name}", sid, kind, img,
                                     kp_root / sub.name / kdir.name / (img.stem + ".json")))
    return out[::stride]


def bbox_from_openpose(json_path: Path, pad: float = 0.15, min_conf: float = 0.1,
                       min_points: int = 4) -> np.ndarray | None:
    """[x1,y1,x2,y2] around the confident keypoints of the best-scoring person (padded by `pad` of
    the box size on every side); None when the file is missing or has no usable person."""
    try:
        people = json.loads(Path(json_path).read_text())["people"]
    except (OSError, ValueError, KeyError):
        return None
    best, best_score = None, 0.0
    for p in people:
        pts = np.asarray(p.get("pose_keypoints_2d", []), dtype=np.float64).reshape(-1, 3)
        sel = pts[pts[:, 2] > min_conf]
        score = float(sel[:, 2].sum()) if len(sel) >= min_points else 0.0
        if score > best_score:
            best, best_score = sel, score
    if best is None:
        return None
    x1, y1 = best[:, 0].min(), best[:, 1].min()
    x2, y2 = best[:, 0].max(), best[:, 1].max()
    w, h = x2 - x1, y2 - y1
    return np.array([x1 - pad * w, y1 - pad * h, x2 + pad * w, y2 + pad * h], dtype=np.float32)


def write_csv(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def load_prediction_csv(path: Path) -> dict[str, dict]:
    """image_id -> {subject, kind, pred (10,)} for rows with status ok."""
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        need = [f"pred_beta_{i}" for i in range(N_BETAS)] + ["subject", "kind"]
        missing = [c for c in need if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path} has no {missing[0]} column -- not a predict_hbw.py CSV.")
        for r in reader:
            if r.get("status") == "ok":
                out[r["image_id"]] = {"subject": r["subject"], "kind": r["kind"],
                                      "pred": np.array([float(r[f"pred_beta_{i}"]) for i in range(N_BETAS)])}
    return out


def self_test() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for sub, kinds in (("012_2_1", ("Photos_Lab", "Pictures_in_the_Wild")), ("017_1_0", ("Photos_Lab",))):
            for k in kinds:
                (d / "images/val_small_resolution" / sub / k).mkdir(parents=True)
                (d / "keypoints/val_small_resolution" / sub / k).mkdir(parents=True)
                for i in range(2):
                    (d / "images/val_small_resolution" / sub / k / f"0000{i}.png").write_bytes(b"x")
                    (d / "images/val_small_resolution" / sub / k / f"._0000{i}.png").write_bytes(b"junk")
        (d / "images/val_small_resolution" / "._012_2_1").mkdir()
        (d / "images/val_small_resolution" / "012_2_1" / "Other").mkdir()
        recs = hbw_records(d)
        assert len(recs) == 6, len(recs)                      # junk, "._" folders and unknown folders skipped
        assert {r.subject for r in recs} == {"012", "017"} and {r.kind for r in recs} == {"lab", "wild"}
        assert len(hbw_records(d, stride=2)) == 3

        # bbox: picks the better person, pads, ignores low-confidence points
        kp = d / "kp.json"
        good = [100, 50, 0.9, 120, 80, 0.9, 110, 200, 0.8, 90, 260, 0.9, 5, 5, 0.01]
        weak = [10, 10, 0.2, 20, 20, 0.2, 30, 30, 0.2, 40, 40, 0.2]
        kp.write_text(json.dumps({"people": [{"pose_keypoints_2d": weak}, {"pose_keypoints_2d": good}]}))
        bb = bbox_from_openpose(kp, pad=0.1)
        assert np.allclose(bb, [90 - 3, 50 - 21, 120 + 3, 260 + 21]), bb   # x 90..120, y 50..260; (5,5) at conf .01 ignored
        assert bbox_from_openpose(d / "missing.json") is None
        kp.write_text(json.dumps({"people": []}))
        assert bbox_from_openpose(kp) is None

        rows = [{"image_id": "a", "subject": "012", "kind": "lab", "status": "ok",
                 **{f"pred_beta_{i}": 0.1 * i for i in range(N_BETAS)}},
                {"image_id": "b", "subject": "012", "kind": "lab", "status": "no_person"}]
        write_csv(rows, d / "p.csv")
        back = load_prediction_csv(d / "p.csv")
        assert set(back) == {"a"} and np.allclose(back["a"]["pred"], 0.1 * np.arange(N_BETAS))
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        raise SystemExit("library module; run with --self-test")
