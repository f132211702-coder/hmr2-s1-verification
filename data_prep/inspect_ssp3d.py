#!/usr/bin/env python3
"""Print what an SSP-3D checkout actually contains before any evaluation is
written against it: files per folder, every key of labels.npz with
shape/dtype, the gender split, how many distinct people (distinct shape
vectors) there are, and the first record in full.

SSP-3D (Sengupta et al., "Synthetic Training for Accurate 3D Human Pose and
Shape Estimation in the Wild"): 311 photos of ~62 people in tight sports
clothes, SMPL shape pseudo-ground-truth + gender.

Usage (self-test, no data needed):
    python data_prep/inspect_ssp3d.py --self-test

Usage:
    python data_prep/inspect_ssp3d.py --dir ~/datasets/SSP-3D
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


def find_labels(root: Path) -> Path | None:
    hits = sorted(root.rglob("labels.npz"))
    return hits[0] if hits else None


def describe_labels(path: Path) -> list[str]:
    lines = [f"{path}"]
    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            arr = data[key]
            desc = f"  {key:18s} shape {str(arr.shape):16s} dtype {arr.dtype}"
            if arr.dtype.kind in "fiub" and arr.size:
                desc += f"  range [{arr.min():.3f}, {arr.max():.3f}]"
            lines.append(desc)
        lines.append("")
        lines.extend(first_record(data))
        lines.append("")
        lines.extend(summarize_people(data))
    return lines


def first_record(data) -> list[str]:
    out = ["first record:"]
    for key in data.files:
        arr = data[key]
        if arr.ndim == 0 or len(arr) == 0:
            continue
        item = np.asarray(arr[0])
        text = item.tolist() if item.dtype.kind in "OUS" else np.array2string(item.reshape(-1)[:12], precision=3)
        out.append(f"  {key:18s} {text}")
    return out


def shape_key(data) -> str | None:
    return next((k for k in data.files if k.lower() in ("shapes", "shape", "betas")), None)


def gender_key(data) -> str | None:
    return next((k for k in data.files if "gender" in k.lower()), None)


def summarize_people(data) -> list[str]:
    out = []
    sk, gk = shape_key(data), gender_key(data)
    if sk is None:
        return ["no shape/betas key found -- look at the key list above"]
    shapes = np.asarray(data[sk], dtype=np.float64).reshape(len(data[sk]), -1)
    distinct = {tuple(np.round(s, 4)) for s in shapes}
    out.append(f"{len(shapes)} records, {len(distinct)} distinct shape vectors "
               f"({shapes.shape[1]} betas each); beta-norm mean {np.linalg.norm(shapes, axis=1).mean():.2f}")
    if gk is not None:
        genders = [str(g) for g in data[gk]]
        out.append(f"gender split: {dict(Counter(genders))}")
        by_gender = {}
        for g, s in zip(genders, shapes):
            by_gender.setdefault(g, set()).add(tuple(np.round(s, 4)))
        out.append(f"distinct people per gender: { {g: len(v) for g, v in by_gender.items()} }")
    else:
        out.append("no gender key found")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    root = Path(args.dir).expanduser()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")
    files = [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    print(f"{len(files)} file(s) under {root}")
    per_folder = Counter(str(p.parent.relative_to(root)) for p in files)
    for folder, n in sorted(per_folder.items()):
        print(f"  {n:6d} files in {folder}")
    print("  extensions:", dict(Counter(p.suffix.lower() or "(none)" for p in files)))
    labels = find_labels(root)
    if labels is None:
        print("\nno labels.npz found -- paste the folder listing above.")
        return
    print()
    print("\n".join(describe_labels(labels)))


def self_test() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        shapes = np.repeat(np.eye(10)[:2], [3, 2], axis=0)
        np.savez(d / "labels.npz", fnames=np.array([f"{i}.png" for i in range(5)]), shapes=shapes,
                 genders=np.array(["m", "m", "m", "f", "f"]))
        assert find_labels(d) == d / "labels.npz"
        text = "\n".join(describe_labels(d / "labels.npz"))
        assert "5 records, 2 distinct" in text, text
        assert "'m': 3" in text and "'f': 2" in text, text
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
