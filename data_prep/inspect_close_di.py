#!/usr/bin/env python3
"""Print what a CloSe-Di folder actually contains, so the loaders written
earlier against the pipeline plan doc's *assumed* layout
(eval/eval_against_gt.py::load_close_di_gt, data_prep/render_close_di_scan.py)
can be checked against a real file instead of guessed.

For a folder it reports: file counts per extension, and for the first
--n .npz files every key with shape/dtype (and the values, when small, e.g.
betas or a gender flag) plus the value range of anything colour-like. Then,
over ALL .npz files that have a betas-like key, how many distinct bodies
there are and their beta norms -- the numbers that decide how useful the
dataset is as ground truth.

Usage (self-test, no data needed):
    python data_prep/inspect_close_di.py --self-test

Usage:
    python data_prep/inspect_close_di.py --dir /path/to/CloSe-Di --n 3
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


def describe_file(path: Path) -> list[str]:
    lines = [f"{path}"]
    with np.load(path, allow_pickle=True) as data:
        for key in data.files:
            arr = data[key]
            desc = f"  {key:22s} shape {str(arr.shape):18s} dtype {arr.dtype}"
            if arr.dtype.kind in "fiub" and arr.size:
                desc += f"  range [{arr.min():.3f}, {arr.max():.3f}]"
                if arr.size <= 12:
                    desc += f"  values {np.array2string(arr.reshape(-1), precision=3)}"
            elif arr.dtype.kind in "OUS" and arr.size <= 3:
                desc += f"  values {arr.reshape(-1).tolist()}"
            lines.append(desc)
    return lines


def summarize_betas(files: list[Path]) -> list[str]:
    betas = []
    for f in files:
        with np.load(f, allow_pickle=True) as data:
            key = next((k for k in data.files if "beta" in k.lower() or k.lower() in ("shape", "betas")), None)
            if key is not None:
                betas.append(np.asarray(data[key], dtype=np.float64).reshape(-1)[:10])
    if not betas:
        return ["no betas-like key found in any .npz (look at the key list above)"]
    B = np.stack(betas)
    distinct = len({tuple(np.round(b, 5)) for b in B})
    norms = np.linalg.norm(B, axis=1)
    return [f"betas-like key found in {len(B)} of {len(files)} files; {distinct} distinct bodies",
            f"first-10-beta norm: mean {norms.mean():.2f}  min {norms.min():.2f}  max {norms.max():.2f}"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--n", type=int, default=3, help="how many files to describe in full")
    args = ap.parse_args()
    root = Path(args.dir).expanduser()
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    print(f"{len(files)} file(s) under {root}")
    for ext, n in Counter(p.suffix.lower() or "(none)" for p in files).most_common():
        print(f"  {n:6d} x {ext}")
    print("\nfirst folders/files:")
    for p in files[:8]:
        print("  ", p.relative_to(root))
    npz = [p for p in files if p.suffix.lower() == ".npz"]
    if not npz:
        print("\nno .npz files -- tell me the extensions listed above.")
        return
    print()
    for p in npz[: args.n]:
        print("\n".join(describe_file(p)))
        print()
    print("\n".join(summarize_betas(npz)))


def self_test() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for i in range(3):
            np.savez(d / f"scan_{i}.npz", points=np.random.rand(50, 3), faces=np.zeros((10, 3), int),
                     colors=np.random.randint(0, 255, (50, 3)), betas=np.full(10, float(i % 2)),
                     gender=np.array("male"))
        lines = describe_file(d / "scan_0.npz")
        assert any("colors" in ln and "range" in ln for ln in lines)
        assert any("gender" in ln and "male" in ln for ln in lines)
        summ = summarize_betas(sorted(d.glob("*.npz")))
        assert "3 of 3" in summ[0] and "2 distinct" in summ[0], summ
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
