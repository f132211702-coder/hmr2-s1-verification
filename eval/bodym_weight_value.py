#!/usr/bin/env python3
"""How much does knowing a person's weight tell us about their chest / waist /
hip circumference, on REAL people (BodyM, 2,505 subjects with measured height,
weight and 14 body measurements)?

Why: eval/fit_betas_from_measurements.py and the SSP-3D evaluation measured
the value of weight with a weight computed from the ground-truth MESH volume
(circular: an upper bound). BodyM has real weights and real circumferences, so
this gives the number the "should we ask users for their weight?" decision
needs, with no tape measure asked of anyone.

Method: per gender, fit a least-squares regression on the train split
(height, weight -> chest / waist / hip), then score it on the held-out testA
(lab photos) and testB (in-the-wild photos) splits. Weights and heights of the
TEST subjects are perturbed by Gaussian noise to simulate self-report error
(--weight-noise-kg, --height-noise-cm); the regression itself is fitted on
clean values. Compared with:
  gender mean      predict the train mean for everyone of that gender (no
                   information -- the floor a method must beat)
  height           regression on height alone
  height+weight    linear
  height+weight (quadratic)  adds squares and the product
Reported: mean absolute error (cm) per measurement; the residual spread of the
real measurement itself (std) is printed so errors can be read against it.

Caveats: BodyM measurements come from lab 3D scans of a US sample (Amazon
customers), and the weights are not self-reports -- people typically under-
report weight, which the noise setting only partly mimics. The circumference
definitions are BodyM's, not SMPL-Anthropometry's. The images are silhouettes,
so this says nothing about photo models; it measures only how far height +
weight alone get.

Usage (self-test, numpy only):
    python eval/bodym_weight_value.py --self-test

Usage:
    python eval/bodym_weight_value.py --dir ~/datasets/BodyM --out results/bodym_weight_value.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data_prep"))
from inspect_bodym import column_map, merge_split, to_float  # noqa: E402

TARGETS = ["chest", "waist", "hip"]
METHODS = ["gender mean", "height", "height+weight", "height+weight (quadratic)"]


def load_split(root: Path, split: str) -> dict:
    """{'gender': array of str, 'height', 'weight', 'chest', 'waist', 'hip': float arrays}, rows
    with any missing value dropped."""
    merged = merge_split(root / split)
    if merged is None:
        raise SystemExit(f"{root / split}: needs measurements.csv and hwg_metadata.csv")
    header, rows, _ = merged
    cols = column_map(header)
    if any(cols[k] is None for k in ["height", "weight", "chest", "waist", "hip", "gender"]):
        raise SystemExit(f"{split}: could not match the needed columns: {cols}")
    keys = ["height", "weight", *TARGETS]
    data = np.array([[to_float(r[cols[k]]) for k in keys] for r in rows])
    ok = ~np.isnan(data).any(axis=1)
    out = {k: data[ok, i] for i, k in enumerate(keys)}
    out["gender"] = np.array([str(r[cols["gender"]]).strip().lower() for r in rows])[ok]
    return out


def features(height: np.ndarray, weight: np.ndarray, method: str) -> np.ndarray | None:
    if method == "gender mean":
        return None
    if method == "height":
        return height[:, None]
    X = np.column_stack([height, weight])
    if method == "height+weight":
        return X
    h, w = (height - 170.0) / 10.0, (weight - 70.0) / 10.0   # centred/scaled for conditioning
    return np.column_stack([h, w, h * h, w * w, h * w])


def fit_predict(train: dict, test_height: np.ndarray, test_weight: np.ndarray, method: str) -> dict:
    """{target: predicted values for the test rows}, fitted on `train` (one gender)."""
    preds = {}
    Xtr = features(train["height"], train["weight"], method)
    Xte = features(test_height, test_weight, method)
    for t in TARGETS:
        if Xtr is None:
            preds[t] = np.full(len(test_height), train[t].mean())
            continue
        A = np.column_stack([Xtr, np.ones(len(Xtr))])
        coef, *_ = np.linalg.lstsq(A, train[t], rcond=None)
        preds[t] = np.column_stack([Xte, np.ones(len(Xte))]) @ coef
    return preds


def evaluate(train: dict, test: dict, weight_noise_kg: float, height_noise_cm: float,
             rng: np.random.Generator, draws: int = 20) -> list[dict]:
    """One row per (gender, method, target): MAE over the test rows (averaged over noise draws)."""
    rows = []
    for gender in sorted(set(test["gender"])):
        tr = {k: v[train["gender"] == gender] for k, v in train.items()}
        te = {k: v[test["gender"] == gender] for k, v in test.items()}
        if len(tr["height"]) < 30 or len(te["height"]) < 10:
            continue
        for method in METHODS:
            maes = {t: [] for t in TARGETS}
            for _ in range(1 if (weight_noise_kg == 0 and height_noise_cm == 0) else draws):
                h = te["height"] + rng.normal(0, height_noise_cm, len(te["height"]))
                w = te["weight"] + rng.normal(0, weight_noise_kg, len(te["weight"]))
                p = fit_predict(tr, h, w, method)
                for t in TARGETS:
                    maes[t].append(np.abs(p[t] - te[t]).mean())
            for t in TARGETS:
                rows.append({"gender": gender, "method": method, "target": t, "n_test": len(te["height"]),
                             "mae_cm": float(np.mean(maes[t])), "real_std_cm": float(te[t].std())})
    return rows


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n=== {title} ===")
    for gender in sorted({r["gender"] for r in rows}):
        n = next(r["n_test"] for r in rows if r["gender"] == gender)
        stds = {r["target"]: r["real_std_cm"] for r in rows if r["gender"] == gender}
        print(f"\n[{gender}] {n} test subjects; spread (std) of the real values: "
              + "  ".join(f"{t} {stds[t]:.1f}" for t in TARGETS))
        print(f"{'method':30s}" + "".join(f"{t:>9s}" for t in TARGETS))
        for m in METHODS:
            vals = {r["target"]: r["mae_cm"] for r in rows if r["gender"] == gender and r["method"] == m}
            print(f"{m:30s}" + "".join(f"{vals[t]:9.2f}" for t in TARGETS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--weight-noise-kg", nargs="+", type=float, default=[0.0, 2.0, 5.0])
    ap.add_argument("--height-noise-cm", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/bodym_weight_value.csv")
    args = ap.parse_args()
    root = Path(args.dir).expanduser()
    train = load_split(root, "train")
    print(f"train: {len(train['height'])} subjects "
          f"({int((train['gender'] == 'female').sum())} female / {int((train['gender'] == 'male').sum())} male)")
    rng = np.random.default_rng(args.seed)
    all_rows = []
    for split in ("testA", "testB"):
        test = load_split(root, split)
        for kg in args.weight_noise_kg:
            hn = 0.0 if kg == 0 else args.height_noise_cm
            rows = evaluate(train, test, kg, hn, rng)
            label = f"{split}, weight error {kg:g} kg, height error {hn:g} cm"
            print_table(label, rows)
            all_rows += [{"split": split, "weight_noise_kg": kg, "height_noise_cm": hn, **r} for r in rows]
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nwrote {out}")


def self_test() -> None:
    rng = np.random.default_rng(0)

    def make(n):
        gender = np.array(["male", "female"] * (n // 2))
        height = rng.normal(172, 8, n)
        weight = 22 * (height / 100) ** 2 * rng.normal(1, 0.2, n)
        waist = 30 + 0.7 * weight + 0.05 * height + rng.normal(0, 2, n)
        return {"gender": gender, "height": height, "weight": weight, "waist": waist,
                "chest": 50 + 0.8 * weight, "hip": 40 + 0.9 * weight}

    train, test = make(1200), make(300)
    rows = evaluate(train, test, 0.0, 0.0, rng)
    mae = {(r["gender"], r["method"], r["target"]): r["mae_cm"] for r in rows}
    for g in ("male", "female"):
        # weight carries almost all the signal in this toy world
        assert mae[(g, "height+weight", "waist")] < 2.0, mae[(g, "height+weight", "waist")]
        assert mae[(g, "height", "waist")] > 3 * mae[(g, "height+weight", "waist")]
        assert mae[(g, "gender mean", "waist")] >= mae[(g, "height", "waist")] - 0.5
        # noiseless chest is an exact linear function of weight
        assert mae[(g, "height+weight", "chest")] < 1e-6
    # weight noise hurts: 5 kg of error must make height+weight clearly worse
    noisy = {(r["gender"], r["method"], r["target"]): r["mae_cm"]
             for r in evaluate(train, test, 5.0, 1.5, rng)}
    assert noisy[("male", "height+weight", "waist")] > 1.5 * mae[("male", "height+weight", "waist")]
    # quadratic runs and does not break a linear world
    assert mae[("male", "height+weight (quadratic)", "waist")] < 2.2

    # loader: writes a fake BodyM-style folder and reads it back
    import csv as _csv
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "train"
        sp.mkdir()
        with open(sp / "measurements.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["subject_id", "chest", "waist", "hip", "height"])
            for i in range(60):
                w.writerow([f"s{i}", 90 + i % 7, 80 + i % 5, 95 + i % 3, 170])
            w.writerow(["bad", "", "", "", ""])
        with open(sp / "hwg_metadata.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["subject_id", "gender", "height_cm", "weight_kg"])
            for i in range(60):
                w.writerow([f"s{i}", "male" if i % 2 else "female", 170 + i % 9, 60 + i % 11])
            w.writerow(["bad", "male", "", ""])
        loaded = load_split(Path(d), "train")
        assert len(loaded["height"]) == 60 and set(loaded["gender"]) == {"male", "female"}
    print_table("self-test", rows)
    print("\n[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
