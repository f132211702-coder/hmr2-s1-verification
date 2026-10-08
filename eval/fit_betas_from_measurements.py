#!/usr/bin/env python3
"""Recover SMPL betas from user-supplied body measurements, and test whether
that beats trusting an image model's predicted betas.

Why: eval/measure_body_error.py showed that HMR2.0b / CameraHMR / TokenHMR
betas are no better than "everyone has the average body" for chest, waist
and hip circumference (the quantities the garment-fit stage needs), and
that predicted bodies come out systematically too slim. If the circumference
can't come from the photo, ask the user for it: height (the plan already
requires it) plus, optionally, chest / waist / hip tape measurements.

Method: sample many random betas (SMPL's shape space is whitened, so
betas ~ N(0, 1)), measure each resulting body with the same SMPL-Anthropometry
tool, and fit measurements -> betas by (ridge) regression. With a Gaussian
prior this regression IS the posterior-mean estimate of betas given the
measurements: 4 numbers cannot pin down 10 betas, and the dimensions the
measurements don't constrain fall back to the average (0) instead of being
guessed. Linear and quadratic variants are compared on held-out samples
(vertex error between the recovered and the true body) and the better one is
kept per input set.

Evaluation (3DPW test, its 5 real men): the "tape measurements" fed in are
SMPL-Anthropometry's measurements of each subject's ground-truth body (male
SMPL + GT betas), optionally perturbed by Gaussian noise (--noise-cm; people
measure themselves with ~1-2 cm error). The recovered body is scored against
the ground-truth body on:
  - the five measurement errors in cm (height, shoulder breadth, chest, waist, hip),
  - `tpose_pve_mm`: mean per-vertex distance between the two T-pose meshes
    after aligning only their centroids -- no scale, no rotation, so unlike
    Procrustes PVE it penalises an overall too-small/too-large body.
Image-model rows (mean-shape baseline, HMR2.0b, CameraHMR, TokenHMR) are
scored identically from their predicted betas (rendered with the NEUTRAL
model, which is what those models output). Every method is averaged per
subject first and then across subjects, so each of the 5 men weighs the
same whatever the number of frames/noise draws.

Two body models for the measurement-based fit:
  NEUTRAL: matches what the image models output / what downstream code receives.
  MALE:    gender-aware; the recovered betas live in the male shape space, the
           same space the 3DPW ground truth is in (valid because the user's
           gender is known). Both are reported.

Caveats: 5 subjects, all male, all measured by the same tool that defines the
regression targets (so the measurement round-trip is optimistic by
construction; tpose_pve_mm is the less circular number). This does NOT test
how accurate real people's tape measurements are.

Status: --self-test checks the regression, sampling, vertex-error and noise
bookkeeping against a toy measurement function. Not yet run with the real
SMPL-Anthropometry measurements.

Usage (self-test, numpy only):
    python eval/fit_betas_from_measurements.py --self-test

Usage (real run; same env as measure_body_error.py, e.g. `camerahmr`):
    python eval/fit_betas_from_measurements.py \\
        --anthro_root ~/SMPL-Anthropometry \\
        --model HMR2.0b=results/eval_3dpw_s20.csv \\
        --model CameraHMR=~/workspace/dresson/CameraHMR/results/camerahmr_eval.csv \\
        --model TokenHMR=~/workspace/dresson/TokenHMR/results/tokenhmr_eval.csv \\
        --out results/betas_from_measurements.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_body_error import (BASELINE, N_BETAS, align_models, build_measurer,  # noqa: E402
                                load_model_csv)

ALL_NAMES = ["height", "shoulder breadth", "chest circumference",
             "waist circumference", "hip circumference"]
INPUT_SETS = {
    "height": ["height"],
    "height+chest/waist/hip": ["height", "chest circumference", "waist circumference",
                               "hip circumference"],
    "height+chest/waist/hip+shoulder": ALL_NAMES,
}


def sample_betas(n: int, rng: np.random.Generator, clip: float = 3.0) -> np.ndarray:
    """Random bodies from SMPL's (whitened) shape prior, clipped to +-clip so no
    near-impossible body ends up in the training set."""
    return np.clip(rng.standard_normal((n, N_BETAS)), -clip, clip)


def poly_features(Z: np.ndarray, degree: int) -> np.ndarray:
    """Z (N, d) standardized inputs -> [Z, all pairwise products Z_i*Z_j (i<=j)] for degree 2."""
    if degree == 1:
        return Z
    d = Z.shape[1]
    quad = [Z[:, i] * Z[:, j] for i in range(d) for j in range(i, d)]
    return np.hstack([Z, np.stack(quad, axis=1)])


class MeasurementsToBetas:
    """Ridge regression measurements (cm) -> betas, inputs standardized, intercept
    handled by centering. degree 1 = linear (exactly the Gaussian posterior mean if
    the measurements were linear in betas), degree 2 adds pairwise products."""

    def __init__(self, names: list[str], degree: int = 1, alpha: float = 1e-3):
        self.names, self.degree, self.alpha = names, degree, alpha

    def _phi(self, M: np.ndarray) -> np.ndarray:
        return poly_features((M - self.mu_m) / self.sd_m, self.degree)

    def fit(self, M: np.ndarray, B: np.ndarray) -> "MeasurementsToBetas":
        self.mu_m, self.sd_m = M.mean(axis=0), M.std(axis=0) + 1e-9
        phi = self._phi(M)
        self.mu_phi, self.mu_b = phi.mean(axis=0), B.mean(axis=0)
        pc, bc = phi - self.mu_phi, B - self.mu_b
        self.W = np.linalg.solve(pc.T @ pc + self.alpha * np.eye(pc.shape[1]), pc.T @ bc)
        return self

    def predict(self, M: np.ndarray) -> np.ndarray:
        return (self._phi(np.atleast_2d(M)) - self.mu_phi) @ self.W + self.mu_b


def tpose_pve_mm(v_pred: np.ndarray, v_gt: np.ndarray) -> float:
    """Mean per-vertex distance (mm) between two same-topology T-pose meshes (meters)
    after subtracting each mesh's centroid. No scale/rotation alignment: an overall
    size error counts."""
    d = (v_pred - v_pred.mean(axis=0)) - (v_gt - v_gt.mean(axis=0))
    return float(np.linalg.norm(d, axis=1).mean() * 1000)


def matrix(measure_dicts: list[dict], names: list[str]) -> np.ndarray:
    return np.array([[m[n] for n in names] for m in measure_dicts])


def noisy(m: np.ndarray, noise_cm: float, rng: np.random.Generator) -> np.ndarray:
    return m + rng.normal(0.0, noise_cm, size=m.shape) if noise_cm > 0 else m


def subject_key(betas: np.ndarray) -> tuple:
    return tuple(np.round(betas, 5))


def macro_average(rows: list[dict], metric_keys: list[str]) -> dict:
    """Average per subject first, then across subjects (equal weight per subject)."""
    subjects = sorted({r["subject"] for r in rows})
    return {k: float(np.mean([np.mean([r[k] for r in rows if r["subject"] == s])
                              for s in subjects])) for k in metric_keys}


def score_body(measure, vertices, betas, gender, gt_m, gt_v, names=ALL_NAMES,
               calibrate: bool = False) -> dict:
    """Errors of one candidate body against one ground-truth body.
    calibrate=True simulates 'the user told us their real height': the candidate
    body is uniformly scaled (about its centroid) so its height equals the real
    one, and all its measurements scale with it. Height error is then 0 by
    construction."""
    m = measure(betas, gender)
    v = vertices(betas, gender)
    if calibrate:
        s = gt_m["height"] / m["height"]
        m = {n: x * s for n, x in m.items()}
        c = v.mean(axis=0)
        v = (v - c) * s + c
    out = {f"err {n}": abs(m[n] - gt_m[n]) for n in names}
    out["tpose_pve_mm"] = tpose_pve_mm(v, gt_v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anthro_root", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CSV",
                    help="image-model eval CSV with beta columns (for the comparison rows and "
                         "for the 3DPW ground-truth subjects)")
    ap.add_argument("--n-samples", type=int, default=4000, help="random bodies for fitting")
    ap.add_argument("--n-val", type=int, default=300, help="held-out bodies for picking linear vs quadratic")
    ap.add_argument("--limit", type=int, default=300,
                    help="image-model records (evenly subsampled) used for their comparison rows")
    ap.add_argument("--noise-cm", type=float, default=1.5, help="std of simulated tape-measure error")
    ap.add_argument("--noise-draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--photo-height-only", action="store_true",
                    help="only the 'one photo + the user's height' comparison: the image-model rows "
                         "(raw and rescaled to the real height) next to the height-only regression "
                         "(NEUTRAL). Skips the tape-measure input sets and the MALE fit (much faster).")
    ap.add_argument("--out", default="results/betas_from_measurements.csv")
    args = ap.parse_args()

    specs = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model expects NAME=CSV, got {spec!r}")
        specs[name] = Path(path).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    anthro_root = Path(args.anthro_root).expanduser().resolve()
    for g in ("NEUTRAL", "MALE"):
        if not (anthro_root / "data" / "smpl" / f"SMPL_{g}.pkl").exists():
            raise SystemExit(f"SMPL_{g}.pkl missing in {anthro_root}/data/smpl/")

    per_model = {n: load_model_csv(p) for n, p in specs.items()}
    keys, gt_betas, pred_betas = align_models(per_model)
    if len(keys) > args.limit:
        keys = [keys[i] for i in np.linspace(0, len(keys) - 1, args.limit).astype(int)]
    subjects = {}
    for k in keys:
        subjects.setdefault(subject_key(gt_betas[k]), gt_betas[k])
    print(f"{len(keys)} records, {len(subjects)} distinct ground-truth subject(s)")

    measure, vertices = build_measurer(anthro_root, ALL_NAMES)
    rng = np.random.default_rng(args.seed)

    gt_body = {}
    for sk, b in subjects.items():
        gt_body[sk] = (measure(b, "MALE"), vertices(b, "MALE"))
    print("GT subjects (male SMPL), cm:")
    for i, (sk, (m, _)) in enumerate(gt_body.items()):
        print(f"  subject {i}: " + "  ".join(f"{n.split()[0]} {m[n]:.1f}" for n in ALL_NAMES))

    metric_keys = [f"err {n}" for n in ALL_NAMES] + ["tpose_pve_mm"]
    results: dict[str, dict] = {}

    # ---- image-based rows -------------------------------------------------
    def image_rows(label, betas_of_key):
        # Each image-based body is scored twice: as predicted, and "+ height" (rescaled to
        # the real height -- the 'one photo plus the user's height' setting).
        for calibrate, suffix in ((False, ""), (True, " + height")):
            rows = []
            for k in keys:
                sk = subject_key(gt_betas[k])
                gt_m, gt_v = gt_body[sk]
                r = score_body(measure, vertices, betas_of_key(k), "NEUTRAL", gt_m, gt_v,
                               calibrate=calibrate)
                r["subject"] = sk
                rows.append(r)
            results[label + suffix] = macro_average(rows, metric_keys)
            print(f"  scored {label + suffix}", flush=True)

    image_rows(BASELINE, lambda k: np.zeros(N_BETAS))
    for name, by_key in pred_betas.items():
        image_rows(name, lambda k, by_key=by_key: by_key[k])

    # ---- measurement-based rows -------------------------------------------
    genders = ("NEUTRAL",) if args.photo_height_only else ("NEUTRAL", "MALE")
    input_sets = {"height": INPUT_SETS["height"]} if args.photo_height_only else INPUT_SETS
    for gender in genders:
        print(f"fitting on {args.n_samples} random {gender} bodies ...", flush=True)
        B = sample_betas(args.n_samples + args.n_val, rng)
        meas = []
        for i, b in enumerate(B):
            meas.append(measure(b, gender))
            if (i + 1) % 500 == 0:
                print(f"  measured {i + 1}/{len(B)}", flush=True)
        n_fit = args.n_samples
        for set_name, names in input_sets.items():
            X = matrix(meas, names)
            best = None
            for degree in (1, 2):
                reg = MeasurementsToBetas(names, degree).fit(X[:n_fit], B[:n_fit])
                pred = reg.predict(X[n_fit:])
                val = np.mean([tpose_pve_mm(vertices(pred[i], gender), vertices(B[n_fit + i], gender))
                               for i in range(args.n_val)])
                print(f"  [{gender}] {set_name:34s} degree {degree}: held-out tpose_pve {val:6.2f} mm")
                if best is None or val < best[0]:
                    best = (val, degree, reg)
            _, degree, reg = best
            for noise in (0.0, args.noise_cm):
                rows = []
                for sk, (gt_m, gt_v) in gt_body.items():
                    x0 = matrix([gt_m], names)
                    for _ in range(1 if noise == 0 else args.noise_draws):
                        b = reg.predict(noisy(x0, noise, rng))[0]
                        r = score_body(measure, vertices, b, gender, gt_m, gt_v)
                        r["subject"] = sk
                        rows.append(r)
                label = f"[{gender}] {set_name} (deg {degree}, noise {noise:g}cm)"
                results[label] = macro_average(rows, metric_keys)
                print(f"  scored {label}", flush=True)

    # ---- report -----------------------------------------------------------
    print("\nMean abs error vs the real body, averaged over subjects (cm; last column mm):")
    head = f"{'method':58s}" + "".join(f"{n.split()[0][:8]:>9s}" for n in ALL_NAMES) + f"{'tpose_pve':>11s}"
    print(head)
    for label, r in results.items():
        print(f"{label:58s}" + "".join(f"{r[f'err {n}']:9.2f}" for n in ALL_NAMES)
              + f"{r['tpose_pve_mm']:11.1f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method"] + metric_keys)
        for label, r in results.items():
            w.writerow([label] + [r[k] for k in metric_keys])
    print(f"wrote {out_path}")


def self_test() -> None:
    """No SMPL needed. Toy world: 5 measurements, each a fixed linear function of the
    first 5 betas (the other 5 betas affect nothing measurable)."""
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (5, 5)) + 3 * np.eye(5)
    toy_names = ALL_NAMES

    def toy_measure(b, gender):
        v = 100 + A @ b[:5]
        return dict(zip(toy_names, v))

    def toy_vertices(b, gender):
        return np.outer(np.arange(1, 7), b[:3])

    # sampling: shape, clipping, determinism
    B = sample_betas(1000, np.random.default_rng(1))
    assert B.shape == (1000, N_BETAS) and np.abs(B).max() <= 3.0
    assert np.array_equal(sample_betas(5, np.random.default_rng(7)), sample_betas(5, np.random.default_rng(7)))

    # regression: with all 5 measurable dims given, the measurable betas are recovered
    # almost exactly and the unmeasurable ones fall back to ~0 (the prior mean).
    M = matrix([toy_measure(b, "X") for b in B], toy_names)
    reg = MeasurementsToBetas(toy_names, degree=1, alpha=1e-6).fit(M[:800], B[:800])
    P = reg.predict(M[800:])
    assert np.abs(P[:, :5] - B[800:, :5]).max() < 1e-3, "measurable betas should be recovered"
    assert np.abs(P[:, 5:]).max() < 0.5, "unmeasurable betas should stay near the prior mean"
    # a single measurement can't recover everything: error must be larger than with all five
    reg1 = MeasurementsToBetas(["height"], degree=1).fit(M[:800, :1], B[:800])
    assert (np.abs(reg1.predict(M[800:, :1]) - B[800:])[:, :5].mean()
            > np.abs(P - B[800:])[:, :5].mean() * 10)
    # quadratic features run and don't break a linear world
    reg2 = MeasurementsToBetas(toy_names, degree=2, alpha=1e-6).fit(M[:800], B[:800])
    assert np.abs(reg2.predict(M[800:])[:, :5] - B[800:, :5]).max() < 1e-2

    # tpose_pve_mm: zero for identical meshes, translation-invariant, scale-sensitive
    v = toy_vertices(np.array([1.0, 2.0, 3.0] + [0] * 7), "X")
    assert tpose_pve_mm(v, v) == 0.0
    assert abs(tpose_pve_mm(v + 5.0, v)) < 1e-9, "centroid alignment must remove translation"
    assert tpose_pve_mm(v * 1.1, v) > 0, "overall size error must be penalised"

    # noise: reproducible, zero-noise is the identity
    x = np.array([[170.0, 100.0]])
    assert np.array_equal(noisy(x, 0.0, rng), x)
    assert np.array_equal(noisy(x, 1.5, np.random.default_rng(3)), noisy(x, 1.5, np.random.default_rng(3)))

    # score_body + macro_average: subjects weigh equally whatever their row counts
    gt_b = np.array([1.0, 0, 0, 0, 0] + [0] * 5)
    gt_m, gt_v = toy_measure(gt_b, "X"), toy_vertices(gt_b, "X")
    r_good = score_body(toy_measure, toy_vertices, gt_b, "X", gt_m, gt_v)
    assert max(r_good.values()) < 1e-9
    rows = [dict(subject="a", e=1.0)] * 9 + [dict(subject="b", e=3.0)]
    assert macro_average(rows, ["e"])["e"] == 2.0

    # calibrate (known real height): candidate 20% too small -> height error becomes 0, other
    # measurements scale up by the same factor, vertices scale about their centroid.
    def small_measure(b, g):
        return {n: 0.8 * x for n, x in toy_measure(gt_b, g).items()}
    def small_vertices(b, g):
        return 0.8 * toy_vertices(gt_b, g)
    raw = score_body(small_measure, small_vertices, gt_b, "X", gt_m, gt_v)
    cal = score_body(small_measure, small_vertices, gt_b, "X", gt_m, gt_v, calibrate=True)
    assert raw["err height"] > 10 and cal["err height"] < 1e-9
    assert max(cal.values()) < 1e-9, "a purely too-small body is exactly fixed by height calibration"
    assert raw["tpose_pve_mm"] > cal["tpose_pve_mm"]

    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
