#!/usr/bin/env python3
"""Recover SMPL betas from user-supplied inputs (height, weight, tape
measurements) and test whether that beats trusting an image model's betas.

Why: eval/measure_body_error.py showed that HMR2.0b / CameraHMR / TokenHMR
betas are no better than "everyone has the average body" for chest, waist
and hip circumference (the quantities the garment-fit stage needs), and
that predicted bodies come out systematically too slim. The project's input
is one photo plus the user's height; this script measures how much each
extra thing a user could be asked for (weight, or chest/waist/hip tape
measurements) would buy, and how the photo + height setting compares.

Method: sample many random betas (SMPL's shape space is whitened, so
betas ~ N(0, 1)), compute each body's inputs (SMPL-Anthropometry
measurements; weight = mesh volume x BODY_DENSITY), and fit inputs -> betas
by (ridge) regression. With a Gaussian prior this regression IS the
posterior-mean estimate of betas given the inputs: a few numbers cannot pin
down 10 betas, and the dimensions they don't constrain fall back to the
average (0) instead of being guessed. Linear and quadratic variants are
compared on held-out samples (vertex error between the recovered and the
true body) and the better one is kept per input set.

Evaluation (3DPW test, its 5 real men): the "user inputs" fed in are the
measurements/weight of each subject's ground-truth body (male SMPL + GT
betas), perturbed by Gaussian noise (--noise-cm for lengths, --weight-noise-kg
for weight; people measure/report with error). The recovered body is scored
against the ground-truth body on:
  - measurement errors in cm (height, shoulder breadth, chest, waist, hip)
    and weight error in kg,
  - `tpose_pve_mm`: mean per-vertex distance between the two T-pose meshes
    after aligning only their centroids -- no scale, no rotation, so an
    overall too-small/too-large body counts.
Image-model rows (mean-shape baseline, HMR2.0b, CameraHMR, TokenHMR) are
scored identically from their predicted betas (NEUTRAL model, what those
models output), both as predicted and "+ height" (rescaled uniformly to the
real height -- the one-photo-plus-height setting). Every method is averaged
per subject first and then across subjects, so each of the 5 men weighs the
same whatever the number of frames/noise draws.

Caveats: 5 subjects, all male. 3DPW has no real weights, so a subject's
"weight" is its ground-truth mesh volume x BODY_DENSITY, and the regression
targets come from the same body model -- this gives an upper bound on how
informative weight could be, not a test with real self-reported weights
(which are biased: see the 5 kg setting). The noise-free rows are circular;
read the noisy ones.

Status: --self-test checks the regression, volume/weight, vertex-error,
noise and calibration bookkeeping against toy data. Weight was added
2026-10; not yet run with the real SMPL-Anthropometry.

Usage (self-test, numpy only):
    python eval/fit_betas_from_measurements.py --self-test

Usage (real run; same env as measure_body_error.py, e.g. `camerahmr`):
    python eval/fit_betas_from_measurements.py --sets height height+weight --genders NEUTRAL \\
        --anthro_root ~/SMPL-Anthropometry \\
        --model HMR2.0b=results/eval_3dpw_s20.csv \\
        --model CameraHMR=~/workspace/dresson/CameraHMR/results/camerahmr_eval.csv \\
        --model TokenHMR=~/workspace/dresson/TokenHMR/results/tokenhmr_eval.csv \\
        --out results/weight_input.csv
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

TOOL_NAMES = ["height", "shoulder breadth", "chest circumference",
              "waist circumference", "hip circumference"]
ALL_NAMES = TOOL_NAMES + ["weight"]   # weight (kg) is added on top of the tool's measurements
BODY_DENSITY = 985.0                  # kg/m^3, the value body-shape-from-image work uses to turn mesh volume into weight
INPUT_SETS = {
    "height": ["height"],
    "height+weight": ["height", "weight"],
    "height+chest/waist/hip": ["height", "chest circumference", "waist circumference",
                               "hip circumference"],
    "height+chest/waist/hip+shoulder": TOOL_NAMES,
}


def mesh_volume_m3(verts: np.ndarray, faces: np.ndarray) -> float:
    """Volume of a closed triangle mesh: sum of signed tetrahedra volumes to the origin.
    abs() so a consistently inward-wound mesh still gives a positive volume."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return float(abs(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum()) / 6.0)


def body_weight_kg(verts: np.ndarray, faces: np.ndarray) -> float:
    return mesh_volume_m3(verts, faces) * BODY_DENSITY


def with_weight(measure, vertices, faces):
    """measure(betas, gender) -> the tool's measurements plus 'weight' in kg."""
    def measure_all(betas: np.ndarray, gender: str) -> dict:
        m = measure(betas, gender)
        m["weight"] = body_weight_kg(vertices(betas, gender), faces)
        return m
    return measure_all


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
    """Ridge regression inputs -> betas, inputs standardized, intercept handled by
    centering. degree 1 = linear (exactly the Gaussian posterior mean if the
    inputs were linear in betas), degree 2 adds pairwise products."""

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


def noise_std(names: list[str], noise_cm: float, noise_kg: float) -> np.ndarray:
    """Per-input noise level: kg for weight, cm for every length/circumference."""
    return np.array([noise_kg if n == "weight" else noise_cm for n in names])


def noisy(m: np.ndarray, std, rng: np.random.Generator) -> np.ndarray:
    std = np.asarray(std, dtype=float)
    return m + rng.normal(0.0, 1.0, size=m.shape) * std if std.any() else m


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
    one. Lengths scale by s, weight by s**3. Height error is then 0 by construction."""
    m = measure(betas, gender)
    v = vertices(betas, gender)
    if calibrate:
        s = gt_m["height"] / m["height"]
        m = {n: x * (s ** 3 if n == "weight" else s) for n, x in m.items()}
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
    ap.add_argument("--sets", nargs="+", default=list(INPUT_SETS), choices=list(INPUT_SETS),
                    help="which user-input sets to fit and score")
    ap.add_argument("--genders", nargs="+", default=["NEUTRAL", "MALE"], choices=["NEUTRAL", "MALE"],
                    help="body model the regression is fitted/rendered with")
    ap.add_argument("--n-samples", type=int, default=4000, help="random bodies for fitting")
    ap.add_argument("--n-val", type=int, default=300, help="held-out bodies for picking linear vs quadratic")
    ap.add_argument("--limit", type=int, default=300,
                    help="image-model records (evenly subsampled) used for their comparison rows")
    ap.add_argument("--noise-cm", type=float, default=1.5, help="std of simulated tape-measure/height error")
    ap.add_argument("--weight-noise-kg", nargs="+", type=float, default=[2.0, 5.0],
                    help="std(s) of the simulated weight error; the first is the main setting, "
                         "the others are extra sensitivity rows for sets that use weight")
    ap.add_argument("--noise-draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
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
    for g in {"NEUTRAL", "MALE"}:  # image rows use NEUTRAL, ground truth is MALE
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

    measure_tool, vertices, faces = build_measurer(anthro_root, TOOL_NAMES)
    measure = with_weight(measure_tool, vertices, faces)
    rng = np.random.default_rng(args.seed)

    gt_body = {}
    for sk, b in subjects.items():
        gt_body[sk] = (measure(b, "MALE"), vertices(b, "MALE"))
    print("GT subjects (male SMPL), cm / kg (weight = mesh volume x density):")
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

    # ---- user-input-based rows --------------------------------------------
    for gender in args.genders:
        print(f"fitting on {args.n_samples} random {gender} bodies ...", flush=True)
        B = sample_betas(args.n_samples + args.n_val, rng)
        meas = []
        for i, b in enumerate(B):
            meas.append(measure(b, gender))
            if (i + 1) % 500 == 0:
                print(f"  measured {i + 1}/{len(B)}", flush=True)
        n_fit = args.n_samples
        for set_name in args.sets:
            names = INPUT_SETS[set_name]
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
            uses_weight = "weight" in names
            levels = [(0.0, 0.0), (args.noise_cm, args.weight_noise_kg[0])]
            if uses_weight:
                levels += [(args.noise_cm, kg) for kg in args.weight_noise_kg[1:]]
            for cm, kg in levels:
                rows = []
                for sk, (gt_m, gt_v) in gt_body.items():
                    x0 = matrix([gt_m], names)
                    for _ in range(1 if (cm == 0 and kg == 0) else args.noise_draws):
                        b = reg.predict(noisy(x0, noise_std(names, cm, kg), rng))[0]
                        r = score_body(measure, vertices, b, gender, gt_m, gt_v)
                        r["subject"] = sk
                        rows.append(r)
                noise_label = f"{cm:g}cm" + (f"/{kg:g}kg" if uses_weight else "")
                label = f"[{gender}] {set_name} (deg {degree}, noise {noise_label})"
                results[label] = macro_average(rows, metric_keys)
                print(f"  scored {label}", flush=True)

    # ---- report -----------------------------------------------------------
    print("\nMean abs error vs the real body, averaged over subjects "
          "(lengths cm, weight kg; last column mm):")
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
    """No SMPL needed. Toy world: 6 inputs (the 5 tool measurements + weight), each a fixed
    linear function of the first 6 betas (the other 4 betas affect nothing observable)."""
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (6, 6)) + 3 * np.eye(6)
    toy_names = ALL_NAMES

    def toy_measure(b, gender):
        return dict(zip(toy_names, 100 + A @ b[:6]))

    def toy_vertices(b, gender):
        return np.outer(np.arange(1, 7), b[:3])

    # volume / weight: unit cube (12 triangles, outward) = 1 m^3 -> BODY_DENSITY kg; winding-agnostic
    cube_v = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], float)
    cube_f = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4],
                       [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
    assert abs(mesh_volume_m3(cube_v, cube_f) - 1.0) < 1e-9
    assert abs(mesh_volume_m3(cube_v, cube_f[:, ::-1]) - 1.0) < 1e-9
    assert abs(body_weight_kg(cube_v * 0.5, cube_f) - 0.125 * BODY_DENSITY) < 1e-9

    # sampling: shape, clipping, determinism
    B = sample_betas(1500, np.random.default_rng(1))
    assert B.shape == (1500, N_BETAS) and np.abs(B).max() <= 3.0
    assert np.array_equal(sample_betas(5, np.random.default_rng(7)), sample_betas(5, np.random.default_rng(7)))

    # regression: with all 6 observable dims given, those betas are recovered almost exactly
    # and the unobservable ones fall back to ~0 (the prior mean).
    M = matrix([toy_measure(b, "X") for b in B], toy_names)
    reg = MeasurementsToBetas(toy_names, degree=1, alpha=1e-6).fit(M[:1200], B[:1200])
    P = reg.predict(M[1200:])
    assert np.abs(P[:, :6] - B[1200:, :6]).max() < 1e-3, "observable betas should be recovered"
    assert np.abs(P[:, 6:]).max() < 0.5, "unobservable betas should stay near the prior mean"
    # fewer inputs recover less: height + weight only must be clearly worse than all six
    two = [0, 5]  # height, weight
    reg2 = MeasurementsToBetas([toy_names[i] for i in two], degree=1).fit(M[:1200, two], B[:1200])
    assert (np.abs(reg2.predict(M[1200:, two]) - B[1200:])[:, :6].mean()
            > np.abs(P - B[1200:])[:, :6].mean() * 10)
    # quadratic features run and don't break a linear world
    reg3 = MeasurementsToBetas(toy_names, degree=2, alpha=1e-6).fit(M[:1200], B[:1200])
    assert np.abs(reg3.predict(M[1200:])[:, :6] - B[1200:, :6]).max() < 1e-2

    # tpose_pve_mm: zero for identical meshes, translation-invariant, scale-sensitive
    v = toy_vertices(np.array([1.0, 2.0, 3.0] + [0] * 7), "X")
    assert tpose_pve_mm(v, v) == 0.0
    assert abs(tpose_pve_mm(v + 5.0, v)) < 1e-9, "centroid alignment must remove translation"
    assert tpose_pve_mm(v * 1.1, v) > 0, "overall size error must be penalised"

    # noise: per-input levels (kg vs cm), zero is the identity, seeded draws repeat
    names = ["height", "weight"]
    assert list(noise_std(names, 1.5, 2.0)) == [1.5, 2.0]
    x = np.array([[170.0, 70.0]])
    assert np.array_equal(noisy(x, noise_std(names, 0.0, 0.0), rng), x)
    d1 = noisy(x, noise_std(names, 1.5, 2.0), np.random.default_rng(3))
    d2 = noisy(x, noise_std(names, 1.5, 2.0), np.random.default_rng(3))
    assert np.array_equal(d1, d2) and not np.array_equal(d1, x)

    # score_body + macro_average: subjects weigh equally whatever their row counts
    gt_b = np.array([1.0, 0, 0, 0, 0, 0] + [0] * 4)
    gt_m, gt_v = toy_measure(gt_b, "X"), toy_vertices(gt_b, "X")
    r_good = score_body(toy_measure, toy_vertices, gt_b, "X", gt_m, gt_v)
    assert max(r_good.values()) < 1e-9
    rows = [dict(subject="a", e=1.0)] * 9 + [dict(subject="b", e=3.0)]
    assert macro_average(rows, ["e"])["e"] == 2.0

    # calibrate (known real height): a body 20% too small in every length (and 0.8^3 in weight)
    # is fixed exactly; vertices scale about their centroid.
    def small_measure(b, g):
        return {n: x * (0.8 ** 3 if n == "weight" else 0.8) for n, x in toy_measure(gt_b, g).items()}
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
