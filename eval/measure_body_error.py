#!/usr/bin/env python3
"""Body measurement error: turn each model's predicted betas into tape-measure
style body measurements (height, chest/waist/hip circumference, shoulder
breadth) and compare them against the same measurements of the 3DPW
ground-truth body.

Why this exists: the team's pipeline plan (3D-VTO-Pipeline-規劃-v3) says S0's
real job is garment-fit analysis, where body *circumferences* matter more
than MPJPE/PVE, and lists "body measurement error" as a required S0 metric
that nothing in eval/ computed yet. PA-MPJPE/PVE can look fine while a
waist circumference is several cm off, because they average over the whole
body.

How measurements are computed: with DavidBoja/SMPL-Anthropometry (MIT),
which defines each measurement on fixed SMPL landmarks -- lengths are
landmark-to-landmark distances, circumferences are the convex-hull
perimeter of a plane cut through the body. It measures the body in a neutral
T-pose from betas alone, which is exactly what we want here: this compares
SHAPE only and ignores whatever pose the model predicted. All values in cm.

What is compared (and why):
  - GT side: 3DPW's ground-truth betas through the GENDERED SMPL model
    (--gt-gender, default MALE: the whole 3DPW test split is male, verified
    in the project report) -- that is the best available estimate of the
    real person's body.
  - Prediction side: each model's predicted betas through the NEUTRAL SMPL
    model, because HMR2/CameraHMR/TokenHMR only output neutral-SMPL betas --
    this is the body the pipeline would hand to downstream garment fitting.
  - "mean_shape_baseline": betas = 0 (the average neutral body) for everyone.
    A model whose error isn't clearly below this baseline isn't adding
    per-person shape information, whatever its other metrics say -- this is
    the measurement-space version of the shape-collapse question in this
    project's report.
  - Each measurement is reported raw AND "height-calibrated": predicted
    values rescaled by (gt_height / predicted_height), simulating the plan's
    v1 decision to require the target's real height as a metric-scale
    anchor. (Height itself is calibrated to zero error by construction, so
    it is reported raw only.)
  All models are scored on exactly the same (image_id, person_id) records;
  a record any model/GT measurement fails on is dropped for all and the
  failures are listed at the end (not silently skipped).

Inputs: the per-record CSVs from eval_against_gt.py / eval_camerahmr_against_gt.py
/ eval_tokenhmr_against_gt.py, which now include pred_beta_0..9 and
gt_beta_0..9 columns (re-run them if yours predate that change). Run all
three with the same --stride so they cover the same frames.

Setup (once): git clone https://github.com/DavidBoja/SMPL-Anthropometry.git,
copy SMPL_NEUTRAL.pkl and SMPL_MALE.pkl into its data/smpl/ folder, and
`pip install plotly trimesh scikit-learn` (measure.py imports its
visualizer, hence plotly, even if nothing is drawn).

Status: --self-test validates the bookkeeping (CSV loading, record
alignment, height calibration, error aggregation) against a toy measurement
function. Not yet run with the real SMPL-Anthropometry measurements.

Usage (self-test, needs only numpy):
    python eval/measure_body_error.py --self-test

Usage (real run; HMR2 env or any env with torch+smplx+trimesh+scipy+plotly):
    python eval/measure_body_error.py \\
        --anthro_root ~/SMPL-Anthropometry \\
        --model HMR2.0b=results/eval_3dpw_s20.csv \\
        --model CameraHMR=~/workspace/dresson/CameraHMR/results/camerahmr_eval.csv \\
        --model TokenHMR=~/workspace/dresson/TokenHMR/results/tokenhmr_eval.csv \\
        --out results/body_measurements.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

DEFAULT_MEASUREMENTS = ["height", "shoulder breadth", "chest circumference",
                        "waist circumference", "hip circumference"]
BASELINE = "mean_shape_baseline"
N_BETAS = 10


def load_model_csv(path: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """(image_id, person_id) -> (pred_betas, gt_betas) for rows with status ok."""
    pred_cols = [f"pred_beta_{i}" for i in range(N_BETAS)]
    gt_cols = [f"gt_beta_{i}" for i in range(N_BETAS)]
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in pred_cols + gt_cols if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path} has no {missing[0]} column (and {len(missing) - 1} more): "
                             f"it predates the change that saves full beta vectors -- re-run "
                             f"the eval script that produced it.")
        for row in reader:
            if row.get("status") != "ok":
                continue
            out[(row["image_id"], row["person_id"])] = (
                np.array([float(row[c]) for c in pred_cols]),
                np.array([float(row[c]) for c in gt_cols]),
            )
    return out


def align_models(per_model: dict[str, dict]) -> tuple[list, dict, dict]:
    """Keep only records present in every model's CSV. Returns (sorted keys,
    gt_betas by key, {model: {key: pred_betas}}). Also checks every CSV
    carries the same GT betas for a key -- different values would mean the
    CSVs came from different GT sources/matching, and comparing them
    would be meaningless."""
    common = set.intersection(*(set(d) for d in per_model.values()))
    keys = sorted(common)
    first = next(iter(per_model.values()))
    gt = {k: first[k][1] for k in keys}
    for name, d in per_model.items():
        for k in keys:
            if not np.allclose(d[k][1], gt[k], atol=1e-5):
                raise SystemExit(f"GT betas for {k} differ between CSVs (at {name}); "
                                 f"the runs don't share a GT source.")
    return keys, gt, {name: {k: d[k][0] for k in keys} for name, d in per_model.items()}


def height_calibrate(pred_m: dict, gt_height: float) -> dict:
    """Rescale every predicted measurement by gt_height / predicted height --
    the plan's 'known target height as metric scale anchor'. (Linear
    measurements scale linearly with body size, so one factor covers both
    lengths and circumferences.)"""
    s = gt_height / pred_m["height"]
    return {name: v * s for name, v in pred_m.items()}


def compute_rows(keys, gt_betas, pred_betas_by_model, measure_gt, measure_pred, names):
    """Long-format rows: one per (record, model, measurement). measure_gt /
    measure_pred: betas -> {measurement name: cm}. Returns (rows, failures)."""
    base = measure_pred(np.zeros(N_BETAS))
    gt_cache: dict = {}
    rows, failures = [], []
    for n, key in enumerate(keys):
        try:
            gk = tuple(np.round(gt_betas[key], 6))
            if gk not in gt_cache:
                gt_cache[gk] = measure_gt(gt_betas[key])
            gt_m = gt_cache[gk]
            preds = {m: measure_pred(by_key[key]) for m, by_key in pred_betas_by_model.items()}
        except Exception as e:  # noqa: BLE001 -- reported below, never silent
            failures.append((key, repr(e)))
            continue
        preds[BASELINE] = base
        for model, pm in preds.items():
            cal = height_calibrate(pm, gt_m["height"])
            for name in names:
                rows.append({"image_id": key[0], "person_id": key[1], "model": model,
                             "measurement": name, "pred_cm": pm[name], "gt_cm": gt_m[name],
                             "pred_cm_calibrated": cal[name]})
        if (n + 1) % 200 == 0:
            print(f"  measured {n + 1}/{len(keys)} records", flush=True)
    return rows, failures


def summarize(rows: list[dict], names: list[str], failures: list) -> None:
    seen_models = list(dict.fromkeys(r["model"] for r in rows))
    models = [BASELINE] + [m for m in seen_models if m != BASELINE]
    n_records = len({(r["image_id"], r["person_id"]) for r in rows})
    print(f"\n{n_records} record(s) scored on identical samples for every model; "
          f"{len(failures)} dropped for measurement failures.")
    for key, err in failures[:5]:
        print(f"  dropped {key}: {err}")

    print("\nGT body measurements (cm) over the scored records -- plausibility check "
          "(adult male heights/waists should look like real people):")
    gt_by_name: dict[str, list[float]] = {n: [] for n in names}
    for r in rows:
        if r["model"] == BASELINE:
            gt_by_name[r["measurement"]].append(r["gt_cm"])
    for n in names:
        v = np.array(gt_by_name[n])
        print(f"  {n:22s} min {v.min():6.1f}  mean {v.mean():6.1f}  max {v.max():6.1f}")

    def stats(model, name, col):
        sel = [r for r in rows if r["model"] == model and r["measurement"] == name]
        pred = np.array([r[col] for r in sel])
        gt = np.array([r["gt_cm"] for r in sel])
        err = pred - gt
        corr = (float(np.corrcoef(pred, gt)[0, 1])
                if len(sel) > 2 and pred.std() > 0 and gt.std() > 0 else float("nan"))
        return np.abs(err).mean(), np.median(np.abs(err)), err.mean(), err.std(), corr

    for name in names:
        print(f"\n=== {name} (cm) ===")
        print(f"{'model':24s}{'MAE':>8s}{'median':>9s}{'bias':>8s}{'err std':>9s}{'corr':>7s}"
              f"{'| MAE, height-calibrated':>26s}")
        for m in models:
            mae, med, bias, std, corr = stats(m, name, "pred_cm")
            corr_s = "n/a" if np.isnan(corr) else f"{corr:+.2f}"
            cal = "n/a (height)" if name == "height" else f"{stats(m, name, 'pred_cm_calibrated')[0]:.2f}"
            print(f"{m:24s}{mae:8.2f}{med:9.2f}{bias:+8.2f}{std:9.2f}{corr_s:>7s}{cal:>26s}")
        gt_std = np.std(gt_by_name[name])
        print(f"  (spread of the real measurement across these records: std {gt_std:.2f} cm -- "
              f"a model that tracks people should show 'err std' below this, and a clearly "
              f"positive 'corr')")
    print("\nbias = mean(pred - gt): negative = the model's body is smaller than the real one. "
          "err std = spread of the error with the bias removed. corr = correlation of predicted "
          "vs real values across records (n/a for the constant mean-shape baseline). MAE alone is "
          "dominated by bias; per-person shape information shows up as err std < real spread "
          f"and positive corr, and MAE below '{BASELINE}'.")


def write_rows(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anthro_root", required=True,
                    help="clone of DavidBoja/SMPL-Anthropometry (its data/ path is relative)")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CSV",
                    help="repeatable; per-record eval CSV that includes the beta columns")
    ap.add_argument("--gt-gender", default="MALE", choices=["MALE", "FEMALE", "NEUTRAL"])
    ap.add_argument("--measurements", nargs="+", default=DEFAULT_MEASUREMENTS)
    ap.add_argument("--limit", type=int, default=None,
                    help="evenly subsample to at most this many records (speed)")
    ap.add_argument("--out", default="results/body_measurements.csv")
    args = ap.parse_args()

    # Resolve user-given paths BEFORE chdir: SMPL-Anthropometry loads models from
    # a path relative to its own root.
    specs = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model expects NAME=CSV, got {spec!r}")
        specs[name] = Path(path).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    anthro_root = Path(args.anthro_root).expanduser().resolve()
    for g in ("NEUTRAL", args.gt_gender):
        pkl = anthro_root / "data" / "smpl" / f"SMPL_{g}.pkl"
        if not pkl.exists():
            raise SystemExit(f"{pkl} missing -- copy it there (e.g. from "
                             f"~/.cache/4DHumans/data/smpl/).")

    per_model = {name: load_model_csv(p) for name, p in specs.items()}
    keys, gt_betas, pred_betas = align_models(per_model)
    print(f"{len(keys)} record(s) common to all {len(specs)} model CSV(s)")
    if args.limit and len(keys) > args.limit:
        keys = [keys[i] for i in np.linspace(0, len(keys) - 1, args.limit).astype(int)]
        print(f"subsampled to {len(keys)} record(s)")

    sys.path.insert(0, str(anthro_root))
    os.chdir(anthro_root)
    import torch
    from measure import MeasureBody, create_model, set_shape  # noqa: E402

    names = list(dict.fromkeys(["height"] + list(args.measurements)))  # height needed for calibration
    measurer = MeasureBody("smpl")
    models: dict = {}

    def measure(betas: np.ndarray, gender: str) -> dict:
        # Same steps as MeasureSMPL.from_body_model, but reusing one smplx model per gender
        # instead of re-reading the .pkl for every record.
        if gender not in models:
            models[gender] = create_model(model_type="smpl", model_root="data", gender=gender,
                                          num_betas=N_BETAS, num_thetas=measurer.num_joints)
        with torch.no_grad():
            out = set_shape(models[gender], torch.tensor(betas, dtype=torch.float32)[None])
        measurer.verts = out.vertices.detach().cpu().numpy().squeeze()
        measurer.joints = out.joints.squeeze().detach().cpu().numpy()
        measurer.gender = gender
        measurer.measurements = {}
        measurer.measure(names)
        return dict(measurer.measurements)

    rows, failures = compute_rows(keys, gt_betas, pred_betas,
                                  measure_gt=lambda b: measure(b, args.gt_gender),
                                  measure_pred=lambda b: measure(b, "NEUTRAL"),
                                  names=names)
    if not rows:
        raise SystemExit("no record could be measured; first failures: " + "; ".join(
            f"{k}: {e}" for k, e in failures[:3]))
    write_rows(rows, out_path)
    summarize(rows, names, failures)


def self_test() -> None:
    """No SMPL / SMPL-Anthropometry needed: toy measurement function in which
    height = 170 + 10*beta0 and waist = 80 + 5*beta1 (cm)."""
    def toy(b):
        return {"height": 170 + 10 * b[0], "waist circumference": 80 + 5 * b[1]}

    names = ["height", "waist circumference"]

    # height_calibrate: waist scales by gt_height / pred_height, height lands on gt.
    cal = height_calibrate({"height": 170.0, "waist circumference": 85.0}, 180.0)
    assert abs(cal["height"] - 180.0) < 1e-9 and abs(cal["waist circumference"] - 90.0) < 1e-9

    # CSV round trip + alignment: two models, one record (b) only in model A.
    def write(path, recs):
        cols = ["image_id", "person_id", "status"] + [f"pred_beta_{i}" for i in range(N_BETAS)] \
            + [f"gt_beta_{i}" for i in range(N_BETAS)]
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for img, pred, gt in recs:
                w.writerow([img, "0", "ok", *pred, *gt])
            w.writerow(["bad", "0", "no_image"] + [""] * (2 * N_BETAS))

    gt = np.zeros(N_BETAS)
    gt[0], gt[1] = 1.0, -1.0
    pa, pb = gt.copy(), gt.copy()
    pa[0], pb[0] = 1.5, 0.5
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a.csv", Path(d) / "b.csv"
        write(a, [("r1", pa, gt), ("r2", pa, gt)])
        write(b, [("r1", pb, gt)])
        per_model = {"A": load_model_csv(a), "B": load_model_csv(b)}
    keys, gts, preds = align_models(per_model)
    assert keys == [("r1", "0")], f"expected only the shared record, got {keys}"

    rows, failures = compute_rows(keys, gts, preds, toy, toy, names)
    assert not failures
    got = {(r["model"], r["measurement"]): r for r in rows}
    # GT height 180, waist 75. A: height 185 (+5); B: 175 (-5); baseline: 170 (-10), waist 80 (+5)
    assert abs(got[("A", "height")]["pred_cm"] - 185) < 1e-9
    assert abs(got[("B", "height")]["pred_cm"] - 175) < 1e-9
    assert abs(got[(BASELINE, "height")]["pred_cm"] - 170) < 1e-9
    assert abs(got[(BASELINE, "waist circumference")]["pred_cm"] - 80) < 1e-9
    # calibration: A's waist is 80 - 5*(-1)... (gt b1=-1, A's b1=-1 too) = 75, scaled 180/185
    assert abs(got[("A", "waist circumference")]["pred_cm_calibrated"] - 75 * 180 / 185) < 1e-9

    # failures are reported and the record dropped for every model
    def flaky(b):
        if b[0] > 1.2:
            raise ValueError("boom")
        return toy(b)
    rows2, failures2 = compute_rows(keys, gts, preds, toy, flaky, names)
    assert not rows2 and len(failures2) == 1, "a failing record must be dropped, not half-scored"

    summarize(rows, names, failures)
    print("\n[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
