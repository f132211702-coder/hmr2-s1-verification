#!/usr/bin/env python3
"""Does a photo tell us anything about body SHAPE beyond the user's height?
Tested on SSP-3D (62 people, 21 women and 41 men, tight clothing, SMPL shape
pseudo-ground-truth + gender) instead of 3DPW's 5 men.

Scores the prediction CSVs from predict_ssp3d.py (HMR2.0b / CameraHMR /
TokenHMR betas) against the pseudo-GT body, in the product setting "one photo
+ the user's height":

  mean body            betas = 0 (the average body), neutral SMPL
  mean body + height   same, uniformly scaled to the real height
  mean body + height + gender
                       same, but the average body of the user's gender
  <model>              the model's betas in the neutral SMPL (what it outputs)
  <model> + height     ... scaled to the real height
  <model> + height + gender
                       ... betas rendered with the user's gendered SMPL
  height only (regression), height + weight (regression, 2 kg / 5 kg error)
                       betas recovered from height (and weight) alone, as in
                       fit_betas_from_measurements.py, one regressor per gender

A model adds shape information only if its "+ height" rows beat "mean body +
height" (the photo-free baseline), and, within one gender, its predicted
circumferences correlate positively with the real ones (corr columns).

Metrics (all vs the pseudo-GT body in the GT person's gendered T-pose):
  err <measurement>  mean abs error, cm (weight: kg = mesh volume x density)
  tpose_pve_mm       T-pose per-vertex error, centroid-aligned only (size counts)
  pve_t_sc_mm        SSP-3D's own PVE-T-SC (scale and translation removed), the
                     number the CameraHMR/SSP-3D papers report
Averaged per person first, then across people (each person weighs the same
whatever the number of photos), for all people and per gender.

Caveats: the GT is a pseudo-GT SMPL fit, not tape measurements, and SSP-3D
people are athletes in tight clothing (easiest case; street clothes are worse).
Weight is mesh volume x density -- circular for the regression rows, an
upper bound. The regressors are trained on betas ~ N(0,1) clipped at +-3
while a few SSP-3D shapes lie beyond that.

Usage (self-test, numpy only): python eval/eval_ssp3d_shape.py --self-test

Usage (needs torch+smplx+trimesh+scipy+plotly, e.g. env camerahmr; SMPL_NEUTRAL,
SMPL_MALE and SMPL_FEMALE .pkl in <anthro_root>/data/smpl/):
    python eval/eval_ssp3d_shape.py --anthro_root ~/SMPL-Anthropometry \\
        --model HMR2.0b=results/ssp3d_hmr2.csv \\
        --model CameraHMR=results/ssp3d_camerahmr.csv \\
        --model TokenHMR=results/ssp3d_tokenhmr.csv \\
        --out results/ssp3d_shape.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fit_betas_from_measurements import (ALL_NAMES, INPUT_SETS, MeasurementsToBetas,  # noqa: E402
                                         macro_average, matrix, noise_std, noisy,
                                         sample_betas, tpose_pve_mm, with_weight)
from measure_body_error import N_BETAS, build_measurer  # noqa: E402
from ssp3d_common import load_prediction_csv, pve_t_sc  # noqa: E402

CIRCUMFERENCES = ["chest circumference", "waist circumference", "hip circumference"]
METRIC_KEYS = [f"err {n}" for n in ALL_NAMES] + ["tpose_pve_mm", "pve_t_sc_mm"]
GROUPS = ["all", "MALE", "FEMALE"]
SHORT = {"height": "height", "shoulder breadth": "shoulder", "chest circumference": "chest",
         "waist circumference": "waist", "hip circumference": "hip", "weight": "weight"}


def candidate_scores(measure, vertices, betas, gender, gt_m, gt_v):
    """Score one candidate body against the GT body. Returns (raw, calibrated):
    calibrated = the candidate uniformly scaled about its centroid so its height
    equals the real one (the user typed their height); lengths scale by s,
    weight by s**3. Each dict holds `err <n>`, the signed `pred <n>`, the GT
    `gt <n>` (for correlations), tpose_pve_mm and pve_t_sc_mm."""
    m = measure(betas, gender)
    v = vertices(betas, gender)
    s = gt_m["height"] / m["height"]
    m_cal = {n: x * (s ** 3 if n == "weight" else s) for n, x in m.items()}
    c = v.mean(axis=0)
    v_cal = (v - c) * s + c

    def one(mm, vv):
        out = {f"err {n}": abs(mm[n] - gt_m[n]) for n in ALL_NAMES}
        out.update({f"pred {n}": mm[n] for n in ALL_NAMES})
        out.update({f"gt {n}": gt_m[n] for n in ALL_NAMES})
        out["tpose_pve_mm"] = tpose_pve_mm(vv, gt_v)
        out["pve_t_sc_mm"] = pve_t_sc(vv, gt_v) * 1000.0
        return out

    return one(m, v), one(m_cal, v_cal)


def image_rows(records, sources, measure, vertices, gt_body) -> list[dict]:
    """records: [(fname, person_id, gender)]. sources: {name: {fname: betas}} where the
    special source "mean body" is added here. Returns per-photo score rows with a
    `method` label."""
    sources = {"mean body": {r[0]: np.zeros(N_BETAS) for r in records}, **sources}
    rows = []
    for fname, pid, gender in records:
        gt_m, gt_v = gt_body[fname]
        for name, by_file in sources.items():
            b = by_file[fname]
            raw, cal = candidate_scores(measure, vertices, b, "NEUTRAL", gt_m, gt_v)
            _, cal_g = candidate_scores(measure, vertices, b, gender, gt_m, gt_v)
            base = {"person": pid, "gender": gender}
            rows.append({**base, "method": name + (", neutral" if name == "mean body" else ""), **raw})
            rows.append({**base, "method": f"{name} + height", **cal})
            rows.append({**base, "method": f"{name} + height + gender", **cal_g})
    return rows


def fit_regressors(measure, vertices, gender, set_names, n_samples, n_val, rng):
    """One MeasurementsToBetas per input set for this gender; linear vs quadratic is chosen
    on held-out random bodies by tpose_pve_mm. Returns {set_name: (degree, regressor)}."""
    B = sample_betas(n_samples + n_val, rng)
    meas = []
    for i, b in enumerate(B):
        meas.append(measure(b, gender))
        if (i + 1) % 500 == 0:
            print(f"  [{gender}] measured {i + 1}/{len(B)}", flush=True)
    out = {}
    for set_name in set_names:
        names = INPUT_SETS[set_name]
        X = matrix(meas, names)
        best = None
        for degree in (1, 2):
            reg = MeasurementsToBetas(names, degree).fit(X[:n_samples], B[:n_samples])
            pred = reg.predict(X[n_samples:])
            val = np.mean([tpose_pve_mm(vertices(pred[i], gender), vertices(B[n_samples + i], gender))
                           for i in range(n_val)])
            print(f"  [{gender}] {set_name:16s} degree {degree}: held-out tpose_pve {val:6.2f} mm")
            if best is None or val < best[0]:
                best = (val, degree, reg)
        out[set_name] = best[1:]
    return out


def regression_rows(people, regressors, measure, vertices, noise_cm, weight_noise_kg,
                    draws, rng) -> list[dict]:
    """people: [(person_id, gender, gt_m, gt_v)] one per distinct person. regressors:
    {gender: {set_name: (degree, reg)}}. The user inputs are the GT body's height (and
    weight) plus Gaussian noise."""
    rows = []
    for pid, gender, gt_m, gt_v in people:
        for set_name, (_, reg) in regressors[gender].items():
            names = INPUT_SETS[set_name]
            kgs = weight_noise_kg if "weight" in names else [0.0]
            for kg in kgs:
                label = (f"{set_name} only (regression, gender known)" if "weight" not in names
                         else f"{set_name} (regression, gender known, weight +-{kg:g}kg)")
                x0 = matrix([gt_m], names)
                for _ in range(draws):
                    b = reg.predict(noisy(x0, noise_std(names, noise_cm, kg), rng))[0]
                    raw, _ = candidate_scores(measure, vertices, b, gender, gt_m, gt_v)
                    rows.append({"person": pid, "gender": gender, "method": label, **raw})
    return rows


def aggregate(rows: list[dict]) -> dict:
    """{(method, group): metrics averaged per person, then over people}."""
    out = {}
    methods = list(dict.fromkeys(r["method"] for r in rows))
    for method in methods:
        for group in GROUPS:
            sel = [dict(r, subject=r["person"]) for r in rows
                   if r["method"] == method and (group == "all" or r["gender"] == group)]
            if sel:
                out[(method, group)] = {"n_people": len({r["subject"] for r in sel}),
                                        **macro_average(sel, METRIC_KEYS)}
    return out


def circumference_corr(rows: list[dict]) -> dict:
    """{(method, gender): {measurement: corr of predicted vs real across photos}} within a
    gender (pooling genders would credit a model for merely knowing men are bigger)."""
    out = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        for gender in ("MALE", "FEMALE"):
            sel = [r for r in rows if r["method"] == method and r["gender"] == gender]
            if len(sel) < 3:
                continue
            c = {}
            for n in CIRCUMFERENCES:
                p = np.array([r[f"pred {n}"] for r in sel])
                g = np.array([r[f"gt {n}"] for r in sel])
                c[n] = float(np.corrcoef(p, g)[0, 1]) if p.std() > 1e-9 and g.std() > 1e-9 else float("nan")
            out[(method, gender)] = c
    return out


def print_tables(agg: dict, corr: dict) -> None:
    for group in GROUPS:
        labels = [m for (m, g) in agg if g == group]
        if not labels:
            continue
        n = agg[(labels[0], group)]["n_people"]
        print(f"\n=== {group}  ({n} people) -- mean abs error vs the pseudo-GT body "
              f"(lengths cm, weight kg, vertex errors mm) ===")
        print(f"{'method':54s}" + "".join(f"{SHORT[x]:>9s}" for x in ALL_NAMES)
              + f"{'tpose_pve':>11s}{'PVE-T-SC':>10s}")
        for m in labels:
            r = agg[(m, group)]
            print(f"{m:54s}" + "".join(f"{r[f'err {x}']:9.2f}" for x in ALL_NAMES)
                  + f"{r['tpose_pve_mm']:11.1f}{r['pve_t_sc_mm']:10.1f}")
    for gender in ("MALE", "FEMALE"):
        labels = [m for (m, g) in corr if g == gender]
        if not labels:
            continue
        print(f"\n=== {gender}: correlation of predicted vs real circumference across photos "
              f"(0 = no per-person information) ===")
        print(f"{'method':54s}" + "".join(f"{SHORT[x]:>9s}" for x in CIRCUMFERENCES))
        for m in labels:
            print(f"{m:54s}" + "".join("      n/a" if np.isnan(corr[(m, gender)][x])
                                       else f"{corr[(m, gender)][x]:+9.2f}" for x in CIRCUMFERENCES))


def write_summary(agg: dict, corr: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "group", "n_people"] + METRIC_KEYS
                   + [f"corr {SHORT[n]}" for n in CIRCUMFERENCES])
        for (method, group), r in agg.items():
            c = corr.get((method, group), {})
            w.writerow([method, group, r["n_people"]] + [r[k] for k in METRIC_KEYS]
                       + [c.get(n, "") for n in CIRCUMFERENCES])
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anthro_root", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CSV")
    ap.add_argument("--limit", type=int, default=None, help="evenly subsample photos (speed)")
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=200)
    ap.add_argument("--noise-cm", type=float, default=1.5)
    ap.add_argument("--weight-noise-kg", nargs="+", type=float, default=[2.0, 5.0])
    ap.add_argument("--noise-draws", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ssp3d_shape.csv")
    args = ap.parse_args()

    specs = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model expects NAME=CSV, got {spec!r}")
        specs[name] = Path(path).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    anthro_root = Path(args.anthro_root).expanduser().resolve()
    for g in ("NEUTRAL", "MALE", "FEMALE"):
        if not (anthro_root / "data" / "smpl" / f"SMPL_{g}.pkl").exists():
            raise SystemExit(f"SMPL_{g}.pkl missing in {anthro_root}/data/smpl/ -- copy it there.")

    per_model = {n: load_prediction_csv(p) for n, p in specs.items()}
    common = sorted(set.intersection(*(set(d) for d in per_model.values())))
    first = per_model[next(iter(per_model))]
    for n, d in per_model.items():
        for k in common:
            if d[k]["gender"] != first[k]["gender"] or not np.allclose(d[k]["gt"], first[k]["gt"], atol=1e-5):
                raise SystemExit(f"{n}: GT/gender for {k} differ from the first CSV -- different runs.")
    if args.limit and len(common) > args.limit:
        common = [common[i] for i in np.linspace(0, len(common) - 1, args.limit).astype(int)]
    records = [(k, first[k]["person_id"], first[k]["gender"]) for k in common]
    print(f"{len(records)} photos common to all {len(specs)} CSV(s), "
          f"{len({r[1] for r in records})} people")

    measure_tool, vertices, faces = build_measurer(anthro_root, ["height"] + [
        n for n in ALL_NAMES if n not in ("height", "weight")])
    measure = with_weight(measure_tool, vertices, faces)
    rng = np.random.default_rng(args.seed)

    gt_body = {k: (measure(first[k]["gt"], first[k]["gender"]), vertices(first[k]["gt"], first[k]["gender"]))
               for k in common}
    print("\nGT bodies (pseudo-GT shapes through the gendered SMPL), mean over photos:")
    for g in ("MALE", "FEMALE"):
        ks = [k for k, _, gg in records if gg == g]
        if ks:
            print(f"  {g:7s} " + "  ".join(f"{SHORT[n]} {np.mean([gt_body[k][0][n] for k in ks]):.1f}"
                                          for n in ALL_NAMES))

    sources = {n: {k: d[k]["pred"] for k in common} for n, d in per_model.items()}
    rows = image_rows(records, sources, measure, vertices, gt_body)
    print(f"scored {len(records)} photos x {len(sources) + 1} body sources", flush=True)

    regressors = {g: fit_regressors(measure, vertices, g, ["height", "height+weight"],
                                    args.n_samples, args.n_val, rng) for g in ("MALE", "FEMALE")}
    seen, people = set(), []
    for k, pid, g in records:
        if pid not in seen:
            seen.add(pid)
            people.append((pid, g, *gt_body[k]))
    rows += regression_rows(people, regressors, measure, vertices, args.noise_cm,
                            args.weight_noise_kg, args.noise_draws, rng)

    agg, corr = aggregate(rows), circumference_corr(rows)
    print_tables(agg, corr)
    write_summary(agg, corr, out_path)


def self_test() -> None:
    """Toy world: 6 'measurements' (the 5 tool ones + weight) are fixed linear functions of
    the betas; the mesh is a 3-parameter blob. No SMPL needed."""
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (6, 6)) + 3 * np.eye(6)

    def measure(b, gender):
        off = 5.0 if gender == "FEMALE" else 0.0
        return dict(zip(ALL_NAMES, 100 + off + A @ b[:6]))

    def vertices(b, gender):
        return np.outer(np.arange(1, 8), b[:3]) + np.arange(21).reshape(7, 3) * 0.1

    # candidate_scores: the exact GT body scores zero, in both views
    gt_b = np.array([0.5, -0.3, 0.2, 0, 0, 0, 0, 0, 0, 0])
    gt_m, gt_v = measure(gt_b, "MALE"), vertices(gt_b, "MALE")
    raw, cal = candidate_scores(measure, vertices, gt_b, "MALE", gt_m, gt_v)
    assert max(raw[k] for k in METRIC_KEYS) < 1e-9 and max(cal[k] for k in METRIC_KEYS) < 1e-9
    # a candidate 20% too small in every length: height calibration removes the height error and
    # PVE-T-SC is scale-free, so it is the same raw and calibrated
    small_m = lambda b, g: {n: x * (0.8 ** 3 if n == "weight" else 0.8) for n, x in gt_m.items()}  # noqa: E731
    small_v = lambda b, g: 0.8 * gt_v  # noqa: E731
    raw, cal = candidate_scores(small_m, small_v, gt_b, "MALE", gt_m, gt_v)
    assert raw["err height"] > 10 and cal["err height"] < 1e-9
    assert raw["tpose_pve_mm"] > cal["tpose_pve_mm"]
    assert abs(raw["pve_t_sc_mm"] - cal["pve_t_sc_mm"]) < 1e-6 and raw["pve_t_sc_mm"] < 1e-6

    # image_rows / aggregate / corr: 2 people (1 woman, 1 man) x 4 photos; a "good" model
    # predicts the GT betas, a "bad" one predicts zeros
    persons = {"p0": ("FEMALE", np.array([1.0, 0.5, 0, 0, 0, 0, 0, 0, 0, 0])),
               "p1": ("MALE", np.array([-1.0, 0.8, 0, 0, 0, 0, 0, 0, 0, 0]))}
    records, gt_betas, gt_body = [], {}, {}
    for pid, (g, b0) in persons.items():
        for i in range(4):
            b = b0 + np.array([0, 0, 0.3 * i, 0, 0, 0, 0, 0, 0, 0])
            fn = f"{pid}_{i}"
            records.append((fn, pid, g))
            gt_betas[fn] = b
            gt_body[fn] = (measure(b, g), vertices(b, g))
    sources = {"good": gt_betas, "bad": {k: np.zeros(N_BETAS) for k in gt_betas}}
    rows = image_rows(records, sources, measure, vertices, gt_body)
    assert len(rows) == len(records) * 3 * 3   # 3 sources (mean, good, bad) x 3 variants
    agg, corr = aggregate(rows), circumference_corr(rows)
    assert agg[("good + height + gender", "all")]["n_people"] == 2
    assert agg[("good + height + gender", "FEMALE")]["n_people"] == 1
    assert agg[("good + height + gender", "all")]["tpose_pve_mm"] < 1e-6
    assert agg[("bad + height + gender", "all")]["tpose_pve_mm"] > 1.0
    assert agg[("mean body, neutral", "all")]["err height"] > agg[("mean body + height", "all")]["err height"]
    assert corr[("good", "FEMALE")]["waist circumference"] > 0.99      # tracks the person
    assert np.isnan(corr[("mean body, neutral", "MALE")]["waist circumference"])  # constant -> n/a

    # regression_rows: height+weight recovers the toy body far better than height alone
    people = [(pid, g, measure(b, g), vertices(b, g)) for pid, (g, b) in persons.items()]
    regs = {}
    for g in ("MALE", "FEMALE"):
        B = sample_betas(2500, np.random.default_rng(1))
        meas = [measure(b, g) for b in B]
        regs[g] = {s: (1, MeasurementsToBetas(INPUT_SETS[s], 1).fit(matrix(meas, INPUT_SETS[s]), B))
                   for s in ("height", "height+weight")}
    rr = regression_rows(people, regs, measure, vertices, 0.0, [0.0], 3, np.random.default_rng(2))
    a2 = aggregate(rr)
    assert len({r["method"] for r in rr}) == 2
    h = a2[("height only (regression, gender known)", "all")]["err weight"]
    hw = a2[("height+weight (regression, gender known, weight +-0kg)", "all")]["err weight"]
    assert hw < 0.05 < h, (hw, h)   # weight given as input is recovered; from height alone it is not

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        write_summary(agg, corr, Path(d) / "s.csv")
        with open(Path(d) / "s.csv") as f:
            assert len(list(csv.reader(f))) == 1 + len(agg)
    print_tables(agg, corr)
    print("\n[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
