#!/usr/bin/env python3
"""Photo + height + weight together: how much does the photo add on top of
height + weight, and how much does weight add on top of the photo?

Method (linear-Gaussian update, one body model per gender):
  prior        the photo model's betas, with an isotropic spread tau per beta
               (tau = how far a photo estimate typically is from the truth)
  observation  y = (height, weight) the user reports, each with noise
  forward map  y ~ H beta + c, fitted by least squares on many random bodies
               (betas ~ N(0,1), weight = mesh volume x density)
  posterior    beta = beta_photo + K (y_obs - (H beta_photo + c)),
               K = tau^2 H^T (tau^2 H H^T + R)^-1,  R = diag(noise std^2)
The photo fixes what height and weight cannot (the other directions of the
shape space); the observations pull the photo's body to the reported height and
weight. With tau -> 0 the result is the photo alone, with tau -> infinity it is
"fit height and weight exactly, ignore the photo's opinion on those".

tau is not known a priori: it is set to the photo model's RMS beta error on
these very photos (a single number estimated on the evaluation data -- mildly
optimistic) and 0.5x / 2x of it are shown as sensitivity.

Caveats (read before quoting any number): SSP-3D has no real weights; weight
here = pseudo-GT mesh volume x density, so every "weight" row is an UPPER
BOUND (BodyM shows what real weight is worth, but it has no photos). The truth
is itself a pseudo-GT fit. Photo betas come from the neutral SMPL and are
rendered/evaluated in the person's gendered model.

Compared (all per person, then averaged, for all / male / female):
  mean body + height + gender       the photo-free floor
  <photo model> + height + gender   the photo alone, scaled to the height
  height+weight (regression, +-kg)  no photo; as in eval_ssp3d_shape.py (linear)
  <photo model> + height + weight (fused, tau x, +-kg)

Usage (self-test, numpy only): python eval/fuse_photo_weight.py --self-test

Usage (same env/inputs as eval_ssp3d_shape.py):
    python eval/fuse_photo_weight.py --anthro_root ~/SMPL-Anthropometry \\
        --model CameraHMR=results/ssp3d_camerahmr.csv --out results/ssp3d_fused.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_ssp3d_shape import (aggregate, candidate_scores, circumference_corr,  # noqa: E402
                              print_tables, write_summary)
from fit_betas_from_measurements import (ALL_NAMES, MeasurementsToBetas, TOOL_NAMES,  # noqa: E402
                                         matrix, sample_betas, with_weight)
from measure_body_error import N_BETAS, build_measurer  # noqa: E402
from ssp3d_common import load_prediction_csv  # noqa: E402

OBS = ["height", "weight"]


def fit_forward(B: np.ndarray, meas: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares y = H beta + c for y = (height, weight). Returns (H (2,10), c (2,))."""
    Y = matrix(meas, OBS)
    A = np.column_stack([B, np.ones(len(B))])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    return coef[:-1].T, coef[-1]


def fuse(beta_photo: np.ndarray, H: np.ndarray, c: np.ndarray, y_obs: np.ndarray,
         tau: float, sigma: np.ndarray) -> np.ndarray:
    """Posterior mean of beta given the photo prior N(beta_photo, tau^2 I) and y_obs = H beta + c + noise(sigma)."""
    R = np.diag(np.asarray(sigma, dtype=float) ** 2)
    S = tau ** 2 * H @ H.T + R
    K = tau ** 2 * H.T @ np.linalg.inv(S)
    return beta_photo + K @ (y_obs - (H @ beta_photo + c))


def photo_tau(pred: dict, fnames: list[str]) -> float:
    """RMS beta error of the photo model over the given photos."""
    err = np.stack([pred[f]["pred"] - pred[f]["gt"] for f in fnames])
    return float(np.sqrt((err ** 2).mean()))


def score_rows(records, gt_body, photo, floor_label, photo_label, measure, vertices, regs, fwd, taus,
               noise_cm, weight_noise_kg, draws, rng) -> list[dict]:
    """records: [(fname, person, gender)]; gt_body[fname] = (measures, vertices); photo[fname] = betas;
    regs[g] = inverse regressor (height, weight -> betas); fwd[g] = (H, c); taus = {label: tau}."""
    rows = []
    for fname, pid, g in records:
        gt_m, gt_v = gt_body[fname]
        base = {"person": pid, "gender": g}
        _, cal0 = candidate_scores(measure, vertices, np.zeros(N_BETAS), g, gt_m, gt_v)
        rows.append({**base, "method": floor_label, **cal0})
        _, cal1 = candidate_scores(measure, vertices, photo[fname], g, gt_m, gt_v)
        rows.append({**base, "method": photo_label, **cal1})
    for kg in weight_noise_kg:
        sigma = np.array([noise_cm, kg])
        for fname, pid, g in records:
            gt_m, gt_v = gt_body[fname]
            base = {"person": pid, "gender": g}
            for _ in range(draws):
                y = np.array([gt_m["height"], gt_m["weight"]]) + rng.normal(0, 1, 2) * sigma
                b_reg = regs[g].predict(y[None])[0]
                raw, _ = candidate_scores(measure, vertices, b_reg, g, gt_m, gt_v)
                rows.append({**base, "method": f"height+weight (regression, +-{kg:g}kg)", **raw})
                H, c = fwd[g]
                for tlabel, tau in taus.items():
                    b = fuse(photo[fname], H, c, y, tau, sigma)
                    raw, _ = candidate_scores(measure, vertices, b, g, gt_m, gt_v)
                    rows.append({**base, "method": f"{photo_label.split(' + ')[0]} + height + weight "
                                                   f"(fused, tau {tlabel}, +-{kg:g}kg)", **raw})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anthro_root", required=True)
    ap.add_argument("--model", required=True, metavar="NAME=CSV", help="the photo model (one)")
    ap.add_argument("--limit", type=int, default=None, help="evenly subsample photos (speed)")
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--noise-cm", type=float, default=1.5)
    ap.add_argument("--weight-noise-kg", nargs="+", type=float, default=[2.0, 5.0])
    ap.add_argument("--draws", type=int, default=3, help="noise draws per photo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ssp3d_fused.csv")
    args = ap.parse_args()

    name, _, csv_path = args.model.partition("=")
    pred = load_prediction_csv(Path(csv_path).expanduser().resolve())
    out_path = Path(args.out).expanduser().resolve()
    anthro = Path(args.anthro_root).expanduser().resolve()
    for g in ("NEUTRAL", "MALE", "FEMALE"):
        if not (anthro / "data" / "smpl" / f"SMPL_{g}.pkl").exists():
            raise SystemExit(f"SMPL_{g}.pkl missing in {anthro}/data/smpl/")

    fnames = sorted(pred)
    if args.limit and len(fnames) > args.limit:
        fnames = [fnames[i] for i in np.linspace(0, len(fnames) - 1, args.limit).astype(int)]
    records = [(f, pred[f]["person_id"], pred[f]["gender"]) for f in fnames]
    tau0 = photo_tau(pred, fnames)
    print(f"{len(records)} photos, {len({r[1] for r in records})} people; {name} RMS beta error "
          f"(tau, per beta) = {tau0:.2f}")

    measure_tool, vertices, faces = build_measurer(anthro, TOOL_NAMES)
    measure = with_weight(measure_tool, vertices, faces)
    rng = np.random.default_rng(args.seed)
    gt_body = {f: (measure(pred[f]["gt"], g), vertices(pred[f]["gt"], g)) for f, _, g in records}

    regs, fwd = {}, {}
    for g in ("MALE", "FEMALE"):
        B = sample_betas(args.n_samples, rng)
        meas = []
        for i, b in enumerate(B):
            meas.append(measure(b, g))
            if (i + 1) % 500 == 0:
                print(f"  [{g}] measured {i + 1}/{len(B)}", flush=True)
        regs[g] = MeasurementsToBetas(OBS, degree=1).fit(matrix(meas, OBS), B)
        fwd[g] = fit_forward(B, meas)

    taus = {f"{m:g}x": m * tau0 for m in (0.5, 1.0, 2.0)}
    photo = {f: pred[f]["pred"] for f in fnames}
    rows = score_rows(records, gt_body, photo, "mean body + height + gender", f"{name} + height + gender",
                      measure, vertices, regs, fwd, taus, args.noise_cm, args.weight_noise_kg,
                      args.draws, rng)
    agg, corr = aggregate(rows), circumference_corr(rows)
    print_tables(agg, corr)
    write_summary(agg, corr, out_path)


def self_test() -> None:
    rng = np.random.default_rng(0)
    A = rng.normal(0, 1, (2, N_BETAS))
    A[:, 6:] = 0.0   # height and weight only look at the first 6 betas

    def toy_measure(b, g):
        h, w = 170 + A[0] @ b, 70 + A[1] @ b
        m = {n: 100.0 + 0.0 * h for n in ALL_NAMES}
        m.update({"height": h, "weight": w, "waist circumference": 80 + 0.5 * (w - 70)})
        return m

    def toy_vertices(b, g):
        return np.outer(np.arange(1, 8), b[:3]) + np.arange(21).reshape(7, 3) * 0.1

    # forward map is recovered
    B = sample_betas(2000, np.random.default_rng(1))
    meas = [toy_measure(b, "X") for b in B]
    H, c = fit_forward(B, meas)
    assert np.abs(H - A).max() < 1e-6 and np.abs(c - [170, 70]).max() < 1e-6

    # fuse: limiting cases
    bp = np.full(N_BETAS, 0.3)
    y = np.array([180.0, 80.0])
    sig = np.array([0.01, 0.01])
    assert np.allclose(fuse(bp, H, c, y, 1e-6, sig), bp, atol=1e-6), "tau -> 0 returns the photo"
    assert np.allclose(fuse(bp, H, c, y, 10.0, np.array([1e6, 1e6])), bp, atol=1e-3), "huge noise -> photo"
    post = fuse(bp, H, c, y, 10.0, sig)
    assert np.abs(H @ post + c - y).max() < 0.1, "tiny noise + wide prior -> observations are matched"
    assert np.abs(post[6:] - bp[6:]).max() < 1e-9, "directions the observations cannot see keep the photo's value"

    # photo_tau
    pred = {"a": {"pred": np.ones(N_BETAS), "gt": np.zeros(N_BETAS)}}
    assert abs(photo_tau(pred, ["a"]) - 1.0) < 1e-12

    # full pipeline on toy data: fusing the photo with height+weight beats both alone
    g_true = np.clip(rng.standard_normal((40, N_BETAS)), -2, 2)
    photo_b = g_true + rng.normal(0, 0.6, g_true.shape)
    records, gt_body, photo, predd = [], {}, {}, {}
    for i, b in enumerate(g_true):
        f = f"f{i}"
        records.append((f, f"p{i}", "MALE" if i % 2 else "FEMALE"))
        gt_body[f] = (toy_measure(b, "X"), toy_vertices(b, "X"))
        photo[f] = photo_b[i]
        predd[f] = {"pred": photo_b[i], "gt": b}
    tau0 = photo_tau(predd, list(predd))
    assert 0.5 < tau0 < 0.7, tau0
    reg = MeasurementsToBetas(OBS, degree=1).fit(matrix(meas, OBS), B)
    regs = {"MALE": reg, "FEMALE": reg}
    fwd = {"MALE": (H, c), "FEMALE": (H, c)}
    rows = score_rows(records, gt_body, photo, "mean", "Photo + height + gender", toy_measure, toy_vertices,
                      regs, fwd, {"1x": tau0}, 1.0, [1.0], 2, np.random.default_rng(3))
    agg = aggregate(rows)
    pve = {m: v["tpose_pve_mm"] for (m, grp), v in agg.items() if grp == "all"}
    fused = pve["Photo + height + weight (fused, tau 1x, +-1kg)"]
    assert fused < pve["Photo + height + gender"], pve
    assert fused < pve["height+weight (regression, +-1kg)"], pve
    assert pve["Photo + height + gender"] < pve["mean"], pve
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
