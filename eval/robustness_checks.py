#!/usr/bin/env python3
"""Three robustness checks on the S0 shape results, on HBW (10 scanned people) and SSP-3D (62 people):

  1. Confidence intervals. Is "CameraHMR beats the average body" (and beats the other models) still
     true once you account for how few people there are? Paired per-person differences in
     circumference error, bootstrapped over PEOPLE (10,000 resamples): mean difference, 95% interval,
     and the share of resamples in which the model is better.
  2. Several photos of one person. Average the predicted betas of k photos of the same person and
     measure the error again. What shrinks with k is noise; what stays at the all-photos level is
     bias -- the part more photos cannot remove.
  3. De-shrinkage. The models pull large bodies toward the average body. Two linear corrections are
     tested, each fitted on one set of people and scored on others (leave-one-person-out inside a
     dataset, and fitted on one dataset / scored on the other):
       a. on the circumference:  c' = c_mean + k (c_pred - c_mean), k fitted per measurement;
       b. on the betas:          beta' = k * beta  (CameraHMR), scored in body space.

Both datasets are measured with the SAME landmark-free definitions as eval_hbw_shape.py (natural
waist = narrowest cross-section at 55-68% of height, chest / hip = widest in their bands), on the
predicted neutral SMPL T-pose scaled to the person's real height versus the person's own body (HBW:
the scan; SSP-3D: the pseudo-GT shape in its gendered SMPL). So SSP-3D circumferences here are NOT
the numbers of eval_ssp3d_shape.py (different definition); compare within this file only.

Caveats: SSP-3D's truth is a fitted pseudo-GT; HBW has only 10 people so its intervals are wide
(that is the point of check 1); k fitted across datasets assumes the shrinkage is similar in both.

Usage (self-test, numpy + scipy): python eval/robustness_checks.py --self-test

Usage (server, env camerahmr; needs the HBW and SSP-3D prediction CSVs):
    python eval/robustness_checks.py --hbw ~/datasets/HBW --ssp3d ~/datasets/SSP-3D \\
        --anthro_root ~/SMPL-Anthropometry \\
        --hbw-model HMR2.0b=results/hbw_hmr2.csv --hbw-model CameraHMR=results/hbw_camerahmr.csv \\
        --hbw-model TokenHMR=results/hbw_tokenhmr.csv \\
        --ssp-model HMR2.0b=results/ssp3d_hmr2.csv --ssp-model CameraHMR=results/ssp3d_camerahmr.csv \\
        --ssp-model TokenHMR=results/ssp3d_tokenhmr.csv --out results/robustness.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_hbw_shape import (CIRC, GTBody, build_gt, measure_mesh, normalize_mesh,  # noqa: E402
                            read_obj_faces, surface_distance_mm, unique_edges)

BASELINE = "mean body"
N_BETAS = 10
K_BETA_GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]


# ---------------------------------------------------------------- pure statistics
def bootstrap_mean(values, rng: np.random.Generator, n_boot: int = 10000) -> tuple[float, float, float, float]:
    """Mean of `values` (one per person), 95% percentile interval over people, and the share of
    resamples whose mean is > 0."""
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float((means > 0).mean())


def fit_scale(P: np.ndarray, M: np.ndarray, G: np.ndarray, lo: float = 0.5, hi: float = 3.0) -> float:
    """Least-squares k in  G - M ~ k (P - M)  (1-D arrays: predicted, mean-body, true circumference)."""
    d = P - M
    den = float((d ** 2).sum())
    return float(np.clip(((G - M) * d).sum() / den, lo, hi)) if den > 1e-12 else 1.0


def apply_scale(P, M, k):
    return M + k * (np.asarray(P) - M)


def macro(per_subject: dict) -> np.ndarray:
    return np.mean([v for v in per_subject.values()], axis=0)


@dataclass
class Dataset:
    name: str
    gts: dict                  # subject -> GTBody
    photos: dict               # model -> subject -> list of betas (10,)
    surface: bool = False      # compute the surface distance too (HBW only)


class Evaluator:
    def __init__(self, vertices_fn, faces, rng, surf_samples: int = 20000):
        self.vertices, self.faces, self.rng = vertices_fn, np.asarray(faces), rng
        self.edges = unique_edges(self.faces)
        self.surf_samples = surf_samples

    def body(self, betas, gt: GTBody, surface: bool = False) -> dict:
        v = normalize_mesh(self.vertices(np.asarray(betas, dtype=float), "NEUTRAL"), target_height=gt.H)
        m = measure_mesh(v, self.edges)
        out = {n: m[n] for n in CIRC}
        if surface:
            out["surf_mm"] = surface_distance_mm(v, self.faces, gt, self.rng, self.surf_samples)
        return out

    def vec(self, betas, gt: GTBody) -> np.ndarray:
        b = self.body(betas, gt)
        return np.array([b[n] for n in CIRC])


def gt_vec(gt: GTBody) -> np.ndarray:
    return np.array([gt.measures[n] for n in CIRC])


# ---------------------------------------------------------------- check 1: bootstrap
def per_photo_arrays(ev: Evaluator, ds: Dataset, model: str) -> dict:
    """subject -> (n_photos, 3) predicted chest/waist/hip."""
    return {s: np.array([ev.vec(b, ds.gts[s]) for b in bl]) for s, bl in ds.photos[model].items()}


def mean_body_vectors(ev: Evaluator, ds: Dataset) -> dict:
    return {s: ev.vec(np.zeros(N_BETAS), g) for s, g in ds.gts.items()}


def subject_errors(P: dict, ds: Dataset) -> dict:
    """subject -> mean abs error (3,) over its photos."""
    return {s: np.abs(p - gt_vec(ds.gts[s])).mean(axis=0) for s, p in P.items()}


def check1(ds: Dataset, P_by_model: dict, M: dict, rng, n_boot: int = 10000) -> list[dict]:
    err = {m: subject_errors(P, ds) for m, P in P_by_model.items()}
    err[BASELINE] = {s: np.abs(M[s] - gt_vec(ds.gts[s])) for s in M}
    people = sorted(set.intersection(*(set(e) for e in err.values())))
    rows = []
    comps = [(m, BASELINE) for m in P_by_model] + [("CameraHMR", m) for m in P_by_model if m != "CameraHMR"]
    for a, b in comps:
        if a not in err or b not in err:
            continue
        for j, name in enumerate(CIRC):
            d = np.array([err[b][s][j] - err[a][s][j] for s in people])      # > 0: a is better than b
            mean, lo, hi, ppos = bootstrap_mean(d, rng, n_boot)
            rows.append({"experiment": "1 bootstrap", "dataset": ds.name, "item": f"{a} vs {b}", "measure": name,
                         "n_people": len(people), "value": mean, "lo": lo, "hi": hi, "p_better": ppos})
    return rows


# ---------------------------------------------------------------- check 2: several photos
def check2(ev: Evaluator, ds: Dataset, model: str, ks: list[int], draws: int, rng) -> list[dict]:
    rows = []
    for k in ks + ["all"]:
        errs = {}
        for s, bl in ds.photos[model].items():
            n = len(bl)
            if k != "all" and n < k:
                continue
            kk = n if k == "all" else k
            nd = 1 if k == "all" else draws
            sample_errs = []
            for _ in range(nd):
                pick = rng.choice(n, size=kk, replace=False)
                avg = np.mean([bl[i] for i in pick], axis=0)
                sample_errs.append(np.abs(ev.vec(avg, ds.gts[s]) - gt_vec(ds.gts[s])))
            errs[s] = np.mean(sample_errs, axis=0)
        if errs:
            m = macro(errs)
            rows.append({"experiment": "2 k photos", "dataset": ds.name, "item": model, "k": k, "n_people": len(errs),
                         **{n: float(m[i]) for i, n in enumerate(CIRC)}})
    return rows


# ---------------------------------------------------------------- check 3: de-shrinkage
def loso_scale(P: dict, M: dict, G: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-person-out circumference scaling. Returns (uncalibrated error, calibrated error, mean k),
    each over the 3 measurements, macro-averaged over people."""
    people = sorted(P)
    e0, e1, ks = [], [], []
    for s in people:
        k = np.ones(3)
        for j in range(3):
            tr = [t for t in people if t != s]
            k[j] = fit_scale(np.concatenate([P[t][:, j] for t in tr]),
                             np.concatenate([np.full(len(P[t]), M[t][j]) for t in tr]),
                             np.concatenate([np.full(len(P[t]), G[t][j]) for t in tr]))
        e0.append(np.abs(P[s] - G[s]).mean(axis=0))
        e1.append(np.abs(apply_scale(P[s], M[s], k) - G[s]).mean(axis=0))
        ks.append(k)
    return np.mean(e0, axis=0), np.mean(e1, axis=0), np.mean(ks, axis=0)


def fit_all_scale(P: dict, M: dict, G: dict) -> np.ndarray:
    people = sorted(P)
    return np.array([fit_scale(np.concatenate([P[t][:, j] for t in people]),
                               np.concatenate([np.full(len(P[t]), M[t][j]) for t in people]),
                               np.concatenate([np.full(len(P[t]), G[t][j]) for t in people])) for j in range(3)])


def scaled_error(P: dict, M: dict, G: dict, k: np.ndarray) -> np.ndarray:
    return np.mean([np.abs(apply_scale(P[s], M[s], k) - G[s]).mean(axis=0) for s in P], axis=0)


def check3a(datasets: dict, arrays: dict, means: dict, rng) -> list[dict]:
    """datasets: name -> Dataset; arrays[name][model] = P dict; means[name] = M dict."""
    rows = []
    gts = {n: {s: gt_vec(g) for s, g in ds.gts.items()} for n, ds in datasets.items()}
    models = sorted(set.intersection(*(set(a) for a in arrays.values())))
    for n in datasets:
        other = [o for o in datasets if o != n]
        for m in models:
            P, M, G = arrays[n][m], means[n], gts[n]
            e0, e1, ks = loso_scale(P, M, G)
            base = {"experiment": "3a circumference scaling", "dataset": n, "item": m}
            rows.append({**base, "variant": "none", **{c: float(e0[j]) for j, c in enumerate(CIRC)}})
            rows.append({**base, "variant": "leave-one-person-out", **{c: float(e1[j]) for j, c in enumerate(CIRC)},
                         **{f"k_{c}": float(ks[j]) for j, c in enumerate(CIRC)}})
            kf = fit_all_scale(P, M, G)
            ef = scaled_error(P, M, G, kf)
            rows.append({**base, "variant": "fitted on itself (optimistic)", **{c: float(ef[j]) for j, c in enumerate(CIRC)},
                         **{f"k_{c}": float(kf[j]) for j, c in enumerate(CIRC)}})
            for o in other:
                ko = fit_all_scale(arrays[o][m], means[o], gts[o])
                eo = scaled_error(P, M, G, ko)
                rows.append({**base, "variant": f"fitted on {o}", **{c: float(eo[j]) for j, c in enumerate(CIRC)},
                             **{f"k_{c}": float(ko[j]) for j, c in enumerate(CIRC)}})
    return rows


def check3b(ev: Evaluator, datasets: dict, model: str, ks: list[float]) -> list[dict]:
    """beta' = k * beta for the given model, scored in body space (macro over people)."""
    rows = []
    for n, ds in datasets.items():
        for k in ks:
            errs = {}
            for s, bl in ds.photos[model].items():
                g = gt_vec(ds.gts[s])
                errs[s] = np.mean([np.abs(ev.vec(k * np.asarray(b), ds.gts[s]) - g) for b in bl], axis=0)
            m = macro(errs)
            rows.append({"experiment": "3b beta scaling", "dataset": n, "item": model, "k": k,
                         **{c: float(m[j]) for j, c in enumerate(CIRC)}})
    return rows


# ---------------------------------------------------------------- reporting
def fmt(rows: list[dict], experiment: str) -> list[dict]:
    return [r for r in rows if r["experiment"] == experiment]


def print_report(rows: list[dict]) -> None:
    print("\n=== 1. Paired per-person difference in circumference error (cm); > 0 = the first model is better ===")
    print(f"{'dataset':8s}{'comparison':26s}{'measure':22s}{'people':>7s}{'mean diff':>10s}{'95% interval':>20s}{'P(better)':>10s}")
    for r in fmt(rows, "1 bootstrap"):
        print(f"{r['dataset']:8s}{r['item']:26s}{r['measure']:22s}{r['n_people']:7d}{r['value']:10.2f}"
              f"{'[' + format(r['lo'], '.2f') + ', ' + format(r['hi'], '.2f') + ']':>20s}{r['p_better']:10.3f}")
    print("\n=== 2. Error vs number of photos averaged (cm, macro over people) ===")
    print(f"{'dataset':8s}{'model':12s}{'k':>6s}{'people':>7s}" + "".join(f"{c.split()[0]:>9s}" for c in CIRC))
    for r in fmt(rows, "2 k photos"):
        print(f"{r['dataset']:8s}{r['item']:12s}{str(r['k']):>6s}{r['n_people']:7d}" + "".join(f"{r[c]:9.2f}" for c in CIRC))
    print("\n=== 3a. Circumference de-shrinkage: error (cm) after c' = c_mean + k (c_pred - c_mean) ===")
    print(f"{'dataset':8s}{'model':12s}{'variant':34s}" + "".join(f"{c.split()[0]:>8s}" for c in CIRC) + "   k (chest/waist/hip)")
    for r in fmt(rows, "3a circumference scaling"):
        ks = "  ".join(f"{r[f'k_{c}']:.2f}" for c in CIRC) if f"k_{CIRC[0]}" in r else ""
        print(f"{r['dataset']:8s}{r['item']:12s}{r['variant']:34s}" + "".join(f"{r[c]:8.2f}" for c in CIRC) + f"   {ks}")
    print("\n=== 3b. Beta scaling beta' = k * beta (error in cm) ===")
    print(f"{'dataset':8s}{'model':12s}{'k':>6s}" + "".join(f"{c.split()[0]:>9s}" for c in CIRC) + f"{'mean':>8s}")
    for r in fmt(rows, "3b beta scaling"):
        print(f"{r['dataset']:8s}{r['item']:12s}{r['k']:6.2f}" + "".join(f"{r[c]:9.2f}" for c in CIRC)
              + f"{np.mean([r[c] for c in CIRC]):8.2f}")


def write_rows(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


# ---------------------------------------------------------------- main
def parse_models(specs):
    out = {}
    for spec in specs:
        name, _, p = spec.partition("=")
        if not p:
            raise SystemExit(f"--*-model expects NAME=CSV, got {spec!r}")
        out[name] = Path(p).expanduser().resolve()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbw", required=True)
    ap.add_argument("--ssp3d", required=True)
    ap.add_argument("--anthro_root", required=True)
    ap.add_argument("--hbw-model", action="append", required=True, metavar="NAME=CSV")
    ap.add_argument("--ssp-model", action="append", required=True, metavar="NAME=CSV")
    ap.add_argument("--checks", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    ap.add_argument("--draws", type=int, default=8, help="random photo subsets per person and k (check 2)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/robustness.csv")
    args = ap.parse_args()

    hbw_specs, ssp_specs = parse_models(args.hbw_model), parse_models(args.ssp_model)
    out_path = Path(args.out).expanduser().resolve()
    hbw_root, ssp_root = Path(args.hbw).expanduser().resolve(), Path(args.ssp3d).expanduser().resolve()
    anthro = Path(args.anthro_root).expanduser().resolve()
    from hbw_common import load_prediction_csv as load_hbw_csv
    from ssp3d_common import load_prediction_csv as load_ssp_csv
    from measure_body_error import build_measurer
    hbw_pred = {n: load_hbw_csv(p) for n, p in hbw_specs.items()}
    ssp_pred = {n: load_ssp_csv(p) for n, p in ssp_specs.items()}
    _, vertices, faces = build_measurer(anthro, ["height"])             # chdirs into the anthro folder
    rng = np.random.default_rng(args.seed)
    ev = Evaluator(vertices, faces, rng)

    # HBW: scans and per-subject photos
    hbw_gts, hbw_photos = {}, {n: {} for n in hbw_pred}
    for sid in sorted({r["subject"] for r in next(iter(hbw_pred.values())).values()}):
        hbw_gts[sid] = build_gt(np.load(hbw_root / "smplx" / "val" / f"{sid}.npy"),
                                read_obj_faces(hbw_root / "smplx" / "val" / f"{sid}.obj"), rng, 20000)
    for n, d in hbw_pred.items():
        for r in d.values():
            hbw_photos[n].setdefault(r["subject"], []).append(r["pred"])
    # SSP-3D: pseudo-GT bodies (gendered SMPL T-pose) and per-person photos
    ssp_gts, ssp_photos = {}, {n: {} for n in ssp_pred}
    first = next(iter(ssp_pred.values()))
    for r in first.values():
        pid = r["person_id"]
        if pid not in ssp_gts:
            ssp_gts[pid] = build_gt(vertices(r["gt"], r["gender"]), np.asarray(faces), rng, 200)
    for n, d in ssp_pred.items():
        for r in d.values():
            ssp_photos[n].setdefault(r["person_id"], []).append(r["pred"])
    datasets = {"HBW": Dataset("HBW", hbw_gts, hbw_photos, surface=True), "SSP-3D": Dataset("SSP-3D", ssp_gts, ssp_photos)}
    print({n: (len(d.gts), sum(len(v) for v in next(iter(d.photos.values())).values())) for n, d in datasets.items()},
          "(people, photos)", flush=True)

    rows: list[dict] = []
    arrays, means = {}, {}
    if 1 in args.checks or 3 in args.checks:
        for n, ds in datasets.items():
            arrays[n] = {m: per_photo_arrays(ev, ds, m) for m in ds.photos}
            means[n] = mean_body_vectors(ev, ds)
            print(f"  measured every photo of {n}", flush=True)
    if 1 in args.checks:
        for n, ds in datasets.items():
            rows += check1(ds, arrays[n], means[n], rng, args.n_boot)
    if 2 in args.checks:
        for n, ds in datasets.items():
            ks = [1, 2, 3, 5, 10] if n == "HBW" else [1, 2, 3, 5]
            for m in ds.photos:
                rows += check2(ev, ds, m, ks, args.draws, rng)
                print(f"  check 2 done: {n} {m}", flush=True)
    if 3 in args.checks:
        rows += check3a(datasets, arrays, means, rng)
        rows += check3b(ev, datasets, "CameraHMR", K_BETA_GRID)
    print_report(rows)
    write_rows(rows, out_path)


# ---------------------------------------------------------------- self-test
def self_test() -> None:
    from eval_hbw_shape import toy_mesh
    rng = np.random.default_rng(0)

    # bootstrap: a clear positive difference has an interval above 0; zero-mean noise straddles it
    mean, lo, hi, p = bootstrap_mean(np.full(10, 2.0) + rng.normal(0, 0.3, 10), rng, 2000)
    assert lo > 0 and p == 1.0 and 1.5 < mean < 2.5
    mean, lo, hi, p = bootstrap_mean(rng.normal(0, 1, 10), rng, 2000)
    assert lo < 0 < hi

    # fit_scale / apply_scale: predictions shrunk by 0.5 toward the mean are restored by k = 2
    M = np.full(200, 80.0)
    G = M + rng.normal(0, 10, 200)
    P = M + 0.5 * (G - M) + rng.normal(0, 1, 200)
    k = fit_scale(P, M, G)
    assert 1.8 < k < 2.2, k
    assert np.abs(apply_scale(P, M, k) - G).mean() < 0.7 * np.abs(P - G).mean()
    assert fit_scale(M, M, G) == 1.0                                       # no deviation: leave alone

    # toy world: waist radius = r0 + 0.02 * beta[0]; a model returns 0.5*truth + noise
    def waist_r(b0):
        return lambda t: (0.15 + 0.02 * b0) - 0.03 * np.exp(-((t - 0.62) / 0.04) ** 2)

    def vertices(betas, gender):
        return toy_mesh(waist_r(betas[0]))[0]

    faces = toy_mesh(waist_r(0))[1]
    ev = Evaluator(vertices, faces, rng, surf_samples=500)
    people = {f"p{i}": float(b) for i, b in enumerate([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])}
    gts = {s: build_gt(vertices(np.array([b]), "M"), faces, rng, 200) for s, b in people.items()}
    photos = {"good": {s: [np.array([b + rng.normal(0, 0.2)]) for _ in range(6)] for s, b in people.items()},
              "shrunk": {s: [np.array([0.5 * b + rng.normal(0, 0.1)]) for _ in range(6)] for s, b in people.items()},
              "noisy": {s: [np.array([0.5 * b + rng.normal(0, 0.8)]) for _ in range(6)] for s, b in people.items()}}
    ds = Dataset("toy", gts, photos)
    P = {m: per_photo_arrays(ev, ds, m) for m in photos}
    M = mean_body_vectors(ev, ds)
    assert all(abs(M[s][1] - M["p2"][1]) < 1e-6 for s in M), "the mean body does not depend on the person"

    # check 1: 'good' beats the average body for every person; 'shrunk' is in between
    rows1 = check1(ds, P, M, rng, 1000)
    r = {(x["item"], x["measure"]): x for x in rows1}
    assert r[("good vs mean body", "waist")]["lo"] > 0
    assert r[("good vs mean body", "waist")]["p_better"] == 1.0

    # check 2: averaging more photos lowers the noisy model's error toward its bias floor
    rows2 = check2(ev, ds, "noisy", [1, 3, 6], 20, rng)
    w = {x["k"]: x["waist"] for x in rows2}
    assert w[1] > w[3] >= w[6] - 0.05, w
    assert abs(w[6] - w["all"]) < 0.05, "k = n photos is the all-photos value"

    # check 3a: leave-one-person-out scaling reduces the shrunk model's error, and ~doubles the deviation
    e0, e1, ks = loso_scale(P["shrunk"], M, {s: gt_vec(g) for s, g in gts.items()})
    assert e1[1] < e0[1] and 1.5 < ks[1] < 2.6, (e0, e1, ks)
    rows3 = check3a({"toy": ds, "toy2": ds}, {"toy": P, "toy2": P}, {"toy": M, "toy2": M}, rng)
    assert {r["variant"] for r in rows3} >= {"none", "leave-one-person-out", "fitted on toy2"}

    # check 3b: beta scaling by 2 fixes the shrunk model; the best k on the grid is near 2
    rows3b = check3b(ev, {"toy": ds}, "shrunk", [1.0, 1.5, 2.0, 3.0])
    wb = {r["k"]: r["waist"] for r in rows3b}
    assert min(wb, key=wb.get) in (1.5, 2.0) and wb[2.0] < wb[1.0], wb

    print_report(rows1 + rows2 + rows3 + rows3b)
    print("\n[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
