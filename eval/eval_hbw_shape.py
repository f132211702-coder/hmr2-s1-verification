#!/usr/bin/env python3
"""Do the photo models get a REAL person's body shape right? Scored against the 3D scans
of HBW (Human Bodies in the Wild; 10 validation subjects, real clothes, scan-aligned
SMPL-X meshes in metres) -- the first ground truth in this project that is neither a
pseudo-GT fit nor missing the photos.

Setting: one photo + the user's height. The predicted betas (neutral SMPL, T-pose) are
scaled uniformly to the SCAN's height ("+ height"), then compared with the scan.

The scan is SMPL-X topology and the models output SMPL, so nothing is compared vertex to
vertex. Two topology-free measurements are used on both meshes:
  circumferences   natural waist = narrowest horizontal cross-section between 55% and 68% of
                   body height; chest = widest between 66% and 74%; hip = widest between 49%
                   and 56%. The cross-section is the convex hull of the plane cut (only
                   points within 0.30 m of the body axis, i.e. the arms are ignored) --
                   the same construction SMPL-Anthropometry uses, but with landmark-free
                   definitions that mean the same thing on both topologies.
  surface distance (mm) mean distance from each mesh's torso+leg points (below 82% of height,
                   within 0.30 m of the axis; head and arms excluded) to the other mesh's
                   surface (dense surface samples + vertices), both directions averaged,
                   after aligning feet and the hip-band centre.

Compared (same photos for every row): "mean body" (betas = 0, the photo-free floor), and
each model, all scaled to the scan height. Averaged per subject first, then over the 10
subjects, for all photos / studio ("lab") photos / candid ("wild") photos; the correlation of
predicted vs real circumference is computed across subjects (10 points: very low power).

Caveats: 10 subjects; gender is not in the data (the neutral body is the baseline for
everyone); no weight/height files (height = scan height); the low-resolution photos are
about 200x300 px; hip depends on the scan's leg pose matching the SMPL rest pose.

Usage (self-test, numpy + scipy): python eval/eval_hbw_shape.py --self-test

Usage (needs torch+smplx+trimesh+scipy; env camerahmr; SMPL_NEUTRAL.pkl in the anthro folder):
    python eval/eval_hbw_shape.py --hbw ~/datasets/HBW --anthro_root ~/SMPL-Anthropometry \\
        --model HMR2.0b=results/hbw_hmr2.csv --model CameraHMR=results/hbw_camerahmr.csv \\
        --model TokenHMR=results/hbw_tokenhmr.csv --out results/hbw_shape.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hbw_common import N_BETAS, load_prediction_csv  # noqa: E402

BANDS = {"chest": (0.66, 0.74, "max"), "waist": (0.55, 0.68, "min"), "hip": (0.49, 0.56, "max")}
CIRC = list(BANDS)
X_LIM = 0.30          # m: points farther from the body axis than this (arms) are ignored
SURF_TOP = 0.82       # fraction of height: surface metric ignores the head and shoulders
LEVEL_STEP = 0.012    # m between cross-section levels
METRICS = [f"err {n}" for n in CIRC] + ["surf_mm"]
BASELINE = "mean body"


# ---------------------------------------------------------------- geometry
def unique_edges(faces: np.ndarray) -> np.ndarray:
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def plane_xz(verts: np.ndarray, edges: np.ndarray, y: float) -> np.ndarray:
    """(x, z) of every mesh edge crossing the plane height y, restricted to |x| <= X_LIM."""
    v0, v1 = verts[edges[:, 0]], verts[edges[:, 1]]
    d0, d1 = v0[:, 1] - y, v1[:, 1] - y
    hit = d0 * d1 < 0
    t = (d0[hit] / (d0[hit] - d1[hit]))[:, None]
    p = v0[hit] + t * (v1[hit] - v0[hit])
    p = p[np.abs(p[:, 0]) <= X_LIM]
    return p[:, [0, 2]]


def hull_perimeter(xz: np.ndarray) -> float:
    if len(xz) < 3:
        return float("nan")
    try:
        return float(ConvexHull(xz).area)      # for 2-D points .area is the perimeter
    except Exception:                           # noqa: BLE001 -- degenerate (collinear) slice
        return float("nan")


def measure_mesh(verts: np.ndarray, edges: np.ndarray) -> dict:
    """chest / waist / hip circumference in cm with the landmark-free band definitions."""
    y0, y1 = verts[:, 1].min(), verts[:, 1].max()
    H = y1 - y0
    out = {}
    for name, (lo, hi, mode) in BANDS.items():
        n = max(3, int(round((hi - lo) * H / LEVEL_STEP)) + 1)
        vals = np.array([hull_perimeter(plane_xz(verts, edges, y0 + f * H)) for f in np.linspace(lo, hi, n)])
        vals = vals[~np.isnan(vals)]
        out[name] = float((vals.min() if mode == "min" else vals.max()) * 100) if len(vals) else float("nan")
    return out


def normalize_mesh(verts: np.ndarray, target_height: float | None = None) -> np.ndarray:
    """Feet on y = 0, uniformly scaled to target_height (if given), x/z centred on the mean of the
    hip band (45-55% of height)."""
    v = verts - np.array([0.0, verts[:, 1].min(), 0.0])
    H = v[:, 1].max()
    if target_height is not None:
        v = v * (target_height / H)
        H = target_height
    band = v[(v[:, 1] >= 0.45 * H) & (v[:, 1] <= 0.55 * H)]
    return v - np.array([band[:, 0].mean(), 0.0, band[:, 2].mean()])


def sample_surface(verts: np.ndarray, faces: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n area-weighted random surface points, plus the vertices themselves (so a mesh measured
    against itself has distance exactly 0)."""
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(faces), n, p=area / area.sum())
    u, w = rng.random((n, 1)), rng.random((n, 1))
    flip = (u + w) > 1
    u, w = np.where(flip, 1 - u, u), np.where(flip, 1 - w, w)
    t = tri[idx]
    return np.vstack([t[:, 0] + u * (t[:, 1] - t[:, 0]) + w * (t[:, 2] - t[:, 0]), verts])


def torso_points(verts: np.ndarray, H: float) -> np.ndarray:
    return verts[(verts[:, 1] <= SURF_TOP * H) & (np.abs(verts[:, 0]) <= X_LIM)]


@dataclass
class GTBody:
    verts: np.ndarray       # normalised scan mesh
    faces: np.ndarray
    edges: np.ndarray
    H: float
    tree: cKDTree
    measures: dict


def build_gt(verts: np.ndarray, faces: np.ndarray, rng: np.random.Generator, n_samples: int) -> GTBody:
    v = normalize_mesh(verts)
    H = float(v[:, 1].max())
    edges = unique_edges(faces)
    return GTBody(v, faces, edges, H, cKDTree(sample_surface(v, faces, n_samples, rng)), measure_mesh(v, edges))


def surface_distance_mm(pred_v: np.ndarray, pred_faces: np.ndarray, gt: GTBody, rng: np.random.Generator,
                        n_samples: int) -> float:
    pred_tree = cKDTree(sample_surface(pred_v, pred_faces, n_samples, rng))
    d_pred = gt.tree.query(torso_points(pred_v, gt.H))[0].mean()
    d_gt = pred_tree.query(torso_points(gt.verts, gt.H))[0].mean()
    return float(0.5 * (d_pred + d_gt) * 1000)


def score_body(pred_raw: np.ndarray, pred_faces: np.ndarray, pred_edges: np.ndarray, gt: GTBody,
               rng: np.random.Generator, n_samples: int) -> dict:
    v = normalize_mesh(pred_raw, target_height=gt.H)            # "+ height": scaled to the scan height
    m = measure_mesh(v, pred_edges)
    out = {f"err {n}": abs(m[n] - gt.measures[n]) for n in CIRC}
    out.update({f"pred {n}": m[n] for n in CIRC})
    out["surf_mm"] = surface_distance_mm(v, pred_faces, gt, rng, n_samples)
    return out


# ---------------------------------------------------------------- aggregation / reporting
def read_obj_faces(path: Path) -> np.ndarray:
    faces = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("f "):
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.array(faces, dtype=np.int64)


def macro(rows: list[dict], method: str, kinds: tuple, keys: list[str]) -> dict:
    """Mean over subjects of the per-subject mean of each key, for the rows of one method / photo kind."""
    sel = [r for r in rows if r["method"] == method and (not kinds or r["kind"] in kinds)]
    subs = sorted({r["subject"] for r in sel})
    return {k: float(np.nanmean([np.nanmean([r[k] for r in sel if r["subject"] == s]) for s in subs]))
            for k in keys} if subs else {}


def subject_means(rows: list[dict], method: str, kinds: tuple, key: str) -> dict:
    sel = [r for r in rows if r["method"] == method and (not kinds or r["kind"] in kinds)]
    return {s: float(np.nanmean([r[key] for r in sel if r["subject"] == s])) for s in sorted({r["subject"] for r in sel})}


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def summary(rows: list[dict], gts: dict, methods: list[str]) -> list[dict]:
    out = []
    for group, kinds in (("all", ()), ("lab", ("lab",)), ("wild", ("wild",))):
        for m in methods:
            r = macro(rows, m, kinds, METRICS)
            if not r:
                continue
            row = {"method": m, "group": group, "n_subjects": len({x["subject"] for x in rows if x["method"] == m and (not kinds or x["kind"] in kinds)}), **r}
            for n in CIRC:
                pm = subject_means(rows, m, kinds, f"pred {n}")
                row[f"corr {n}"] = pearson([pm[s] for s in pm], [gts[s].measures[n] for s in pm])
            out.append(row)
    return out


def print_tables(summ: list[dict], rows: list[dict], gts: dict, methods: list[str]) -> None:
    print("\nScan bodies (landmark-free measurements, cm):  subject  height   chest   waist    hip")
    for s, g in sorted(gts.items()):
        print(f"{'':44s}{s:>7s}{g.H * 100:8.1f}{g.measures['chest']:8.1f}{g.measures['waist']:8.1f}{g.measures['hip']:8.1f}")
    for group in ("all", "lab", "wild"):
        sel = [r for r in summ if r["group"] == group]
        if not sel:
            continue
        print(f"\n=== {group} photos ({sel[0]['n_subjects']} subjects): error vs the scan, all bodies scaled to the scan height ===")
        print(f"{'method':22s}{'chest':>8s}{'waist':>8s}{'hip':>8s}{'surface mm':>12s}   corr (across subjects) chest/waist/hip")
        for r in sel:
            cs = "   ".join("  n/a" if np.isnan(r[f"corr {n}"]) else f"{r[f'corr {n}']:+.2f}" for n in CIRC)
            print(f"{r['method']:22s}{r['err chest']:8.2f}{r['err waist']:8.2f}{r['err hip']:8.2f}{r['surf_mm']:12.1f}   {cs}")
    print("\nPer subject, waist (cm), mean over all photos:  subject   scan " + "".join(f"{m[:10]:>12s}" for m in methods))
    per = {m: subject_means(rows, m, (), "pred waist") for m in methods}
    for s in sorted(gts):
        print(f"{'':48s}{s:>3s}{gts[s].measures['waist']:8.1f}" + "".join(f"{per[m].get(s, float('nan')):12.1f}" for m in methods))


def write_summary(summ: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "group", "n_subjects"] + METRICS + [f"corr {n}" for n in CIRC]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summ)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hbw", required=True)
    ap.add_argument("--anthro_root", required=True, help="SMPL-Anthropometry clone with data/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CSV")
    ap.add_argument("--n-samples", type=int, default=30000, help="surface samples per mesh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/hbw_shape.csv")
    args = ap.parse_args()

    specs = {}
    for spec in args.model:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model expects NAME=CSV, got {spec!r}")
        specs[name] = Path(path).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    hbw = Path(args.hbw).expanduser().resolve()
    anthro = Path(args.anthro_root).expanduser().resolve()
    per_model = {n: load_prediction_csv(p) for n, p in specs.items()}
    ids = sorted(set.intersection(*(set(d) for d in per_model.values())))
    print(f"{len(ids)} photo(s) predicted by every model, {len({per_model[next(iter(per_model))][i]['subject'] for i in ids})} subject(s)")

    from measure_body_error import build_measurer
    _, vertices, faces = build_measurer(anthro, ["height"])        # chdirs into the anthro folder
    pred_edges = unique_edges(np.asarray(faces))
    rng = np.random.default_rng(args.seed)

    gts = {}
    for sid in sorted({per_model[next(iter(per_model))][i]["subject"] for i in ids}):
        v = np.load(hbw / "smplx" / "val" / f"{sid}.npy")
        gts[sid] = build_gt(v, read_obj_faces(hbw / "smplx" / "val" / f"{sid}.obj"), rng, args.n_samples)
    print(f"{len(gts)} scan(s) loaded")

    methods = [BASELINE] + list(specs)
    rows = []
    base_cache = {}
    for n, i in enumerate(ids):
        first = per_model[next(iter(per_model))][i]
        sid, kind = first["subject"], first["kind"]
        for m in methods:
            betas = np.zeros(N_BETAS) if m == BASELINE else per_model[m][i]["pred"]
            if m == BASELINE and sid in base_cache:
                s = base_cache[sid]
            else:
                s = score_body(vertices(betas, "NEUTRAL"), faces, pred_edges, gts[sid], rng, args.n_samples)
                if m == BASELINE:
                    base_cache[sid] = s
            rows.append({"subject": sid, "kind": kind, "method": m, **s})
        if (n + 1) % 100 == 0:
            print(f"  scored {n + 1}/{len(ids)}", flush=True)

    summ = summary(rows, gts, methods)
    print_tables(summ, rows, gts, methods)
    write_summary(summ, out_path)


# ---------------------------------------------------------------- self-test
def toy_mesh(r_fn, H: float = 1.7, n_levels: int = 86, n_ring: int = 24):
    ys = np.linspace(0, H, n_levels)
    v = np.array([[r_fn(y / H) * np.cos(a), y, r_fn(y / H) * np.sin(a)]
                  for y in ys for a in np.linspace(0, 2 * np.pi, n_ring, endpoint=False)])
    f = []
    for i in range(n_levels - 1):
        for k in range(n_ring):
            a, b = i * n_ring + k, i * n_ring + (k + 1) % n_ring
            f += [[a, b, b + n_ring], [a, b + n_ring, a + n_ring]]
    return v, np.array(f)


def self_test() -> None:
    rng = np.random.default_rng(0)
    # plane section + perimeter on a cylinder of radius 0.15: 24-gon perimeter
    v, f = toy_mesh(lambda t: 0.15)
    e = unique_edges(f)
    poly = 2 * 24 * 0.15 * np.sin(np.pi / 24)
    assert abs(hull_perimeter(plane_xz(v, e, 0.91)) - poly) < 5e-3
    # arms are ignored: a spike far from the axis must not change the cross-section
    v2 = np.vstack([v, [[0.9, 0.9, 0.0]]])
    assert abs(hull_perimeter(plane_xz(v2, e, 0.91)) - poly) < 5e-3

    # band definitions: waist = narrowest in 55-68%, chest/hip = widest in their bands
    prof = lambda t: 0.15 - 0.03 * np.exp(-((t - 0.62) / 0.04) ** 2) + 0.02 * np.exp(-((t - 0.52) / 0.03) ** 2)  # noqa: E731
    v, f = toy_mesh(prof)
    e = unique_edges(f)
    m = measure_mesh(v, e)
    k = 2 * 24 * np.sin(np.pi / 24)
    assert abs(m["waist"] - k * (0.15 - 0.03) * 100) < 1.0, m           # narrowest at t = 0.62
    assert abs(m["hip"] - k * (0.15 + 0.02) * 100) < 1.0, m             # widest at t = 0.52
    assert m["chest"] > m["waist"]

    # normalise: scale to a target height, translation invariance
    base = normalize_mesh(v)
    shifted = normalize_mesh(v + np.array([5.0, 3.0, -2.0]))
    assert np.allclose(base, shifted, atol=1e-9)
    big = normalize_mesh(v * 1.2, target_height=base[:, 1].max())
    assert abs(big[:, 1].max() - base[:, 1].max()) < 1e-9

    # surface distance: identical mesh = 0; a body 2 cm fatter = about 20 mm; skinnier is symmetric
    gt = build_gt(v, f, rng, 20000)
    assert abs(gt.H - 1.7) < 1e-9
    same = score_body(v, f, e, gt, rng, 20000)
    assert same["surf_mm"] < 0.5 and max(same[f"err {n}"] for n in CIRC) < 1e-6, same
    fat_v, fat_f = toy_mesh(lambda t: prof(t) + 0.02)
    fat = score_body(fat_v, fat_f, unique_edges(fat_f), gt, rng, 20000)
    assert 15 < fat["surf_mm"] < 25, fat
    assert abs(fat["err waist"] - k * 0.02 * 100) < 1.0, fat
    # a uniformly scaled (taller) copy of the scan scores zero once scaled back to the scan height
    tall = score_body(v * (1.9 / 1.7), f, e, gt, rng, 20000)
    assert tall["surf_mm"] < 0.5 and max(tall[f"err {n}"] for n in CIRC) < 1e-6, tall

    # OBJ faces (1-based, optional /vt/vn)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "a.obj"
        p.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 2 3\nf 2//1 3//1 4//1\n")
        assert read_obj_faces(p).tolist() == [[0, 1, 2], [1, 2, 3]]

    # aggregation: per-subject mean first; correlation across subjects
    gts = {"a": gt, "b": gt}
    rows = []
    for sid, kinds in (("a", ["lab", "lab", "wild"]), ("b", ["lab"])):
        for kind in kinds:
            for meth, off in ((BASELINE, 10.0), ("good", 0.0)):
                rows.append({"subject": sid, "kind": kind, "method": meth, "err chest": off, "err waist": off,
                             "err hip": off, "surf_mm": off, "pred chest": 1.0, "pred waist": 2.0, "pred hip": 3.0})
    summ = summary(rows, gts, [BASELINE, "good"])
    s_all = {(r["method"], r["group"]): r for r in summ}
    assert s_all[("good", "all")]["err waist"] == 0.0 and s_all[(BASELINE, "lab")]["n_subjects"] == 2
    assert s_all[("good", "wild")]["n_subjects"] == 1
    assert np.isnan(s_all[("good", "all")]["corr waist"])              # 2 subjects: not enough points
    assert abs(pearson([1, 2, 3, 4], [2, 4, 6, 8]) - 1.0) < 1e-12 and np.isnan(pearson([1, 1, 1], [1, 2, 3]))
    print_tables(summ, rows, gts, [BASELINE, "good"])
    print("\n[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
