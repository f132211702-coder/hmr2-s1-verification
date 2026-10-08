#!/usr/bin/env python3
"""Figures for the report: SSP-3D photos, CameraHMR's mesh drawn on them, and a
front-view comparison of the real body shape against what each model predicted.

Selection is by a stated rule, not by eye: one photo per person, ranked by
CameraHMR's height-calibrated WAIST error against the pseudo-GT body, and per
gender the best, the median and the worst person are shown.

Three steps (the CameraHMR mesh overlay comes from CameraHMR's own demo.py, so no
renderer is re-implemented here and the picture is what the model really did):
  1. pick     scores every photo, picks 3 people per gender, copies their photos to
              <out-dir>/photos and writes <out-dir>/cases.csv (numbers for captions)
  2. (you)    run CameraHMR's demo.py on <out-dir>/photos  -> an overlay folder
  3. compose  builds <out-dir>/ssp3d_cases_female.png and ssp3d_cases_male.png:
              [photo crop] [CameraHMR overlay] [T-pose front view: real body as the grey
              filled shape; contours of CameraHMR (blue), HMR2.0b (green) and the
              average body (orange), all scaled to the real body's height]

Needs the same environment as eval_ssp3d_shape.py (torch, smplx, trimesh, plotly,
cv2). Labels in the images are English on purpose: OpenCV cannot draw CJK text.

Usage (self-test, numpy + cv2):
    python eval/make_ssp3d_figures.py --self-test

Usage:
    python eval/make_ssp3d_figures.py pick --anthro_root ~/SMPL-Anthropometry --ssp3d ~/datasets/SSP-3D \\
        --model HMR2.0b=results/ssp3d_hmr2.csv --model CameraHMR=results/ssp3d_camerahmr.csv \\
        --out-dir results/ssp3d_cases
    cd ~/workspace/dresson/CameraHMR && python demo.py --image_folder <out-dir>/photos --output_folder <out-dir>/camerahmr_demo
    python eval/make_ssp3d_figures.py compose --anthro_root ... --ssp3d ... --model ... --out-dir results/ssp3d_cases
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ssp3d_common import load_prediction_csv, load_ssp3d  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg"}
COLORS = {"CameraHMR": (220, 90, 20), "HMR2.0b": (60, 170, 60), "mean body": (0, 140, 255)}  # BGR
LABELS = ["best", "median", "worst"]
CELL = 420
PANEL_W = 260   # width of the shape panel (its height is CELL)


# ---------------------------------------------------------------- selection
def choose_cases(entries: list[dict]) -> list[dict]:
    """entries: [{fname, person_id, gender, abs_err}]. One photo per person (the first seen),
    then per gender the smallest, the median and the largest error."""
    first = {}
    for e in entries:
        first.setdefault(e["person_id"], e)
    out = []
    for gender in ("FEMALE", "MALE"):
        people = sorted((e for e in first.values() if e["gender"] == gender), key=lambda e: e["abs_err"])
        if not people:
            continue
        for label, e in zip(LABELS, (people[0], people[len(people) // 2], people[-1])):
            out.append({**e, "label": label})
    return out


# ---------------------------------------------------------------- silhouettes
def silhouette_masks(verts_by_name: dict, faces: np.ndarray, gt_name: str = "real body",
                     size=(PANEL_W, CELL)) -> dict:
    """Front-view (x right, y up) filled silhouettes on a shared canvas. Every mesh is uniformly
    scaled to the GT mesh's height with its feet on a common baseline and its x-centre on the
    canvas centre."""
    import cv2
    W, H = size
    gt = verts_by_name[gt_name]
    gt_h = gt[:, 1].max() - gt[:, 1].min()
    px_per_m = (H - 60) / gt_h
    masks = {}
    for name, v in verts_by_name.items():
        h = v[:, 1].max() - v[:, 1].min()
        f = gt_h / h
        x = (v[:, 0] - v[:, 0].mean()) * f * px_per_m + W / 2
        y = (H - 25) - (v[:, 1] - v[:, 1].min()) * f * px_per_m
        tri = np.round(np.stack([x, y], axis=1)[faces]).astype(np.int32)
        m = np.zeros((H, W), np.uint8)
        for t in tri:
            cv2.fillConvexPoly(m, t, 255)
        masks[name] = m
    return masks


def shape_panel(masks: dict, gt_name: str = "real body") -> np.ndarray:
    import cv2
    H, W = masks[gt_name].shape
    img = np.full((H, W, 3), 255, np.uint8)
    img[masks[gt_name] > 0] = (205, 205, 205)
    for name, m in masks.items():
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (90, 90, 90) if name == gt_name else COLORS[name], 2)
    y = 22
    for name in [gt_name] + [n for n in masks if n != gt_name]:
        col = (90, 90, 90) if name == gt_name else COLORS[name]
        cv2.rectangle(img, (8, y - 11), (24, y + 1), col, -1)
        cv2.putText(img, name, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        y += 20
    return img


# ---------------------------------------------------------------- composition
def find_overlay(overlay_dir: Path, stem: str) -> Path | None:
    cands = sorted(p for p in Path(overlay_dir).rglob("*") if p.suffix.lower() in IMG_EXT and stem in p.name)
    for key in ("overlay", "mesh", "render"):
        for p in cands:
            if key in p.name.lower():
                return p
    return cands[0] if cands else None


def fit_cell(img: np.ndarray | None, text: str = "") -> np.ndarray:
    """Letterbox into a CELL x CELL white tile (or a grey placeholder when img is None)."""
    import cv2
    tile = np.full((CELL, CELL, 3), 255 if img is not None else 235, np.uint8)
    if img is None:
        cv2.putText(tile, text or "missing", (20, CELL // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1, cv2.LINE_AA)
        return tile
    h, w = img.shape[:2]
    s = CELL / max(h, w)
    r = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    y0, x0 = (CELL - r.shape[0]) // 2, (CELL - r.shape[1]) // 2
    tile[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return tile


def crop_to_bbox(img: np.ndarray, bbox) -> np.ndarray:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    return img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]


def case_row(photo, overlay, shapes, header: str) -> np.ndarray:
    import cv2
    row = np.hstack([fit_cell(photo, "photo missing"), fit_cell(overlay, "demo overlay missing"), shapes])
    bar = np.full((34, row.shape[1], 3), 245, np.uint8)
    cv2.putText(bar, header, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    return np.vstack([bar, row])


# ---------------------------------------------------------------- CLI
def parse_models(specs):
    out = {}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not path:
            raise SystemExit(f"--model expects NAME=CSV, got {spec!r}")
        out[name] = Path(path).expanduser().resolve()
    return out


def setup_tool(args):
    from eval_ssp3d_shape import candidate_scores  # noqa: E402
    from fit_betas_from_measurements import with_weight  # noqa: E402
    from measure_body_error import build_measurer  # noqa: E402
    anthro = Path(args.anthro_root).expanduser().resolve()
    measure_tool, vertices, faces = build_measurer(anthro, ["height", "waist circumference", "chest circumference"])
    return with_weight(measure_tool, vertices, faces), vertices, faces, candidate_scores


def cmd_pick(args) -> None:
    models = parse_models(args.model)
    out_dir = Path(args.out_dir).expanduser().resolve()
    ssp3d = Path(args.ssp3d).expanduser().resolve()
    per_model = {n: load_prediction_csv(p) for n, p in models.items()}
    if args.rank_model not in per_model:
        raise SystemExit(f"--rank-model {args.rank_model} is not among --model names {list(per_model)}")
    common = sorted(set.intersection(*(set(d) for d in per_model.values())))
    first = per_model[args.rank_model]
    measure, vertices, _, candidate_scores = setup_tool(args)

    scored = {}
    for fname in common:
        gender, gt_b = first[fname]["gender"], first[fname]["gt"]
        gt_m, gt_v = measure(gt_b, gender), vertices(gt_b, gender)
        sources = {n: d[fname]["pred"] for n, d in per_model.items()}
        sources["mean body"] = np.zeros(len(gt_b))
        scored[fname] = {"gt": gt_m}
        for n, b in sources.items():
            _, cal = candidate_scores(measure, vertices, b, "NEUTRAL", gt_m, gt_v)
            scored[fname][n] = cal
    entries = [{"fname": f, "person_id": first[f]["person_id"], "gender": first[f]["gender"],
                "abs_err": scored[f][args.rank_model]["err waist circumference"]} for f in common]
    cases = choose_cases(entries)

    (out_dir / "photos").mkdir(parents=True, exist_ok=True)
    rows = []
    for c in cases:
        s = scored[c["fname"]]
        row = {"gender": c["gender"], "label": c["label"], "fname": c["fname"], "person_id": c["person_id"],
               "gt waist": s["gt"]["waist circumference"], "gt chest": s["gt"]["chest circumference"]}
        for n in list(models) + ["mean body"]:
            row[f"{n} waist"] = s[n]["pred waist circumference"]
            row[f"{n} chest"] = s[n]["pred chest circumference"]
        rows.append(row)
        shutil.copy(ssp3d / "ssp_3d" / "images" / c["fname"], out_dir / "photos" / c["fname"])
    with open(out_dir / "cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_dir / 'cases.csv'} and {len(rows)} photos in {out_dir / 'photos'}\n")
    print(f"{'gender':8s}{'case':8s}{'GT waist':>10s}" + "".join(f"{n[:10]:>12s}" for n in list(models) + ['mean body']))
    for r in rows:
        print(f"{r['gender']:8s}{r['label']:8s}{r['gt waist']:10.1f}"
              + "".join(f"{r[f'{n} waist']:12.1f}" for n in list(models) + ["mean body"]))
    print("\nnext: run CameraHMR's demo.py on the photos folder, then the compose step.")


def cmd_compose(args) -> None:
    import cv2
    models = parse_models(args.model)
    out_dir = Path(args.out_dir).expanduser().resolve()
    ssp3d = Path(args.ssp3d).expanduser().resolve()
    overlay_dir = Path(args.overlay_dir).expanduser().resolve() if args.overlay_dir else out_dir / "camerahmr_demo"
    per_model = {n: load_prediction_csv(p) for n, p in models.items()}
    records = {r.fname: r for r in load_ssp3d(ssp3d)}
    with open(out_dir / "cases.csv", newline="") as f:
        cases = list(csv.DictReader(f))
    _, vertices, faces, _ = setup_tool(args)

    rows_by_gender: dict[str, list] = {}
    for c in cases:
        fname, gender = c["fname"], c["gender"]
        rec = records[fname]
        verts = {"real body": vertices(per_model[next(iter(per_model))][fname]["gt"], gender)}
        for n, d in per_model.items():
            verts[n] = vertices(d[fname]["pred"], "NEUTRAL")
        verts["mean body"] = vertices(np.zeros(10), "NEUTRAL")
        shapes = shape_panel(silhouette_masks(verts, faces))

        photo = cv2.imread(str(rec.img_path))
        photo_crop = crop_to_bbox(photo, rec.bbox) if photo is not None else None
        ov_path = find_overlay(overlay_dir, Path(fname).stem)
        overlay = cv2.imread(str(ov_path)) if ov_path else None
        if overlay is not None and photo is not None and overlay.shape[:2] == photo.shape[:2]:
            overlay = crop_to_bbox(overlay, rec.bbox)
        print(f"{fname}: overlay {'-> ' + ov_path.name if ov_path else 'NOT FOUND in ' + str(overlay_dir)}")
        head = f"{gender.lower()} / {c['label']} (by CameraHMR waist error)  waist cm: real {float(c['gt waist']):.0f}" + "".join(
            f"  {n} {float(c[f'{n} waist']):.0f}" for n in list(models) + ["mean body"])
        rows_by_gender.setdefault(gender, []).append(case_row(photo_crop, overlay, shapes, head))
    for gender, rows in rows_by_gender.items():
        path = out_dir / f"ssp3d_cases_{gender.lower()}.png"
        cv2.imwrite(str(path), np.vstack(rows))
        print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pick", "compose"])
    ap.add_argument("--anthro_root", required=True)
    ap.add_argument("--ssp3d", required=True)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CSV")
    ap.add_argument("--rank-model", default="CameraHMR")
    ap.add_argument("--out-dir", default="results/ssp3d_cases")
    ap.add_argument("--overlay-dir", default=None, help="CameraHMR demo output folder (default <out-dir>/camerahmr_demo)")
    args = ap.parse_args()
    {"pick": cmd_pick, "compose": cmd_compose}[args.cmd](args)


def self_test() -> None:
    import tempfile

    import cv2

    # choose_cases: one photo per person, best / median / worst per gender
    entries = [{"fname": f"f{i}_{k}", "person_id": f"f{i}", "gender": "FEMALE", "abs_err": float(i)} for i in range(5) for k in (0, 1)]
    entries += [{"fname": f"m{i}", "person_id": f"m{i}", "gender": "MALE", "abs_err": 10.0 - i} for i in range(4)]
    cases = choose_cases(entries)
    assert [(c["gender"], c["label"], c["person_id"]) for c in cases] == [
        ("FEMALE", "best", "f0"), ("FEMALE", "median", "f2"), ("FEMALE", "worst", "f4"),
        ("MALE", "best", "m3"), ("MALE", "median", "m1"), ("MALE", "worst", "m0")], cases
    assert all(c["fname"].endswith("_0") for c in cases if c["gender"] == "FEMALE"), "first photo per person"

    # silhouettes: same height after scaling, narrower body has the smaller area
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    box = lambda w, h: np.array([[-w, 0, 0], [w, 0, 0], [w, h, 0], [-w, h, 0]], float)  # noqa: E731
    verts = {"real body": box(0.3, 1.7), "CameraHMR": box(0.2, 1.5), "HMR2.0b": box(0.3, 1.9), "mean body": box(0.25, 1.7)}
    masks = silhouette_masks(verts, faces)
    rows_of = {n: np.where(m.any(axis=1))[0] for n, m in masks.items()}
    ext = {n: r.max() - r.min() for n, r in rows_of.items()}
    assert max(ext.values()) - min(ext.values()) <= 2, ext
    assert masks["CameraHMR"].sum() < masks["mean body"].sum() < masks["real body"].sum()
    panel = shape_panel(masks)
    assert panel.shape == (CELL, PANEL_W, 3) and panel.min() < 255

    # find_overlay prefers a file whose name says overlay; none -> None
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for name in ("a_001.png", "a_001_overlay.png", "b_002.png"):
            cv2.imwrite(str(d / name), np.zeros((4, 4, 3), np.uint8))
        assert find_overlay(d, "a_001").name == "a_001_overlay.png"
        assert find_overlay(d, "b_002").name == "b_002.png"
        assert find_overlay(d, "zzz") is None

    # a row: three tiles wide plus the header bar; missing images become placeholders
    row = case_row(np.full((300, 200, 3), 120, np.uint8), None, panel, "header")
    assert row.shape == (34 + CELL, 2 * CELL + PANEL_W, 3)
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
