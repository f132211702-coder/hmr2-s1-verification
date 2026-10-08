#!/usr/bin/env python3
"""Inspect the BodyM tables downloaded by fetch_bodym.py: what columns exist,
how many subjects, the gender split, the distribution of height / weight /
chest / waist / hip, and how well height + weight alone already predict the
circumferences on REAL people -- the number that decides whether asking users
for their weight is worth it (our simulation used a circular weight computed
from a mesh; BodyM has real ones).

Column names are matched by substring (height, weight, chest, waist, hip,
gender/sex) because the real header is not known in advance; the script
prints every header it finds so a wrong match is visible.

Usage (self-test, numpy only):
    python data_prep/inspect_bodym.py --self-test

Usage:
    python data_prep/inspect_bodym.py --dir ~/datasets/BodyM
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

WANTED = ["height", "weight", "chest", "waist", "hip"]


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        delim = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(f, delimiter=delim)
        return list(reader.fieldnames or []), list(reader)


def find_column(header: list[str], word: str) -> str | None:
    """First column whose name contains `word`; prefers one that also says
    'girth' or 'circ' for body-part words, and avoids 'length'/'to' columns."""
    cands = [h for h in header if word in h.lower() and "length" not in h.lower() and "_to_" not in h.lower()]
    cands.sort(key=lambda h: (0 if any(t in h.lower() for t in ("girth", "circ")) else 1, len(h)))
    return cands[0] if cands else None


def to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def column_map(header: list[str]) -> dict:
    cols = {w: find_column(header, w) for w in WANTED}
    cols["gender"] = next((h for h in header if h.lower() in ("gender", "sex")), None) \
        or next((h for h in header if "gender" in h.lower() or h.lower() == "sex"), None)
    return cols


def describe_table(path: Path) -> list[str]:
    header, rows = read_csv(path)
    lines = [f"{path}  ({len(rows)} rows)", f"  columns: {header}"]
    for r in rows[:2]:
        lines.append("  " + str({k: r[k] for k in header[:8]}))
    return lines


def fit(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares fit y ~ X + const -> (R^2, residual std)."""
    A = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return r2, float(res.std())


def analyse(header: list[str], rows: list[dict]) -> list[str]:
    cols = column_map(header)
    lines = [f"  matched columns: {cols}"]
    missing = [w for w in WANTED if cols[w] is None]
    if missing:
        return lines + [f"  no column matched for {missing} -- this file is not the measurement table "
                        f"(or the header uses other words; tell me the header above)."]
    data = np.array([[to_float(r[cols[w]]) for w in WANTED] for r in rows])
    gender = [str(r[cols["gender"]]).strip().lower() if cols["gender"] else "all" for r in rows]
    ok = ~np.isnan(data).any(axis=1)
    lines.append(f"  usable rows (all five values present): {int(ok.sum())} of {len(rows)}")
    lines.append(f"  gender values: { {g: gender.count(g) for g in sorted(set(gender))} }")
    for g in sorted(set(gender)):
        sel = ok & np.array([x == g for x in gender])
        if sel.sum() < 20:
            continue
        d = data[sel]
        lines.append(f"\n  [{g}] {int(sel.sum())} rows -- mean (std):  " + "   ".join(
            f"{w} {d[:, i].mean():.1f} ({d[:, i].std():.1f})" for i, w in enumerate(WANTED)))
        hw = d[:, :2]
        for i, w in enumerate(WANTED[2:], start=2):
            r2_h, sd_h = fit(hw[:, :1], d[:, i])
            r2_hw, sd_hw = fit(hw, d[:, i])
            lines.append(f"    {w:6s}: height alone R^2 {r2_h:.2f} (residual {sd_h:.1f} cm)   "
                         f"height+weight R^2 {r2_hw:.2f} (residual {sd_hw:.1f} cm)")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    root = Path(args.dir).expanduser()
    tables = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".csv", ".tsv"))
    if not tables:
        raise SystemExit(f"no csv/tsv under {root} -- run fetch_bodym.py --download first")
    for p in tables:
        print("\n".join(describe_table(p)))
        header, rows = read_csv(p)
        print("\n".join(analyse(header, rows)))
        print()


def self_test() -> None:
    rng = np.random.default_rng(0)
    n = 200
    height = rng.normal(170, 8, n)
    weight = 22 * (height / 100) ** 2 * rng.normal(1, 0.15, n)
    waist = 40 + 0.55 * weight + 0.1 * height + rng.normal(0, 2, n)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "subject_measurements.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject_id", "gender", "height_cm", "weight_kg", "chest_girth", "waist_girth",
                        "hip_girth", "arm-length"])
            for i in range(n):
                w.writerow([i, "male" if i % 2 else "female", height[i], weight[i],
                            60 + 0.6 * weight[i], waist[i], 50 + 0.7 * weight[i], 60])
            w.writerow([n, "male", "", "", "", "", "", ""])   # a row with missing values
        header, rows = read_csv(p)
        cols = column_map(header)
        assert cols == {"height": "height_cm", "weight": "weight_kg", "chest": "chest_girth",
                        "waist": "waist_girth", "hip": "hip_girth", "gender": "gender"}, cols
        text = "\n".join(analyse(header, rows))
        assert "usable rows (all five values present): 200 of 201" in text, text
        assert "[male]" in text and "[female]" in text
        # weight is built into waist, so height+weight must explain far more than height alone
        line = next(ln for ln in text.splitlines() if ln.strip().startswith("waist"))
        r2_h = float(line.split("height alone R^2")[1].split()[0])
        r2_hw = float(line.split("height+weight R^2")[1].split()[0])
        assert r2_hw > r2_h + 0.3 and r2_hw > 0.8, line
        (Path(d) / "other.csv").write_text("a,b\n1,2\n")
        h2, r2 = read_csv(Path(d) / "other.csv")
        assert "no column matched" in "\n".join(analyse(h2, r2))
    print("[self-test] all checks passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
