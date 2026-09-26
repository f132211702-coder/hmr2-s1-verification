#!/usr/bin/env python3
"""Check whether the beta_l2 shape-error numbers from eval_against_gt.py
mostly reflect HMR2's estimation skill, or are dominated by a structural
gap between the neutral-SMPL space HMR2 predicts in and the gendered-SMPL
space 3DPW's ground truth uses.

Motivation: a real 3DPW-TEST run (results/eval_3dpw.csv, 2026-09-25) showed
beta_l2 clustered tightly per sequence (each 3DPW sequence has ~1-2 real
people), with office_phoneCall_00 sitting far below every downtown_*
sequence (1.50 vs 2.6-3.1) despite unremarkable pose (pa_mpjpe) error.
Since HMR2 only ever outputs neutral-SMPL betas, but 3DPW's GT betas are
gendered, a person whose *true* gendered shape happens to sit close to the
neutral model's zero-shape average gets an unearned head start (small
beta_l2) regardless of how well HMR2 actually "sees" their body -- and the
reverse for someone far from average. This script tests that hypothesis
directly, with no HMR2 prediction involved on the GT side at all: for each
(sequence, person), compute how far their GT betas already sit from the
neutral zero-shape baseline (gt_neutral_dist = ||gt_betas[:10]||), and
check whether that correlates with the beta_l2 eval_against_gt.py actually
measured. A strong positive correlation would mean: yes, a meaningful
chunk of "beta error" is really "how unusual this particular person's body
is relative to the neutral average", not HMR2's estimation error.

Status: the merge/correlation logic is validated against synthetic data
(see --self-test). The 3DPW pkl's `genders` field is assumed present per
the standard 3DPW format (list of 'm'/'f' per person) but NOT yet
confirmed against a real file -- if it's missing, gender-based grouping is
silently skipped rather than failing.

Usage (self-test, no real data needed):
    python eval/verify_neutral_gender_bias.py --self-test

Usage (real check, needs eval_against_gt.py's 3DPW output + the GT pkls):
    python eval/verify_neutral_gender_bias.py \\
        --eval_csv results/eval_3dpw.csv \\
        --gt_dir /path/to/3DPW/sequenceFiles/test
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_gt_baseline(gt_dir: Path) -> pd.DataFrame:
    """For every (sequence, person_id) in the 3DPW GT, return their gender
    (if available) and how far their true betas (first 10 dims, matching
    HMR2's output size -- same truncation eval_against_gt.py's
    load_3dpw_gt() uses) sit from the neutral model's zero-shape baseline.
    No HMR2 prediction is read here at all -- this is purely a property of
    the GT data."""
    rows = []
    for pkl_path in sorted(gt_dir.glob("*.pkl")):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        seq_name = pkl_path.stem
        genders = data.get("genders")
        for person_id, betas300 in enumerate(data["betas"]):
            betas10 = np.asarray(betas300[:10], dtype=np.float64)
            rows.append({
                "sequence": seq_name,
                "person_id": person_id,
                "gender": genders[person_id] if genders is not None else None,
                "gt_neutral_dist": float(np.linalg.norm(betas10)),
            })
    return pd.DataFrame(rows)


def summarize_per_person(eval_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse eval_against_gt.py's per-frame CSV rows down to one row per
    (sequence, person_id), since a person's GT betas are constant across
    their whole sequence -- comparing per-frame would just repeat the same
    GT point many times."""
    df = eval_df[eval_df["status"] == "ok"].copy()
    df["sequence"] = df["image_id"].str.split("__image_").str[0]
    return df.groupby(["sequence", "person_id"]).agg(
        n_frames=("beta_l2", "size"),
        beta_l2_mean=("beta_l2", "mean"),
        pa_mpjpe_mean=("pa_mpjpe_mm", "mean"),
    ).reset_index()


def analyze(per_person: pd.DataFrame, gt_baseline: pd.DataFrame) -> pd.DataFrame:
    """Pure merge + join step, kept separate from file I/O so --self-test
    can exercise it with synthetic data."""
    return per_person.merge(gt_baseline, on=["sequence", "person_id"], how="inner")


def self_test() -> None:
    """No real 3DPW data needed. Builds two small synthetic tables with a
    known, perfect linear relationship between gt_neutral_dist and
    beta_l2_mean, and checks analyze() recovers that relationship --
    validates the merge key logic and the correlation math, independent of
    whatever the real correlation turns out to be."""
    per_person = pd.DataFrame({
        "sequence": ["seqA", "seqA", "seqB", "seqC"],
        "person_id": [0, 1, 0, 0],
        "n_frames": [100, 100, 50, 80],
        "beta_l2_mean": [1.0, 2.0, 3.0, 4.0],
        "pa_mpjpe_mean": [70.0, 71.0, 69.0, 72.0],  # deliberately unrelated to beta_l2_mean
    })
    gt_baseline = pd.DataFrame({
        "sequence": ["seqA", "seqA", "seqB", "seqC"],
        "person_id": [0, 1, 0, 0],
        "gender": ["f", "m", "f", "m"],
        "gt_neutral_dist": [1.0, 2.0, 3.0, 4.0],  # == beta_l2_mean, by construction
    })

    merged = analyze(per_person, gt_baseline)
    assert len(merged) == 4, f"expected all 4 rows to merge, got {len(merged)}"

    corr = merged[["beta_l2_mean", "gt_neutral_dist"]].corr().iloc[0, 1]
    print(f"[self-test] beta_l2_mean vs gt_neutral_dist correlation: {corr:.6f} (should be ~1.0)")
    assert corr > 0.999, "constructed a perfect linear relationship; correlation math is wrong"

    corr_unrelated = merged[["beta_l2_mean", "pa_mpjpe_mean"]].corr().iloc[0, 1]
    print(f"[self-test] beta_l2_mean vs pa_mpjpe_mean correlation: {corr_unrelated:.6f} (unrelated by construction)")
    assert abs(corr_unrelated) < 0.9, "columns were constructed to NOT be strongly correlated"

    print("[self-test] all checks passed.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                     help="validate the merge/correlation logic against synthetic data; needs no real dataset")
    ap.add_argument("--eval_csv", type=str, help="eval_against_gt.py's 3DPW output CSV")
    ap.add_argument("--gt_dir", type=str, help="3DPW's sequenceFiles/test (or train/validation)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not (args.eval_csv and args.gt_dir):
        raise SystemExit("A real check needs --eval_csv and --gt_dir, or use --self-test first.")

    eval_df = pd.read_csv(args.eval_csv)
    per_person = summarize_per_person(eval_df)
    gt_baseline = load_gt_baseline(Path(args.gt_dir))
    merged = analyze(per_person, gt_baseline)

    print(f"matched {len(merged)}/{len(per_person)} (sequence, person_id) pairs against GT betas")
    print()
    print("=== per (sequence, person), sorted by beta_l2_mean, descending ===")
    print(merged.sort_values("beta_l2_mean", ascending=False).to_string(index=False))
    print()
    print("=== correlation: beta_l2_mean vs gt_neutral_dist (THE hypothesis test) ===")
    print(merged[["beta_l2_mean", "gt_neutral_dist"]].corr())
    print()
    print("=== correlation: beta_l2_mean vs pa_mpjpe_mean (sanity check -- should stay near 0) ===")
    print(merged[["beta_l2_mean", "pa_mpjpe_mean"]].corr())
    if merged["gender"].notna().any():
        print()
        print("=== beta_l2_mean / gt_neutral_dist, grouped by gender ===")
        print(merged.groupby("gender")[["beta_l2_mean", "gt_neutral_dist"]].agg(["mean", "count"]))
    else:
        print()
        print("(no `genders` field found in the GT pkls -- skipped gender grouping)")


if __name__ == "__main__":
    main()
