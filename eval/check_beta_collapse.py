#!/usr/bin/env python3
"""Test whether HMR2's predicted betas actually track each person's real
body shape, or collapse toward a near-constant "average" shape regardless
of who's in the image.

Motivation: verify_neutral_gender_bias.py found beta_l2_mean and
gt_neutral_dist (= ||gt_betas||, how far the real person's shape sits from
the neutral zero-shape average) not just correlated (r=0.99) but close in
*magnitude* on 3DPW. Since ||pred - gt|| ~= ||gt|| only holds when pred is
small, that's consistent with HMR2 predicting betas close to zero -- i.e.
close to the average shape -- for every person, rather than actually
tracking individual body shape. This script checks that directly, using
only the predictions (results/s1_raw_3dpw/*.npz), no GT betas needed for
the core check:
    1. How big is ||pred_betas|| itself, per real person?
    2. How much does ||pred_betas|| vary ACROSS different real people,
       compared to how much ||gt_betas|| varies across those same people?
       If HMR2 tracked real shape, these two spreads should be comparable.
       If HMR2 collapses toward average, pred's spread should be much
       smaller than GT's.
    3. Do different people's mean pred_betas vectors all point in roughly
       the same direction (high pairwise cosine similarity)? That would
       mean HMR2 isn't just predicting a similarly-sized correction, it's
       predicting nearly the *same* specific shape for everyone.

Status: the grouping/stats logic is validated against synthetic data (see
--self-test), covering both a "collapsed" and a "shape-tracking" synthetic
case so the script can't just always print the same conclusion regardless
of input.

Usage (self-test, no real data needed):
    python eval/check_beta_collapse.py --self-test

Usage (real check):
    python eval/check_beta_collapse.py \\
        --pred_dir results/s1_raw_3dpw \\
        --gt_dir /home/intern/datasets/3DPW/sequenceFiles/test
"""
from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_against_gt import load_predictions  # noqa: E402
from verify_neutral_gender_bias import load_gt_baseline  # noqa: E402


def summarize_predictions(pred_by_image: dict) -> pd.DataFrame:
    """One row per (sequence, person_id): how big is this person's
    predicted betas on average, how consistent is it across their frames,
    and what's the mean predicted betas *vector* (needed for the
    cosine-similarity check across people)."""
    rows = []
    by_person: dict[tuple[str, int], list[np.ndarray]] = {}
    for image_id, records in pred_by_image.items():
        sequence = image_id.split("__image_")[0]
        for r in records:
            by_person.setdefault((sequence, r.person_id), []).append(r.betas)

    for (sequence, person_id), betas_list in by_person.items():
        betas_arr = np.stack(betas_list)  # (n_frames, 10)
        norms = np.linalg.norm(betas_arr, axis=1)
        rows.append({
            "sequence": sequence,
            "person_id": person_id,
            "n_frames": len(betas_list),
            "pred_beta_norm_mean": float(norms.mean()),
            "pred_beta_norm_std": float(norms.std()),
            "pred_beta_mean_vec": betas_arr.mean(axis=0),
        })
    return pd.DataFrame(rows)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def analyze(pred_summary: pd.DataFrame, gt_baseline) -> tuple[pd.DataFrame, float]:
    """Pure computation, kept separate from file I/O so --self-test can
    exercise it with synthetic data. Returns (merged table, mean pairwise
    cosine similarity between different people's mean predicted-beta
    vectors)."""
    merged = pred_summary.merge(gt_baseline, on=["sequence", "person_id"], how="inner")

    vecs = list(merged["pred_beta_mean_vec"])
    if len(vecs) >= 2:
        sims = [cosine_sim(a, b) for a, b in combinations(vecs, 2)]
        mean_pairwise_cosine = float(np.mean(sims))
    else:
        mean_pairwise_cosine = float("nan")

    return merged, mean_pairwise_cosine


def self_test() -> None:
    """No real data needed. Builds two synthetic scenarios and checks the
    stats tell them apart -- if this script always reported "collapsed"
    regardless of input, that would be a bug, not a finding."""
    rng = np.random.default_rng(0)
    n_people = 6
    gt_norms = np.linspace(0.5, 4.0, n_people)  # 6 people, deliberately varied true shape
    sequences = [f"seq{i}" for i in range(n_people)]

    def make_pred_summary(pred_norms: np.ndarray, shared_direction: bool) -> pd.DataFrame:
        base_dir = rng.normal(size=10)
        base_dir /= np.linalg.norm(base_dir)
        rows = []
        for i, norm in enumerate(pred_norms):
            if shared_direction:
                vec = base_dir * norm
            else:
                d = rng.normal(size=10)
                d /= np.linalg.norm(d)
                vec = d * norm
            rows.append({"sequence": sequences[i], "person_id": 0, "n_frames": 50,
                         "pred_beta_norm_mean": norm, "pred_beta_norm_std": 0.05,
                         "pred_beta_mean_vec": vec})
        return pd.DataFrame(rows)

    gt_baseline = pd.DataFrame({
        "sequence": sequences, "person_id": [0] * n_people,
        "gender": ["m"] * n_people, "gt_neutral_dist": gt_norms,
    })

    # Scenario A: "collapsed" -- pred norm barely varies across people and
    # all point the same direction, regardless of how much gt varies.
    collapsed_pred = make_pred_summary(np.full(n_people, 0.3) + rng.normal(0, 0.02, n_people),
                                        shared_direction=True)
    merged_a, cosine_a = analyze(collapsed_pred, gt_baseline)
    spread_ratio_a = merged_a["pred_beta_norm_mean"].std() / merged_a["gt_neutral_dist"].std()
    print(f"[self-test] collapsed scenario: pred/gt spread ratio={spread_ratio_a:.4f} "
          f"(should be small), mean pairwise cosine={cosine_a:.4f} (should be high)")
    assert spread_ratio_a < 0.2, "collapsed scenario should show much less spread in pred than gt"
    assert cosine_a > 0.9, "collapsed scenario's predictions should all point the same direction"

    # Scenario B: "tracking" -- pred norm scales with gt (with noise) and
    # directions are independent per person.
    tracking_pred = make_pred_summary(gt_norms * 0.8 + rng.normal(0, 0.1, n_people),
                                       shared_direction=False)
    merged_b, cosine_b = analyze(tracking_pred, gt_baseline)
    spread_ratio_b = merged_b["pred_beta_norm_mean"].std() / merged_b["gt_neutral_dist"].std()
    corr_b = merged_b[["pred_beta_norm_mean", "gt_neutral_dist"]].corr().iloc[0, 1]
    print(f"[self-test] tracking scenario: pred/gt spread ratio={spread_ratio_b:.4f} "
          f"(should be ~0.8), corr={corr_b:.4f} (should be high), "
          f"mean pairwise cosine={cosine_b:.4f} (should be low/near 0)")
    assert spread_ratio_b > 0.5, "tracking scenario should show comparable spread to gt"
    assert corr_b > 0.9, "tracking scenario was constructed to correlate with gt"
    assert abs(cosine_b) < 0.5, "tracking scenario's directions were constructed to be independent"

    print("[self-test] all checks passed -- the stats correctly distinguish "
          "a collapsed model from one that actually tracks shape.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                     help="validate the stats against synthetic collapsed/tracking scenarios; needs no real data")
    ap.add_argument("--pred_dir", type=str, help="s1_infer.py's --out folder for the 3DPW run")
    ap.add_argument("--gt_dir", type=str, help="3DPW's sequenceFiles/test (or train/validation)")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not (args.pred_dir and args.gt_dir):
        raise SystemExit("A real check needs --pred_dir and --gt_dir, or use --self-test first.")

    pred_by_image = load_predictions(Path(args.pred_dir))
    pred_summary = summarize_predictions(pred_by_image)
    gt_baseline = load_gt_baseline(Path(args.gt_dir))
    merged, mean_pairwise_cosine = analyze(pred_summary, gt_baseline)

    print(f"matched {len(merged)}/{len(pred_summary)} (sequence, person_id) pairs against GT betas")
    print()
    print("=== per (sequence, person), sorted by gt_neutral_dist, descending ===")
    cols = ["sequence", "person_id", "n_frames", "pred_beta_norm_mean",
            "pred_beta_norm_std", "gt_neutral_dist"]
    print(merged[cols].sort_values("gt_neutral_dist", ascending=False).to_string(index=False))
    print()

    pred_std = merged["pred_beta_norm_mean"].std()
    gt_std = merged["gt_neutral_dist"].std()
    print(f"spread of pred_beta_norm_mean across different real people: std={pred_std:.4f}")
    print(f"spread of gt_neutral_dist across those same people:         std={gt_std:.4f}")
    print(f"ratio (pred spread / gt spread): {pred_std / gt_std:.4f}  "
          f"(near 0 => HMR2 barely differentiates people's shape; near 1 => it tracks real spread)")
    print()
    print("=== correlation: pred_beta_norm_mean vs gt_neutral_dist ===")
    print(merged[["pred_beta_norm_mean", "gt_neutral_dist"]].corr())
    print()
    print(f"mean pairwise cosine similarity between different people's mean predicted-beta vectors: "
          f"{mean_pairwise_cosine:.4f}")
    print("(near 1 => HMR2 predicts nearly the same specific shape for everyone, "
          "regardless of who they are; near 0 => directions vary independently per person)")


if __name__ == "__main__":
    main()
