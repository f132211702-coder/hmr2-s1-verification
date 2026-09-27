#!/usr/bin/env python3
"""Before committing to training a shape-correction model on top of HMR2:
does the feature HMR2's own shape head reads from actually carry
recoverable, person-specific shape signal, or is the signal already gone
by that point in the network?

check_beta_collapse.py / verify_neutral_gender_bias.py showed HMR2's final
betas output barely varies between real people and doesn't correlate with
their true shape. That's a fact about the OUTPUT. This script probes one
layer earlier: `smpl_head.decshape` is a single nn.Linear(1024, 10) that
reads a 1024-dim feature (`token_out` in smpl_head.py -- the transformer
decoder's output, shared by the pose/shape/cam heads) and was itself
trained to predict shape from it. If a freshly-fit probe on that same
feature, evaluated on people it never saw, still can't predict real shape
above a trivial baseline, training a bigger/better head on top of this
FROZEN feature won't help either -- the signal has to come from somewhere
earlier (or a retrained backbone), which is a different, larger project.
If the probe CAN recover signal the existing linear decshape head isn't
using, that's a concrete, much smaller "train a head" case to pursue.

Method: hook decshape's input for each (person, frame). Group by real
subject identity (exact GT betas match -- 3DPW reuses few actors across
many sequences). Leave-one-subject-out ridge regression: fit on all other
subjects' frames, predict the held-out subject's shape by averaging
predictions over their frames, compare to a trivial baseline that ignores
the image entirely (predicts the training subjects' mean shape).

Known limitation, stated up front: 3DPW's test split has only a handful of
distinct real actors (this script reports exactly how many it found). A
leave-one-out test across that few subjects has very low statistical
power -- this is a quick, cheap first check, not a definitive answer
either way. Read the result as "worth a bigger investment" or "probably
not", not as proof.

Status: the ridge-regression/leave-one-out bookkeeping is validated
against synthetic data (see --self-test), covering a "signal present" and
a "no signal" synthetic case. Not yet run against a real HMR2 feature.

Usage (self-test, no model/data needed):
    python eval/probe_shape_features.py --self-test

Usage (real check, GPU server):
    python eval/probe_shape_features.py \\
        --img_root /home/intern/datasets/3DPW/imageFiles \\
        --gt_dir /home/intern/datasets/3DPW/sequenceFiles/test \\
        --stride 10 --out results/shape_features.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def ridge_fit_predict(X_train: np.ndarray, Y_train: np.ndarray,
                      X_test: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge regression, centered (so an intercept is implicit).
    X_train: (N,D), Y_train: (N,K), X_test: (M,D) -> (M,K) predictions."""
    mu_x, mu_y = X_train.mean(axis=0), Y_train.mean(axis=0)
    Xc, Yc = X_train - mu_x, Y_train - mu_y
    D = Xc.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(D), Xc.T @ Yc)
    return (X_test - mu_x) @ W + mu_y


def group_by_identity(betas: np.ndarray) -> np.ndarray:
    """Real subjects repeat their exact GT betas across every frame they
    appear in (it's read straight from the source pkl, no computation in
    between) -- so exact-match grouping recovers subject identity without
    needing sequence/person_id bookkeeping. Returns an integer id per row."""
    seen: list[np.ndarray] = []
    ids = np.empty(len(betas), dtype=np.int64)
    for i, b in enumerate(betas):
        match = next((j for j, s in enumerate(seen) if np.allclose(s, b, atol=1e-6)), None)
        if match is None:
            seen.append(b)
            match = len(seen) - 1
        ids[i] = match
    return ids


def leave_one_subject_out(features: np.ndarray, betas: np.ndarray, subject_ids: np.ndarray,
                          alpha: float) -> list[dict]:
    """For each unique subject, fit on every OTHER subject's frames and
    predict this one's shape (averaged over its frames). Returns one row
    per subject: {subject, n_frames, pred_beta, true_beta,
    baseline_beta (trivial: mean of the other subjects' true betas, using
    no image information at all)}."""
    rows = []
    for s in np.unique(subject_ids):
        train = subject_ids != s
        test = subject_ids == s
        pred = ridge_fit_predict(features[train], betas[train], features[test], alpha).mean(axis=0)
        rows.append({
            "subject": int(s),
            "n_frames": int(test.sum()),
            "pred_beta": pred,
            "true_beta": betas[test][0],
            "baseline_beta": betas[train].mean(axis=0),
        })
    return rows


def summarize(rows: list[dict], alpha: float) -> None:
    n = len(rows)
    print(f"\n{n} distinct real subject(s) found (leave-one-out over these)")
    if n < 4:
        print("WARNING: too few distinct subjects for this to be conclusive either way.")
    true_norm = np.array([np.linalg.norm(r["true_beta"]) for r in rows])
    pred_norm = np.array([np.linalg.norm(r["pred_beta"]) for r in rows])
    base_norm = np.array([np.linalg.norm(r["baseline_beta"]) for r in rows])

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else float("nan")

    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print(f"\n=== held-out shape prediction, ridge alpha={alpha} (this repo's model output for "
          f"comparison: corr -0.07, see verify_neutral_gender_bias.py) ===")
    print(f"probe:    corr(||pred||, ||true||) = {corr(pred_norm, true_norm):+.3f}   "
          f"mean cosine(pred, true) = {np.mean([cos(r['pred_beta'], r['true_beta']) for r in rows]):+.3f}")
    print(f"baseline: corr(||pred||, ||true||) = {corr(base_norm, true_norm):+.3f}   "
          f"mean cosine(pred, true) = {np.mean([cos(r['baseline_beta'], r['true_beta']) for r in rows]):+.3f}   "
          f"(no image used at all -- what 'no signal' looks like)")
    print("\nInterpretation: probe clearly beating baseline => the feature has recoverable shape "
          "signal the current linear head isn't using, worth training a real head on. Probe ~= "
          "baseline => the signal isn't in this feature; a bigger head won't fix it.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--gt_dir", type=str, required=True)
    ap.add_argument("--stride", type=int, default=10, help="use every Nth frame per sequence")
    ap.add_argument("--alpha", type=float, default=10.0, help="ridge regularization strength")
    ap.add_argument("--out", type=str, default="results/shape_features.npz")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    import cv2
    import torch
    from s1_infer import HMR2Estimator

    est = HMR2Estimator(gender="neutral", device=args.device)

    captured: list[torch.Tensor] = []
    hook = est.model.smpl_head.decshape.register_forward_pre_hook(
        lambda module, inp: captured.append(inp[0].detach().cpu())
    )

    features, betas_list = [], []
    for pkl_path in sorted(Path(args.gt_dir).glob("*.pkl")):
        from eval_against_gt import load_3dpw_gt
        seq = pkl_path.stem
        n_before = len(features)
        for rec in load_3dpw_gt(pkl_path):
            frame = int(rec.image_id.rsplit("_", 1)[1])
            if frame % args.stride != 0 or rec.bbox is None:
                continue
            img = cv2.imread(str(Path(args.img_root) / seq / f"image_{frame:05d}.jpg"))
            if img is None:
                continue
            captured.clear()
            people = est.estimate(img, boxes=rec.bbox[None])
            if not people or not captured:
                continue
            features.append(captured[-1][0].numpy())  # last IEF iteration, only box -> index 0
            betas_list.append(rec.betas.astype(np.float64))
        print(f"{seq}: {len(features) - n_before} frame(s) added, {len(features)} total", flush=True)

    hook.remove()
    features_arr = np.stack(features)
    betas_arr = np.stack(betas_list)
    subject_ids = group_by_identity(betas_arr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, features=features_arr, betas=betas_arr, subject_ids=subject_ids)
    print(f"wrote {out_path} ({len(features_arr)} frames, {len(np.unique(subject_ids))} subjects)")

    rows = leave_one_subject_out(features_arr, betas_arr, subject_ids, args.alpha)
    summarize(rows, args.alpha)


def self_test() -> None:
    """No model/data needed. Builds a 'signal present' case (features
    linearly predict shape) and a 'no signal' case (features are pure
    noise, independent of shape) and checks the probe tells them apart."""
    rng = np.random.default_rng(0)
    n_subjects, frames_per_subject, feat_dim, beta_dim = 8, 15, 32, 10

    true_betas = rng.normal(0, 1, (n_subjects, beta_dim))
    subject_ids = np.repeat(np.arange(n_subjects), frames_per_subject)
    betas_arr = true_betas[subject_ids]

    def make_features(signal: bool) -> np.ndarray:
        if signal:
            # features are a fixed linear function of the true betas plus small
            # per-frame noise -- so the features -> betas mapping is genuinely
            # linear and recoverable, unlike the "no signal" case below.
            base = betas_arr @ rng.normal(0, 1, (beta_dim, feat_dim))
            return base + rng.normal(0, 0.1, base.shape)
        return rng.normal(0, 1, (len(betas_arr), feat_dim))

    for signal, label in ((True, "signal present"), (False, "no signal")):
        feats = make_features(signal)
        rows = leave_one_subject_out(feats, betas_arr, subject_ids, alpha=1.0)
        true_norm = np.array([np.linalg.norm(r["true_beta"]) for r in rows])
        pred_norm = np.array([np.linalg.norm(r["pred_beta"]) for r in rows])
        corr = float(np.corrcoef(pred_norm, true_norm)[0, 1])
        print(f"[self-test] {label}: corr(||pred||, ||true||) = {corr:+.3f}")
        if signal:
            assert corr > 0.6, f"probe should recover strong signal when it's really there, got {corr}"
        else:
            assert abs(corr) < 0.5, f"probe should not find signal in pure noise, got {corr}"

    ids = group_by_identity(np.array([[1, 2], [1, 2], [3, 4], [1, 2.0000001]]))
    assert list(ids) == [0, 0, 1, 0], f"near-identical rows should group together, got {ids}"

    print("[self-test] all checks passed -- the probe distinguishes real signal from noise, "
          "and identity grouping works.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        main()
