#!/usr/bin/env python3
"""S1 quantitative evaluation: compare s1_infer.py's predictions against the
ground-truth SMPL parameters shipped with a dataset, and report MPJPE /
PA-MPJPE (joint/pose error) and beta error (shape error).

Status (updated 2026-09-25): 3DPW is downloaded and `load_3dpw_gt()` has
been verified against a real file (`sequenceFiles/test/outdoors_fencing_01.pkl`
— see that function's docstring for exactly what was checked). CloSe-Di is
still not downloaded, so `load_close_di_gt()` remains an unverified
skeleton. The error math (MPJPE / PA-MPJPE / Procrustes alignment / beta
error) is pure math and was validated against synthetic data separately
(see --self-test).

Known issues, not yet solved:
    - 3DPW's GT is computed with a gendered SMPL model (male/female); HMR2
      only outputs neutral-SMPL parameters, and only SMPL_NEUTRAL.pkl is
      available locally. Comparing joints computed from different body
      models introduces a systematic error that is not HMR2's estimation
      error. Either download the gendered models separately from
      https://smpl.is.tue.mpg.de/ and evaluate against those, or accept
      this as a known error source and note it in the finding.
    - A 3DPW clip can have more than one person; match_prediction() only
      does bbox-IoU matching so far and hasn't been tested on a multi-person
      case.

Usage (skeleton self-test, no real dataset or GPU needed):
    python eval/eval_against_gt.py --self-test

Usage (once data is downloaded and s1_infer.py has produced --pred_dir):
    python eval/eval_against_gt.py --dataset 3dpw \\
        --pred_dir results/s1_raw_3dpw --gt_dir /path/to/3DPW/sequenceFiles/test \\
        --out results/eval_3dpw.csv

    python eval/eval_against_gt.py --dataset close-di \\
        --pred_dir results/s1_raw_close_di --gt_dir /path/to/CloSe-Di \\
        --out results/eval_close_di.csv
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import torch  # noqa: E402
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    # Same reason as s1_infer.py: smplx's SMPL_NEUTRAL.pkl loader goes
    # through pickle, and the pinned pytorch-lightning version predates
    # torch>=2.6's weights_only=True default.
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


# ---------------------------------------------------------------------------
# Shared data structure
# ---------------------------------------------------------------------------

@dataclass
class PoseRecord:
    """One person's SMPL parameters — used for both predictions and GT, so
    the matching/scoring logic below is shared regardless of source."""
    image_id: str
    person_id: int
    betas: np.ndarray                      # (10,)
    body_pose_aa: np.ndarray                # (23,3) axis-angle, excludes global_orient
    global_orient_aa: np.ndarray            # (3,) axis-angle
    bbox: np.ndarray | None = None          # (4,) [x1,y1,x2,y2], used for matching
    joints_3d: np.ndarray | None = None     # (J,3), if the dataset provides it directly


# ---------------------------------------------------------------------------
# Pose representation conversion / SMPL forward (for computing joints)
# ---------------------------------------------------------------------------

def rotmat_to_aa(rotmat: np.ndarray) -> np.ndarray:
    """(...,3,3) rotation matrices -> (...,3) axis-angle."""
    from scipy.spatial.transform import Rotation
    shape = rotmat.shape[:-2]
    aa = Rotation.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec()
    return aa.reshape(*shape, 3).astype(np.float32)


def build_smpl_layer(device: "torch.device" = None, gender: str = "neutral"):
    """Load an SMPL layer. Only SMPL_NEUTRAL.pkl is available locally (the
    only one HMR2's official release ships). gender="male"/"female" needs
    SMPL_MALE.pkl / SMPL_FEMALE.pkl downloaded separately from
    https://smpl.is.tue.mpg.de/ into the same folder."""
    import smplx
    from hmr2.configs import CACHE_DIR_4DHUMANS

    filename = {"neutral": "SMPL_NEUTRAL.pkl", "male": "SMPL_MALE.pkl",
                "female": "SMPL_FEMALE.pkl"}[gender]
    smpl_path = Path(CACHE_DIR_4DHUMANS) / "data" / "smpl" / filename
    if not smpl_path.exists():
        raise FileNotFoundError(
            f"{smpl_path} does not exist. The gender={gender!r} SMPL model "
            f"must be downloaded separately from https://smpl.is.tue.mpg.de/ "
            f"and placed at that path."
        )
    return smplx.SMPLLayer(model_path=str(smpl_path), num_betas=10).to(device).eval()


def get_joints(smpl_layer, record: PoseRecord, device: "torch.device" = None) -> np.ndarray:
    """Prefer the dataset's own precomputed joints_3d if present; otherwise
    compute them via an SMPL forward pass."""
    if record.joints_3d is not None:
        return record.joints_3d

    from scipy.spatial.transform import Rotation

    body_pose_rotmat = Rotation.from_rotvec(record.body_pose_aa).as_matrix()  # (23,3,3)
    global_orient_rotmat = Rotation.from_rotvec(record.global_orient_aa[None]).as_matrix()  # (1,3,3)

    betas_t = torch.tensor(record.betas, dtype=torch.float32, device=device)[None]
    body_pose_t = torch.tensor(body_pose_rotmat, dtype=torch.float32, device=device)[None]
    global_orient_t = torch.tensor(global_orient_rotmat, dtype=torch.float32, device=device)[None]

    with torch.no_grad():
        out = smpl_layer(betas=betas_t, body_pose=body_pose_t, global_orient=global_orient_t)
    return out.joints[0].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Error metrics (pure math, independent of any dataset format — this is
# what --self-test validates)
# ---------------------------------------------------------------------------

def compute_similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Procrustes alignment: find the rotation + scale + translation that
    minimizes the error mapping source onto target (the standard PA-MPJPE
    approach, matching the SPIN/HMR family of papers). source, target:
    (J,3). Returns the aligned source, same shape."""
    mu1 = source.mean(axis=0, keepdims=True)
    mu2 = target.mean(axis=0, keepdims=True)
    X1 = source - mu1
    X2 = target - mu2

    var1 = np.sum(X1 ** 2)
    K = X1.T @ X2
    U, s, Vt = np.linalg.svd(K)
    Z = np.eye(U.shape[0])
    Z[-1, -1] = np.sign(np.linalg.det(U @ Vt))
    R = Vt.T @ Z @ U.T
    scale = np.trace(R @ K) / var1
    t = mu2.T - scale * (R @ mu1.T)
    aligned = (scale * (R @ source.T) + t).T
    return aligned


def mpjpe(pred_joints: np.ndarray, gt_joints: np.ndarray, pelvis_idx: int = 0) -> float:
    """Mean Per-Joint Position Error, in mm (assumes meter inputs).
    Root-relative: both sides are re-centered on the pelvis joint first, so
    this only reflects relative-pose error, not overall translation."""
    pred = pred_joints - pred_joints[pelvis_idx:pelvis_idx + 1]
    gt = gt_joints - gt_joints[pelvis_idx:pelvis_idx + 1]
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000)


def pa_mpjpe(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
    """Procrustes-Aligned MPJPE, in mm. Differs from mpjpe() by aligning
    rotation/scale first, so it isolates pose-shape error from global
    rotation/scale/translation — usually smaller than mpjpe(), and the
    metric most commonly reported in papers."""
    aligned = compute_similarity_transform(pred_joints, gt_joints)
    return float(np.linalg.norm(aligned - gt_joints, axis=-1).mean() * 1000)


def beta_error(pred_betas: np.ndarray, gt_betas: np.ndarray) -> dict:
    """Beta (shape) error, per-dimension and overall. No alignment needed —
    betas are already pose/position-independent shape coefficients."""
    diff = pred_betas - gt_betas
    return {
        "beta_mae": float(np.abs(diff).mean()),
        "beta_l2": float(np.linalg.norm(diff)),
        "beta_per_dim_abs": np.abs(diff).tolist(),
    }


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Standard IoU, boxes as [x1,y1,x2,y2]. Used to match predicted people to GT."""
    xa1, ya1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xa2, ya2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Prediction loading (reads s1_infer.py's npz output; format is ours, not a TODO)
# ---------------------------------------------------------------------------

def load_predictions(pred_dir: Path) -> dict[str, list[PoseRecord]]:
    """Scan `<image_id>_<person_id>_smpl_params.npz` files produced by
    s1_infer.py, grouped by image_id."""
    by_image: dict[str, list[PoseRecord]] = {}
    for npz_path in sorted(pred_dir.glob("*_smpl_params.npz")):
        stem = npz_path.stem  # "<image_id>_<person_id>_smpl_params"
        # "smpl_params" itself contains an underscore, so this has to split
        # off 3 trailing "_"-parts ("<person_id>", "smpl", "params"), not 2
        # — splitting off only 2 cuts between "smpl" and "params" instead
        # and misreads "smpl" as the person_id.
        image_id, person_id = stem.rsplit("_", 3)[0], stem.rsplit("_", 3)[1]
        data = np.load(npz_path)
        record = PoseRecord(
            image_id=image_id,
            person_id=int(person_id),
            betas=data["betas"],
            body_pose_aa=rotmat_to_aa(data["body_pose"]),
            global_orient_aa=rotmat_to_aa(data["global_orient"])[0],
            bbox=data["bbox"] if "bbox" in data else None,
        )
        by_image.setdefault(image_id, []).append(record)
    return by_image


# ---------------------------------------------------------------------------
# GT loading
# ---------------------------------------------------------------------------

def load_3dpw_gt(seq_pkl_path: Path) -> list[PoseRecord]:
    """Load one 3DPW sequence's GT
    (`sequenceFiles/{train,validation,test}/*.pkl`).

    Verified against a real file (`sequenceFiles/test/outdoors_fencing_01.pkl`
    on 2026-09-25): the fields used here (poses/betas/jointPositions/
    campose_valid) exist with the expected shapes. `betas` is (300,) per
    person, not (10,) — HMR2 only outputs 10, so we keep just the first 10
    (SMPL's shape space is nested/PCA-ordered, so this is a valid truncation,
    not an arbitrary slice).

    image_id: uses the pose array's own frame index (0..num_frames-1), NOT
    the `img_frame_ids` field. `img_frame_ids` records each pose's frame
    number in the original 60Hz video (e.g. [0, 2, 4, 6, ...] — 3DPW's pose
    annotations are downsampled from 60Hz), but the *shipped* `imageFiles/`
    directory is already renumbered to match that downsampled sequence
    (verified: outdoors_fencing_01 has exactly 942 poses and exactly 942
    images named image_00000.jpg..image_00941.jpg, contiguous). So
    `poses[i]` corresponds to `image_{i:05d}.jpg`, and `img_frame_ids` isn't
    needed for anything here. The "__" (not "/") separator matches
    s1_infer.py's --img_folder naming (see its module docstring), which
    replaces path separators the same way to avoid different sequences'
    identically-numbered frames colliding.
    """
    import pickle

    with open(seq_pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    seq_name = seq_pkl_path.stem
    records: list[PoseRecord] = []

    num_people = len(data["poses"])
    for person_id in range(num_people):
        poses = data["poses"][person_id]              # (num_frames, 72)
        betas = data["betas"][person_id][:10]           # (300,) -> keep first 10 to match HMR2
        joint_positions = data.get("jointPositions", [None] * num_people)[person_id]  # (num_frames, 24*3) or None
        valid = data.get("campose_valid", [None] * num_people)[person_id]

        num_frames = poses.shape[0]
        for frame_idx in range(num_frames):
            if valid is not None and not valid[frame_idx]:
                continue
            # "__" (not "/") to match s1_infer.py's --img_folder naming,
            # which replaces path separators the same way to avoid
            # different sequences' identically-numbered frames colliding.
            image_id = f"{seq_name}__image_{frame_idx:05d}"
            pose_frame = poses[frame_idx].reshape(24, 3)  # axis-angle, joint 0 = global_orient
            joints_3d = None
            if joint_positions is not None:
                joints_3d = joint_positions[frame_idx].reshape(24, 3)
            records.append(PoseRecord(
                image_id=image_id,
                person_id=person_id,
                betas=betas.astype(np.float32),
                body_pose_aa=pose_frame[1:].astype(np.float32),
                global_orient_aa=pose_frame[0].astype(np.float32),
                joints_3d=joints_3d,
            ))
    return records


def load_close_di_gt(npz_path: Path) -> PoseRecord:
    """Load one CloSe-Di scan's GT (fields per the pipeline plan doc's
    section 3.2: betas/pose/trans/canon_pose).

    TODO not yet verified against real data: the plan doc's field names
    were compiled from the CloSe-D paper/repo docs, not confirmed against
    an actual .npz file. image_id currently assumes it equals the filename
    stem (CloSe-Di is one file per scan rather than a video sequence, so
    matching is much simpler than 3DPW and lower risk, but still needs a
    real-file check).
    """
    data = np.load(npz_path)
    pose = data["pose"]  # (72,) axis-angle
    pose = pose.reshape(24, 3)
    return PoseRecord(
        image_id=npz_path.stem,
        person_id=0,
        betas=data["betas"][:10].astype(np.float32),
        body_pose_aa=pose[1:].astype(np.float32),
        global_orient_aa=pose[0].astype(np.float32),
    )


# ---------------------------------------------------------------------------
# Matching + evaluation driver
# ---------------------------------------------------------------------------

def match_prediction(preds: list[PoseRecord], gt: PoseRecord) -> PoseRecord | None:
    """An image's number of predictions and GT people can differ (missed or
    extra detections); match by highest bbox IoU (must be > 0). If there's
    only one prediction (most CloSe-Di cases), it's returned directly."""
    if not preds:
        return None
    if len(preds) == 1:
        return preds[0]
    if gt.bbox is None:
        return preds[0]  # no GT bbox to compare against; fall back to the first one (TODO: not ideal)
    best, best_iou = None, 0.0
    for p in preds:
        if p.bbox is None:
            continue
        iou = bbox_iou(p.bbox, gt.bbox)
        if iou > best_iou:
            best, best_iou = p, iou
    return best


def evaluate(pred_by_image: dict[str, list[PoseRecord]], gt_records: list[PoseRecord],
             smpl_layer, device: "torch.device" = None) -> list[dict]:
    """Generic evaluation loop shared by 3DPW and CloSe-Di — once both are
    converted to PoseRecord, matching/scoring is identical."""
    rows = []
    for gt in gt_records:
        preds = pred_by_image.get(gt.image_id, [])
        pred = match_prediction(preds, gt)
        if pred is None:
            rows.append({"image_id": gt.image_id, "person_id": gt.person_id,
                         "status": "no_prediction"})
            continue

        gt_joints = get_joints(smpl_layer, gt, device)
        pred_joints = get_joints(smpl_layer, pred, device)
        err = beta_error(pred.betas, gt.betas)
        rows.append({
            "image_id": gt.image_id,
            "person_id": gt.person_id,
            "status": "ok",
            "mpjpe_mm": mpjpe(pred_joints, gt_joints),
            "pa_mpjpe_mm": pa_mpjpe(pred_joints, gt_joints),
            "beta_mae": err["beta_mae"],
            "beta_l2": err["beta_l2"],
        })
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    print(f"{len(rows)} GT record(s) total, {len(ok_rows)} matched and scored.")
    if ok_rows:
        for key in ("mpjpe_mm", "pa_mpjpe_mm", "beta_mae", "beta_l2"):
            values = [r[key] for r in ok_rows]
            print(f"  {key}: mean={np.mean(values):.2f}  median={np.median(values):.2f}")


# ---------------------------------------------------------------------------
# Self-test: no real dataset needed, validates the error math itself
# ---------------------------------------------------------------------------

def self_test() -> None:
    rng = np.random.default_rng(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    smpl_layer = build_smpl_layer(device=device, gender="neutral")

    def random_record(image_id: str, person_id: int, seed_offset: float) -> PoseRecord:
        betas = rng.normal(0, 1, 10).astype(np.float32)
        body_pose_aa = (rng.normal(0, 0.1, (23, 3)) + seed_offset).astype(np.float32)
        global_orient_aa = rng.normal(0, 0.1, 3).astype(np.float32)
        return PoseRecord(image_id=image_id, person_id=person_id, betas=betas,
                           body_pose_aa=body_pose_aa, global_orient_aa=global_orient_aa,
                           bbox=np.array([0, 0, 100, 200], dtype=np.float32))

    gt = random_record("dummy_0001", 0, seed_offset=0.0)

    # Case 1: prediction == GT, error should be 0 (sanity-checks
    # mpjpe/pa_mpjpe's sign/units aren't wrong).
    joints_gt = get_joints(smpl_layer, gt, device)
    err_zero_mpjpe = mpjpe(joints_gt, joints_gt)
    err_zero_pa = pa_mpjpe(joints_gt, joints_gt)
    print(f"[self-test] prediction=GT: mpjpe={err_zero_mpjpe:.6f}mm, "
          f"pa_mpjpe={err_zero_pa:.6f}mm (both should be ~0)")
    assert err_zero_mpjpe < 1e-3, "identical input should give MPJPE=0; math is wrong"
    assert err_zero_pa < 1e-3, "identical input should give PA-MPJPE=0; math is wrong"

    # Case 2: prediction is GT + noise, error should be > 0, and
    # pa_mpjpe <= mpjpe (Procrustes alignment can only reduce error, by definition).
    pred_noisy = random_record("dummy_0001", 0, seed_offset=0.02)
    pred_noisy.betas = gt.betas + rng.normal(0, 0.3, 10).astype(np.float32)
    joints_pred = get_joints(smpl_layer, pred_noisy, device)
    m = mpjpe(joints_pred, joints_gt)
    pa = pa_mpjpe(joints_pred, joints_gt)
    beta_err = beta_error(pred_noisy.betas, gt.betas)
    print(f"[self-test] prediction≈GT+noise: mpjpe={m:.2f}mm, pa_mpjpe={pa:.2f}mm, "
          f"beta_mae={beta_err['beta_mae']:.4f}, beta_l2={beta_err['beta_l2']:.4f}")
    assert m > 0 and pa > 0, "noisy input gave 0 error; math is wrong"
    assert pa <= m + 1e-4, "PA-MPJPE should never exceed MPJPE; math is wrong"

    # Case 3: run the full evaluate() flow (matching + CSV) too, to confirm
    # the data actually flows end to end.
    pred_by_image = {"dummy_0001": [pred_noisy]}
    rows = evaluate(pred_by_image, [gt], smpl_layer, device)
    out_path = Path("results") / "eval_self_test.csv"
    write_csv(rows, out_path)
    print(f"[self-test] all checks passed, sample output at {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                     help="validate the error math against synthetic data; needs no real dataset/download")
    ap.add_argument("--dataset", type=str, choices=["3dpw", "close-di"])
    ap.add_argument("--pred_dir", type=str, help="s1_infer.py's --out folder")
    ap.add_argument("--gt_dir", type=str, help="3DPW's sequenceFiles/test, or a CloSe-Di folder")
    ap.add_argument("--out", type=str, default="results/eval.csv")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not (args.dataset and args.pred_dir and args.gt_dir):
        raise SystemExit("A real evaluation needs --dataset --pred_dir --gt_dir, "
                          "or use --self-test to validate the math first.")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    smpl_layer = build_smpl_layer(device=device, gender="neutral")
    pred_by_image = load_predictions(Path(args.pred_dir))

    gt_dir = Path(args.gt_dir)
    if args.dataset == "3dpw":
        gt_records = []
        for pkl_path in sorted(gt_dir.glob("*.pkl")):
            gt_records.extend(load_3dpw_gt(pkl_path))
    else:
        gt_records = [load_close_di_gt(p) for p in sorted(gt_dir.glob("*.npz"))]

    rows = evaluate(pred_by_image, gt_records, smpl_layer, device)
    write_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
