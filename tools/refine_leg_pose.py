#!/usr/bin/env python3
"""S1 post-processing: correct S1's estimated leg pose (theta) using 2D pose
keypoints (ViTPose).

Background: HMR2 is a single-image 3D pose regressor with no
stereo/multi-view information, so it can misjudge knee joint rotation on
poses that are rare in its training data (e.g. sitting, or a knee bent
sharply toward the camera) — the projected calf ends up compressed and the
foot doesn't land where it actually is in the photo. This is a limitation
of the model itself, not a bug in s1_infer.py (already ruled out "the
detection box missed the foot" by inspecting tools/visualize.py's boxes
output). This directly affects the pipeline's S3 draping stage — S3 fits
the garment mesh to the target body's pose using (beta, theta); wrong theta
means the draped garment won't line up with the actual knee position in the
photo, which then throws off S4's per-region ease calculation. So leg pose
accuracy is not a negligible edge case, especially since pants/shorts are
exactly the garment categories this pipeline covers.

Method (an SMPLify-style post-process; no model is retrained):
    1. Run ViTPose (HuggingFace `transformers`, COCO 17-point format) inside
       S1's detected person box to get real 2D positions for the 6 leg
       joints (left/right hip, knee, ankle).
    2. Turn just those 6 joints' rotation matrices in S1's body_pose into
       trainable parameters (axis-angle parameterization; each gradient
       step converts back to a rotation matrix via aa_to_rotmat(), so it
       always stays a valid rotation). The other 17 joints (plus beta,
       global_orient, everything else) are frozen.
    3. Project those 6 joints back to 2D with the same perspective_projection()
       used in hmr2/utils/geometry.py, and minimize a confidence-weighted L2
       distance to the ViTPose targets (weights = ViTPose's own confidence
       scores), with a small regularizer keeping the result close to S1's
       original estimate (so low-confidence/occluded keypoints don't pull
       the joint to an implausible angle).
    4. Only those 6 joints are overwritten — arms, torso, and toe tips
       (SMPL has no toe joints) are all untouched.

Usage (run s1_infer.py first; this consumes its output):
    python tools/refine_leg_pose.py --img path/to/image.jpg --s1_out results/s1_raw --out results/s1_refined

Environment: same as s1_infer.py, plus `pip install transformers` if not
already installed (it should be, since the default RT-DETR detector needs
it too — this reuses that same install).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import torch  # noqa: E402
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    # Same reason as s1_infer.py: smplx's SMPL_NEUTRAL.pkl loader also goes
    # through pickle, and the pinned pytorch-lightning version predates
    # torch>=2.6's weights_only=True default.
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# Same pyrender stub s1_infer.py needs: hmr2.utils.geometry itself has no
# pyrender dependency, but importing the hmr2.utils package (e.g. via
# `from hmr2.utils.geometry import ...`) runs hmr2/utils/__init__.py first,
# which imports renderer.py, which unconditionally does `import pyrender` —
# this fails hard on machines without OpenGL/EGL. Same no-op stub fixes it.
import sys as _sys
import types as _types


class _DummyPyrenderAttr:
    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def _pyrender_stub_getattr(name: str):
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _DummyPyrenderAttr()


if "pyrender" not in _sys.modules:
    _pyrender_stub = _types.ModuleType("pyrender")
    _pyrender_stub.__getattr__ = _pyrender_stub_getattr
    _sys.modules["pyrender"] = _pyrender_stub


# Reuse tools/visualize.py's drawing/export helpers rather than duplicating
# the same logic.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_THIS_DIR))
from visualize import draw_mesh_overlay, write_obj, render_mesh_shaded  # noqa: E402


# Standard COCO 17-keypoint order (the convention essentially every "COCO
# format" 2D pose model follows). Hardcoded here rather than read from the
# transformers VitPose config, which doesn't expose per-point names.
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# The 6 leg joints to correct: (COCO keypoint name, SMPL joint index, body_pose array index)
# SMPL's 24 joints (pelvis=0): ..., 1 left_hip, 2 right_hip, ..., 4 left_knee,
# 5 right_knee, ..., 7 left_ankle, 8 right_ankle, ... (cross-checked against
# smplx's joint definitions). body_pose only stores the 23 non-pelvis
# joints, so body_pose index = SMPL joint index - 1.
LEG_JOINTS = [
    ("left_hip", 1, 0),
    ("right_hip", 2, 1),
    ("left_knee", 4, 3),
    ("right_knee", 5, 4),
    ("left_ankle", 7, 6),
    ("right_ankle", 8, 7),
]


def _rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """(3,3) rotation matrix -> axis-angle (3,). Called once before the
    optimization loop to build the initial value; doesn't need to be
    differentiable (the loop itself converts back via the torch aa_to_rotmat())."""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rotmat).as_rotvec().astype(np.float32)


class ViTPoseLegDetector:
    """Wraps ViTPose model + processor loading. Build once, call detect()
    repeatedly — batch-processing many images/people should not reload the
    model for every single person."""

    def __init__(self, model_name: str = "usyd-community/vitpose-base-simple",
                 device: "torch.device" = None):
        from transformers import AutoImageProcessor, VitPoseForPoseEstimation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_name).to(device).eval()

    def detect(self, img_bgr: np.ndarray, bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run ViTPose inside the given person box.
        Returns (leg_keypoints_2d, leg_scores), shapes (6,2) / (6,), ordered
        to match LEG_JOINTS."""
        from PIL import Image

        img_rgb = img_bgr[:, :, ::-1]
        pil_img = Image.fromarray(img_rgb)

        x1, y1, x2, y2 = bbox
        box_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
        inputs = self.processor(pil_img, boxes=[[box_xywh]], return_tensors="pt")
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # boxes are full-image absolute pixel coordinates (not normalized to
        # 0-1), so target_sizes is intentionally omitted here —
        # post_process_pose_estimation then returns full-image pixel
        # coordinates directly (matches transformers' own docstring example).
        results = self.processor.post_process_pose_estimation(outputs, boxes=[[box_xywh]])[0][0]
        keypoints = results["keypoints"].cpu().numpy()  # (17, 2)
        scores = results["scores"].cpu().numpy()  # (17,)

        name_to_idx = {name: i for i, name in enumerate(COCO_KEYPOINT_NAMES)}
        leg_kp = np.zeros((len(LEG_JOINTS), 2), dtype=np.float32)
        leg_scores = np.zeros(len(LEG_JOINTS), dtype=np.float32)
        for i, (coco_name, _, _) in enumerate(LEG_JOINTS):
            idx = name_to_idx[coco_name]
            leg_kp[i] = keypoints[idx]
            leg_scores[i] = scores[idx]
        return leg_kp, leg_scores


def refine_leg_pose(smpl_layer, betas: np.ndarray, body_pose: np.ndarray,
                     global_orient: np.ndarray, cam_t: np.ndarray,
                     scaled_focal_length: float, img_w: int, img_h: int,
                     target_kp_2d: np.ndarray, target_scores: np.ndarray,
                     iters: int = 150, lr: float = 0.02, reg_weight: float = 5.0,
                     score_thresh: float = 0.3, device: "torch.device" = None) -> np.ndarray:
    """The core SMPLify-lite optimization loop. Adjusts only the 6
    LEG_JOINTS; everything else stays frozen. Returns the corrected full
    body_pose (23,3,3)."""
    from hmr2.utils.geometry import aa_to_rotmat, perspective_projection

    betas_t = torch.tensor(betas, dtype=torch.float32, device=device)[None]
    global_orient_t = torch.tensor(global_orient, dtype=torch.float32, device=device)[None]
    body_pose_frozen = torch.tensor(body_pose, dtype=torch.float32, device=device)[None]  # (1,23,3,3)

    leg_body_pose_idx = [bp_idx for _, _, bp_idx in LEG_JOINTS]
    leg_joint_idx = [smpl_idx for _, smpl_idx, _ in LEG_JOINTS]

    init_theta = np.stack([
        _rotmat_to_axis_angle(body_pose[bp_idx]) for _, _, bp_idx in LEG_JOINTS
    ])  # (6,3)
    theta_leg = torch.nn.Parameter(
        torch.tensor(init_theta, dtype=torch.float32, device=device)
    )
    theta_leg_init = theta_leg.detach().clone()

    cam_t_t = torch.tensor(cam_t, dtype=torch.float32, device=device)[None]  # (1,3)
    focal_t = torch.tensor([scaled_focal_length, scaled_focal_length],
                            dtype=torch.float32, device=device)[None]  # (1,2)
    cam_center_t = torch.tensor([img_w / 2.0, img_h / 2.0], dtype=torch.float32, device=device)[None]

    target_kp_t = torch.tensor(target_kp_2d, dtype=torch.float32, device=device)  # (6,2)
    weights = torch.tensor(target_scores, dtype=torch.float32, device=device)
    weights = torch.where(weights > score_thresh, weights, torch.zeros_like(weights))  # (6,)

    if weights.sum().item() <= 0:
        # ViTPose has essentially no confidence for this person's legs
        # (e.g. heavy occlusion) — there's no reliable signal to refine
        # against, so give up cleanly and return S1's original estimate
        # rather than force an unreliable correction.
        print("  [warn] all leg-keypoint confidence scores are below "
              "threshold; skipping refinement, keeping S1's original estimate.")
        return body_pose

    def _project_leg_joints(theta: torch.Tensor) -> torch.Tensor:
        """Given the 6 leg joints' axis-angle, return their 2D pixel
        projection (6,2). Diagnostic use only, not part of the gradient
        path (callers wrap this in no_grad)."""
        rotmats_leg = aa_to_rotmat(theta)
        body_pose_diag = body_pose_frozen.clone()
        body_pose_diag[0, leg_body_pose_idx] = rotmats_leg
        smpl_out = smpl_layer(betas=betas_t, body_pose=body_pose_diag,
                               global_orient=global_orient_t)
        joints = smpl_out.joints[0, leg_joint_idx]
        return perspective_projection(
            joints[None], translation=cam_t_t, focal_length=focal_t,
            camera_center=cam_center_t,
        )[0]

    # Diagnostic: the "before" reprojection (using the initial theta, i.e.
    # S1's original angles), compared against the ViTPose target — this
    # quantifies "how wrong was S1 to begin with".
    with torch.no_grad():
        proj_before = _project_leg_joints(theta_leg_init)
        err_before = (proj_before - target_kp_t).norm(dim=1)  # (6,) pixel distance

    optimizer = torch.optim.Adam([theta_leg], lr=lr)
    for _ in range(iters):
        optimizer.zero_grad()
        rotmats_leg = aa_to_rotmat(theta_leg)  # (6,3,3), re-projected onto valid rotations every step
        body_pose_iter = body_pose_frozen.clone()
        body_pose_iter[0, leg_body_pose_idx] = rotmats_leg

        smpl_out = smpl_layer(betas=betas_t, body_pose=body_pose_iter,
                               global_orient=global_orient_t)
        joints = smpl_out.joints[0, leg_joint_idx]  # (6,3)

        proj = perspective_projection(
            joints[None], translation=cam_t_t, focal_length=focal_t,
            camera_center=cam_center_t,
        )[0]  # (6,2)

        diff = (proj - target_kp_t)
        data_loss = (weights[:, None] * diff.pow(2)).sum() / weights.sum().clamp(min=1e-6)
        reg_loss = (theta_leg - theta_leg_init).pow(2).mean()
        loss = data_loss + reg_weight * reg_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        rotmats_leg = aa_to_rotmat(theta_leg)
        body_pose_final = body_pose_frozen.clone()
        body_pose_final[0, leg_body_pose_idx] = rotmats_leg

        # Diagnostic: the "after" reprojection compared against the target
        # (how much error remains), plus how far each joint moved from
        # before to after — if this number is tiny, the optimizer barely
        # moved anything, which is the real thing to be suspicious of
        # (rather than eyeballing the render).
        proj_after = _project_leg_joints(theta_leg.detach())
        err_after = (proj_after - target_kp_t).norm(dim=1)
        moved = (proj_after - proj_before).norm(dim=1)

    print("  Diagnostics (pixel distance, smaller is better):")
    for i, (name, _, _) in enumerate(LEG_JOINTS):
        flag = "" if target_scores[i] > score_thresh else "(confidence below threshold, not used)"
        print(f"    {name:12s} before={err_before[i]:6.1f}px  "
              f"after={err_after[i]:6.1f}px  moved={moved[i]:6.1f}px {flag}")

    return body_pose_final[0].cpu().numpy()


def process_image(img_path: Path, s1_out_dir: Path, out_dir: Path,
                   smpl_layer, leg_detector: ViTPoseLegDetector,
                   iters: int, lr: float, reg_weight: float, score_thresh: float,
                   device: "torch.device", save_render: bool) -> None:
    """Process one image: read every S1 npz for it in s1_out_dir and refine
    each person's leg pose. A standalone function (rather than inline in
    main()'s loop) so batch mode can call it repeatedly and one bad image
    doesn't abort the whole batch (warns and skips instead of raising)."""
    npz_paths = sorted(s1_out_dir.glob(f"{img_path.stem}_*_smpl_params.npz"))
    if not npz_paths:
        print(f"[skip] no {img_path.stem}_*_smpl_params.npz found in {s1_out_dir}; "
              f"run s1_infer.py on this image first.")
        return

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[warn] could not read image {img_path}, skipping")
        return
    img_h, img_w = img_bgr.shape[:2]

    all_people_before = []
    all_people_after = []

    for npz_path in npz_paths:
        data = np.load(npz_path)
        if "bbox" not in data or "scaled_focal_length" not in data:
            print(f"  [skip] {npz_path.name} is missing bbox/scaled_focal_length "
                  f"(re-run s1_infer.py to regenerate it).")
            continue

        person_id = npz_path.stem.split("_")[-3]  # <stem>_<id>_smpl_params.npz
        betas = data["betas"]
        body_pose = data["body_pose"]
        global_orient = data["global_orient"]
        cam_t = data["cam_t"]
        bbox = data["bbox"]
        scaled_focal_length = float(data["scaled_focal_length"])

        print(f"[{npz_path.name}] detecting 2D leg keypoints...")
        leg_kp_2d, leg_scores = leg_detector.detect(img_bgr, bbox)
        for (name, _, _), score in zip(LEG_JOINTS, leg_scores):
            print(f"    {name}: score={score:.2f}")

        print(f"[{npz_path.name}] optimizing leg pose ({iters} iterations)...")
        refined_body_pose = refine_leg_pose(
            smpl_layer, betas, body_pose, global_orient, cam_t, scaled_focal_length,
            img_w, img_h, leg_kp_2d, leg_scores,
            iters=iters, lr=lr, reg_weight=reg_weight,
            score_thresh=score_thresh, device=device,
        )

        out_npz = out_dir / npz_path.name.replace("_smpl_params.npz", "_smpl_params_refined.npz")
        np.savez(out_npz, betas=betas, body_pose=refined_body_pose,
                 global_orient=global_orient, cam_t=cam_t, bbox=bbox,
                 scaled_focal_length=scaled_focal_length)
        print(f"  saved {out_npz}")

        with torch.no_grad():
            betas_t = torch.tensor(betas, dtype=torch.float32, device=device)[None]
            global_orient_t = torch.tensor(global_orient, dtype=torch.float32, device=device)[None]

            body_pose_before_t = torch.tensor(body_pose, dtype=torch.float32, device=device)[None]
            verts_before = smpl_layer(betas=betas_t, body_pose=body_pose_before_t,
                                       global_orient=global_orient_t).vertices[0].cpu().numpy()

            body_pose_after_t = torch.tensor(refined_body_pose, dtype=torch.float32, device=device)[None]
            verts_after = smpl_layer(betas=betas_t, body_pose=body_pose_after_t,
                                      global_orient=global_orient_t).vertices[0].cpu().numpy()

        out_obj = out_dir / npz_path.name.replace("_smpl_params.npz", "_smpl_params_refined.obj")
        write_obj(verts_after, smpl_layer.faces, out_obj)
        print(f"  saved {out_obj}")

        if save_render:
            out_render = out_dir / npz_path.name.replace("_smpl_params.npz", "_smpl_params_refined_render.png")
            render_mesh_shaded(verts_after, smpl_layer.faces, out_render)
            print(f"  saved {out_render}")

        all_people_before.append({"person_id": int(person_id), "pred_vertices": verts_before,
                                   "cam_t": cam_t, "scaled_focal_length": scaled_focal_length})
        all_people_after.append({"person_id": int(person_id), "pred_vertices": verts_after,
                                  "cam_t": cam_t, "scaled_focal_length": scaled_focal_length})

    if all_people_before:
        overlay_before = draw_mesh_overlay(img_bgr, all_people_before)
        overlay_after = draw_mesh_overlay(img_bgr, all_people_after)
        side_by_side = np.concatenate([overlay_before, overlay_after], axis=1)
        out_path = out_dir / f"{img_path.stem}_before_after.jpg"
        cv2.imwrite(str(out_path), side_by_side)
        print(f"Done, before/after comparison saved to {out_path} (left: before, right: after)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, help="single image path, same one fed to s1_infer.py "
                     "(mutually exclusive with --img_folder)")
    ap.add_argument("--img_folder", type=str, help="batch mode: process every *.jpg/*.png in "
                     "this folder, each looked up against --s1_out")
    ap.add_argument("--s1_out", type=str, required=True, help="s1_infer.py's --out folder")
    ap.add_argument("--out", type=str, default="results/s1_refined")
    ap.add_argument("--vitpose-model", type=str, default="usyd-community/vitpose-base-simple")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--reg-weight", type=float, default=5.0,
                     help="regularization strength; higher trusts the 2D keypoints less "
                          "and stays closer to S1's original estimate.")
    ap.add_argument("--score-thresh", type=float, default=0.3,
                     help="ViTPose confidence threshold; joints below this are not used for refinement.")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-render", action="store_true",
                     help="by default also saves <stem>_<id>_smpl_params_refined_render.png "
                          "(the final refined result); this flag disables that.")
    args = ap.parse_args()

    if not args.img and not args.img_folder:
        raise SystemExit("Provide either --img or --img_folder")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    img_paths: list[Path] = []
    if args.img:
        img_paths.append(Path(args.img))
    if args.img_folder:
        folder = Path(args.img_folder)
        img_paths.extend(sorted(folder.rglob("*.jpg")) + sorted(folder.rglob("*.png")))

    s1_out_dir = Path(args.s1_out)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SMPL model...")
    import smplx
    from hmr2.configs import CACHE_DIR_4DHUMANS
    smpl_path = Path(CACHE_DIR_4DHUMANS) / "data" / "smpl" / "SMPL_NEUTRAL.pkl"
    smpl_layer = smplx.SMPLLayer(model_path=str(smpl_path), num_betas=10).to(device).eval()

    print(f"Loading ViTPose ({args.vitpose_model})... ({len(img_paths)} image(s) to process)")
    leg_detector = ViTPoseLegDetector(model_name=args.vitpose_model, device=device)

    for img_path in img_paths:
        process_image(img_path, s1_out_dir, out_dir, smpl_layer, leg_detector,
                       iters=args.iters, lr=args.lr, reg_weight=args.reg_weight,
                       score_thresh=args.score_thresh, device=device,
                       save_render=not args.no_render)


if __name__ == "__main__":
    main()
