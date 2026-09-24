#!/usr/bin/env python3
"""S1: estimate SMPL parameters (betas, pose, camera) from an RGB image using
the pretrained HMR2.0 model (https://github.com/shubham-goel/4D-Humans).

This is the single entry point for this pipeline stage. It is intentionally
minimal: detection + regression only, no visualization. For debugging a
single image with rendered overlays, use tools/visualize.py on top of this
script's output instead of adding flags here.

    Input:  one or more RGB images (person photos)
    Output: for each detected person, one .npz file with
        betas          (10,)        SMPL shape coefficients
        body_pose      (23, 3, 3)   joint rotation matrices (excludes root)
        global_orient  (1, 3, 3)    root joint rotation matrix
        cam_t          (3,)         weak-perspective camera translation,
                                     already converted to full-image space
        bbox           (4,)         [x1, y1, x2, y2], full-image pixel coords
        scaled_focal_length  ()     scalar, needed to re-project pred_vertices
        pred_vertices  (6890, 3)    SMPL mesh vertices (model space)

Pipeline: a person detector finds bounding boxes, then HMR2 (pure
ViT + transformer decoder, no dependency on the detector backend) regresses
SMPL parameters per box.

Detector backends:
    transformers (default) - HuggingFace RT-DETR (PekingU/rtdetr_r50vd).
        Pure pip install, no C++/CUDA build step, works on CPU. Empirically
        more accurate than yolos-tiny on photos with overlapping people.
    detectron2 - same detector as the official demo.py. Usually more
        accurate but needs a compiled build environment.

Environment:
    - 4D-Humans installed as a package: `pip install -e .` (see setup.py,
      or configs/environment.yml for the full conda environment)
    - transformers backend (default): `pip install transformers`
    - checkpoint + SMPL support data in ~/.cache/4DHumans/: run
      `bash scripts/download_checkpoint.sh` once
    - SMPL neutral model: register at https://smplify.is.tue.mpg.de/ and
      place it at ~/.cache/4DHumans/data/smpl/SMPL_NEUTRAL.pkl (license
      restriction, not redistributed with this repo)

Usage (a folder of images, e.g. an extracted dataset):
    python s1_infer.py --img_folder /path/to/images --out results/s1_raw

Usage (as a library, e.g. from an evaluation or downstream-pipeline script):
    from s1_infer import HMR2Estimator
    est = HMR2Estimator()
    results = est.estimate(image_bgr)   # BGR numpy array, e.g. cv2.imread()
    # results: list[dict], one dict per detected person, keys as above
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# torch>=2.6 defaults to weights_only=True. The official pretrained
# checkpoint loaded here is a trusted source, but the pytorch-lightning
# version this repo pins predates that default change, so force it back off.
import torch  # noqa: E402
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# hmr2/utils/renderer.py unconditionally does `import pyrender`, which needs
# OpenGL/EGL and fails hard on machines without it (e.g. some servers),
# even though we never call any rendering function from here — we only need
# cam_crop_to_full(), a pure-math function defined in the same file. Install
# a no-op stub module before hmr2 is imported anywhere, so the import
# succeeds without touching real OpenGL/EGL.
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


class _CPUPredictorLazy:
    """Copy of detectron2's official DefaultPredictor_Lazy, with the one
    hardcoded `.cuda()` call replaced by the given device — needed on
    machines without an NVIDIA GPU."""

    def __init__(self, cfg, device):
        import detectron2.data.transforms as T
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import instantiate
        from detectron2.data import MetadataCatalog
        from omegaconf import OmegaConf

        self.model = instantiate(cfg.model)
        test_dataset = OmegaConf.select(cfg, "dataloader.test.dataset.names", default=None)
        if isinstance(test_dataset, (list, tuple)):
            test_dataset = test_dataset[0]

        DetectionCheckpointer(self.model).load(OmegaConf.select(cfg, "train.init_checkpoint", default=""))

        mapper = instantiate(cfg.dataloader.test.mapper)
        self.aug = mapper.augmentations
        self.input_format = mapper.image_format

        self.model.eval().to(device)
        self.device = device
        if test_dataset:
            self.metadata = MetadataCatalog.get(test_dataset)
        assert self.input_format in ["RGB", "BGR"], self.input_format
        self._T = T

    def __call__(self, original_image):
        with torch.no_grad():
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug(self._T.AugInput(original_image)).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            return self.model([inputs])[0]


class _TransformersPersonDetector:
    """Person-bounding-box detector using HuggingFace `transformers`, as an
    alternative to detectron2. Pure Python + pip install, no C++/CUDA build
    step required.

    Default model: PekingU/rtdetr_r50vd (RT-DETR). Empirically more accurate
    than hustvl/yolos-tiny on photos with overlapping/crowded people; only
    struggles on extremely dense crowds, which this project's photos
    (single person / small groups) are not expected to have. Swap models
    freely via --detector-model — anything in the
    `AutoModelForObjectDetection` family shares this same interface.

    Output box format matches the detectron2 path: (N,4) numpy array,
    [x1, y1, x2, y2], full-image pixel coordinates.
    """

    def __init__(self, model_name: str = "PekingU/rtdetr_r50vd", device: "torch.device" = None,
                 score_thresh: float = 0.5):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.device = device
        self.score_thresh = score_thresh
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name).to(device).eval()
        # Keep whichever COCO label id is named "person" — id2label numbering
        # can differ between models, so match by name rather than a hardcoded id.
        self.person_label_ids = {
            i for i, name in self.model.config.id2label.items() if name == "person"
        }

    def __call__(self, img_bgr: np.ndarray, score_thresh: float | None = None) -> np.ndarray:
        from PIL import Image

        # Allow the caller to override the threshold at call time (e.g. for
        # a threshold sweep during quantitative evaluation); falls back to
        # the value fixed at construction time otherwise.
        threshold = self.score_thresh if score_thresh is None else score_thresh

        img_rgb = img_bgr[:, :, ::-1]
        pil_img = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # target_sizes wants (height, width); PIL's .size is (width, height).
        target_sizes = torch.tensor([pil_img.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]
        boxes = [
            box.detach().cpu().numpy()
            for label, box in zip(results["labels"], results["boxes"])
            if int(label) in self.person_label_ids
        ]
        if not boxes:
            return np.zeros((0, 4), dtype=np.float32)
        return np.stack(boxes).astype(np.float32)


class HMR2Estimator:
    """Wraps model + detector loading. Build once, call estimate() repeatedly
    without reloading weights — this is the object to instantiate once per
    batch job over a whole dataset."""

    def __init__(self, checkpoint: str | None = None,
                 detector_backend: str = "transformers",
                 detector_model_name: str = "PekingU/rtdetr_r50vd",
                 detector_weights: str | None = None,
                 score_thresh: float = 0.5, device: str | None = None,
                 batch_size: int = 8):
        from hmr2.configs import CACHE_DIR_4DHUMANS
        from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT

        assert detector_backend in ("transformers", "detectron2"), detector_backend
        self.detector_backend = detector_backend
        self.batch_size = batch_size

        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Fall back to the package's own auto-download if the checkpoint
        # isn't there yet (normally you'd run scripts/download_checkpoint.sh
        # first; this is just a safety net).
        if not Path(checkpoint or DEFAULT_CHECKPOINT).exists():
            download_models(CACHE_DIR_4DHUMANS)
        self.model, self.model_cfg = load_hmr2(checkpoint or DEFAULT_CHECKPOINT)
        self.model = self.model.to(self.device).eval()

        if detector_backend == "transformers":
            self.detector = _TransformersPersonDetector(
                model_name=detector_model_name, device=self.device, score_thresh=score_thresh
            )
        else:
            from detectron2 import model_zoo
            resolved_weights = detector_weights or str(
                Path.home() / ".cache" / "4DHumans" / "detectron2" / "model_final_ef3a80.pkl"
            )
            det_cfg = model_zoo.get_config(
                "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
            )
            det_cfg.train.init_checkpoint = resolved_weights
            det_cfg.model.roi_heads.box_predictor.test_score_thresh = score_thresh
            det_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = _CPUPredictorLazy(det_cfg, self.device)

    def estimate(self, img_bgr: np.ndarray, score_thresh: float = 0.5) -> list[dict]:
        """Input one BGR numpy array (as returned by cv2.imread). Returns the
        SMPL parameters for every detected person (keys documented at the
        top of this file)."""
        from hmr2.datasets.vitdet_dataset import ViTDetDataset
        from hmr2.utils import recursive_to
        from hmr2.utils.renderer import cam_crop_to_full

        if self.detector_backend == "transformers":
            boxes = self.detector(img_bgr, score_thresh=score_thresh)  # already filtered (N,4) person boxes
        else:
            det_out = self.detector(img_bgr)
            det_instances = det_out["instances"]
            valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > score_thresh)
            boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        if len(boxes) == 0:
            return []

        dataset = ViTDetDataset(self.model_cfg, img_bgr, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        results: list[dict] = []
        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            pred_cam = out["pred_cam"]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            )
            cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            ).detach().cpu().numpy()

            for n in range(batch["img"].shape[0]):
                # `boxes` holds every person detected in this image, but `n`
                # is only the local index within this batch. When an image
                # has more people than batch_size, the DataLoader splits
                # into a second batch whose local n restarts at 0 — indexing
                # boxes[n] there would silently grab the wrong person's box.
                # `personid` is the global index into `boxes` that
                # ViTDetDataset assigns (see hmr2/datasets/vitdet_dataset.py:
                # self.personid = np.arange(len(boxes))), so use that instead.
                person_idx = int(batch["personid"][n])
                results.append({
                    "person_id": person_idx,
                    "betas": out["pred_smpl_params"]["betas"][n].detach().cpu().numpy(),
                    "body_pose": out["pred_smpl_params"]["body_pose"][n].detach().cpu().numpy(),
                    "global_orient": out["pred_smpl_params"]["global_orient"][n].detach().cpu().numpy(),
                    "cam_t": cam_t_full[n],
                    "scaled_focal_length": float(scaled_focal_length),
                    "pred_vertices": out["pred_vertices"][n].detach().cpu().numpy(),
                    "bbox": boxes[person_idx],
                })
        return results


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, help="path to a single image")
    ap.add_argument("--img_folder", type=str, help="batch mode: process every *.jpg/*.png in this folder")
    ap.add_argument("--out", type=str, default="results/s1_raw")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--detector-backend", type=str, default="transformers",
                     choices=["transformers", "detectron2"])
    ap.add_argument("--detector-model", type=str, default="PekingU/rtdetr_r50vd")
    ap.add_argument("--detector-weights", type=str, default=None,
                     help="detectron2 backend only. Defaults to "
                          "~/.cache/4DHumans/detectron2/model_final_ef3a80.pkl if omitted.")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8,
                     help="HMR2 inference batch size; increase on a GPU for throughput.")
    args = ap.parse_args()

    if not args.img and not args.img_folder:
        raise SystemExit("Provide either --img or --img_folder")

    img_paths: list[Path] = []
    if args.img:
        img_paths.append(Path(args.img))
    if args.img_folder:
        folder = Path(args.img_folder)
        img_paths.extend(sorted(folder.rglob("*.jpg")) + sorted(folder.rglob("*.png")))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading HMR2 model + detector (backend={args.detector_backend})... "
          f"({len(img_paths)} image(s) to process)")
    est = HMR2Estimator(checkpoint=args.checkpoint,
                         detector_backend=args.detector_backend,
                         detector_model_name=args.detector_model,
                         detector_weights=args.detector_weights,
                         score_thresh=args.score_thresh,
                         batch_size=args.batch_size)

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[warn] could not read {img_path}, skipping")
            continue
        people = est.estimate(img_bgr, score_thresh=args.score_thresh)
        print(f"{img_path.name}: {len(people)} person(s) detected")
        for p in people:
            npz_path = out_dir / f"{img_path.stem}_{p['person_id']}_smpl_params.npz"
            np.savez(npz_path, betas=p["betas"], body_pose=p["body_pose"],
                     global_orient=p["global_orient"], cam_t=p["cam_t"],
                     bbox=p["bbox"], scaled_focal_length=p["scaled_focal_length"],
                     pred_vertices=p["pred_vertices"])

    print(f"Done. Output written to {out_dir}")


if __name__ == "__main__":
    main()
