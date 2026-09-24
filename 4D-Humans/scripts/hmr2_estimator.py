#!/usr/bin/env python3
"""S1：用 pretrained HMR 2.0（4D-Humans）從影像估計 SMPL 參數 (β, θ, π)。

給大規模資料集跑量化評估用的精簡版（3DPW／CloSe-Di 對 GT SMPL 的比對）。
只留推論本體，不含 s1_hmr2_infer.py 那邊的視覺化／debug 功能（畫框、疊圖、
渲染、匯出 .obj）——批次跑整個資料集不需要這些，也會拖慢速度、佔硬碟。
如果要肉眼檢查單張圖片的估計品質，請用 s1_hmr2_infer.py。

每個偵測到的人輸出：
    betas          (10,)        SMPL shape 係數
    body_pose      (23, 3, 3)   23 個關節的旋轉矩陣
    global_orient  (1, 3, 3)    根關節旋轉矩陣
    cam_t          (3,)         相機平移（弱透視相機，已換算到全圖座標）
    pred_vertices  (6890, 3)    SMPL 頂點（模型座標系），算 MPJPE 等指標時要從這裡取關節

不用自己訓練，直接用官方 pretrained HMR2 checkpoint。流程分兩段：
    1. 物件偵測器抓每個人的 bounding box（預設 HuggingFace RT-DETR，實測比
       yolos-tiny 準，純 pip 安裝不需要編譯 C++/CUDA；也支援 --detector-backend
       detectron2 走官方 demo.py 同款偵測器，精度通常更高但需要編譯環境）
    2. HMR2 本體（純 ViT + transformer decoder）算這個人的 SMPL 參數

環境需求：
    - 獨立虛擬環境，裝好 4D-Humans（pip install -e .）
    - transformers backend（預設）：另外 pip install transformers
    - ~/.cache/4DHumans/ 下要有 checkpoint（跑一次會自動下載）
    - ./data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl（SMPL neutral model，
      需自行到 https://smplify.is.tue.mpg.de/ 註冊下載）

用法（整個資料夾，例如展開後的 3DPW / CloSe-Di 影像）：
    python scripts/hmr2_estimator.py --img_folder /path/to/images --out results/s1_raw

當模組 import 用（寫比對/評估腳本時用這個介面）：
    from scripts.hmr2_estimator import HMR2Estimator
    est = HMR2Estimator()
    results = est.estimate(image_bgr)   # cv2.imread 讀進來的 BGR numpy array
    # results: list[dict]，每個偵測到的人一個 dict，keys 見上面
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# torch>=2.6 預設 weights_only=True，這裡讀的官方 pretrained checkpoint是可信
# 來源，但舊版 pytorch-lightning 沒跟上這個新預設值，強制蓋掉成 False。
import torch  # noqa: E402
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# hmr2/utils/renderer.py 一被 import 就會硬拉 `import pyrender`，而 pyrender
# 需要 OpenGL/EGL，在沒裝這些的機器（例如某些學校 server）上會直接讓 import
# 整串失敗——即使根本用不到畫圖功能也一樣。這裡只需要同一支檔案裡
# cam_crop_to_full() 這個純數學函式，所以搶在 hmr2 任何東西被 import 之前，
# 先塞一個空殼「假 pyrender」進 sys.modules，讓 `import pyrender` 直接成功。
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
    """detectron2 官方 DefaultPredictor_Lazy 的複製版本，唯一差別是把寫死的
    .cuda() 換成傳入的 device——沒有 NVIDIA GPU 的環境需要這個修法。"""

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
    """用 HuggingFace transformers 的物件偵測模型抓「人」的 bounding box，取代
    detectron2。純 Python + pip 安裝，不需要編譯 C++/CUDA 擴充功能。預設
    PekingU/rtdetr_r50vd（實測比 yolos-tiny 準，尤其人物重疊的照片）。

    吐出來的 box 格式跟 detectron2 版本一致：(N,4) numpy array，
    [x1, y1, x2, y2]，原圖像素座標。
    """

    def __init__(self, model_name: str = "PekingU/rtdetr_r50vd", device: "torch.device" = None,
                 score_thresh: float = 0.5):
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        self.device = device
        self.score_thresh = score_thresh
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name).to(device).eval()
        # COCO 類別裡凡是叫 "person" 的 label id 都留下（不同模型的 id2label
        # 編號可能不同，用名字比對比較保險，不要寫死數字）
        self.person_label_ids = {
            i for i, name in self.model.config.id2label.items() if name == "person"
        }

    def __call__(self, img_bgr: np.ndarray, score_thresh: float | None = None) -> np.ndarray:
        from PIL import Image

        # 允許呼叫端在推論當下覆寫門檻（量化分析要掃不同 threshold時用得到），
        # 沒給就用建構子當初設定的值。
        threshold = self.score_thresh if score_thresh is None else score_thresh

        img_rgb = img_bgr[:, :, ::-1]
        pil_img = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # target_sizes 要給 (height, width)，PIL 的 .size 是 (width, height)
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
    """封裝好 model + detector 的載入，重複呼叫 estimate() 不用重新載入權重
    ——批次跑整個資料集時，這個物件只建立一次，對每張圖呼叫 estimate()。"""

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
        # 第一次執行時自動下載+解壓縮 HMR2 checkpoint 到 ~/.cache/4DHumans。
        # download_models() 只檢查壓縮檔在不在、不檢查解壓縮後的內容，這裡先看
        # checkpoint 檔案存不存在，已經有就跳過，避免重複下載。
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
            # 預設路徑用 Path.home()，不要寫死特定使用者路徑，換一台機器
            # （例如學校 server）才不會找不到檔案。
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
        """輸入一張 BGR numpy array（cv2.imread 的格式），回傳偵測到的每個人的
        SMPL 參數清單（keys 見檔頭）。"""
        from hmr2.datasets.vitdet_dataset import ViTDetDataset
        from hmr2.utils import recursive_to
        from hmr2.utils.renderer import cam_crop_to_full

        if self.detector_backend == "transformers":
            boxes = self.detector(img_bgr, score_thresh=score_thresh)  # 已過濾好的 (N,4) person box
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
                # boxes 是這張圖全部偵測到的人，但 n 只是這個 batch 內的局部
                # 索引；一張圖人數超過 batch_size 時要用 personid（ViTDetDataset
                # 裡對應到 boxes 的全域索引）才能正確取回這個人的 bbox。
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
    ap.add_argument("--img", type=str, help="單張圖片路徑")
    ap.add_argument("--img_folder", type=str, help="批次處理整個資料夾（找 *.jpg/*.png）")
    ap.add_argument("--out", type=str, default="results/s1_raw")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--detector-backend", type=str, default="transformers",
                     choices=["transformers", "detectron2"])
    ap.add_argument("--detector-model", type=str, default="PekingU/rtdetr_r50vd")
    ap.add_argument("--detector-weights", type=str, default=None,
                     help="detector-backend=detectron2 時的權重檔路徑。"
                          "不給的話預設 ~/.cache/4DHumans/detectron2/model_final_ef3a80.pkl。")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8,
                     help="HMR2 推論的 batch size；有 GPU 時可調大加速。")
    args = ap.parse_args()

    if not args.img and not args.img_folder:
        raise SystemExit("要給 --img 或 --img_folder 其中一個")

    img_paths: list[Path] = []
    if args.img:
        img_paths.append(Path(args.img))
    if args.img_folder:
        folder = Path(args.img_folder)
        img_paths.extend(sorted(folder.rglob("*.jpg")) + sorted(folder.rglob("*.png")))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"載入 HMR2 model + detector（backend={args.detector_backend}）..."
          f"（共 {len(img_paths)} 張圖片待處理）")
    est = HMR2Estimator(checkpoint=args.checkpoint,
                         detector_backend=args.detector_backend,
                         detector_model_name=args.detector_model,
                         detector_weights=args.detector_weights,
                         score_thresh=args.score_thresh,
                         batch_size=args.batch_size)

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[警告] 讀不到 {img_path}，跳過")
            continue
        people = est.estimate(img_bgr, score_thresh=args.score_thresh)
        print(f"{img_path.name}: 偵測到 {len(people)} 個人")
        for p in people:
            npz_path = out_dir / f"{img_path.stem}_{p['person_id']}_smpl_params.npz"
            np.savez(npz_path, betas=p["betas"], body_pose=p["body_pose"],
                     global_orient=p["global_orient"], cam_t=p["cam_t"],
                     bbox=p["bbox"], scaled_focal_length=p["scaled_focal_length"],
                     pred_vertices=p["pred_vertices"])

    print(f"完成，輸出於 {out_dir}")


if __name__ == "__main__":
    main()
