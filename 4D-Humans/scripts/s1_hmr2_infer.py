#!/usr/bin/env python3
"""S1：用 pretrained HMR 2.0（4D-Humans）從影像估計 SMPL 參數 (β, θ, π)。

依規劃文件 6.0 節輸入格式，每個偵測到的人輸出：
    betas          (10,)        SMPL shape 係數
    body_pose      (23, 3, 3)   23 個關節的旋轉矩陣
    global_orient  (1, 3, 3)    根關節旋轉矩陣
    cam_t          (3,)         相機平移（弱透視相機，cam_crop_to_full 換算到全圖座標）

不用自己訓練，直接用官方 pretrained HMR2 checkpoint。整個流程分兩段：
    1. 「人在哪裡」——一個物件偵測器，從整張照片框出每個人的 bounding box
    2. 「這個人的 SMPL 參數」——HMR2 本體（純 ViT + transformer decoder，
       這段完全不依賴 detectron2，見 hmr2/datasets/vitdet_dataset.py）

第 1 段偵測器有兩種可選 backend：

    --detector-backend transformers（預設，建議 Windows / 沒有 C++ 編譯器的機器用）
        用 HuggingFace `transformers` 套件的物件偵測模型（預設 PekingU/rtdetr_r50vd，
        RT-DETR，準確度高、也支援 CPU）抓人的 bounding box，純 Python + pip 安裝，
        不需要編譯任何 C++/CUDA 擴充功能。實測比 hustvl/yolos-tiny 準很多——
        yolos-tiny 在人多、彼此重疊的照片裡容易重複框、框不準，換成
        rtdetr_r50vd 之後在真實照片（含真人擁抱、重疊姿勢）上都正確分開，
        只有在极度擁擠（十幾人緊貼在一起）的極端情況才會出現把多人框在一起
        的問題——這種情況在本專案實際會用到的照片（使用者單人或小群體照片）
        裡不太會遇到。如果想換回更快但較不準的 yolos-tiny，用
        `--detector-model hustvl/yolos-tiny`。

    --detector-backend detectron2（原本雲端 sandbox 用的方式）
        用官方 4D-Humans demo.py 同款的 detectron2 RegNetY Mask R-CNN。
        detectron2 需要編譯 C++ 擴充功能，Windows 上要另外裝 Visual Studio
        Build Tools 才裝得起來，比較麻煩，但偵測精度通常較高。

兩種 backend 吐出來的 bounding box 格式完全一樣（(N,4)，[x1,y1,x2,y2]，
原圖像素座標），後面 HMR2 本體的推論邏輯共用，不受影響。

輸出驗證：除了 <檔名>_boxes.jpg（畫偵測框），預設還會另外存一張
<檔名>_mesh_overlay.jpg——把估計出來的 SMPL 頂點（(β,θ,π) 算出來的 6890 個
點）用跟 hmr2 內部一樣的透視投影公式（純 numpy 重寫，不需要 pyrender/OpenGL，
見 project_vertices_to_image()）投影回原圖，疊成一堆小點畫在照片上。如果
(β,θ,π) 估計正確，這些點應該會準確貼合照片裡本人的輪廓、姿勢；如果明顯偏移
或形狀不對，就代表估計有問題。兩個旗標 --no-boxes-viz / --no-mesh-viz 可以
分別關掉。

環境需求（跟主 pipeline 的 requirements.txt 分開裝，這條路徑相依套件不同）：
    - 獨立虛擬環境，裝好 4D-Humans（pip install -e . 即可，不需要 [all]/detectron2
      這個 extra，除非你要用 --detector-backend detectron2）
    - transformers backend：另外 `pip install transformers`
    - ~/.cache/4DHumans/ 下要有 hmr2_data 解壓縮後的內容（checkpoint 等，
      正常執行 demo 時會自動下載）
    - ~/4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl（SMPL neutral
      模型，需要自己去 https://smplify.is.tue.mpg.de/ 註冊下載，程式第一次執行
      時會自動轉換、複製到 cache 目錄，不用手動處理）
    - detectron2 backend 才需要：detector 權重本地檔案路徑（見 --detector-weights）

用法（單張圖片）：
    python scripts/s1_hmr2_infer.py --img path/to/image.jpg --out data/s1_out

用法（整個資料夾，例如 render_pipeline.py 產出的 view_00.png 們）：
    python scripts/s1_hmr2_infer.py --img_folder data/deepFashion3DV2_rendered --out data/s1_out

也可以當模組直接 import 用（S2 接資料時用這個介面，不用另外寫 subprocess）：
    from scripts.s1_hmr2_infer import HMR2Estimator
    est = HMR2Estimator()   # 預設 detector_backend="transformers"
    results = est.estimate(image_bgr)   # cv2.imread 讀進來的 BGR numpy array
    # results: list[dict]，每個偵測到的人一個 dict，keys 見上面
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# torch>=2.6 預設 weights_only=True，但這裡讀的官方 pretrained checkpoint
# 是可信來源（已用 weights_only=False 手動驗證過內容），舊版 pytorch-lightning
# 沒跟上這個新預設值，強制蓋掉成 False。
import torch  # noqa: E402
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# 跟官方 demo.py 用同一組淺藍色人偶顏色（LIGHT_BLUE），render_mesh_shaded() 與
# write_obj() 的 .mtl 材質都用這個常數，避免兩處各寫一份數值、之後要調色時漏改。
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


# hmr2/utils/renderer.py 一被 import 就會 `import pyrender`，而 pyrender 底層要
# 借助 OpenGL/EGL 才能載入——這在 Linux（雲端 sandbox）沒問題，但 Windows 上沒有
# 這個東西，會直接讓 import 整串炸掉（就算你根本沒用到畫圖功能也一樣，因為是
# hmr2 套件自己在 import 階段就硬拉這個依賴）。
# 我們的 S1 流程完全不需要畫渲染圖，只需要同一支檔案裡 cam_crop_to_full() 這個
# 純數學函式，所以搶在 hmr2 任何東西被 import 之前，先塞一個空殼「假 pyrender」
# 進 sys.modules，讓 `import pyrender` 直接成功、不會真的去載入 OpenGL/EGL。
# 兩邊平台都用這個 stub 沒差——反正這支腳本從頭到尾都不會真的呼叫 pyrender。
import sys as _sys
import types as _types


class _DummyPyrenderAttr:
    """假的 pyrender 屬性/類別，存取或呼叫都回傳自己，不做任何事。只有真的要
    渲染畫面時才會用到 pyrender 的東西，S1 這條路徑用不到。"""

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def _pyrender_stub_getattr(name: str):
    # dunder 屬性（__file__/__spec__/__path__ 之類）交給 Python 自己的預設行為
    # 處理（跟內建模組一樣「沒有就沒有」），不要連這些也回傳假物件——
    # inspect/importlib 這類工具會去讀 __file__ 之類的屬性、預期拿到字串或
    # AttributeError，回傳一個假物件會讓它們自己內部邏輯壞掉（曾經在這裡踩到
    # `TypeError: expected str, bytes or os.PathLike object, not
    # _DummyPyrenderAttr` 這個錯誤，就是這樣造成的）。
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    return _DummyPyrenderAttr()


if "pyrender" not in _sys.modules:
    _pyrender_stub = _types.ModuleType("pyrender")
    _pyrender_stub.__getattr__ = _pyrender_stub_getattr
    _sys.modules["pyrender"] = _pyrender_stub


class _CPUPredictorLazy:
    """detectron2 官方 DefaultPredictor_Lazy 的複製版本，唯一差別是把寫死的
    .cuda() 換成傳入的 device——沒有 NVIDIA GPU 的環境需要這個修法，
    detectron2/hmr2 官方程式碼本身沒有處理這個情況。
    """

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
    """用 HuggingFace `transformers` 的物件偵測模型抓「人」的 bounding box，
    取代 detectron2。純 Python + pip 安裝，不需要編譯 C++/CUDA 擴充功能，
    Windows 上特別有感（detectron2 在 Windows 上要另外裝 Visual Studio Build
    Tools 才編得起來）。

    預設用 PekingU/rtdetr_r50vd（RT-DETR，準確度高，CPU 上也還算能接受的速度）。
    這是實測過的結論：hustvl/yolos-tiny 最快最小，但在人多重疊的照片裡容易重複框、
    抓不清楚；facebook/detr-resnet-50 中規中矩；rtdetr_r50vd 在真實照片（含真人
    擁抱、遮擋姿勢）上表現最好，只有在十幾人緊貼在一起這種極端擁擠場景才會出現
    把多人框在一起的情況——這在本專案實際會用到的照片（使用者單人或小群體照片）
    裡不太會遇到。這幾個都是 `AutoModelForObjectDetection` 家族，介面完全一樣，
    要換隨時可以透過 `--detector-model` 換掉。

    吐出來的 box 格式跟 detectron2 版本一致：(N,4) numpy array，
    [x1, y1, x2, y2]，原圖（未裁切、未縮放）的像素座標——這樣下游
    ViTDetDataset 的邏輯完全不用改。
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

        # 允許呼叫端在推論當下覆寫門檻（例如量化分析要掃不同 threshold），
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
    """封裝好 model + detector 的載入，重複呼叫 estimate() 不用重新載入權重。"""

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
        # 跟官方 demo.py 一樣，第一次執行時自動下載+解壓縮 HMR2 checkpoint
        # 到 ~/.cache/4DHumans（雲端 sandbox 因為網路白名單擋掉這個網域才需要
        # 手動搬檔案，你自己機器上網路正常的話這行會自動處理，不用手動做）。
        # 注意：download_models() 自己只檢查 hmr2_data.tar.gz 這個壓縮檔在不在，
        # 不會檢查解壓縮後的 checkpoint 在不在——如果你像雲端這邊一樣手動放好
        # 解壓縮後的內容、但沒留著原始 .tar.gz，它每次都會想重新下載。這裡先看
        # checkpoint 檔案存不存在，已經有就直接跳過，不去踩這個重複下載的問題。
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
            # 預設路徑用 Path.home()，不寫死 /root——這是原本雲端 sandbox（root
            # 使用者）專用的路徑，換一台機器（例如學校 server，非 root 使用者）
            # 就會找不到檔案。
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
        SMPL 參數清單。"""
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
                # 注意：boxes 是「這張圖全部偵測到的人」，但 n 只是這個 batch
                # 內的局部索引（0..batch_size-1）。一張圖偵測到的人數超過
                # batch_size 時（DataLoader 會分成第二個 batch），直接用 boxes[n]
                # 會拿到錯的人的 bbox（例如第二個 batch 的 n=0 其實是全域第
                # batch_size 個人，卻會被誤配到 boxes[0]）。personid 才是
                # ViTDetDataset 裡對應到 boxes 的原始全域索引（見
                # hmr2/datasets/vitdet_dataset.py 的 self.personid =
                # np.arange(len(boxes))），要用它來索引 boxes 才對。
                person_idx = int(batch["personid"][n])
                results.append({
                    "person_id": person_idx,
                    "betas": out["pred_smpl_params"]["betas"][n].detach().cpu().numpy(),
                    "body_pose": out["pred_smpl_params"]["body_pose"][n].detach().cpu().numpy(),
                    "global_orient": out["pred_smpl_params"]["global_orient"][n].detach().cpu().numpy(),
                    "cam_t": cam_t_full[n],
                    # 這張圖的 scaled focal length（scaled_focal_length 是對整個
                    # batch 的 img_size 取 max 算出來的純量，不是逐人索引的
                    # tensor——同一張圖裡每個人本來就共用同一個值，直接轉成
                    # float 就好，不能用 [n] 索引，0-dim tensor 索引會噴
                    # IndexError）。驗證用的 mesh 投影疊圖需要這個值，
                    # 見 project_vertices_to_image()。
                    "scaled_focal_length": float(scaled_focal_length),
                    "pred_vertices": out["pred_vertices"][n].detach().cpu().numpy(),
                    "bbox": boxes[person_idx],
                })
        return results


def project_vertices_to_image(vertices: np.ndarray, cam_t: np.ndarray,
                               focal_length: float, img_w: int, img_h: int) -> np.ndarray:
    """驗證用：把 SMPL 頂點（模型座標系，相機在原點、看向 +Z）投影回全圖 2D
    像素座標，純 numpy 實作，不需要 pyrender/OpenGL——邏輯完全對應
    hmr2/utils/geometry.py 的 perspective_projection()（rotation=單位矩陣，
    camera_center=圖片中心），只是那邊是 torch 版、這裡改成 numpy 方便單張圖
    快速畫圖驗證。

    vertices: (N, 3)
    cam_t: (3,)     cam_crop_to_full() 換算過的全圖空間相機平移
    focal_length: 純量，estimate() 回傳的 scaled_focal_length
    回傳: (N, 2) 像素座標（float，尚未取整數/裁切範圍）
    """
    cam_center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float64)
    pts = vertices.astype(np.float64) + cam_t.astype(np.float64)[None, :]
    projected = pts[:, :2] / pts[:, 2:3]
    projected = projected * focal_length + cam_center[None, :]
    return projected


def draw_mesh_overlay(img_bgr: np.ndarray, people: list[dict]) -> np.ndarray:
    """把每個人的 SMPL 頂點投影回原圖，畫成一堆小點疊在原圖上，方便肉眼確認
    估計出來的姿勢/體型跟照片本人準不準（S1 輸出正確性的視覺化驗證方法）。
    不同人用不同顏色循環區分。"""
    img_h, img_w = img_bgr.shape[:2]
    viz = img_bgr.copy()
    palette = [
        (0, 255, 0), (0, 128, 255), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (255, 0, 0), (128, 0, 255), (0, 200, 100),
    ]
    for p in people:
        color = palette[p["person_id"] % len(palette)]
        pts = project_vertices_to_image(
            p["pred_vertices"], p["cam_t"], p["scaled_focal_length"], img_w, img_h
        )
        x = np.clip(np.round(pts[:, 0]).astype(int), 0, img_w - 1)
        y = np.clip(np.round(pts[:, 1]).astype(int), 0, img_h - 1)
        # 每個投影點畫成 3x3 小方塊（單一像素太難用肉眼看清楚），用 numpy
        # 向量化位移取代逐點 cv2.circle，6890 個頂點也能瞬間畫完。
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx = np.clip(x + dx, 0, img_w - 1)
                yy = np.clip(y + dy, 0, img_h - 1)
                viz[yy, xx] = color
    return viz


def render_mesh_overlay_solid(img_bgr: np.ndarray, people: list[dict], faces: np.ndarray) -> np.ndarray:
    """把 SMPL 網格畫成「實心白色人偶」直接疊在原圖上（跟官方 demo.py 的
    overlay 渲染同一種效果——例如把花式滑冰選手直接換成白色人偶站在原本的
    冰場背景上），不是 draw_mesh_overlay() 那種一堆小點的散點疊圖。

    做法：不用 pyrender/OpenGL，改用純 numpy + OpenCV 的「軟體光柵化」——
    把每個三角形投影到 2D 螢幕座標，依照相機空間的深度由遠到近排序（畫家
    演算法 painter's algorithm），一個一個畫上去，這樣近的面自然會蓋掉遠
    的面，達到跟真的 3D 渲染一樣的正確遮擋效果；每個三角形的亮度用法向量
    跟「這個面朝向相機的方向」算內積做平面著色（跟 render_mesh_shaded() 同
    樣原理），背面（法向量背對相機的面，也就是身體內側/看不到的那一面）直
    接剔除不畫，這樣才會是實心人偶而不是穿透看得到背後輪廓的線框感。

    多人的話所有人的三角形會合併一起做深度排序（不是一個人畫完換下一個
    人），這樣人跟人之間互相重疊/遮擋時才會正確。
    """
    import cv2

    img_h, img_w = img_bgr.shape[:2]
    viz = img_bgr.copy()
    base_color_bgr = np.array([235.0, 233.0, 230.0])  # 淺灰白色人偶，BGR

    all_depths = []
    all_screen_tris = []
    all_intensity = []

    cam_center = np.array([img_w / 2.0, img_h / 2.0])
    for p in people:
        cam_verts = p["pred_vertices"].astype(np.float64) + p["cam_t"].astype(np.float64)[None, :]
        projected = cam_verts[:, :2] / cam_verts[:, 2:3] * p["scaled_focal_length"] + cam_center[None, :]

        tri_cam = cam_verts[faces]        # (F, 3, 3) 相機座標系，算法向量/深度用
        tri_screen = projected[faces]     # (F, 3, 2) 螢幕像素座標，畫圖用

        normals = np.cross(tri_cam[:, 1] - tri_cam[:, 0], tri_cam[:, 2] - tri_cam[:, 0])
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
        norm_len[norm_len == 0] = 1.0
        normals = normals / norm_len

        centroid = tri_cam.mean(axis=1)  # (F, 3)
        centroid_len = np.linalg.norm(centroid, axis=1, keepdims=True)
        centroid_len[centroid_len == 0] = 1.0
        view_dir = -centroid / centroid_len  # 面中心指向相機（原點）的方向

        intensity = np.sum(normals * view_dir, axis=1)
        depth = centroid[:, 2]

        # 背面剔除：只留法向量朝向相機的面（正面），不然實心人偶會變成
        # 半透明線框感（背後看不到的面也被畫出來，互相打架）。
        front_mask = intensity > 0.05
        all_depths.append(depth[front_mask])
        all_screen_tris.append(tri_screen[front_mask])
        all_intensity.append(np.clip(intensity[front_mask], 0.0, 1.0))

    if not all_depths:
        return viz

    depths = np.concatenate(all_depths)
    screen_tris = np.concatenate(all_screen_tris)
    intensities = np.concatenate(all_intensity)

    # 畫家演算法：由遠到近排序（深度大=遠，先畫；深度小=近，後畫，自然蓋過遠的）。
    order = np.argsort(-depths)
    screen_tris_int = np.round(screen_tris).astype(np.int32)

    for idx in order:
        shade = 0.35 + 0.65 * intensities[idx]  # 留個底，避免暗面死黑一片
        color = tuple(float(c) * shade for c in base_color_bgr)
        cv2.fillConvexPoly(viz, screen_tris_int[idx], color)

    return viz


def render_mesh_shaded(vertices: np.ndarray, faces: np.ndarray, out_path: Path,
                        elev: float = -90.0, azim: float = -90.0) -> None:
    """把 SMPL 網格畫成跟官方 demo.py 同一種風格的「淺藍色、白色背景、有立體
    光影」渲染圖——但不透過 pyrender/OpenGL（Windows 用不了），純用 matplotlib
    的 Poly3DCollection 手動做平面著色（flat shading）：每個三角形算法向量，
    跟「攝影機方向」（headlight，光源固定跟著攝影機走，不管哪個角度看，正對
    鏡頭的面永遠會被照亮，不用每個視角手動調光源方向）算內積當亮度，混合
    demo.py 原本用的同一組淺藍色 base color（LIGHT_BLUE = (0.651,0.741,0.859)）。
    只需要 numpy + matplotlib，兩邊平台都能跑，不需要額外安裝任何要編譯的套件。

    elev/azim 是 matplotlib 3D 視角參數，預設值是試出來的「正面站姿」角度
    （SMPL 網格的座標系是 Y 軸朝上，這組角度剛好對到人臉朝鏡頭、頭上腳下）。
    """
    tris = vertices[faces]  # (F, 3, 3)
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
    norm_len[norm_len == 0] = 1.0
    normals = normals / norm_len

    elev_r, azim_r = np.radians(elev), np.radians(azim)
    # headlight 方向：跟攝影機看過去的方向同一條軸，這樣不管 elev/azim 怎麼調，
    # 正對著鏡頭的面永遠有光，不用每個角度重新試光源方向。
    cam_dir = np.array([
        np.cos(elev_r) * np.cos(azim_r),
        np.cos(elev_r) * np.sin(azim_r),
        np.sin(elev_r),
    ])
    intensity = np.clip(normals @ cam_dir, 0.2, 1.0)

    colors = np.clip(np.array(LIGHT_BLUE)[None, :] * intensity[:, None], 0, 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(4, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.add_collection3d(Poly3DCollection(tris, facecolor=colors, edgecolor=None, linewidths=0))

    center = vertices.mean(0)
    span = (vertices.max(0) - vertices.min(0)).max() / 2 * 1.1
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=elev, azim=azim)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, facecolor="white")
    plt.close(fig)


def write_obj(vertices: np.ndarray, faces: np.ndarray, out_path: Path) -> None:
    """把 SMPL 頂點 + 固定不變的三角形拓樸，寫成標準 Wavefront .obj 檔案，方便
    直接用 Windows「3D 檢視器」或 Blender/MeshLab 打開檢視、量測。純文字輸出，
    不需要 pyrender/OpenGL——SMPL 的三角形拓樸（哪三個頂點組成一個面）是模型
    定義的一部分，跟這個人的姿勢/體型完全無關，隨便一組 (β,θ) 都能配上同一份
    faces。.obj 格式索引是 1-based，SMPL 的 faces 是 0-based，這裡要 +1。

    .obj 本身只存幾何（頂點座標+三角形），完全不含顏色資訊——單獨這個檔案
    在任何看檔軟體打開，都只會是預設的白/灰色，這是正常的，不是漏了什麼。
    這裡額外配一份同名的 .mtl 材質檔（標準搭配格式，Windows 3D 檢視器、
    Blender、MeshLab 都認得），把顏色設成跟官方 demo.py/render_mesh_shaded()
    同一組淺藍色，這樣打開才會直接看到有顏色的模型，不用再自己上材質。
    """
    mtl_name = out_path.stem + ".mtl"

    with open(out_path, "w") as f:
        f.write("# exported by s1_hmr2_infer.py (3dvto S1 pipeline)\n")
        f.write(f"mtllib {mtl_name}\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        f.write("usemtl smpl_body\n")
        for face in faces:
            f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    mtl_path = out_path.parent / mtl_name
    with open(mtl_path, "w") as f:
        f.write("# exported by s1_hmr2_infer.py (3dvto S1 pipeline)\n")
        f.write("newmtl smpl_body\n")
        f.write(f"Kd {LIGHT_BLUE[0]:.6f} {LIGHT_BLUE[1]:.6f} {LIGHT_BLUE[2]:.6f}\n")  # 漫反射色（主要顏色）
        f.write("Ka 0.0 0.0 0.0\n")   # 環境光反射
        f.write("Ks 0.1 0.1 0.1\n")   # 鏡面反射（弱一點，避免看起來太亮/塑膠感）
        f.write("Ns 10.0\n")          # 鏡面反射的集中度
        f.write("d 1.0\n")            # 不透明


def main() -> None:
    import cv2

    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, help="單張圖片路徑")
    ap.add_argument("--img_folder", type=str, help="批次處理整個資料夾（找 *.jpg/*.png）")
    ap.add_argument("--out", type=str, default="data/s1_out")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--detector-backend", type=str, default="transformers",
                     choices=["transformers", "detectron2"],
                     help="人物偵測器。transformers（預設）不需要 C++ 編譯器，"
                          "Windows 上建議用這個；detectron2 是原本雲端 sandbox 用的版本。")
    ap.add_argument("--detector-model", type=str, default="PekingU/rtdetr_r50vd",
                     help="detector-backend=transformers 時用的 HuggingFace 模型名稱，"
                          "預設 PekingU/rtdetr_r50vd（準確度高，實測結論）；"
                          "想換更快但較不準的可以用 hustvl/yolos-tiny。")
    ap.add_argument("--detector-weights", type=str, default=None,
                     help="detector-backend=detectron2 時的權重檔路徑。"
                          "不給的話預設 ~/.cache/4DHumans/detectron2/model_final_ef3a80.pkl。")
    ap.add_argument("--score-thresh", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8,
                     help="HMR2 推論的 batch size（同一張圖裡多個人一起送進模型）。"
                          "本機 CPU 用預設值即可；學校 server 有 GPU 時可以調大加速。")
    ap.add_argument("--no-boxes-viz", action="store_true",
                     help="預設會另外存一張畫出偵測框的 <檔名>_boxes.jpg，方便肉眼確認"
                          "偵測結果準不準；加這個旗標可以關掉這個行為。")
    ap.add_argument("--no-mesh-viz", action="store_true",
                     help="預設會另外存一張把 SMPL 頂點投影回原圖的 <檔名>_mesh_overlay.jpg"
                          "，這是驗證 S1 輸出的 (β,θ,π) 是否正確的方法之一——如果估計"
                          "正確，疊上去的點雲應該會準確貼合照片裡本人的輪廓/姿勢；"
                          "加這個旗標可以關掉這個行為。")
    ap.add_argument("--no-obj", action="store_true",
                     help="預設每個人另外存一份 <檔名>_<id>_smpl_params.obj"
                          "（SMPL 頂點+三角形，標準 Wavefront 格式，可以直接用"
                          "Windows 內建 3D 檢視器或 Blender/MeshLab 打開），"
                          "加這個旗標可以關掉這個行為。")
    ap.add_argument("--no-render", action="store_true",
                     help="預設每個人另外存一份 <檔名>_<id>_render.png"
                          "（淺藍色、白色背景、有立體光影的網格渲染圖，"
                          "跟官方 demo.py 用 pyrender 產生的風格一致，"
                          "但純用 matplotlib 畫，不需要 pyrender/OpenGL），"
                          "加這個旗標可以關掉這個行為。")
    ap.add_argument("--no-mesh-solid", action="store_true",
                     help="預設每張圖另外存一張 <檔名>_mesh_solid.jpg"
                          "（把所有人的 SMPL 網格畫成實心白色人偶、直接疊在"
                          "原圖背景上，效果跟官方 demo.py 的 overlay 渲染一致，"
                          "純用 OpenCV 軟體光柵化+畫家演算法排序，不需要"
                          "pyrender/OpenGL），加這個旗標可以關掉這個行為。")
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
            # bbox、scaled_focal_length 也一起存起來——不是規劃文件要求的 (β,θ,π)
            # 本體,但下游要做視覺化驗證/姿勢校正(例如 s1b_refine_leg_pose.py)
            # 都要重新投影回原圖,沒有這兩個值沒辦法算,省得下游還要重新跑一次偵測。
            np.savez(npz_path, betas=p["betas"], body_pose=p["body_pose"],
                     global_orient=p["global_orient"], cam_t=p["cam_t"],
                     bbox=p["bbox"], scaled_focal_length=p["scaled_focal_length"])

            if not args.no_obj:
                # faces 是 SMPL 固定拓樸，跟這個人的姿勢/體型無關，每個人都能
                # 共用同一份，不用重算。
                obj_path = out_dir / f"{img_path.stem}_{p['person_id']}_smpl_params.obj"
                write_obj(p["pred_vertices"], est.model.smpl.faces, obj_path)

            if not args.no_render:
                render_path = out_dir / f"{img_path.stem}_{p['person_id']}_render.png"
                render_mesh_shaded(p["pred_vertices"], est.model.smpl.faces, render_path)

        if not args.no_boxes_viz:
            # 把每個偵測到的框畫在原圖上另外存一份，方便肉眼確認偵測結果準不準
            # （框太多/重複框到同一個人，一眼就看得出來，比單看數字準）。
            viz = img_bgr.copy()
            for p in people:
                x1, y1, x2, y2 = [int(round(v)) for v in p["bbox"]]
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz, str(p["person_id"]), (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            viz_path = out_dir / f"{img_path.stem}_boxes.jpg"
            cv2.imwrite(str(viz_path), viz)

        if not args.no_mesh_viz and people:
            # SMPL 頂點投影疊圖——驗證 (β,θ,π) 估計正確性用，見 draw_mesh_overlay()。
            mesh_viz = draw_mesh_overlay(img_bgr, people)
            mesh_viz_path = out_dir / f"{img_path.stem}_mesh_overlay.jpg"
            cv2.imwrite(str(mesh_viz_path), mesh_viz)

        if not args.no_mesh_solid and people:
            # 實心白色人偶疊圖——跟官方 demo.py 的 overlay 渲染同一種效果，
            # 見 render_mesh_overlay_solid()。左右並排：原圖在左、疊圖在右，
            # 跟參考圖（論文常見的 before/after 對照排版）同一種呈現方式，
            # 中間留一條細白線分隔方便肉眼比對。
            solid_viz = render_mesh_overlay_solid(img_bgr, people, est.model.smpl.faces)
            divider = np.full((img_bgr.shape[0], 4, 3), 255, dtype=np.uint8)
            side_by_side = cv2.hconcat([img_bgr, divider, solid_viz])
            solid_viz_path = out_dir / f"{img_path.stem}_mesh_solid.jpg"
            cv2.imwrite(str(solid_viz_path), side_by_side)

    print(f"完成，輸出於 {out_dir}")


if __name__ == "__main__":
    main()
