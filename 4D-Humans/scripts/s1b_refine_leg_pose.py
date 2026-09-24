#!/usr/bin/env python3
"""S1 後處理：用 2D 姿態關鍵點（ViTPose）校正 S1 估計出來的腿部姿勢 θ。

背景（跟使用者討論過的結論，記在這裡方便之後回頭看）：
    HMR2 是單張影像的 3D 姿態回歸模型，沒有立體視覺/多視角資訊，遇到訓練資料
    裡少見的姿勢（例如坐姿、膝蓋大幅彎曲朝鏡頭方向伸）容易把膝關節的 3D 旋轉角度
    估錯，導致投影回 2D 之後小腿比例被壓縮、腳沒有落在照片裡真實腳的位置——這是
    模型準確度的限制，不是 s1_hmr2_infer.py 的 bug（已經用 _boxes.jpg 排除過
    「偵測框沒框到腳」這個可能性）。

    這個問題會直接影響規劃文件 S3 披覆階段的結果：S3 吃 (β, θ) 把服裝網格貼合到
    「目標人體的姿勢」，S1 的 θ 錯了，S3 畫出來的褲子就不會貼合照片裡真實的膝蓋
    位置，S4 算出來的分區鬆量也會跟著錯——所以腿部姿勢的準確度不是可以忽略的
    邊緣案例，尤其長褲/短褲本來就是 DeepFashion3D V2 涵蓋的服裝類別。

做法（SMPLify 風格的後處理，只修腿部，不重訓任何模型）：
    1. 用 ViTPose（HuggingFace `transformers` 內建，COCO 17 點格式）在 S1 已經
       算好的人物框內，抓出這個人左右髖、膝、踝共 6 個關節的真實 2D 位置。
    2. 把 S1 估計出來的 body_pose 裡，這 6 個關節對應的旋轉矩陣改成可訓練參數
       （用 axis-angle 參數化，梯度下降時每一步用 aa_to_rotmat() 轉回旋轉矩陣，
       確保過程中一直是合法的旋轉矩陣，不會跑出 SO(3) 之外），其餘 17 個關節
       （包含 β、global_orient、其他姿勢）全部凍結不動。
    3. 用 hmr2/utils/geometry.py 同一套 perspective_projection() 公式，把這 6 個
       關節投影回 2D，跟 ViTPose 偵測到的真實位置算加權 L2 距離（權重用 ViTPose
       自己吐出來的信心分數），反覆迭代到收斂，另外加一個小的正則化項讓結果不要
       離 S1 原始估計太遠（避免關鍵點被遮擋、信心低的時候被拉去很奇怪的角度）。
    4. 只有這 6 個關節被覆寫，其他姿勢、β、相機參數完全不動——手臂、軀幹、腳趾
       末端（跟之前討論過的，SMPL 本來就沒有腳趾關節）都不受影響。

用法（先跑過 s1_hmr2_infer.py，這裡吃它的輸出）：
    python scripts/s1b_refine_leg_pose.py --img path/to/image.jpg --s1_out data/s1_out --out data/s1_out_refined

環境需求：除了 s1_hmr2_infer.py 原本的環境，多一個 `pip install transformers`
（如果還沒裝的話——用 RT-DETR 偵測器時應該已經裝過了，這裡直接重複利用同一套
transformers 安裝，不需要另外裝任何新套件）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

import torch  # noqa: E402
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    # 跟 s1_hmr2_infer.py 同樣的理由：舊版 pytorch-lightning 沒跟上
    # torch>=2.6 weights_only=True 的新預設值。這裡雖然不讀 lightning checkpoint，
    # 但 smplx 讀 SMPL_NEUTRAL.pkl 也是走 pickle，保險起見一起處理掉。
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

# 跟 s1_hmr2_infer.py 一樣需要的 pyrender stub：hmr2.utils.geometry 這個模組本身
# 沒有 pyrender 依賴，但只要 import 到 hmr2.utils 這個套件（例如透過
# `from hmr2.utils.geometry import ...`），Python 就會先執行 hmr2/utils/__init__.py，
# 而它會 import 同目錄下的 renderer.py，renderer.py 一被 import 就會硬拉
# `import pyrender`——這在 Windows 上會直接炸掉（詳細原因見 s1_hmr2_infer.py
# 檔頭的說明）。用同一招假 pyrender stub 解決。
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


# 跟 s1_hmr2_infer.py 同目錄，直接重複利用它裡面的投影/畫圖函式，
# 不要重寫第二份一樣的邏輯。
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_THIS_DIR))
from s1_hmr2_infer import (  # noqa: E402
    project_vertices_to_image, draw_mesh_overlay, write_obj, render_mesh_shaded,
)


# COCO 17 點關鍵點順序——這是業界標準慣例（几乎所有輸出「COCO 格式」的 2D 姿態
# 模型都遵守這個順序），不是從 transformers 的 VitPose config 讀出來的（它沒有存
# 逐點名稱），這裡手動寫死方便對照。
COCO_KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# 要校正的 6 個腿部關節：(COCO 關鍵點名稱, SMPL joint 編號, body_pose 陣列索引)
# SMPL 24 個關節順序（pelvis=0 開始）：...,1 left_hip,2 right_hip,...,4 left_knee,
# 5 right_knee,...,7 left_ankle,8 right_ankle,...——已經用 smplx 的 joint 定義核對過。
# body_pose 只存 pelvis 以外的 23 個關節，所以 body_pose 索引 = SMPL joint 編號 - 1。
LEG_JOINTS = [
    ("left_hip", 1, 0),
    ("right_hip", 2, 1),
    ("left_knee", 4, 3),
    ("right_knee", 5, 4),
    ("left_ankle", 7, 6),
    ("right_ankle", 8, 7),
]


def _rotmat_to_axis_angle(rotmat: np.ndarray) -> np.ndarray:
    """(3,3) 旋轉矩陣轉 axis-angle (3,)，只在迭代開始前呼叫一次，用來當優化變數的
    初始值，不需要可微分（之後迭代都是用 aa_to_rotmat() 這個 torch 版反過來算）。"""
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(rotmat).as_rotvec().astype(np.float32)


def detect_leg_keypoints_2d(img_bgr: np.ndarray, bbox: np.ndarray,
                             model_name: str = "usyd-community/vitpose-base-simple",
                             device: "torch.device" = None) -> tuple[np.ndarray, np.ndarray]:
    """用 ViTPose 在給定的人物框內抓 2D 關鍵點。
    回傳 (leg_keypoints_2d, leg_scores)，形狀都是 (6,2) / (6,)，順序對應 LEG_JOINTS。
    """
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation
    from PIL import Image

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = VitPoseForPoseEstimation.from_pretrained(model_name).to(device).eval()

    img_rgb = img_bgr[:, :, ::-1]
    pil_img = Image.fromarray(img_rgb)

    x1, y1, x2, y2 = bbox
    box_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
    inputs = processor(pil_img, boxes=[[box_xywh]], return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # 注意：boxes 是原圖絕對像素座標（不是 0~1 正規化），所以這裡不傳
    # target_sizes，post_process_pose_estimation 才會直接吐出原圖像素座標
    # （跟 transformers 官方 docstring 範例的用法一致）。
    results = processor.post_process_pose_estimation(outputs, boxes=[[box_xywh]])[0][0]
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
    """核心的 SMPLify-lite 優化迴圈。只調整 LEG_JOINTS 這 6 個關節，其餘凍結。
    回傳修正後的完整 body_pose (23,3,3)。"""
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
        # ViTPose 對這個人幾乎沒信心（例如遮擋太嚴重），校正沒有可靠的依據，
        # 與其硬套一個不可靠的結果，不如原樣放棄、回傳原本 S1 的估計。
        print("  [警告] ViTPose 對腿部關鍵點的信心分數全部低於門檻，跳過校正，"
              "沿用 S1 原始估計。")
        return body_pose

    def _project_leg_joints(theta: torch.Tensor) -> torch.Tensor:
        """給定 6 個腿部關節的 axis-angle，回傳它們投影到 2D 的像素座標 (6,2)。
        只用來算診斷用的重投影誤差，不影響訓練梯度（呼叫端會包在 no_grad 裡）。"""
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

    # 診斷用：校正「前」的重投影位置（用初始 theta，也就是 S1 原始角度），
    # 拿來跟 ViTPose 抓到的 target 位置比對，量化「S1 原本錯多少」。
    with torch.no_grad():
        proj_before = _project_leg_joints(theta_leg_init)
        err_before = (proj_before - target_kp_t).norm(dim=1)  # (6,) 像素距離

    optimizer = torch.optim.Adam([theta_leg], lr=lr)
    for _ in range(iters):
        optimizer.zero_grad()
        rotmats_leg = aa_to_rotmat(theta_leg)  # (6,3,3)，每一步都重新投影回合法旋轉矩陣
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

        # 診斷用：校正「後」的重投影位置，跟 target 比對，量化「校正完還剩多少誤差」；
        # 另外也印出「校正前 vs 校正後」關節本身移動了多少像素——如果這個數字很小，
        # 代表優化過程根本沒怎麼動，才是真正該懷疑的異常（而不是肉眼看渲染圖看不出來）。
        proj_after = _project_leg_joints(theta_leg.detach())
        err_after = (proj_after - target_kp_t).norm(dim=1)
        moved = (proj_after - proj_before).norm(dim=1)

    print("  診斷（像素距離，quad 越小越好）：")
    for i, (name, _, _) in enumerate(LEG_JOINTS):
        flag = "" if target_scores[i] > score_thresh else "（信心分數過低，未採用）"
        print(f"    {name:12s} 校正前誤差={err_before[i]:6.1f}px  "
              f"校正後誤差={err_after[i]:6.1f}px  關節位移={moved[i]:6.1f}px {flag}")

    return body_pose_final[0].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", type=str, required=True, help="跟餵給 s1_hmr2_infer.py 同一張圖")
    ap.add_argument("--s1_out", type=str, required=True, help="s1_hmr2_infer.py 的 --out 資料夾")
    ap.add_argument("--out", type=str, default="data/s1_out_refined")
    ap.add_argument("--vitpose-model", type=str, default="usyd-community/vitpose-base-simple")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--reg-weight", type=float, default=5.0,
                     help="正則化強度，越大代表越不信任 2D 關鍵點、越貼近 S1 原始估計。")
    ap.add_argument("--score-thresh", type=float, default=0.3,
                     help="ViTPose 關鍵點信心分數門檻，低於這個值的關節不納入校正依據。")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-render", action="store_true",
                     help="預設每個人另外存一份 <檔名>_<id>_smpl_params_refined_render.png"
                          "（淺藍色、白色背景、有立體光影的網格渲染圖，"
                          "校正後的最終結果），加這個旗標可以關掉這個行為。")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    img_path = Path(args.img)
    s1_out_dir = Path(args.s1_out)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_paths = sorted(s1_out_dir.glob(f"{img_path.stem}_*_smpl_params.npz"))
    if not npz_paths:
        raise SystemExit(f"在 {s1_out_dir} 找不到 {img_path.stem}_*_smpl_params.npz，"
                          f"要先跑過 s1_hmr2_infer.py（而且要是加了 bbox/"
                          f"scaled_focal_length 這個新版本存出來的 npz）。")

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise SystemExit(f"讀不到圖片 {img_path}")
    img_h, img_w = img_bgr.shape[:2]

    print("載入 SMPL model...")
    import smplx
    from hmr2.configs import CACHE_DIR_4DHUMANS
    smpl_path = Path(CACHE_DIR_4DHUMANS) / "data" / "smpl" / "SMPL_NEUTRAL.pkl"
    smpl_layer = smplx.SMPLLayer(model_path=str(smpl_path), num_betas=10).to(device).eval()

    print(f"載入 ViTPose（{args.vitpose_model}）...")
    all_people_before = []
    all_people_after = []

    for npz_path in npz_paths:
        data = np.load(npz_path)
        if "bbox" not in data or "scaled_focal_length" not in data:
            print(f"  [跳過] {npz_path.name} 是舊版 s1_hmr2_infer.py 存出來的，"
                  f"沒有 bbox/scaled_focal_length，重新跑一次 S1 再來校正。")
            continue

        person_id = npz_path.stem.split("_")[-3]  # <stem>_<id>_smpl_params.npz
        betas = data["betas"]
        body_pose = data["body_pose"]
        global_orient = data["global_orient"]
        cam_t = data["cam_t"]
        bbox = data["bbox"]
        scaled_focal_length = float(data["scaled_focal_length"])

        print(f"[{npz_path.name}] 偵測 2D 腿部關鍵點...")
        leg_kp_2d, leg_scores = detect_leg_keypoints_2d(
            img_bgr, bbox, model_name=args.vitpose_model, device=device
        )
        for (name, _, _), score in zip(LEG_JOINTS, leg_scores):
            print(f"    {name}: score={score:.2f}")

        print(f"[{npz_path.name}] 優化腿部姿勢中（{args.iters} 次迭代）...")
        refined_body_pose = refine_leg_pose(
            smpl_layer, betas, body_pose, global_orient, cam_t, scaled_focal_length,
            img_w, img_h, leg_kp_2d, leg_scores,
            iters=args.iters, lr=args.lr, reg_weight=args.reg_weight,
            score_thresh=args.score_thresh, device=device,
        )

        out_npz = out_dir / npz_path.name.replace("_smpl_params.npz", "_smpl_params_refined.npz")
        np.savez(out_npz, betas=betas, body_pose=refined_body_pose,
                 global_orient=global_orient, cam_t=cam_t, bbox=bbox,
                 scaled_focal_length=scaled_focal_length)
        print(f"  已存 {out_npz}")

        # 用修正前後兩組 body_pose 各算一次頂點，方便直接肉眼比較有沒有真的改善。
        with torch.no_grad():
            betas_t = torch.tensor(betas, dtype=torch.float32, device=device)[None]
            global_orient_t = torch.tensor(global_orient, dtype=torch.float32, device=device)[None]

            body_pose_before_t = torch.tensor(body_pose, dtype=torch.float32, device=device)[None]
            verts_before = smpl_layer(betas=betas_t, body_pose=body_pose_before_t,
                                       global_orient=global_orient_t).vertices[0].cpu().numpy()

            body_pose_after_t = torch.tensor(refined_body_pose, dtype=torch.float32, device=device)[None]
            verts_after = smpl_layer(betas=betas_t, body_pose=body_pose_after_t,
                                      global_orient=global_orient_t).vertices[0].cpu().numpy()

        # 這才是整條 pipeline目前的「最終結果」（S1 + 腿部校正之後）——直接寫成
        # .obj，跟 s1_hmr2_infer.py 用同一份 write_obj()，faces 是 SMPL 固定
        # 拓樸，共用同一個 smpl_layer.faces 即可。
        out_obj = out_dir / npz_path.name.replace("_smpl_params.npz", "_smpl_params_refined.obj")
        write_obj(verts_after, smpl_layer.faces, out_obj)
        print(f"  已存 {out_obj}")

        if not args.no_render:
            out_render = out_dir / npz_path.name.replace("_smpl_params.npz", "_smpl_params_refined_render.png")
            render_mesh_shaded(verts_after, smpl_layer.faces, out_render)
            print(f"  已存 {out_render}")

        all_people_before.append({"person_id": int(person_id), "pred_vertices": verts_before,
                                   "cam_t": cam_t, "scaled_focal_length": scaled_focal_length})
        all_people_after.append({"person_id": int(person_id), "pred_vertices": verts_after,
                                  "cam_t": cam_t, "scaled_focal_length": scaled_focal_length})

    if all_people_before:
        overlay_before = draw_mesh_overlay(img_bgr, all_people_before)
        overlay_after = draw_mesh_overlay(img_bgr, all_people_after)
        side_by_side = np.concatenate([overlay_before, overlay_after], axis=1)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_before_after.jpg"), side_by_side)
        print(f"完成，校正前後對照圖存在 {out_dir / f'{img_path.stem}_before_after.jpg'}"
              f"（左：校正前，右：校正後）")


if __name__ == "__main__":
    main()
