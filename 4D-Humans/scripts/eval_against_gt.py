#!/usr/bin/env python3
"""S1 量化評估：把 hmr2_estimator.py 的預測結果跟資料集提供的 GT SMPL 比對，
算 MPJPE / PA-MPJPE（姿勢/關節誤差）與 β 誤差（體型誤差）。

現況：3DPW、CloSe-Di 都還沒下載，這支是骨架——誤差計算（MPJPE/PA-MPJPE/
β 誤差、Procrustes 對齊）是純數學，已經用假資料驗證過邏輯正確（見
--self-test）。GT 載入函式（load_3dpw_gt / load_close_di_gt）的欄位名稱是
照公開文件記的格式寫的，**還沒有拿真實檔案驗證過**，標了 TODO 的地方拿到
真實資料後要先確認欄位名稱、再視情況調整。

已知需要之後確認/處理的事（不是本次骨架能解決的）：
    - 3DPW 的 GT 是用「性別化」SMPL（male/female）算的，HMR2 只輸出 neutral
      SMPL 的參數。目前本機只有 SMPL_NEUTRAL.pkl，兩邊用不同模型算出來的
      關節位置本身就有系統性差異，不是 HMR2 的誤差。要嘛額外去
      https://smpl.is.tue.mpg.de/ 下載性別化模型分開算，要嘛先接受這個
      已知誤差來源、在 finding 裡註明。
    - 3DPW 一段影片可能不只一個人，目前的 match_predictions_to_gt() 只用
      bbox IoU 配對，多人場景還沒有拿真實資料測過。
    - 3DPW 影像的抽幀檔名慣例（image_%05d.jpg）是官方釋出腳本的常見慣例，
      不是從這台機器上的真實檔案確認過，路徑對不上要調整
      _3dpw_frame_image_id()。

用法（骨架自我測試，不需要任何真實資料/GPU）：
    python scripts/eval_against_gt.py --self-test

用法（資料下載好、hmr2_estimator.py 已經跑過產生 --pred_dir 之後）：
    python scripts/eval_against_gt.py --dataset 3dpw \\
        --pred_dir results/s1_raw_3dpw --gt_dir /path/to/3DPW/sequenceFiles/test \\
        --out results/eval_3dpw.csv

    python scripts/eval_against_gt.py --dataset close-di \\
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
    # 跟 hmr2_estimator.py 同樣的理由：smplx 讀 SMPL_NEUTRAL.pkl 走 pickle，
    # 舊版 pytorch-lightning 沒跟上 torch>=2.6 weights_only=True 的新預設值。
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


# ---------------------------------------------------------------------------
# 共用資料結構
# ---------------------------------------------------------------------------

@dataclass
class PoseRecord:
    """一個人的 SMPL 參數，不管是預測還是 GT 都用這個格式，方便共用同一套
    比對/計算邏輯。"""
    image_id: str
    person_id: int
    betas: np.ndarray                      # (10,)
    body_pose_aa: np.ndarray                # (23,3) axis-angle，不含 global_orient
    global_orient_aa: np.ndarray            # (3,) axis-angle
    bbox: np.ndarray | None = None          # (4,) [x1,y1,x2,y2]，配對用
    joints_3d: np.ndarray | None = None     # (J,3)，若資料集直接提供就不用重算


# ---------------------------------------------------------------------------
# 姿勢表示轉換 / SMPL forward（算 joints 用）
# ---------------------------------------------------------------------------

def rotmat_to_aa(rotmat: np.ndarray) -> np.ndarray:
    """(...,3,3) 旋轉矩陣 → (...,3) axis-angle。"""
    from scipy.spatial.transform import Rotation
    shape = rotmat.shape[:-2]
    aa = Rotation.from_matrix(rotmat.reshape(-1, 3, 3)).as_rotvec()
    return aa.reshape(*shape, 3).astype(np.float32)


def build_smpl_layer(device: "torch.device" = None, gender: str = "neutral"):
    """載入 SMPL layer。目前本機只有 SMPL_NEUTRAL.pkl（HMR2 官方釋出的就只有
    這個），gender="male"/"female" 需要自行從 https://smpl.is.tue.mpg.de/
    另外下載 SMPL_MALE.pkl / SMPL_FEMALE.pkl 放到同一個資料夾才能用。"""
    import smplx
    from hmr2.configs import CACHE_DIR_4DHUMANS

    filename = {"neutral": "SMPL_NEUTRAL.pkl", "male": "SMPL_MALE.pkl",
                "female": "SMPL_FEMALE.pkl"}[gender]
    smpl_path = Path(CACHE_DIR_4DHUMANS) / "data" / "smpl" / filename
    if not smpl_path.exists():
        raise FileNotFoundError(
            f"{smpl_path} 不存在。gender={gender!r} 的 SMPL 模型需要自行到 "
            f"https://smpl.is.tue.mpg.de/ 註冊下載，放到這個路徑。"
        )
    return smplx.SMPLLayer(model_path=str(smpl_path), num_betas=10).to(device).eval()


def get_joints(smpl_layer, record: PoseRecord, device: "torch.device" = None) -> np.ndarray:
    """優先用資料集直接提供的 joints_3d；沒有的話才用 SMPL forward 算。"""
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
# 誤差計算（純數學，不依賴任何資料集格式，--self-test 驗證的就是這一段）
# ---------------------------------------------------------------------------

def compute_similarity_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Procrustes 對齊：找一個旋轉+縮放+平移，把 source 對到 target 上誤差最小
    （PA-MPJPE 用的標準做法，跟 SPIN/HMR 系列論文的實作邏輯一致）。
    source, target: (J,3)。回傳對齊後的 source，形狀不變。"""
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
    """Mean Per-Joint Position Error，單位 mm（假設輸入是公尺）。root-relative
    ：兩邊都先減掉骨盆關節座標，只看相對姿勢誤差，排除整體平移的影響。"""
    pred = pred_joints - pred_joints[pelvis_idx:pelvis_idx + 1]
    gt = gt_joints - gt_joints[pelvis_idx:pelvis_idx + 1]
    return float(np.linalg.norm(pred - gt, axis=-1).mean() * 1000)


def pa_mpjpe(pred_joints: np.ndarray, gt_joints: np.ndarray) -> float:
    """Procrustes-Aligned MPJPE，單位 mm。跟 mpjpe() 的差別是先做旋轉/縮放
    對齊，只看姿勢本身的形狀誤差，排除全域旋轉/縮放/平移——通常比 mpjpe() 小，
    是論文最常報的指標。"""
    aligned = compute_similarity_transform(pred_joints, gt_joints)
    return float(np.linalg.norm(aligned - gt_joints, axis=-1).mean() * 1000)


def beta_error(pred_betas: np.ndarray, gt_betas: np.ndarray) -> dict:
    """β（shape）誤差，逐維度 + 整體。不需要對齊，betas 本身就是跟姿勢/位置
    無關的體型係數。"""
    diff = pred_betas - gt_betas
    return {
        "beta_mae": float(np.abs(diff).mean()),
        "beta_l2": float(np.linalg.norm(diff)),
        "beta_per_dim_abs": np.abs(diff).tolist(),
    }


def bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """標準 IoU，box 格式 [x1,y1,x2,y2]。配對預測/GT 的人用。"""
    xa1, ya1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xa2, ya2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# 預測結果載入（讀 hmr2_estimator.py 存出來的 npz，格式已知、不是 TODO）
# ---------------------------------------------------------------------------

def load_predictions(pred_dir: Path) -> dict[str, list[PoseRecord]]:
    """掃 hmr2_estimator.py 存出來的 `<image_id>_<person_id>_smpl_params.npz`，
    依 image_id 分組回傳。"""
    by_image: dict[str, list[PoseRecord]] = {}
    for npz_path in sorted(pred_dir.glob("*_smpl_params.npz")):
        stem = npz_path.stem  # "<image_id>_<person_id>_smpl_params"
        image_id, person_id = stem.rsplit("_", 2)[0], stem.rsplit("_", 2)[1]
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
# GT 載入 —— TODO：這兩個函式還沒有拿真實資料測過，見檔頭說明
# ---------------------------------------------------------------------------

def load_3dpw_gt(seq_pkl_path: Path) -> list[PoseRecord]:
    """載入一個 3DPW sequence 的 GT（`sequenceFiles/{train,validation,test}/*.pkl`）。

    TODO 待真實資料驗證：欄位名稱（poses/betas/trans/jointPositions/genders/
    campose_valid）是官方論文與釋出腳本記載的標準格式，但還沒有拿真實 .pkl
    檔案跑過 `pickle.load` 確認過。拿到資料後第一步應該是：
        import pickle
        d = pickle.load(open(seq_pkl_path, "rb"), encoding="latin1")
        print(d.keys())
    確認欄位名稱一致再繼續，不一致要照實際欄位調整這裡。

    image_id 目前假設抽幀慣例是 `<sequence_name>/image_%05d`（3DPW 官方抽幀
    腳本的常見命名），要跟 load_predictions() 用同一套 image_id 對得上，
    才配對得到同一張圖的預測。
    """
    import pickle

    with open(seq_pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    seq_name = seq_pkl_path.stem
    records: list[PoseRecord] = []

    num_people = len(data["poses"])
    for person_id in range(num_people):
        poses = data["poses"][person_id]              # (num_frames, 72)
        betas = data["betas"][person_id][:10]           # (10,) 或 (300,)，只取前 10 維跟 HMR2 對齊
        joint_positions = data.get("jointPositions", [None] * num_people)[person_id]  # (num_frames, 24*3) 或 None
        valid = data.get("campose_valid", [None] * num_people)[person_id]

        num_frames = poses.shape[0]
        for frame_idx in range(num_frames):
            if valid is not None and not valid[frame_idx]:
                continue
            image_id = f"{seq_name}/image_{frame_idx:05d}"
            pose_frame = poses[frame_idx].reshape(24, 3)  # axis-angle，joint 0 = global_orient
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
    """載入一個 CloSe-Di scan 的 GT（規劃文件 3.2 節記載的欄位：betas/pose/
    trans/canon_pose）。

    TODO 待真實資料驗證：規劃文件是根據 CloSe-D 論文/repo 文件整理的欄位名稱，
    還沒有拿真實 .npz 檔案確認過。image_id 目前假設等於檔名 stem（CloSe-Di
    是逐 scan 一個檔案，不像 3DPW 是影片序列，配對邏輯簡單很多，風險較低，
    但一樣要拿到真實資料後跑一次確認)。
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
# 配對 + 評估主流程
# ---------------------------------------------------------------------------

def match_prediction(preds: list[PoseRecord], gt: PoseRecord) -> PoseRecord | None:
    """一張圖裡預測跟 GT 的人數可能不一樣（漏偵測/多偵測），用 bbox IoU 配對，
    取 IoU 最高且 > 0 的那個。只有一個人的情況（CloSe-Di 大多數 case）直接
    回傳唯一的預測。"""
    if not preds:
        return None
    if len(preds) == 1:
        return preds[0]
    if gt.bbox is None:
        return preds[0]  # 沒有 GT bbox 可比對，退而求其次選第一個（TODO：不理想）
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
    """通用評估迴圈，3DPW／CloSe-Di 共用——兩邊都已經轉成 PoseRecord 之後，
    後面的配對/算誤差邏輯完全一樣。"""
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
    print(f"共 {len(rows)} 筆 GT，成功配對並算出誤差 {len(ok_rows)} 筆。")
    if ok_rows:
        for key in ("mpjpe_mm", "pa_mpjpe_mm", "beta_mae", "beta_l2"):
            values = [r[key] for r in ok_rows]
            print(f"  {key}: mean={np.mean(values):.2f}  median={np.median(values):.2f}")


# ---------------------------------------------------------------------------
# 自我測試：不需要任何真實資料集，驗證誤差計算本身邏輯正確
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

    # case 1：預測 = GT，誤差應該是 0（驗證 mpjpe/pa_mpjpe 本身沒有寫錯正負號/單位）
    joints_gt = get_joints(smpl_layer, gt, device)
    err_zero_mpjpe = mpjpe(joints_gt, joints_gt)
    err_zero_pa = pa_mpjpe(joints_gt, joints_gt)
    print(f"[self-test] 預測=GT：mpjpe={err_zero_mpjpe:.6f}mm，"
          f"pa_mpjpe={err_zero_pa:.6f}mm（應該都 ~0）")
    assert err_zero_mpjpe < 1e-3, "相同輸入 MPJPE 應該是 0，數學寫錯了"
    assert err_zero_pa < 1e-3, "相同輸入 PA-MPJPE 應該是 0，數學寫錯了"

    # case 2：預測跟 GT 有一點雜訊，誤差應該 > 0，且 pa_mpjpe <= mpjpe
    # （Procrustes 對齊後誤差不會比對齊前大，這是這個指標的定義性質）
    pred_noisy = random_record("dummy_0001", 0, seed_offset=0.02)
    pred_noisy.betas = gt.betas + rng.normal(0, 0.3, 10).astype(np.float32)
    joints_pred = get_joints(smpl_layer, pred_noisy, device)
    m = mpjpe(joints_pred, joints_gt)
    pa = pa_mpjpe(joints_pred, joints_gt)
    beta_err = beta_error(pred_noisy.betas, gt.betas)
    print(f"[self-test] 預測≈GT+雜訊：mpjpe={m:.2f}mm，pa_mpjpe={pa:.2f}mm，"
          f"beta_mae={beta_err['beta_mae']:.4f}，beta_l2={beta_err['beta_l2']:.4f}")
    assert m > 0 and pa > 0, "有雜訊卻算出 0 誤差，數學寫錯了"
    assert pa <= m + 1e-4, "PA-MPJPE 理論上不該大於 MPJPE，數學寫錯了"

    # case 3：整條 evaluate() 流程（配對 + 寫 CSV）也跑一次，確認資料流通順
    pred_by_image = {"dummy_0001": [pred_noisy]}
    rows = evaluate(pred_by_image, [gt], smpl_layer, device)
    out_path = Path("results") / "eval_self_test.csv"
    write_csv(rows, out_path)
    print(f"[self-test] 全部通過，範例輸出見 {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                     help="用假資料驗證誤差計算邏輯，不需要任何真實資料集/下載")
    ap.add_argument("--dataset", type=str, choices=["3dpw", "close-di"])
    ap.add_argument("--pred_dir", type=str, help="hmr2_estimator.py 的 --out 資料夾")
    ap.add_argument("--gt_dir", type=str, help="3DPW 的 sequenceFiles/test，或 CloSe-Di 的資料夾")
    ap.add_argument("--out", type=str, default="results/eval.csv")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not (args.dataset and args.pred_dir and args.gt_dir):
        raise SystemExit("正式評估要給 --dataset --pred_dir --gt_dir，"
                          "或用 --self-test 先驗證邏輯。")

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
