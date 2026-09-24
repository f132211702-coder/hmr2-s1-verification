# hmr2.0 — S1（人體幾何估計）驗證

3D 虛擬試穿 pipeline 的 **S1** 階段：RGB 影像 → SMPL (β, θ, π)，使用官方 pretrained **HMR 2.0**（4D-Humans），不重訓。完整 pipeline 規劃見《3D-VTO-Pipeline-規劃.md》（S2–S4 由同組同學負責，不在本 repo 範圍）。

本 repo 依循 `knowledge / inbox / tasks / experiments / runs / views / work` 的知識管理骨架，詳見 [knowledge/PROJECT_MODEL.md](knowledge/PROJECT_MODEL.md) 與 [AGENTS.md](AGENTS.md)。目前狀態見 [knowledge/PROJECT_STATE.md](knowledge/PROJECT_STATE.md)。

## 快速開始

本機開發用 pip 即可；**正式跑分請照 `4D-Humans/environment.yml`**（conda + GPU + CUDA 11.8），本機 Mac 是 CPU-only 開發環境，不是部署設定。

```bash
cd 4D-Humans
python -m venv ../hmr2_venv
source ../hmr2_venv/bin/activate
pip install torch
pip install -e .[all]
pip install transformers   # s1_hmr2_infer.py / s1b_refine_leg_pose.py 需要
```

SMPL neutral model 需自行到 https://smplify.is.tue.mpg.de/ 註冊下載，放到 `4D-Humans/data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`。

## 執行 S1 推論

```bash
cd 4D-Humans
python scripts/s1_hmr2_infer.py --img path/to/image.jpg --out data/s1_out
python scripts/s1b_refine_leg_pose.py --img path/to/image.jpg --s1_out data/s1_out --out data/s1_out_refined
```

也可以當模組 import：

```python
from scripts.s1_hmr2_infer import HMR2Estimator
est = HMR2Estimator()
results = est.estimate(image_bgr)  # cv2.imread 讀進來的 BGR numpy array
```

## 批次跑資料集（3DPW / CloSe-Di 量化評估）

`s1_hmr2_infer.py` 含大量視覺化/debug 功能，批次跑上千張圖太慢也太佔硬碟。跑資料集評估請用精簡版 [`hmr2_estimator.py`](4D-Humans/scripts/hmr2_estimator.py)（同一套 `HMR2Estimator`，只留推論本體，見 [knowledge/DECISIONS.md](knowledge/DECISIONS.md) D-005）：

```bash
python scripts/hmr2_estimator.py --img_folder /path/to/3DPW/images --out results/s1_raw
```

每個人輸出一份 `.npz`（`betas`/`body_pose`/`global_orient`/`cam_t`/`bbox`/`scaled_focal_length`/`pred_vertices`），再用 [`eval_against_gt.py`](4D-Humans/scripts/eval_against_gt.py) 跟資料集提供的 GT SMPL 比對，算 MPJPE／PA-MPJPE／β 誤差：

```bash
# 不需要任何真實資料，先驗證誤差計算邏輯本身正確
python scripts/eval_against_gt.py --self-test

# 資料下載好之後（見 knowledge/QUESTIONS.md 待確認的欄位假設）
python scripts/eval_against_gt.py --dataset 3dpw \
    --pred_dir results/s1_raw_3dpw --gt_dir /path/to/3DPW/sequenceFiles/test \
    --out results/eval_3dpw.csv
```

目前 GT 載入的部分（3DPW／CloSe-Di 各自的檔案格式）還是骨架，還沒拿真實資料驗證過，見 [knowledge/DECISIONS.md](knowledge/DECISIONS.md) D-006 與 [experiments/](experiments/)。

## 目錄結構

```
knowledge/     canonical project knowledge（PROJECT_MODEL / PROJECT_STATE / QUESTIONS / FINDINGS / DECISIONS）
inbox/         尚未整理的筆記、參考資料
tasks/         工作項目（TASK-xxx）
experiments/   實驗定義（EXP-xxx）
runs/          執行紀錄與 evidence（RUN-xxx）
views/         由 knowledge/runs 產生的唯讀衍生視圖
work/          可丟棄的暫存工作區
src/           非 vendored 的獨立程式碼（目前為空）
scripts/       repo 層級自動化腳本（S1 腳本本身在 4D-Humans/scripts/）
tests/         程式碼正確性測試
configs/       版本控制下的設定（含本機環境 pip freeze 供參考）
4D-Humans/     vendored 官方 HMR2.0 code（不重訓；見 knowledge/DECISIONS.md D-004）
```

## 驗證計畫

見 [experiments/EXP-001-3dpw-baseline](experiments/EXP-001-3dpw-baseline/EXPERIMENT.md)（3DPW-TEST baseline 重現）與 [experiments/EXP-002-close-di-beta-occlusion](experiments/EXP-002-close-di-beta-occlusion/EXPERIMENT.md)（CloSe-Di 遮蔽下的 β 誤差）。
