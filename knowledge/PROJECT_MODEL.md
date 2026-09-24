# S1：人體幾何估計（HMR 2.0）

本 repo 只涵蓋 3D-VTO pipeline 的 **S1** 階段：RGB 影像 → SMPL (β, θ, π)。
對應規劃文件《3D-VTO-Pipeline-規劃.md》第一節 Pipeline 總覽、第四節子任務↔Dataset 對應總表。
S2–S4 由同組同學負責，不在本 repo 範圍內。

## 輸入 / 輸出

輸入：單張 RGB 影像（人物照片）。

輸出：每個偵測到的人一組 SMPL 參數
- `betas` (10,) — SMPL shape 係數，對應規劃文件的 β
- `body_pose` (23,3,3) — 23 個關節旋轉矩陣
- `global_orient` (1,3,3) — 根關節旋轉矩陣（與 `body_pose` 合稱規劃文件的 θ）
- `cam_t` (3,) — 弱透視相機平移，對應規劃文件的 π

## 依賴

- **不重訓**：直接使用官方 4D-Humans / HMR2.0 pretrained checkpoint（vendored 於 `4D-Humans/`，見 [[DECISIONS]] D-004）
- 物件偵測：預設 HuggingFace `transformers` 的 RT-DETR（`PekingU/rtdetr_r50vd`），可選 detectron2 backend

## 流程（三支腳本，皆在 `4D-Humans/scripts/`）

1. **`s1_hmr2_infer.py`** — 偵測 + HMR2 回歸，含畫框/疊圖/渲染/匯出 `.obj` 等視覺化驗證功能，肉眼檢查單張圖片估計品質用。可 CLI 執行，也可 `from scripts.s1_hmr2_infer import HMR2Estimator` 當模組 import。
2. **`hmr2_estimator.py`** — 同一套 `HMR2Estimator`，但拿掉所有視覺化/debug 功能，只留推論本體，給大規模資料集批次跑（3DPW、CloSe-Di 量化評估）用；上傳 GitHub、部署到學校 server 跑評估走這支。輸出多存一份 `pred_vertices`，供之後算 MPJPE 等指標時取關節用。見 [[DECISIONS]] D-005。
3. **`s1b_refine_leg_pose.py`** — 後處理，用 ViTPose 2D 關鍵點校正 6 個腿部關節（左右髖/膝/踝）的旋轉，SMPLify 風格優化，只修這 6 個關節，其餘（β、其他姿勢、相機參數）完全凍結。
4. **`eval_against_gt.py`** — 把 `hmr2_estimator.py` 的預測跟資料集提供的 GT SMPL 比對，算 MPJPE／PA-MPJPE／β 誤差，對應 EXP-001／EXP-002。目前是骨架（3DPW/CloSe-Di 都還沒下載），誤差計算已用假資料驗證過（`--self-test`），GT 載入函式標了 TODO。見 [[DECISIONS]] D-006。

## 為什麼需要 s1b（腿部校正）

HMR2 是單張影像的 3D 姿態回歸模型，沒有立體視覺/多視角資訊。訓練資料裡少見的姿勢（例如坐姿、膝蓋大幅彎曲朝鏡頭方向伸）容易把膝關節的 3D 旋轉角度估錯。這個誤差會直接傳到規劃文件的 S3（披覆階段要用 θ 把服裝貼合姿勢）與 S4（分區鬆量計算），所以列為 S1 驗證的一部分，不是可忽略的邊緣案例——尤其長褲/短褲本來就是 DeepFashion3D V2 涵蓋的服裝類別。詳見 [[DECISIONS]] D-003。

## 驗證目標（對應規劃文件表四 S1 列）

- **3DPW-TEST**：標準 benchmark，確認 pretrained 模型架設正確、數值與論文一致（sanity check）
- **CloSe-Di**：遮蔽情境下的 β 誤差（其掃描含 GT betas，可直接比對）

對應實驗：`../experiments/EXP-001-3dpw-baseline/EXPERIMENT.md`、`../experiments/EXP-002-close-di-beta-occlusion/EXPERIMENT.md`。
