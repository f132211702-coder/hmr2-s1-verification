# EXP-002：CloSe-Di 遮蔽情境下的 β 誤差

對應 research question：**RQ-001-does-occlusion-degrade-beta**

## Research Question

影像中人體遮蔽程度上升時，HMR2.0 估計的 SMPL β（shape）誤差是否隨之上升？

## 方法

1. 從 CloSe-Di 取樣一批掃描，取其 GT `betas`（見規劃文件 3.2 節，`.npz` 裡的 `betas` 欄位）與對應影像（或用 CloSe-Di 提供的 scan 渲染出影像，視資料可得性調整，做法確定後補在此處）
2. 依遮蔽程度分組（例如：無遮蔽／部分遮蔽／嚴重遮蔽；分組依據——mask 面積比例或人工分級——待定，確定後補充於此，並同步記一筆到 `knowledge/QUESTIONS.md` 的答覆）
3. 對每組影像跑 `4D-Humans/scripts/hmr2_estimator.py --img_folder <分組資料夾> --out results/s1_raw_close_di` 取得估計 β̂
4. 用 `eval_against_gt.py --dataset close-di --pred_dir results/s1_raw_close_di --gt_dir <CloSe-Di 資料夾> --out results/eval_close_di.csv` 計算 β̂ 與 GT betas 的誤差（逐維度與整體 L1/L2/MAE），跟 3DPW（EXP-001）共用同一套 `beta_error()` 邏輯。`load_close_di_gt()` 目前是骨架，欄位未經真實資料驗證，見 `knowledge/DECISIONS.md` D-006

## 變數

- 遮蔽分組定義（待定）
- 是否連動測 s1b 腿部校正的影響（β 理論上不受腿部校正影響，可作對照組）

## Metrics

- β 誤差（MAE／L2）依遮蔽分組列出
- 是否有系統性偏移（某些 shape 維度誤差特別大）

## 驗收條件

屬探索性實驗，非證明性：只要能穩定產出「遮蔽程度 vs β 誤差」的量化關係即算成功執行。若 hypothesis 不成立（遮蔽程度與誤差無明顯相關）記錄為 `negative`，不是 `failed`。

## 狀態

`planned` — 尚未執行，等待 CloSe-Di 下載 + 遮蔽分組方法確定。

## 對應 Run

（尚無）
