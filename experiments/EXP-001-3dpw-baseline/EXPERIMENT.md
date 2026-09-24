# EXP-001：3DPW-TEST baseline 重現

## 問題 / Hypothesis

官方 HMR2.0 checkpoint 在 3DPW-TEST 上的 MPJPE／PA-MPJPE 應與論文（Goel et al., "Humans in 4D", ICCV 2023）報告值一致（合理重現誤差範圍內）。這是環境與模型架設正確性的 sanity check，不是新方法驗證。

## 方法

主要路徑（跟 3DPW 官方釋出的 SMPL 標註直接比對，也是唯一能延伸到 CloSe-Di 的路徑）：

1. 下載 3DPW，用 `hmr2_estimator.py --img_folder <3DPW影像> --out results/s1_raw_3dpw` 批次跑 S1
2. 用 `eval_against_gt.py --dataset 3dpw --pred_dir results/s1_raw_3dpw --gt_dir <3DPW>/sequenceFiles/test --out results/eval_3dpw.csv` 跟 GT SMPL 比對，算 MPJPE/PA-MPJPE/β 誤差
3. `load_3dpw_gt()` 目前是骨架（欄位名稱未經真實資料驗證），跑之前要先確認 `.pkl` 實際欄位，見 `knowledge/DECISIONS.md` D-006

交叉驗證路徑（可選，官方標準流程，用來確認上面自製腳本沒寫錯）：

1. 下載 `hmr2_evaluation_data`，路徑寫入 `4D-Humans/hmr2/configs/datasets_eval.yaml`
2. 執行 `python eval.py --dataset 3DPW-TEST`，結果 append 到 `4D-Humans/results/eval_regression.csv`

把兩邊結果摘要複製到對應的 `runs/RUN-xxx/` 並在此檔案底部連結。

## 變數

- checkpoint：官方預設（HMR2.0b，`DEFAULT_CHECKPOINT`）
- `batch_size`、`num_samples`：先用預設值；若學校 server 資源不足才調整，調整值記錄在對應 run

## Metrics

- `mode_re`（PA-MPJPE）
- `mode_mpjpe`（MPJPE）

## 驗收條件

數值與論文對照表落在合理重現誤差範圍內視為通過；若落差明顯，記錄為 finding 並排查（checkpoint 版本、資料前處理、`datasets_eval.yaml` 路徑設定等）。

## 狀態

`planned` — 尚未執行，等待學校 server 環境 + 3DPW 資料下載。

## 對應 Run

（尚無，執行後建立 `runs/RUN-001-...` 並在此連結）
