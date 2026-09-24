# 現況（2026-09-24）

- [x] 環境建置（Mac 本機，CPU-only，`hmr2_venv/`，Python 3.14，editable install 指向 `4D-Humans/hmr2`）
- [x] `demo.py` 手動跑過幾張測試圖（輸出在 `4D-Humans/data/{test_out,test_out_refined,test01_out,s1_out}/`；屬於手動 sanity check，非正式 run，未納入 `runs/`）
- [x] 自製 `s1_hmr2_infer.py`（偵測 + HMR2 回歸，含 transformers/detectron2 雙 backend）
- [x] 自製 `s1b_refine_leg_pose.py`（ViTPose 腿部姿勢校正）
- [x] 依 SUMMARY.md 骨架整理 repo 結構
- [x] 另存精簡版 `hmr2_estimator.py`（給批次跑資料集用，見 DECISIONS D-005）
- [x] `eval_against_gt.py` 骨架：MPJPE/PA-MPJPE/β 誤差計算已用假資料自我測試通過；GT 載入函式（3DPW/CloSe-Di）待真實資料驗證（見 DECISIONS D-006）
- [ ] 正式 `eval.py` benchmark 尚未執行（`hmr2_evaluation_data` 與資料集影像都還沒下載）
- [ ] 3DPW 尚未下載
- [ ] CloSe-Di 尚未下載
- [ ] 尚未上傳 GitHub
- [ ] 尚未在學校 server 上跑

## 已知環境落差

本機 venv 用 Python 3.14 + CPU-only pip 安裝；官方 `4D-Humans/environment.yml` 指定 `python=3.10` + `pytorch-cuda=11.8`（conda）。**學校 server 上應照官方 environment.yml 重新建置**，不要直接搬本機 venv 或 `configs/requirements-local-mac-dev.txt`（該檔僅供參考/偵錯）。

## 下一步（依使用者指示的順序）

1. 依骨架整理程式碼結構 ← 本次完成
2. 上傳 GitHub（待使用者確認 repo 名稱與公開/私有）
3. 在學校 server 上跑 EXP-001（3DPW baseline）與 EXP-002（CloSe-Di β 誤差）
