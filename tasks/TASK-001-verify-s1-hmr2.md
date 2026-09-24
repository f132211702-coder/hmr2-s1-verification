# TASK-001：驗證 S1（HMR 2.0）

**負責人**：使用者本人（S1 owner）
**狀態**：active

## 目標

依規劃文件《3D-VTO-Pipeline-規劃.md》第四節子任務對應表，驗證 pretrained HMR2.0 在 S1 任務上的表現：

1. 3DPW-TEST 標準 benchmark 數值需與論文一致（sanity check，確認環境/模型架設正確）
2. CloSe-Di 遮蔽情境下的 β 誤差

## 產出

- `experiments/EXP-001-3dpw-baseline/`
- `experiments/EXP-002-close-di-beta-occlusion/`
- 更新 `knowledge/PROJECT_STATE.md`、`knowledge/FINDINGS.md`

## 依賴

- 環境：學校 server（GPU），依 `4D-Humans/environment.yml` 重建（見 `knowledge/PROJECT_STATE.md` 的環境落差說明）
- 資料：3DPW、CloSe-Di（下載狀態見 `knowledge/QUESTIONS.md`）

## 子步驟

- [x] 整理 repo 結構（依骨架）
- [ ] 上傳 GitHub
- [ ] 學校 server 環境建置
- [ ] 下載 3DPW + hmr2_evaluation_data，跑 EXP-001
- [ ] 下載 CloSe-Di，設計遮蔽分組方法，跑 EXP-002
- [ ] 整理 findings，回報結果給同學（S2 需要知道 S1 輸出品質/已知限制）
