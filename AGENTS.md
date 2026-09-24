# AGENTS.md

本 repo 遵循以下知識管理骨架（原始完整版是使用者的 `SUMMARY.md`；此處為工作慣例摘要，供任何在這個 repo 工作的人／agent 參考）。

## 目錄責任

- `knowledge/` — canonical，controlled-write，改動需可審查（PROJECT_MODEL / PROJECT_STATE / QUESTIONS / FINDINGS / DECISIONS）
- `inbox/` — staging，不是 knowledge base
- `tasks/` — 工作項目描述（`TASK-xxx`），本身不是 evidence
- `experiments/` — hypothesis／method／metrics 定義（`EXP-xxx-slug`），重要 research question 用 `RQ-xxx-slug`
- `runs/` — 實際執行紀錄（`RUN-xxx-slug`），需可由 code／config／data／env／seed 重現，並保留 TensorBoard／W&B 追蹤資訊
- `views/` — 唯讀衍生視圖，不手改，改來源後重新產生
- `work/` — 可丟棄的暫存區，不是 shared truth
- `4D-Humans/` — vendored 官方 HMR2.0 code（見 `knowledge/DECISIONS.md` D-004），**不隨意改動內部結構**；我們自己的推論／後處理腳本放在 `4D-Humans/scripts/`
- `src/`、`scripts/`、`tests/`、`configs/` — 見各自的 `README.md`

## 工作流

inbox → task/question → experiment or implementation → run/result → evidence → finding → decision → 更新 knowledge → 重新產生 views

## 不變規則（節錄自 SUMMARY.md）

- ID 用於 identity；short slug 用於辨識，不要把所有參數塞進 ID
- `EXP` 是 intellectual unit，`RUN` 是一次 execution；失敗的 run 不會自動變成新 experiment
- `negative`（假說不成立但實驗成功執行）≠ `failed`（實驗無法提供有效 evidence）
- finding 必須說明 scope、conditions、evidence、confidence/limitations，並連結相關 runs；decision 記錄行動與理由，不等同於 finding
- canonical knowledge 為 controlled-write；derived views 為唯讀且可替換
- 不要把未審查的 inbox note、scratch file、generated view 或 agent claim 當成 canonical truth
- 不要隨意覆寫其他人（含同學）的工作或 shared knowledge；衝突必須明確處理
- WIP 需要明確 status（`planned`／`active`／`blocked`／`completed`／`archived`）
- 架構變更前，必須同步更新本檔案、`README.md` 與受影響的 `knowledge/` 文件

## 本 repo 特別注意

- `4D-Humans/` 是 flatten 進來的 vendored code（見 `knowledge/DECISIONS.md` D-004），不是我們的 `src/`；要改東西前先想清楚是要 patch vendor 還是寫新的 wrapper（優先寫 wrapper，盡量不動 vendor 內部）
- 學校 server 部署請照 `4D-Humans/environment.yml`（python 3.10 + conda + CUDA 11.8），**不要照抄本機 Mac 的 CPU-only venv**（見 `knowledge/PROJECT_STATE.md`）
- 本 repo 只負責 S1；跟 S2-S4 對接時的介面是 `scripts.s1_hmr2_infer.HMR2Estimator`，輸出格式定義在 `knowledge/PROJECT_MODEL.md`
