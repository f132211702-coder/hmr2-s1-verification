# scripts（top-level）

跨階段／repo 層級的自動化與維護腳本（例如環境檢查、資料同步腳本）。

S1 本身的推論與後處理腳本目前留在 vendored `4D-Humans/scripts/`（`s1_hmr2_infer.py`、`s1b_refine_leg_pose.py`），因為它們依賴 4D-Humans 內部的相對路徑與 `hmr2` 套件，沒有搬到這裡以避免破壞既有行為。
