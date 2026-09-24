# runs

實際執行紀錄與 evidence。每個 run 屬於 `experiments/` 底下的一個 experiment，命名建議 `RUN-001-seed-42` 這類。應保存 status、outcome、metrics、logs 與 provenance（code version／configuration／data version／environment/dependencies／random seed），並記錄對應的 TensorBoard／W&B 連結（TensorBoard 用於本地訓練 debug，W&B 用於遠端 tracking／跨 run 比較）。

目前尚無 run；EXP-001、EXP-002 執行後在此建立對應資料夾。
