# 待釐清問題

- [ ] 學校 server 的 GPU 型號／CUDA 版本？需對照 `environment.yml` 的 `pytorch-cuda=11.8` 是否相容
- [ ] 3DPW、CloSe-Di 的下載／授權流程由誰申請、預計何時取得
- [ ] GitHub repo 是否要包含完整 vendored `4D-Humans/` 原始碼，或改用 submodule/fork？目前選擇 flatten 進同一個 repo（見 [[DECISIONS]] D-004），如需改回 submodule 方式要在推上 GitHub 前決定
- [ ] 與同學共用的整體 pipeline repo（S1-S4）是否已存在？本 repo 未來是否要合併進去，或維持獨立、由 S2 那邊來 import/呼叫
- [ ] setup.py 移除 `chumpy` 依賴的確切原因（見 [[DECISIONS]] D-002，目前只有推測）
- [ ] 3DPW `sequenceFiles/*.pkl` 的實際欄位名稱是否跟 `eval_against_gt.py` 的 `load_3dpw_gt()` 假設一致（見 [[DECISIONS]] D-006，下載後第一步要 `pickle.load` 確認）
- [ ] 3DPW 的性別化 SMPL（male/female）GT 要不要另外下載 SMPL_MALE/FEMALE.pkl 分開算，還是先接受跟 HMR2 neutral SMPL 比較的系統性誤差（見 D-006）
