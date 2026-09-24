# Decisions

## D-001　人物偵測 backend 預設用 transformers/RT-DETR，非官方預設的 detectron2

**理由**：detectron2 需要編譯 C++ 擴充功能，Mac／Windows 上沒有對應 build tools 的機器裝不起來；RT-DETR（`PekingU/rtdetr_r50vd`）純 pip 安裝即可、支援 CPU，且實測在人多、彼此重疊的照片裡比 `hustvl/yolos-tiny` 準確很多（yolos-tiny 容易重複框、框不準）。仍保留 `--detector-backend detectron2` 作為可選項（原本雲端 sandbox 用的方式，精度通常較高）。

**狀態**：已採用（`s1_hmr2_infer.py` 的 `--detector-backend` 預設值）

---

## D-002　移除 setup.py 對 chumpy 的 git 依賴

`git diff setup.py` 顯示 `install_requires` 移除了 `chumpy @ git+https://github.com/mattloper/chumpy`。但本機 venv 的 `pip freeze` 顯示 chumpy 其實還是裝著的（另外手動裝的）。

**理由（推測，待作者本人確認並補上正式說明）**：常見原因是 `pip install -e .` 在新版 pip 的 resolver 下對 `install_requires` 裡直接寫 git URL 的依賴容易解析失敗／變慢，所以從 setup.py 拿掉，改成另外手動 `pip install git+https://github.com/mattloper/chumpy`。

**狀態**：已套用於本機環境。**TODO**：確認真正原因後更新本條，並視情況決定要不要在 README/environment 說明裡註明「chumpy 需另外手動安裝」。

---

## D-003　新增 s1b 腿部姿勢後處理（ViTPose 校正）

**理由**：HMR2 是單視角 3D 回歸模型，沒有立體/多視角資訊，訓練資料裡少見的姿勢（例如坐姿、膝蓋大幅彎曲朝鏡頭方向伸）容易把膝關節的 3D 旋轉角度估錯，導致投影回 2D 後小腿比例壓縮、腳沒有落在照片裡真實腳的位置。這不是 `s1_hmr2_infer.py` 的 bug（已用 `_boxes.jpg` 排除「偵測框沒框到腳」的可能性），而是模型本身的限制。這個誤差會直接傳到規劃文件的 S3（披覆用 θ 貼合姿勢，腿部姿勢錯了褲子就貼合不上真實膝蓋位置）與 S4（分區鬆量跟著算錯）。

**做法**：用 ViTPose（HuggingFace transformers，COCO 17 點）偵測左右髖/膝/踝共 6 個關節的 2D 位置，只把 `body_pose` 裡這 6 個關節對應的旋轉矩陣改成可訓練參數（axis-angle 參數化 + 梯度下降，每步轉回合法旋轉矩陣），其餘 17 個關節、β、global_orient、相機參數全部凍結。用 `hmr2/utils/geometry.py` 同一套 `perspective_projection()` 投影回 2D，跟 ViTPose 偵測位置算加權 L2（權重用 ViTPose 信心分數），加小正則化避免被低信心關鍵點拉去奇怪角度。

**狀態**：已實作（`s1b_refine_leg_pose.py`）

---

## D-004　vendored `4D-Humans` 用 flatten 方式併入本 repo，非 git submodule

**理由**：本次整理目的是「整理乾淨 → 上傳 GitHub → 學校 server 直接跑」。Submodule 需要額外的 `git submodule update --init` 初始化步驟，而且無法直接夾帶本機對 `setup.py` 的 patch（除非另外 fork 一份 4D-Humans）。Flatten 後 `git clone` 完直接可用，對課程/專題規模的協作成本較低。

**Provenance**：vendored from `https://github.com/shubham-goel/4D-Humans.git`，commit `efe18deff163b29dff87ddbd575fa29b716a356c`（`main` branch，2024 年後某次 "Update Links" commit）。原始 `.git` 歷史已於整理時移除；上游隨時可重新對照/re-clone 比對差異。本機在此 commit 之上的實際差異：移除 `setup.py` 的 chumpy 依賴（見 D-002）、新增 `scripts/s1_hmr2_infer.py`、`scripts/s1b_refine_leg_pose.py`、`example_data/images/test01.jpg` 與 `test02.jpg`。

**狀態**：已套用。若之後想改回 submodule/fork 方式（例如要頻繁同步上游更新），這是可逆的決定，但要在正式推上 GitHub 前決定，見 [[QUESTIONS]]。

---

## D-005　另存 `hmr2_estimator.py`，作為批次跑量化評估的精簡版

**理由**：`s1_hmr2_infer.py` 是給人肉眼檢查單張圖片估計品質用的（畫偵測框、SMPL 頂點疊圖、實心人偶疊圖、matplotlib 渲染、匯出 `.obj`/`.mtl`）。要跑 3DPW、CloSe-Di 這種上千張圖的批次量化評估、跟資料集提供的 GT SMPL 做數值比對時，這些視覺化功能完全用不到，只會拖慢速度、多佔硬碟。拆成兩支檔案，`hmr2_estimator.py` 只保留 `HMR2Estimator`（含 D-001~D-004 已經修好的 bug：bbox 索引、`score_thresh` 一致性、`batch_size` 參數化、`detector_weights` 可攜路徑）跟一個只存 npz 的最小 CLI，程式碼從 696 行降到 339 行。兩支檔案共用同一套 `HMR2Estimator` 邏輯（複製過去時沒有再改動核心推論程式碼），已用 `example_data/images/test01.jpg` 驗證兩邊輸出的 betas/body_pose/global_orient/cam_t/bbox 數值完全一致（max abs diff = 0.0）。

**狀態**：已套用。上傳 GitHub、部署學校 server 跑 EXP-001/EXP-002 用 `hmr2_estimator.py`；手動抽查估計品質時才用 `s1_hmr2_infer.py`。

---

## D-006　`eval_against_gt.py`：獨立寫比對腳本，不只依賴官方 `eval.py`

**理由**：官方 `eval.py` 只認 `hmr2_evaluation_data`（預先處理過的 `.npz` metadata，含 25-45 號 OpenPose+SMPL 關節索引），不支援 CloSe-Di，且不會直接跟資料集原始釋出的 SMPL 檔（3DPW 的 `sequenceFiles/*.pkl`）比對。我們的目標是「HMR2 估出來的 (β,θ,π) 跟資料集官方 SMPL 標註差多少」，兩個資料集（3DPW、CloSe-Di）都要用同一套邏輯跑，所以另外寫一支通用比對腳本，共用同一套 `PoseRecord` 抽象、`mpjpe()`/`pa_mpjpe()`/`beta_error()`。官方 `eval.py` 仍可以留著當作交叉驗證（如果兩邊算出來的 3DPW 數字差很多，代表這支自製腳本哪裡寫錯了)。

**做法**：`compute_similarity_transform()`（Procrustes/Umeyama 對齊）、`mpjpe()`、`pa_mpjpe()`、`beta_error()` 是純數學，不依賴資料集格式，已經用 `--self-test`（合成假資料）驗證：預測=GT 時誤差為 0、加雜訊後誤差 >0 且 PA-MPJPE ≤ MPJPE（Procrustes 對齊的定義性質）。`load_3dpw_gt()` / `load_close_di_gt()` 這兩個 GT 載入函式的欄位名稱是照公開文件/規劃文件記的格式寫的，**還沒有拿真實檔案驗證過**，屬於骨架、待補。

**已知但這次骨架沒解決的問題**（記錄下來，拿到真實資料時第一件事就是處理這些）：
1. 3DPW 的 GT 用性別化 SMPL（male/female）算，HMR2 只輸出 neutral SMPL 參數，本機也只有 `SMPL_NEUTRAL.pkl`——兩邊用不同模型算關節位置本身就有系統性誤差，不是 HMR2 的估計誤差。
2. 3DPW 一段影片可能不只一個人，`match_prediction()` 目前只用 bbox IoU 配對，還沒拿多人場景測過。
3. 3DPW 抽幀檔名慣例（`<sequence>/image_%05d`）是官方釋出腳本的常見慣例，不是從真實檔案確認的，要跟 `hmr2_estimator.py --img_folder` 實際產生的檔名對上。

**狀態**：骨架已建立、誤差計算已自我測試通過。3DPW/CloSe-Di 下載後要先手動確認 GT 檔案欄位名稱，再視情況調整 `load_3dpw_gt()`/`load_close_di_gt()`。
