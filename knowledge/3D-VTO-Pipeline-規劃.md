# 3D 虛擬試穿 Pipeline — 資料與任務規劃

> **架構決策**：Template mesh 頂點回歸（非 UDF）
> **服裝範圍**：上身服裝與下身服裝；洋裝暫緩
> **文件版本**：v2 — 以 CloSe-D 取代 4D-DRESS

---

## 目錄

1. [Pipeline 總覽](#一pipeline-總覽)
2. [資料組成與規模](#二資料組成與規模)
3. [包含哪些資訊、什麼格式](#三包含哪些資訊什麼格式)
4. [子任務 ↔ Dataset 對應總表](#四子任務--dataset-對應總表)
5. [子任務 TODO](#五子任務-todo)
6. [Phase 2 詳細規格：Template 頂點回歸網路](#六phase-2-詳細規格template-頂點回歸網路)
7. [風險與備案](#七風險與備案)
8. [附錄：被排除的資料集與理由](#八附錄被排除的資料集與理由)

---

## 一、Pipeline 總覽

| Stage | 內容 | 方法基礎 |
|---|---|---|
| **S1** | RGB 影像 → SMPL (β, θ, π) 人體幾何 | HMR 2.0（預訓練，不重訓） |
| **S2** | 服裝影像 + mask + β → 帶顏色的標準姿態服裝網格 | Design2Cloth 的**條件注入設計** + Template 頂點回歸 |
| **S3** | 服裝網格 + (β, θ) → 貼合目標人體的變形網格 | DrapeNet 概念，自訓輕量 draping head |
| **S4** | 服裝網格 + 人體網格 → 分區鬆量與合身度指標 | 幾何計算，無需訓練 |

### 兩個關鍵架構決策

**決策一：放棄 UDF + MeshUDF，改用 Template 頂點回歸。**

Design2Cloth 實測出現破洞、雜訊、邊界髒三種症狀，根因同一：

| 症狀 | 根因 | 本方案的解法 |
|---|---|---|
| 破洞 | MeshUDF 靠梯度估計偽符號，薄層／折疊處判錯就不生成面片 | 面片來自模板，拓撲固定，**結構上不可能有洞** |
| 雜訊 | UDF 零位集是尖點（左右梯度反向），網路抹平成淺谷，等值面位置模糊 | 直接回歸頂點座標，用 `L_normal` + `L_lap` 明確約束表面 |
| 邊界髒 | 開放邊界是 UDF 的 ridge，非光滑，被網路抹圓 | 邊界由模板定義且精確，`L_bnd` 額外加權 |

可行前提：Deep Fashion3D V2 的 registered mesh 是**類別專屬三角化，同類別內拓撲一致**，頂點天然對應。

代價是失去任意拓撲生成能力（不對稱剪裁、開衩、鏤空、多層次）。但本計畫目標是**合身度量化**而非無限制服裝設計，固定拓撲反而提供跨體型的頂點對應，對合身度熱力圖的一致性是加分。

**決策二：放棄對抗式訓練，改為純回歸。**

Design2Cloth 用 GAN 是因為 UDF 沒有逐點對應監督。固定拓撲後有逐頂點 ground truth，回歸即可。訓練穩定性提升一個數量級。

### 與 FIT / Fit-VTO 的定位差異（提案核心論點）

Fit-VTO 是 Flux.1-dev + LoRA 的 2D 擴散模型，把量測數值當條件 embedding，合身度是**生成出來的**。
本計畫輸出顯式 3D 幾何，合身度是從服裝與人體網格**直接計算的物理量**。可解釋、可驗證。

---

## 二、資料組成與規模

只列實際會用到的七個。

| Dataset | 角色 | 規模組成 | 性質 | 取得 |
|---|---|---|---|---|
| **Deep Fashion3D V2** | **核心訓練資料**（S2a/S2b/S2c） | 2,078 件服裝點雲、10 個類別；registered mesh 經人工篩選後數量較少（**待清點**）；貼圖 2048×2048 | 真實多視角重建、靜態、單件衣物 | Google 表單取解壓密碼 |
| **CloSe-D**（Di 子集） | β 來源、S3/S4 驗證、S2 補充 | 全集 3,167 個掃描 / 18 類；**Di 子集約 1,455 個掃描**含完整幾何 | 真實掃描、靜態、逐頂點人工精修標籤 | HuggingFace 直接下載 |
| **CLOTH3D**（DrapeNet 預處理版） | 披覆訓練（S3） | 原始 7,000+ 序列 × 300 幀 @30fps ≈ 210 萬幀，7 類，平均約 20K 頂點/套；DrapeNet 取用 **600 上身 + 300 下身** | 合成、MoCap 驅動 SMPL + 物理模擬 | DrapeNet repo 直接下載 |
| **FIT** | 影像端預訓練 + 合身度校準 + 評估基準 | 公開版 `fitvto-100k`：100k train + 5k eval（論文全量 1.13M）；168 種體型（82 男 86 女，XS–3XL）、528 種姿勢、158,483 種服裝設計 | 合成 3D → 物理模擬 → 照片級 re-texturing | HuggingFace 直接下載 |
| **SIZER** | 合身度 ground truth（S4 主要） | 100 位受試者、約 2,000 個掃描；8 種服裝類型；**同款服裝多尺碼**；姿勢約 A-pose | 真實 3D 掃描、靜態 | 機構信箱申請 |
| **VITON-HD** | 真實照片域測試（S2c 評估） | 1024×768 正面女性 + 上衣配對，11,647 train / 2,032 test（共 13,679） | 電商去背棚拍 | 直接下載 |
| **3DPW** | S1 標準 benchmark | in-the-wild 影片 + GT SMPL | 真實 | 直接下載 |

### CloSe-Di 類別分佈（與本計畫聚焦範圍相關）

| 類別 | 掃描數 |
|---|---|
| 長褲 Pants | 897 |
| 襯衫 Shirt | 556 |
| 短褲 Short-Pants | 500 |
| T-shirt | 415 |
| Hoodies | 209 |
| 外套 Coat | 191 |
| 背心 Vest | 85 |
| 裙 Skirts | 59 |

> **注意子集授權差異**
> - **CloSe-Di**：掃描幾何 + SMPL + 標籤，全部釋出 → **本計畫使用此子集**
> - **CloSe-Dc**：僅標籤 + SMPL，商用掃描（Renderpeople / Twindom / AXYZ）不轉發
> - **CloSe-D++**：約 1,000 個標籤，掃描需另向 THuman2.0 / HuMMan / 3DHumans 取得（三者皆公開可申請，可順手拿）

---

## 三、包含哪些資訊、什麼格式

### 3.1 Deep Fashion3D V2 — 主力訓練資料

```
deepFashion3DV2/
├── point_cloud/
│   ├── 1/1-1.ply, 1-2.ply         # 高解析彩色點雲，1-1 = 服裝#1 的第一個姿勢
│   └── cloth_type_list.txt         # 服裝編號 ↔ 類別
├── filtered_registered_mesh/
│   └── 1-1/
│       ├── model_cleaned.obj       # ★ 類別專屬三角化的註冊網格
│       ├── model_cleaned.obj.mtl
│       └── 1-1_tex.png             # ★ 2048×2048 貼圖，同類別內對齊
├── featureline_annotation/
│   └── 1/1_1/1_1_1.ply, 1_1_2.ply  # 服裝最外圍邊界曲線
└── pose_estimation/
    └── 1/1_1.pkl                   # SMPL pose 參數 + scale + translation
```

★ 標記的兩項是整個計畫的基石：同類別拓撲一致 → 頂點對應 → template 回歸可行、外觀監督對齊。

**Feature line 標籤語意**（template 邊界對齊時使用）：

| 服裝類別 | 標籤定義 |
|---|---|
| 上身衣物與洋裝 | 1 = 領口線、2 = 左袖口、3 = 右袖口、4 = 下擺 |
| 長／短褲 | 1 = 腰線、2 = 左褲口、3 = 右褲口 |
| 長／短裙 | 1 = 腰線、2 = 下擺 |

> **座標系警告**：V2 的服裝已粗略對齊到 T-pose SMPL，**rotation 與 translation 與 V1 不同**。網路上針對 V1 撰寫的載入程式碼不能直接套用。

### 3.2 CloSe-D — β 來源與驗證

每個 scan 一個 `.npz`：

```
points     (N,3)   掃描頂點
normals    (N,3)
colors     (N,3)   ★ 有顏色
faces      (F,3)
labels     (N,)    ★ 逐頂點服裝標籤，18 類
garments   (18,)   該 scan 含哪些服裝的 binary 編碼
pose       (72,)   SMPL pose 參數
betas      (10,)   ★★ SMPL shape 參數
trans      (3,)    SMPL translation
canon_pose (N,3)   每個掃描點對應的 T-pose SMPL 頂點位置
scale      (1,)
```

`betas` 直接可用，**這是解決 Phase 0 第三項阻斷性問題（DF3D V2 是否含 β）的備案來源**。
`canon_pose` 提供 scan → T-pose 的對應，做 template 註冊時很有用。

與 SIZER 同屬 MPI 團隊（Tiwari、Pons-Moll），SMPL 慣例一致，兩者併用不會有座標系衝突。

### 3.3 CLOTH3D / DrapeNet 預處理版

原始格式（供參考，本計畫不直接處理）：

- 靜態版型 `.obj`：rest pose 頂點 + 拓撲
- 動態序列 `.pc16`：PC2 的 16-bit 版本，頂點座標相對 SMPL root joint 儲存以保證落在 [-2,2]
- `info.mat`：SMPL 參數、服裝名稱、布料材質

**實際使用**：DrapeNet 官方 repo 提供預處理好的網格下載，已 drape 到 neutral female body、canonical 狀態。**零處理成本**。

### 3.4 FIT

HuggingFace parquet，欄位固定：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `cloth` / `target` / `person` | image | 768px |
| `body_bust` / `body_height` / `body_hips` / `body_waist` | float32 | 公分 |
| `garment_bust` / `garment_length` / `garment_sleeve_length` | float32 | 公分 |

**沒有 3D 資產釋出** — GarmentCode 的 sewing pattern 與 draped mesh 是中間產物，公開版看不到。

**授權** CC-BY-NC-ND-4.0。個人研究、訓練自用模型不受限（ND 限制的是**散布**改作物，不是製作）。若日後要公開權重，在 model card 註明來源與非商業限制即可。

### 3.5 SIZER

- 網格 `.obj` + 同名 `.jpg` 貼圖
- SMPL / SMPL+D / SMPL+G 三種註冊
- 分割成上衣／下著／皮膚的掃描
- 服裝類別與尺碼標籤、性別標籤
- 官方 repo 附 renderer（預設 72 個固定視角），可產生 {image, scan, SMPL params} 配對
- 掃描原始影像不公開

### 3.6 VITON-HD

```
train/ 與 test/ 各含：
  image/                      1024×768 人物
  cloth/                      平攤服裝
  cloth-mask/
  image-parse-v3/             P mode 影像，實際為 0~19 uint 標籤（顯示顏色僅為視覺化）
  image-parse-agnostic-v3.2/
  agnostic-v3.2/
  agnostic-mask/
  image-densepose/
  openpose_img/               PNG
  openpose_json/              JSON
train_pairs.txt / test_pairs.txt
```

---

## 四、子任務 ↔ Dataset 對應總表

| # | 子任務 | 輸入 → 輸出 | 訓練資料 | 評估資料 |
|---|---|---|---|---|
| **S1** | 人體幾何估計 | RGB → SMPL (β, θ, π) | 無（HMR 2.0 預訓練權重） | **3DPW**（標準 benchmark）+ CloSe-Di（遮蔽下 β 誤差） |
| **S2a** | 幾何分支 | mask + β + 影像特徵 → template 頂點位移 | Deep Fashion3D V2 registered mesh | DF3D V2 held-out |
| **S2b** | 外觀分支 | 頂點特徵 → 頂點 RGB | DF3D V2 貼圖採樣至頂點 | DF3D V2 held-out |
| **S2c** | 影像編碼器 | 服裝照片 → 外觀特徵 | 預訓練：FIT `cloth`；監督：DF3D V2 render 配對 | VITON-HD `cloth`（真實域泛化） |
| **S3** | 披覆 | 服裝頂點 + (β, θ) → 變形後頂點 | DrapeNet 預處理網格，self-supervised | **CloSe-Di**（真實 scan、多樣姿態） |
| **S4** | 合身度量化 | 服裝網格 + 人體網格 → 分區鬆量 | 無需訓練（幾何計算） | **SIZER**（尺碼 GT，主）、CloSe-Di（真實距離分佈）、FIT benchmark |

**S2 資料不足時的補充來源**：CloSe-Di，靠 `labels` 切出服裝區域。標籤經人工精修，品質遠優於從 SMPL 頂點位移切割。

---

## 五、子任務 TODO

### Phase 0 — 阻斷性驗證（最優先，半天到兩天）

任何一項失敗都會改變整體架構，必須最先確認。

- [ ] **驗證同類別拓撲一致性** — 載入同一類別的兩個 `model_cleaned.obj`，比對頂點數與面片索引是否完全相同
  - **這是 template 路線的前提**，不成立則整個方案需重新設計
- [ ] **清點各類別可用數量** — 統計 `filtered_registered_mesh/` 下每類別實際件數（篩選後 < 2,078）
  - 若某類別 < 100 件 → 該類別放棄或與相近類別合併
- [ ] **確認 `pose_estimation/*.pkl` 是否含 shape 參數 β** — 官方僅明確提到 pose、scale、translation
  - 先查 repo issue #12 的載入討論
  - 若無 β → 改用 CloSe-Di 的 `betas`，或跑 SMPL shape fitting，或以 mask 尺度比例作 proxy

### Phase 1 — 資料準備

**下載**

- [ ] Deep Fashion3D V2（Google 表單取解壓密碼）
- [ ] CloSe-D（HuggingFace `anticdimi/CloSe-D`，只需 Di 子集）
- [ ] DrapeNet 預處理網格
- [ ] FIT `fitvto-100k`
- [ ] VITON-HD、3DPW
- [ ] SIZER（機構信箱申請，前置時間長，先送出）

**處理**

- [ ] 建立 render pipeline：DF3D V2 網格 → 多視角 RGB + binary mask + 頂點色
  - 視角策略：正面為主 + 小角度擾動（方位角 ±30°、仰角 ±15°），不需 SIZER 那樣的 72 視角
  - 產出即為 S2c 的 (影像, 網格) 配對與 S2a 的 mask 條件
- [ ] 貼圖 → 頂點色：用 obj 的 UV 從 2048² 貼圖採樣，每頂點得一組 RGB
- [ ] 為每類別建立 template mesh：取該類別平均網格或指定一件為基準
- [ ] 用 feature line 標註定出邊界頂點集合 `B_c`（領口／袖口／下擺）
- [ ] 切分 train / val / test，**依服裝 ID 切分**，避免同件衣物不同姿勢跨集合洩漏

### Phase 2 — S2 幾何與外觀分支

詳見[第六節規格](#六phase-2-詳細規格template-頂點回歸網路)。

- [ ] 實作三路編碼端（mask / β / 影像）與融合
- [ ] 實作 triplane 生成器（modulation 注入條件，**非 input concat**）
- [ ] 實作頂點特徵查詢（三平面 concat 聚合 + Fourier PE + 逐頂點 embedding）
- [ ] 實作幾何 head 與外觀 head
- [ ] Loss 逐項加入並調權重
- [ ] 三階段訓練（A → B → C）
- [ ] **條件強度診斷**（`S_mask`、`S_β`、`S_img`）
- [ ] 消融實驗

### Phase 3 — S3 披覆

- [ ] 用 DrapeNet 預處理網格重訓輕量 draping head：吃「頂點座標 + 條件」預測位移
  - **不要直接對接 DrapeNet 原本的 UDF latent space**，介面不相容
  - 固定拓撲後自訓一個更簡單的版本
- [ ] Self-supervised loss：物理項（strain / bending）+ 碰撞懲罰 + 重力
- [ ] 姿態從 AMASS 取樣，β 從 [-3,3]¹⁰ 均勻取樣（沿用 DrapeNet 設定）
- [ ] 驗證頂點對應在披覆前後保持 → 顏色可直接沿用
- [ ] CloSe-Di 真實 scan 驗證

### Phase 4 — S4 合身度與評估

- [ ] 定義分區：接觸區（肩、胸、腰、髖）vs 懸垂區（袖身、下襬）
- [ ] 接觸區算**鬆量偏差**（相對該件衣服的設計鬆量，非相對零）
- [ ] 懸垂區改看輪廓正確性與長度落點
- [ ] 失敗指標：肩線掉落、胸口張力、腰線位移
- [ ] **SIZER 驗證**：同一件衣服放大一號，鬆量是否對應增加 — 最直接的正確性檢驗
- [ ] CloSe-Di 驗證：真實掃描的 garment-to-body 距離作為數值範圍對照
- [ ] FIT benchmark：沿用其測試集與 Size-Aware 指標，與 IDM-VTON、CatVTON、Any2AnyTryOn、Fit-VTO 比較
- [ ] VITON-HD 定性測試：真實電商照片輸入的泛化表現

---

## 六、Phase 2 詳細規格：Template 頂點回歸網路

### 6.0 符號與資料流

```
輸入
  M   ∈ R^{1×256×256}    服裝二值 mask
  β   ∈ R^{10}            SMPL shape 參數
  I   ∈ R^{3×224×224}     服裝 RGB 影像
  c   ∈ {0..K-1}          服裝類別 index

模板（每類別一份，訓練時固定）
  T_c = (V_c ∈ R^{N_c×3}, F_c ∈ Z^{M_c×3}, UV_c ∈ R^{N_c×2}, B_c ⊂ [N_c])
  B_c = 邊界頂點集合（由 feature line 標註求得）

輸出
  V̂ ∈ R^{N_c×3}          變形後頂點座標
  Ĉ ∈ R^{N_c×3}          逐頂點 RGB
```

流程：三路編碼 → 融合成 latent `w` → triplane 生成 → 模板頂點投影查詢 → 雙 head 解碼。

### 6.1 編碼端

**Mask encoder**

```
Conv(1→32,  k4 s2) → GN(8)  → SiLU      # 128
Conv(32→64, k4 s2) → GN(8)  → SiLU      # 64
Conv(64→128,k4 s2) → GN(16) → SiLU      # 32
Conv(128→256,k4 s2)→ GN(32) → SiLU      # 16
Conv(256→256,k4 s2)→ GN(32) → SiLU      # 8
AdaptiveAvgPool(1) → Flatten → Linear(256→256)
→ z_mask ∈ R^256
```

用 GroupNorm 而非 BatchNorm，batch size 小時穩定得多。

**Shape embedding**

```
Linear(10→128) → SiLU → Linear(128→128) → z_β ∈ R^128
```

β 先用訓練集 mean/std 標準化。若無 β，改吃 proxy（mask bounding box 長寬比 + 面積比，2 維），維度不變。

**Image encoder**

Backbone 選 **DINOv2 ViT-B/14（凍結）**。資料量僅一兩千件，從頭訓 ViT 不可行；DINOv2 的 patch 特徵對材質、織紋、圖案的區辨力優於 CLIP（CLIP 偏語意）。若日後要加文字編輯功能再換。

```
I → DINOv2-B/14 (frozen) → CLS token ∈ R^768
CLS → Linear(768→384) → SiLU → Linear(384→384) → z_img ∈ R^384
```

**融合**

```
z = concat[z_mask(256), z_β(128), z_img(384)]        # 768
w = MLP(768 → 512 → 512)                             # 512
e_c = CategoryEmbedding(c)                           # 64
```

`w` 以 StyleGAN2 的 weight demodulation 調變 triplane 生成器。

> **不要用 concat-to-input 的方式注入條件。** Design2Cloth 條件控制力弱很可能出在這裡。Modulation 讓條件影響每一層每個 channel，比在輸入端拼接強得多。

### 6.2 Triplane 生成器

```
const ∈ R^{512×4×4}  (learned)
for res in [8, 16, 32, 64, 128]:
    ModulatedConv2d(w) → Upsample → LeakyReLU(0.2)
ToPlanes: ModulatedConv2d(→ 3 × (C_g + C_a), k1)
輸出 reshape → P ∈ R^{3 × (C_g+C_a) × 128 × 128}
拆成 P_g ∈ R^{3×32×128×128}, P_a ∈ R^{3×32×128×128}
```

| 參數 | 建議值 | 說明 |
|---|---|---|
| 平面解析度 | 128 | 256 更細但顯存 ×4；資料量不大時 128 足夠 |
| `C_g` 幾何通道 | 32 | |
| `C_a` 外觀通道 | 32 | |
| 平面配置 | XY / XZ / YZ | |

幾何與外觀**共用生成器主幹、只在最後一層分出兩組通道** — 這即是「共享空間特徵同時保持解耦」的落地方式。

### 6.3 頂點特徵查詢

對模板頂點 `p = (x, y, z)`（已正規化至 `[-1,1]³`）：

```python
f_g = concat[grid_sample(P_g[0], (x,y)),
             grid_sample(P_g[1], (x,z)),
             grid_sample(P_g[2], (y,z))]        # 96 維
f_a = concat[...同樣三面, 用 P_a...]             # 96 維
γ(p) = Fourier positional encoding, L=6         # 36 維
ε_v = VertexEmbedding[c][v]                     # 32 維
```

**兩個關鍵設計選擇**

*三平面用 concat 而非 sum。* EG3D 與 Design2Cloth 都用 sum。但服裝是薄層幾何，當曲面與某平面近乎平行時（如裙面對 XY 平面），該平面特徵嚴重混疊；sum 會把混疊特徵與良好特徵混在一起，concat 則保留平面身分，讓 MLP 自行決定信任誰。代價是輸入維度 ×3，可忽略。

*逐頂點 learnable embedding `ε_v`。* 這是 template 路線的獨有能力 — 拓撲固定，每個頂點有穩定身分，網路可學到「第 1247 號頂點是左袖口」這種先驗。UDF 表示完全做不到。實作為 `nn.Embedding(N_c, 32)` per category，參數量極小但對邊界品質幫助明顯。

### 6.4 解碼 head

**幾何 head**

```
in = concat[f_g(96), γ(p)(36), ε_v(32), e_c(64)]     # 228
MLP: 228 → 256 → SiLU → 256 → SiLU → 256 → SiLU → 3
Δp = output × s          # s 為可學純量，初始 0.1
V̂ = V_c + Δp
```

最後一層權重初始化接近零（`weight *= 0.01`），讓網路從「輸出等於模板」起步。對早期收斂幫助很大，也避免第一個 epoch 把模板扯爛。

**外觀 head（v1）**

```
in = concat[f_a(96), γ(p)(36), ε_v(32), e_c(64)]
MLP: 228 → 256 → SiLU → 128 → SiLU → 3 → Sigmoid
Ĉ = output
```

**外觀 head（v2，建議後續升級）**

因 DF3D V2 的 UV 在同類別內對齊，可改為從 `w` 直接解出 512×512 UV 貼圖，再用 `UV_c` 採樣。逐頂點顏色的解析度受限於網格密度（幾千頂點），貼圖能表達 logo、印花、細織紋。先用 v1 跑通，v2 作為明確改進項寫進提案。

### 6.5 Loss 設計

```
L = λ_v  · L_vert
  + λ_n  · L_normal
  + λ_l  · L_lap
  + λ_e  · L_edge
  + λ_b  · L_bnd
  + λ_c  · L_chamfer      (optional)
  + λ_rgb· L_rgb
  + λ_Δ  · L_disp
  + λ_tv · L_tv
```

| 項 | 定義 | 目的 | 權重 |
|---|---|---|---|
| `L_vert` | `mean‖V̂ − V_gt‖₁` | 主監督。用 L1 不用 L2，對 registration 誤差較 robust | 1.0 |
| `L_normal` | `mean_f (1 − cos(n̂_f, n_f^gt))` | **表面品質關鍵項**，直接對抗「雜訊」 | 0.1 |
| `L_lap` | `‖L(V̂) − L(V_gt)‖²` | Differential coordinate 對齊。注意是**對齊 GT 的 Laplacian**，非最小化自身 Laplacian（後者會抹平皺褶） | 0.05 |
| `L_edge` | `mean_{ij} \|‖e_ij(V̂)‖ − ‖e_ij(V_gt)‖\|` | 防止三角形退化、拉伸 | 0.05 |
| `L_bnd` | `mean_{v∈B_c}‖V̂_v − V_gt,v‖₁` | **直接對抗「邊界髒」**。邊界頂點通常僅佔 5~10%，需放大權重 | 3.0 |
| `L_chamfer` | `Chamfer(V̂, PC_gt)` | 對**原始 dense point cloud** 而非 registered mesh。registration 本身有誤差，此項把細節拿回來 | 0.1（Stage C） |
| `L_rgb` | `mean‖Ĉ − C_gt‖₁` | 顏色監督 | 1.0（Stage B） |
| `L_disp` | `mean‖Δp‖²` | 位移幅度正則，僅防極端變形 | 0.001 |
| `L_tv` | triplane total variation | 抑制特徵場高頻雜訊 | 1e-4 |

> **調參順序**：先跑 `λ_n = λ_l = λ_e = 0`，確認 `L_vert` 能降下來，再逐項加入。一次全開難以 debug。

**可微渲染 loss（Stage C 選配）**

```
用 nvdiffrast 或 PyTorch3D 把 (V̂, F_c, Ĉ) render 成 256×256
L_render = L1(render, GT_render) + 0.1 · LPIPS(render, GT_render)
```

同時改善幾何與外觀，且是唯一能懲罰「顏色對但貼在錯位置」的 loss。實作成本不低，等前面跑通再加。

### 6.6 訓練策略

**分三階段。** 影像分支資訊量遠大於 mask，若一開始就聯合訓練，網路會走捷徑只看影像，mask 與 β 條件學不起來 — 這正是「條件控制力弱」的成因。

| Stage | 開啟 | 凍結 | Epochs | LR | 目標 |
|---|---|---|---|---|---|
| **A** | mask enc、β emb、triplane、幾何 head | 影像 enc（`z_img` 設零向量）、外觀 head | ~100 | 1e-4 | 學會從 mask + β 生出正確輪廓 |
| **B** | 加入影像 enc 投影層、外觀 head | DINOv2 backbone | ~100 | 1e-4 | 加入外觀，幾何微調 |
| **C** | 全部；加 `L_chamfer`、render loss | 無（DINOv2 解凍最後 4 block，LR 降十倍） | ~50 | 3e-5 | 精修 |

Stage A 時將 `z_img` 設為零向量而非移除，結構不變，Stage B 可直接接續。

**其他設定**

```
optimizer  AdamW(betas=(0.9,0.99), weight_decay=0.01)
scheduler  cosine + 5 epoch linear warmup
batch      8~16（受頂點數與 triplane 解析度限制）
AMP        bf16
EMA        decay 0.999，evaluation 用 EMA 權重
```

**資料增強**

| 對象 | 增強 |
|---|---|
| Mask | 隨機 dilation/erosion (±3px)、小幅仿射、邊界加雜訊 |
| 影像 | color jitter、隨機裁切、隨機背景、JPEG 壓縮 |
| 視角 | render 時方位角 ±30°、仰角 ±15° |
| β | 小幅擾動（若有 GT） |

mask 形變增強特別重要 — 實際使用時 mask 來自分割模型，不會像 render 出來那麼乾淨。影像的 JPEG 與背景增強則為縮小與 VITON-HD 真實電商照的 domain gap。

### 6.7 多類別處理

**單一模型 + 類別條件**，而非每類別一個模型。

```
共享：mask enc、β emb、影像 enc、triplane 生成器、兩個 head 的 MLP 權重
per-category：模板 T_c、頂點 embedding ε_v、類別 embedding e_c
```

資料稀缺（每類別可能僅一兩百件），獨立訓練必然過擬合。共享主幹讓「布料如何隨體型變化」這類通用知識跨類別遷移。

因不同類別頂點數 `N_c` 不一，建議用 **category-balanced sampler**，每個 batch 只放同一類別，簡單且避免 padding 浪費。

### 6.8 條件強度診斷（務必實作）

針對「條件控制力弱」這個核心痛點，定義明確指標，訓練中定期跑：

```python
# Mask 敏感度
固定 (I, β, c)，取兩個不同 mask M₁, M₂
S_mask = ‖V̂(M₁) − V̂(M₂)‖ / ‖V_gt(M₁) − V_gt(M₂)‖

# β 敏感度
固定 (M, I, c)，β 沿主成分方向 ±2σ
S_β = ‖V̂(β⁺) − V̂(β⁻)‖ / ‖V_gt(β⁺) − V_gt(β⁻)‖

# 影像敏感度（應只影響顏色，不應大幅影響幾何）
S_img^geo 應接近 0，S_img^rgb 應接近 1
```

理想值皆接近 1。

- `S_mask` 明顯低於 1 → mask 條件被忽略，加大 modulation 深度或 Stage A 多訓幾輪
- `S_img^geo` 偏高 → 幾何與外觀未解耦，考慮在幾何 head 輸入移除影像來源成分

這組數字也適合放進論文的消融表。

### 6.9 消融實驗規劃

| 消融 | 驗證什麼 |
|---|---|
| 無影像分支（回到原始 Design2Cloth 條件） | 第一項擴展的價值 |
| 幾何／外觀不分通道（共用同一組 triplane 特徵） | 第二項擴展的價值 |
| 三平面 sum vs concat | 薄層幾何的聚合方式 |
| 有無 `ε_v` 頂點 embedding | template 路線的獨有優勢 |
| 有無 `L_normal` / `L_bnd` | 對應「雜訊」「邊界髒」的量化改善 |
| 逐頂點 RGB vs UV 貼圖解碼 | v1 → v2 的升級幅度 |
| **UDF baseline**（重跑 Design2Cloth） | **最重要的一項** |

> **關於 UDF baseline**：用同一份 DF3D V2 資料訓 UDF 版與 template 版，量化破洞率、法向誤差、邊界 Chamfer。這是換表示這個決策的直接證據 — 「Design2Cloth 效果差」目前是主觀觀察，做成受控比較後就變成論文貢獻。
>
> 建議指標：非流形邊數、連通分量數（破洞的代理）、法向一致性、邊界頂點到 GT feature line 的 Chamfer 距離。

---

## 七、風險與備案

| 風險 | 徵兆 | 備案 |
|---|---|---|
| 同類別拓撲不一致 | Phase 0 第一項失敗 | 改用單一 template + 非剛性註冊到每件衣物（多一道處理，仍優於 UDF） |
| 某類別資料量不足 | Phase 0 第二項清點結果偏低 | 合併相近類別（長袖/短袖上衣共用 template + sleeve length 條件）；或用 CloSe-Di 靠 labels 切出服裝補充 |
| DF3D V2 無 β | Phase 0 第三項 | **改用 CloSe-Di 的 `betas`**；或 SMPL shape fitting；或 mask bounding box 比例當 proxy |
| S2c 真實照片域泛化差 | VITON-HD 測試崩壞 | 增加 render 的光照／背景／相機擾動；加大 FIT 預訓練比重 |
| 條件控制力仍弱 | `S_mask` 或 `S_β` 遠低於 1 | 加深 modulation；Stage A 延長；檢查是否 z_img 洩漏到幾何 head |
| SIZER 申請未過或延遲 | — | 合身度驗證改用 CloSe-Di + FIT 量測校準，SIZER 為後補 |
| 洋裝需求回歸 | — | 切換至 sewing pattern 表示（GarmentCode），需另評估建置成本 |

---

## 八、附錄：被排除的資料集與理由

記錄於此，以便日後回顧或在提案的 related work 中引用。

| 資料集 | 排除理由 |
|---|---|
| **CAPE** | 四層問題：(1) 幾何表示不匹配 — SMPL 拓撲（6,890 頂點）的頂點位移，服裝僅為 SMPL 頂點子集，切出的網格邊界是三角面硬切的鋸齒，抵消 UDF/template 表達開放拓撲的優勢；(2) 多樣性不足 — 僅 15 位受試者、4 種穿搭組合，無寬鬆服裝；(3) **完全無外觀資訊**，外觀分支無法訓練；(4) β 條件的體型分佈從數千人縮到 15 人 |
| **Design2Cloth 官方資料集** | 資料本身可用（2,010 identities、2,000+ 服裝、已公開），但實測模型輸出破洞、雜訊、邊界髒。診斷後確認根因在 UDF + MeshUDF 表示而非資料，故保留其**條件注入概念**、改用 DF3D V2 作為訓練資料（真實掃描、有貼圖、拓撲一致、經人工篩選） |
| **4D-DRESS** | **過度規格**。本 pipeline 從頭到尾不產生時間序列（單張影像 → 靜態 3D → 單一姿態披覆），4D 特性用不到。其獨門優勢（真實 4D 掃描的逐頂點語義標籤）對本計畫無附加價值，卻是申請門檻最高者（機構信箱、簽授權、明確禁止商用與訓練商用模型）。CloSe-D 以更低成本提供所需的全部資訊（真實掃描 + 逐頂點標籤 + SMPL 含 β + 顏色），且 HuggingFace 直接下載 |
| **MVHumanNet** | 無服裝層級 3D ground truth（僅整體人體 mask、keypoints、SMPL-X）。下載成本高，本計畫每個子任務都用不到其核心價值 |
| **DNA-Rendering** | 同上，且 Part1–6 全量約 11TB，需每週五統一回覆的申請流程。僅在未來做多視角渲染品質比較時才有意義 |

### 曾評估過的 4D 替代方案（若日後需要動態驗證）

| 資料集 | 規模 | 服裝分割 | 備註 |
|---|---|---|---|
| X-Humans | 20 位參與者、233 段序列、約 35,500 幀，含 texture map 與 SMPL-X | 無 | ETH AIT 出品，4D-DRESS 的前身級資料 |
| 4DHumanOutfit | 20 位演員 × 7 套服裝 × 11 種動作 | 無 | **同一人穿不同服裝**，做合身度比較有價值 |
| BUFF | 6 位受試者 × 2 種穿搭，13,632 個掃描 | 有（含衣下真實體型） | 規模小，但有 ground truth body under clothing |

三者皆無 4D-DRESS 的逐頂點語義標籤 — 那確實是其獨門，短期內無真正替代品，但本計畫用不到。

---

## 九、關鍵決策速查

| 問題 | 決策 | 理由 |
|---|---|---|
| 幾何表示？ | Template mesh 頂點回歸 | UDF 的破洞／雜訊／邊界髒是先天缺陷，換資料救不了 |
| 訓練範式？ | 純回歸，無 GAN | 固定拓撲後有逐頂點 GT，不需對抗式 |
| 主訓練資料？ | Deep Fashion3D V2 | 真實掃描 + 2048² 貼圖 + 同類別拓撲一致 + 經人工篩選 |
| 條件注入？ | StyleGAN2 weight modulation | concat-to-input 是條件控制力弱的可能成因 |
| 三平面聚合？ | concat 而非 sum | 薄層幾何在近平行平面上會混疊 |
| 影像 backbone？ | DINOv2 ViT-B/14 凍結 | 資料量小；材質區辨力優於 CLIP |
| 4D 資料集？ | 不使用 | Pipeline 無時序輸出，過度規格 |
| 洋裝？ | 暫緩 | 資料不足（4D-DRESS 僅 4 件、DF3D V2 待清點）且 template 對不對稱／開衩／多層次失效 |
