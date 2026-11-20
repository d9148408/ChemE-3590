# Unit00 Python學習環境設定教學 (Environment Setup)

## 📚 單元簡介

在開始 AI 與機器學習的學習之旅前，建立一個穩定且功能完整的 Python 開發環境是至關重要的基礎工作。本單元將帶領您完成兩種主流的 Python 開發環境設定：**Google Colab 雲端環境**與 **Windows 本地開發環境 (`PY310`)**，並額外介紹可輔助程式開發與文獻研究的自主型 AI 代理工具（**Hermes Agent**）。

Google Colab 提供了免安裝的雲端運算資源與免費 GPU 加速，適合快速開始學習與原型開發；而 Windows 本地環境則透過輕量化 Miniconda 與本課程專屬的 `PY310_environment.yml` 配置檔，一鍵部署包含科學計算、梯度提升樹、深度學習（TensorFlow 2.10.1）與原生 Windows GPU 硬體加速（CUDA 11.2 / cuDNN 8.1）的獨立虛擬環境，提供無時間限制與高安全性的開發體驗。同時，本課程所有範例 Notebook 皆採用**跨環境（Colab 與 Local）雙相容標準路徑範本**，讓您能根據需求在雲端與本地端之間無縫切換。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解 Python 環境管理的核心概念**：掌握虛擬環境（Virtual Environment）隔離、套件相依性控制與實驗可重現性
2. **掌握雙重開發環境與跨環境工作流**：熟練操作 Google Colab 雲端環境與 Windows 本地端 Conda + JupyterLab/Notebook 環境，並理解本課程跨環境路徑設定範本
3. **完成課程專案同步與本地環境建置**：在 Colab 中透過 `git clone` 同步課程儲存庫至 Google Drive，並在本地端使用 `PY310_environment.yml` 建立支援原生 Windows GPU 的 `PY310` 環境
4. **執行端到端化工數據分析與環境驗證**：透過連續攪拌槽反應器（CSTR）與間歇式反應器（Batch Reactor）實作案例，完整測試 NumPy、Pandas、Matplotlib、Seaborn、Scikit-learn 與 TensorFlow GPU 運算功能
5. **建立 AI Agent 使用能力（核心教材）**：安裝並操作 Hermes Agent，擴充技能、外掛與 MCP 工具，協助化工資料清理、程式除錯與研究自動化

---

## 📖 單元內容架構

```
Part_0/
├── README.md                                  # 本單元總覽與快速指南
├── PY310_environment.yml                      # Windows 本地端 Conda 虛擬環境定義檔
├── Unit00_Colab_Environment_Setup.md          # Google Colab 雲端環境設定完整講義
├── Unit00_Colab_Environment_Setup.ipynb       # Google Colab 環境實作與 CSTR 數據分析範例
├── Unit00_Local_Environment_Setup.md          # Windows 本地環境設定完整講義
├── Unit00_Local_Environment_Setup.ipynb       # Windows 本地環境驗證與批次反應器分析範例
├── Unit00_Hermes_Agent_Setup.md               # [核心] Hermes Agent 安裝與使用教學
├── Unit00_Hermes_Agent_Command.md             # [核心] Hermes Agent 常用指令詳細教學
├── Unit00_Hermes_Agent_Skills&Plugins.md      # [核心] Hermes Agent 技能、外掛與 MCP 工具指南
└── outputs/                                   # 範例 Notebook 執行產出結果與圖表目錄
    ├── P0_Unit00_Colab_Environment_Setup/
    │   ├── reactor_data.csv
    │   └── figs/
    │       ├── reactor_data_distribution.png
    │       ├── correlation_heatmap.png
    │       └── actual_vs_predicted.png
    └── P0_Unit00_Local_Environment_Setup/
        ├── batch_reactor_data.csv
        └── figs/
            ├── matplotlib_basic_plots.png
            └── sklearn_linear_regression.png
```

---

### 1️⃣ Google Colab 雲端環境設定

**檔案**：
- 📄 講義：[Unit00_Colab_Environment_Setup.md](Unit00_Colab_Environment_Setup.md)
- 📓 程式範例：[Unit00_Colab_Environment_Setup.ipynb](Unit00_Colab_Environment_Setup.ipynb)

**內容重點**：
- **Google Colab 簡介與運作機制**：
  - 什麼是 Google Colab 及其核心特色（免安裝、雲端硬碟整合、免費 T4 GPU）
  - 虛擬機運作限制（`/content/` 暫存空間、最長 12 小時運行、閒置 90 分鐘斷線與動態配額）

- **基本介面與儲存格操作**：
  - 從 Google Drive、Colab 官網或 GitHub 開啟筆記本（含「在雲端硬碟中儲存副本」重要提醒）
  - 頂部選單、左側邊欄（目錄、變數檢查器 `{x}`、密鑰 Secrets、檔案管理）
  - 程式碼儲存格（Code Cell）與文字儲存格（Markdown / MathJax）操作及 `Ctrl + M` 系列快捷鍵

- **檔案管理與本課程 GitHub 儲存庫同步 SOP**：
  - 掛載個人 Google Drive（`drive.mount('/content/drive')`）、本地檔案上傳與下載
  - **課程專案同步三步驟**：首次 `git clone` 下載全課程教材至雲端硬碟、`git reset --hard origin/main` 更新全庫、以及單一單元局部更新
  - **本課程標準跨環境路徑設定範本**：自動偵測 Colab / Local 環境、建立 `/content/ChemE-3590` 符號連結（Symlink），並統一管理 `OUTPUT_DIR` 與 `FIG_DIR`

- **Python 套件管理與 GPU / TPU 硬體加速**：
  - 檢查預裝科學計算與機器學習套件版本
  - 使用 `!pip install` 與跨環境自動檢查安裝（`importlib` + `subprocess`），以及化工常用擴充套件（`rdkit`、`chempy`、`coolprop`、`thermo`、`optuna`）
  - 啟用 NVIDIA T4 GPU、`!nvidia-smi` 硬體狀態檢視，以及 TensorFlow GPU 動態記憶體成長（`set_memory_growth`）設定

- **實作範例：連續攪拌槽反應器 (CSTR) 數據分析與多元線性回歸建模**：
  - 產生 $N = 200$ 組包含反應溫度 $T$ 、壓力 $P$ 、進料流量 $F$ 與產物轉化率 $y$ 的合成反應器數據
  - 繪製 $2 \times 3$ 單變量直方圖分布與雙變量散佈趨勢圖，解析為何單變量視角會掩蓋次要變數效應
  - 皮爾森相關係數熱力圖（Correlation Heatmap）與自變數獨立性（無多重共線性）檢驗
  - Scikit-learn 多元線性回歸建模（測試集 $R^2 = 0.9236$ 、 $\mathrm{RMSE} = 0.0437$ ）、理論物理參數還原驗證（相對誤差 $< 9\%$ ）、對角預測圖（Parity Plot）與殘差圖（Residual Plot）診斷

- **Colab 進階技巧**：
  - IPython Magic Commands（單行 `%timeit`、`%whos` 與儲存格第一行 `%%time`、`%%writefile`）
  - Linux Shell 指令（`!pwd`、`!ls -lh`、`!df -h`）與互動式表單模擬器（`#@param` 滑桿與下拉選單）

**適合場景**：
- 初學者快速開始學習（零安裝門檻）
- 個人電腦無獨立 NVIDIA 顯示卡，需要免費雲端 GPU 進行 Part 4 深度學習訓練
- 跨裝置存取與團隊即時協作分享

---

### 2️⃣ Windows 本地環境設定 ⭐

**檔案**：
- 📄 講義：[Unit00_Local_Environment_Setup.md](Unit00_Local_Environment_Setup.md)
- 📓 程式範例：[Unit00_Local_Environment_Setup.ipynb](Unit00_Local_Environment_Setup.ipynb)

**內容重點**：
- **本地開發環境與虛擬環境核心概念**：
  - 本地開發環境的六大工程優勢（無連線時間限制、完全離線可用、製程資料保密、高速本地 SSD I/O）
  - 虛擬環境（Virtual Environment）沙盒隔離機制與環境可重現性
  - Miniconda vs. Anaconda 深度比較（為何本課程採用極致輕量的 Miniconda）

- **Miniconda 下載與 Windows 安裝詳解**：
  - 官方安裝檔選擇與觀念釐清（`base` 環境僅負責運行 Conda，`PY310` 環境將獨立配置 Python 3.10）
  - Windows 10/11 圖文安裝選項解析（強烈建議選擇 `Just Me` 與純英文無空白安裝路徑）
  - 開啟 Anaconda Prompt 驗證安裝與更新基礎工具

- **建立本課程專屬虛擬環境 (`PY310`) 與工程背景解析**：
  - `PY310_environment.yml` 完整架構解析
  - **關鍵工程設計背景**：為何精確鎖定 `Python 3.10` + `tensorflow==2.10.1` + Conda 內建 `cudatoolkit=11.2` / `cudnn=8.1` + `protobuf==3.19.6`（原生 Windows 最後支援 GPU 之 TensorFlow 版本，由 Conda 自動配置底層 `.dll`，免除學生手動安裝 3 GB 系統層級 CUDA Toolkit 之痛點）
  - 一鍵建立環境標準 SOP 與手動逐步建立替代方案

- **Conda / Pip 管理與 JupyterLab / Notebook (v7+) 實務操作**：
  - 虛擬環境啟動（`conda activate PY310`）、套件安裝查詢、快取清理（`conda clean --all`）與 YAML 備份匯出
  - JupyterLab（多視窗並排 IDE，推薦）與 Jupyter Notebook (v7+) 介面比較、標準四步驟啟動流程、快捷鍵速查與安全關閉伺服器

- **環境設定驗證與實作演練（6 大完整測試）**：
  1. **系統硬體、進階套件與本地 GPU 張量運算測試**：驗證 `sys.executable` 路徑、10 大進階 ML/DL 套件版本，並於本地 GPU 執行 $2000 \times 2000$ 矩陣乘法計時測試
  2. **NumPy 數值與線性代數測試**：一維陣列統計、二維矩陣乘法 $C = A B$ 、行列式與線性聯立方程組 $A \mathbf{x} = \mathbf{b}$ 求解
  3. **Pandas 間歇式反應器數據處理與本地 CSV 讀寫**：模擬 $12$ 批次反應器數據、描述性統計、高產率批次篩選（ $Y > 85\%$ ）、相關性矩陣（ $r_{T,Y} = 0.921$ ）與 `utf-8-sig` 本地 CSV 讀寫驗證
  4. **Matplotlib $2 \times 2$ 多子圖科學繪圖**：折線圖、散佈趨勢圖、催化劑批次長條圖（含目標門檻線與數值標註）及常態分布直方圖（`300 DPI` 高解析度輸出）
  5. **Scikit-learn 線性回歸與殘差診斷**：訓練集/測試集切分、OLS 參數估計 vs. 真實物理定律比較（斜率相對誤差 $5.35\%$ ）、決定係數（測試集 $R^2 = 0.8740$ 、 $\mathrm{RMSE} = 1.6170\%$ ）評估與殘差圖診斷
  6. **本地環境 10 項自動檢核清單（Final Readiness Checklist）**：一鍵輸出 `[PASS]` 驗證報告

- **常見問題故障排除（Troubleshooting Top 10）與最佳實務**：
  - 涵蓋 PowerShell 初始化、SSL 憑證更新、`protobuf` 版本衝突、`cudart64_110.dll` 載入機制、`imbalanced-learn` 相容性、Jupyter Kernel 註冊與中文路徑處理

**適合場景**：
- 長期課程學習、專題研究與畢業論文開發
- 需要長時間連續運算（超參數搜尋、大型深度神經網路訓練）
- 具備 NVIDIA 獨立顯示卡，希望發揮本地硬體極致效能
- 離線開發或處理具機密性之化工廠製程數據

---

### 3️⃣ 環境配置定義檔

**檔案**：⚙️ [PY310_environment.yml](PY310_environment.yml)

**內容說明**：
- **基礎環境與頻道**：`name: PY310`、優先使用 `conda-forge` 頻道、`python=3.10`
- **原生 Windows GPU 加速底層庫**：`cudatoolkit=11.2`、`cudnn=8.1`（由 Conda 自動配置於環境內部 `Library/bin`）
- **科學計算與數值分析**：`numpy=1.23.5`、`scipy`
- **表格資料與試算表處理**：`pandas`、`openpyxl`（Excel `.xlsx` 讀寫支援）
- **資料視覺化與圖表匯出**：`matplotlib`、`seaborn`、`python-kaleido`、`pydot`
- **傳統機器學習與非監督式學習**：`scikit-learn`、`imbalanced-learn`（SMOTE 不平衡數據處理）、`umap-learn`（流形降維）、`mlxtend`（Apriori / FP-Growth 關聯規則）、`networkx`（網路拓樸分析）
- **梯度提升樹三劍客**：`py-xgboost-gpu=1.7.6`（支援 GPU 加速之 XGBoost）、`lightgbm`、`catboost`
- **深度學習與 TensorFlow 生態系（透過 `pip` 精確鎖定版本）**：
  - `tensorflow==2.10.1`、`tensorboard==2.10.1`
  - `protobuf==3.19.6`、`tensorflow-datasets==4.6.0`、`tensorflow-metadata==1.13.0`、`googleapis-common-protos==1.56.4`
- **互動式開發工具**：`jupyterlab`、`notebook`、`ipykernel`、`ipywidgets`

**使用方式**：
```powershell
conda env create -f PY310_environment.yml
conda activate PY310
```

---

### 4️⃣ AI Agent 使用能力：Hermes Agent 教學（核心教材）

**檔案**：
- 📄 安裝與基礎操作講義：[Unit00_Hermes_Agent_Setup.md](Unit00_Hermes_Agent_Setup.md)
- 📄 常用指令與實戰講義：[Unit00_Hermes_Agent_Command.md](Unit00_Hermes_Agent_Command.md)
- 📄 技能與外掛推薦指南：[Unit00_Hermes_Agent_Skills&Plugins.md](Unit00_Hermes_Agent_Skills&Plugins.md)

**內容重點**：
- **認識自主型 AI 代理（Hermes Agent）**：
  - 由 Nous Research 開發的開源自主型 AI Agent，具備**持久記憶（Persistent Memory）**、**程序性技能系統（`SKILL.md`）**、**多步驟工具呼叫**與**子代理平行委派（Subagents）**能力
  - 同一份代理核心與記憶可同時在 Windows 桌面應用（Desktop App）、命令列（CLI `hermes`）與現代終端介面（TUI `hermes --tui`）間無縫接續

- **Windows 安裝與模型供應商設定**：
  - 桌面引導式安裝程式（`Hermes-Setup.exe`）與 PowerShell 命令列安裝步驟
  - Windows 資料儲存架構（`%LOCALAPPDATA%\hermes\` 下的 `config.yaml`、`.env`、`skills\` 與 `sessions\`）
  - 連接課程實驗室的地端模型服務（Custom Endpoint）完成模型與工具設定，不需購買商業模型；技能與 MCP 以本機佈置為主，會把內容送到第三方的雲端工具只在教師指定的練習中使用

- **四大類常用指令詳解與化工實戰應用**：
  - **對話管理**：`hermes`、`hermes --tui --in <路徑>`、`hermes --continue (-c)`、`hermes -r "<標題>"`、`hermes sessions list`
  - **設定管理**：`hermes model`（選 Custom Endpoint）、`hermes setup`、`hermes config`、`hermes tools`
  - **功能擴充**：`hermes skills`（瀏覽、搜尋與安裝程序性技能）、`hermes gateway`、`hermes pm`、`hermes desktop`
  - **維護診斷與斜線指令**：`hermes doctor`（萬用環境健康診斷）、`hermes update`、`hermes uninstall`，以及對話內斜線指令（`/title`、`/compress`、`/usage`、`/model`、`/tools` 等）與反應器溫度數據分析實戰工作流

---

## 🎓 環境比較與選擇指南

| 比較項目 | Google Colab 雲端環境 | Windows 本地環境 (`PY310`) |
| :--- | :--- | :--- |
| **環境建置難度** | 極低（免安裝，瀏覽器開啟即用） | 中等（需安裝 Miniconda 與建立虛擬環境） |
| **成本** | 免費（另有 Colab Pro 付費升級方案） | 免費（使用個人電腦硬體） |
| **網路連線需求** | 必須全程保持穩定網路連線 | 環境建置後支援完全離線開發 |
| **運算硬體資源** | 免費共享雲端 T4 GPU / CPU（有動態配額） | 取決於個人電腦 CPU / RAM / NVIDIA GPU 配置 |
| **單次執行時間** | 最長 12 小時 | 無時間限制，可連續執行數天 |
| **閒置斷線限制** | 閒置約 90 分鐘自動中斷並回收虛擬機 | 無限制 |
| **儲存空間與持久性** | `/content/` 為暫存；需掛載 Google Drive 持久化 | 直接讀寫本地 SSD/HDD，環境與套件永久保存 |
| **套件管理自由度** | 額外安裝之套件於虛擬機回收後需重新安裝 | 安裝於 `PY310` 環境後永久有效，完全可控 |
| **協作與分享功能** | ✅ 優秀（雲端連結共用、修訂歷史與留言） | 需透過 Git / GitHub 或傳送檔案分享 |
| **製程資料機密性** | 數據需上傳至雲端伺服器 | 數據保存於本地加密硬碟，機密不外流 |
| **適合對象** | 初學者、無獨顯筆電使用者、快速原型驗證 | 長期專案開發、具備 NVIDIA 獨顯者、實務研發 |

**選擇建議**：
1. **初學入門階段** $\to$ **Google Colab**（零安裝門檻，立即專注於 Python 與 AI 語法學習）
2. **課程日常作業** $\to$ **Google Colab** 或 **Windows 本地環境** 皆可（本課程 Notebook 雙向完全相容）
3. **無獨立顯卡但需訓練深度學習（Part 4）** $\to$ **Google Colab**（啟用免費 T4 GPU 加速）
4. **長時間超參數搜尋或大型模型訓練** $\to$ **Windows 本地環境**（無 90 分鐘閒置與 12 小時斷線限制）
5. **學士專題、碩士論文與工廠機密數據分析** $\to$ **Windows 本地環境**（完整環境控制權與資料安全性）

---

## 💻 實作環境需求

### 本地環境硬體建議

**最低配置需求**：
- **CPU**：Intel Core i3 / AMD Ryzen 3 或同等級處理器
- **RAM**：8 GB
- **硬碟空間**：15 GB 以上可用空間（用於存放 Miniconda、`PY310` 虛擬環境與課程數據集）
- **作業系統**：Windows 10 / 11 (64-bit)

**建議配置（流暢執行全課程與深度學習）**：
- **CPU**：Intel Core i5/i7 或 AMD Ryzen 5/7 以上多核心處理器
- **RAM**：16 GB 以上（處理大型資料集與影像資料更流暢）
- **硬碟**：SSD 固態硬碟 25 GB 以上可用空間
- **GPU**：NVIDIA 獨立顯示卡（選配，支援 CUDA Compute Capability 3.5+；若無獨立顯卡，全課程皆可以 CPU 模式正常完成）
- **作業系統**：Windows 10 / 11 (64-bit)

### 本課程標準環境配置 (`PY310_environment.yml`)

```yaml
name: PY310
channels:
  - conda-forge
dependencies:
  - python=3.10
  - cudatoolkit=11.2
  - cudnn=8.1
  - numpy=1.23.5
  - pandas
  - openpyxl
  - scipy
  - scikit-learn
  - umap-learn
  - imbalanced-learn
  - mlxtend
  - networkx
  - py-xgboost-gpu=1.7.6
  - lightgbm
  - catboost
  - matplotlib
  - seaborn
  - python-kaleido
  - jupyterlab
  - notebook
  - ipykernel
  - ipywidgets
  - pydot
  - pip
  - pip:
      - tensorflow==2.10.1
      - tensorboard==2.10.1
      - protobuf==3.19.6
      - tensorflow-datasets==4.6.0
      - tensorflow-metadata==1.13.0
      - googleapis-common-protos==1.56.4
```

---

## 📈 學習路徑建議

### 🔰 第一階段：建立主開發環境（建議時間：1–2 小時）

**路線 A：使用 Google Colab 雲端環境（推薦初學者或無獨顯筆電）**
1. 閱讀 [Unit00_Colab_Environment_Setup.md](Unit00_Colab_Environment_Setup.md)（30 分鐘）
2. 依照第 3.3 節 SOP 將本課程 GitHub 儲存庫 `git clone` 至個人 Google Drive（10 分鐘）
3. 開啟並逐格執行 [Unit00_Colab_Environment_Setup.ipynb](Unit00_Colab_Environment_Setup.ipynb)（30 分鐘），完成 CSTR 反應器數據分析與多元回歸驗證

**路線 B：建立 Windows 本地開發環境（推薦有基礎或配備 NVIDIA 獨顯者）**
1. 閱讀 [Unit00_Local_Environment_Setup.md](Unit00_Local_Environment_Setup.md)（40 分鐘）
2. 安裝 Miniconda，並使用 `conda env create -f PY310_environment.yml` 建立 `PY310` 虛擬環境（20–30 分鐘）
3. 啟動 `jupyter lab`，開啟並執行 [Unit00_Local_Environment_Setup.ipynb](Unit00_Local_Environment_Setup.ipynb)（30 分鐘），確認最後 10 項檢核清單全數顯示 `[PASS]`

### 🔧 第二階段：熟悉雙環境切換與核心操作（建議時間：1–2 小時）

1. **掌握跨環境路徑架構**：理解各單元開頭 `### 0. 環境設定` 如何透過 `IN_COLAB` 自動切換雲端 Symlink 與本地 `Path.cwd()`，並將結果輸出至 `OUTPUT_DIR` 與 `FIG_DIR`
2. **熟練日常操作指令**：練習啟動/退出環境、查詢套件版本、切換 GPU 硬體加速器，以及使用快捷鍵高效操作儲存格
3. **建立故障排除能力**：瀏覽講義中的 Troubleshooting Top 10，了解遇到 `ModuleNotFoundError`、路徑錯誤或顯存不足時的標準解法

### 🎯 第三階段：建立 AI Agent 使用能力（核心教材，第 1 週完成安裝與連線，之後各單元持續練習）

1. 閱讀 [Unit00_Hermes_Agent_Setup.md](Unit00_Hermes_Agent_Setup.md)，完成安裝並連接課程實驗室端點，詳讀 4.5 節的課程使用規定
2. 參考 [Unit00_Hermes_Agent_Command.md](Unit00_Hermes_Agent_Command.md)，嘗試使用 `hermes --tui` 輔助整理數據或檢查程式碼
3. 依 [Unit00_Hermes_Agent_Skills&Plugins.md](Unit00_Hermes_Agent_Skills&Plugins.md) 安裝課程工具清單中的技能與本機 MCP，並在之後各單元的 Homework 中持續擴充

---

## 🔍 核心操作指令速查

### Google Colab 常用指令

**課程專案同步與檔案操作**：
```python
# 1. 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# 2. 首次下載本課程 GitHub 儲存庫至 Google Drive（全學期只需執行一次）
%cd "/content/drive/My Drive/Colab Notebooks/"
!git clone https://github.com/d9148408/ChemE-3590.git

# 3. 更新本課程儲存庫至官方最新版本
%cd "/content/drive/My Drive/Colab Notebooks/ChemE-3590/"
!git fetch origin
!git reset --hard origin/main

# 4. 上傳本地檔案與下載雲端檔案
from google.colab import files
uploaded = files.upload()
files.download('filename.csv')

# 5. 檢查雲端虛擬機 NVIDIA GPU 狀態
!nvidia-smi
```

**套件管理**：
```python
# 安裝套件（加上 -q 可靜默安裝）
!pip install -q package_name

# 安裝指定版本
!pip install package_name==1.2.3

# 升級套件
!pip install --upgrade package_name

# 列出已安裝套件
!pip list
```

### Windows 本地環境 (`PY310`) 常用指令

**虛擬環境管理（於 Anaconda Prompt 執行）**：
```powershell
# 啟動本課程環境（極重要！每次工作前必做）
conda activate PY310

# 退出目前環境
conda deactivate

# 列出所有虛擬環境與路徑
conda env list

# 從 YAML 定義檔建立環境
conda env create -f PY310_environment.yml

# 匯出環境備份檔
conda env export > environment_backup.yml

# 清理下載快取壓縮包（釋放硬碟空間）
conda clean --all -y
```

**套件管理**：
```powershell
# 優先使用 conda-forge 頻道安裝套件
conda install -c conda-forge package_name

# 使用 pip 安裝套件
pip install package_name

# 安裝指定版本（注意：conda 用單等號 =，pip 用雙等號 ==）
conda install -c conda-forge numpy=1.23.5
pip install tensorflow==2.10.1

# 查詢目前環境已安裝之套件
conda list
```

**啟動與關閉 Jupyter**：
```powershell
# 1. 啟動環境並切換至課程專案目錄
conda activate PY310
cd /d D:\MyGit\ChemE-3590

# 2. 啟動 JupyterLab（推薦）或 Jupyter Notebook
jupyter lab
# 或
jupyter notebook

# 3. 指定連接埠啟動（當預設 8888 埠被佔用時）
jupyter lab --port 8889

# 4. 安全停止服務：在 Anaconda Prompt 視窗連按兩次 Ctrl + C
```

### Hermes Agent 常用指令

```powershell
# 1. 設定模型供應商：選 Custom Endpoint，填入教師提供的實驗室端點
hermes model

# 2. 萬用環境健康診斷（遇到問題時優先執行）
hermes doctor

# 3. 啟動現代終端介面（TUI）並指定課程工作目錄
hermes --tui --in D:\MyGit\ChemE-3590

# 4. 接續最近一次對話或依標題接續對話
hermes -c
hermes -r "reactor-temp-analysis"

# 5. 啟動桌面應用程式
hermes desktop
```

---

## 📝 環境驗證檢查清單

### ✅ 快速驗證腳本（Colab 與本地環境皆適用）

在 Notebook 儲存格中執行以下程式碼，確認核心套件與 GPU 狀態正常：

```python
import sys
import numpy as np
import scipy
import pandas as pd
import matplotlib
import seaborn as sns
import sklearn
import tensorflow as tf

print(f"Python       : {sys.version.split()[0]}")
print(f"Executable   : {sys.executable}")
print(f"NumPy        : {np.__version__}")
print(f"SciPy        : {scipy.__version__}")
print(f"Pandas       : {pd.__version__}")
print(f"Matplotlib   : {matplotlib.__version__}")
print(f"Seaborn      : {sns.__version__}")
print(f"Scikit-learn : {sklearn.__version__}")
print(f"TensorFlow   : {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
```

### ✅ Windows 本地環境 10 項完整檢核清單

執行 [Unit00_Local_Environment_Setup.ipynb](Unit00_Local_Environment_Setup.ipynb) 最後一節的自動檢核程式，確認以下 10 個項目全數顯示 `[PASS]`：

- [x] `1. Python 3.10+ interpreter active`
- [x] `2. Running inside 'PY310' environment`
- [x] `3. NumPy & SciPy scientific computing`
- [x] `4. Pandas & OpenPyXL data processing`
- [x] `5. Matplotlib & Seaborn visualization`
- [x] `6. Scikit-learn machine learning`
- [x] `7. Gradient Boosting (XGBoost/LightGBM/CatBoost)`
- [x] `8. Unsupervised & Graph tools (UMAP/mlxtend/NetworkX)`
- [x] `9. TensorFlow 2.x deep learning framework`
- [x] `10. Local CSV & Figure outputs verified`

---

## 🚀 故障排除指南 (Troubleshooting Top 10)

### 1. 安裝完 Miniconda 後找不到 Anaconda Prompt
- **解決方案**：
  - 在 Windows 底部搜尋列輸入 `Anaconda Prompt`，點選開啟 **Anaconda Prompt (miniconda3)**。
  - 若想在一般 Windows PowerShell 中直接使用 `conda`，請先在 Anaconda Prompt 執行一次初始化：
    ```powershell
    conda init powershell
    conda init cmd.exe
    ```

### 2. 在 PowerShell 輸入 `conda activate PY310` 出現腳本禁止執行錯誤
- **原因**：Windows PowerShell 預設的執行原則（Execution Policy）阻擋了 Conda 啟動腳本。
- **解決方案**：直接使用 **Anaconda Prompt**，或在 PowerShell 中放寬目前使用者的執行權限：
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### 3. 建立環境時出現 `CondaSSLError` 或網路中斷
- **解決方案**：更新 `base` 環境的安全憑證庫並清理損壞快取後重試（請勿隨意加入已停止維護的舊版 `pkgs/free/` 鏡像頻道）：
  ```powershell
  conda install -n base -c conda-forge ca-certificates certifi openssl -y
  conda clean --all -y
  conda env create -f PY310_environment.yml
  ```

### 4. 匯入 TensorFlow 時出現 `TypeError: Descriptors cannot not be created directly`
- **原因**：環境中的 `protobuf` 被其他新套件升級至 `4.x` 以上版本，與 TensorFlow 2.10.1 不相容。
- **解決方案**：在 `PY310` 環境中將 `protobuf` 降回鎖定版本：
  ```powershell
  conda activate PY310
  pip install protobuf==3.19.6
  ```

### 5. 本地匯入 TensorFlow 時出現 `Could not load dynamic library 'cudart64_110.dll'` 或偵測不到 GPU
- **原因**：在 Windows Conda 環境中，CUDA 11.2 與 cuDNN 8.1 的 `.dll` 動態連結庫位於 `envs\PY310\Library\bin`。若未先執行 `conda activate PY310` 就直接呼叫 `python.exe`，該目錄不會載入 `PATH` 環境變數。
- **解決方案**：務必先在 Anaconda Prompt 執行 `conda activate PY310`，再從該視窗啟動 `jupyter lab`。

### 6. 匯入 `imblearn` (`imbalanced-learn`) 時出現 `ImportError: cannot import name '_safe_tags'`
- **原因**：`scikit-learn 1.6+` 調整了內部標籤 API，舊版 `imbalanced-learn 0.12.x` 需同步升級。
- **解決方案**：
  ```powershell
  conda activate PY310
  pip install --upgrade imbalanced-learn
  ```

### 7. 在 Notebook 中匯入套件卻顯示 `ModuleNotFoundError: No module named 'tensorflow'`
- **原因**：在預設的 `(base)` 環境下啟動了 Jupyter，或 Notebook 綁定到了錯誤的 Kernel。
- **解決方案**：
  1. 在儲存格執行 `import sys; print(sys.executable)`，確認路徑包含 `\envs\PY310\python.exe`。
  2. 將 `PY310` 註冊為獨立 Kernel（**註冊前必須先啟動 `PY310` 環境**）：
     ```powershell
     conda activate PY310
     python -m ipykernel install --user --name PY310 --display-name "Python 3.10 (PY310)"
     ```
     隨後於 Jupyter 頂部選單點選 `Kernel → Change Kernel...` 切換為 `Python 3.10 (PY310)`。

### 8. Windows 使用者名稱含中文導致 Jupyter 啟動報錯 `UnicodeDecodeError`
- **解決方案**：確保 Miniconda 與課程專案皆位於純英文無空白路徑（如 `C:\miniconda3` 與 `D:\MyGit\ChemE-3590`）。若系統暫存目錄含中文，可先設定英文 Runtime 目錄再啟動：
  - **於 Anaconda Prompt (`cmd.exe`) 執行**：
    ```cmd
    set JUPYTER_RUNTIME_DIR=C:\temp\jupyter_runtime
    jupyter lab
    ```
  - **於 Windows PowerShell 執行**：
    ```powershell
    $env:JUPYTER_RUNTIME_DIR = "C:\temp\jupyter_runtime"
    jupyter lab
    ```

### 9. Colab 出現 `⚠️ 找不到雲端 ChemE-3590 路徑` 或連線中斷遺失檔案
- **解決方案**：
  - 確認已依照 [Unit00_Colab_Environment_Setup.md](Unit00_Colab_Environment_Setup.md) 第 3.3 節，在 `/content/drive/My Drive/Colab Notebooks/` 下執行過 `!git clone https://github.com/d9148408/ChemE-3590.git`。
  - 切勿在儲存格中使用 `while True: time.sleep(60)` 無窮迴圈防斷線（這會阻塞整個 Python Kernel 導致後續程式無法執行）；應透過掛載 Google Drive 將所有輸出與模型檢查點（`ModelCheckpoint`）直接存入 `OUTPUT_DIR` 實現持久化。

### 10. 執行大型模型時發生記憶體耗盡（RAM / GPU OOM）
- **解決方案**：
  - 調降訓練批次大小（`batch_size`，例如由 `128` 降至 `32`）。
  - 刪除不再使用的大型陣列並觸發垃圾回收：
    ```python
    import gc
    del large_dataframe
    gc.collect()
    ```

---

## 📚 參考資源

### 官方文件
1. [Conda Official Documentation](https://docs.conda.io/) - Conda 環境與套件管理完整指南
2. [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html) - Conda 常用指令速查表
3. [JupyterLab Documentation](https://jupyterlab.readthedocs.io/en/stable/) - JupyterLab 官方操作手冊
4. [Google Colab FAQ](https://research.google.com/colaboratory/faq.html) - Google Colab 官方常見問題解答
5. [Python 3.10 Documentation](https://docs.python.org/3.10/) - Python 官方標準函式庫文件

### 核心套件文件
- [NumPy Documentation](https://numpy.org/doc/stable/) - 多維陣列與線性代數運算
- [Pandas Documentation](https://pandas.pydata.org/docs/) - 表格資料與時間序列處理
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html) - 科學繪圖與視覺化
- [Seaborn Documentation](https://seaborn.pydata.org/) - 統計資料視覺化
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html) - 機器學習演算法與評估指標
- [TensorFlow API Documentation](https://www.tensorflow.org/api_docs) - 深度學習與神經網路框架
- [Hermes Agent Documentation](https://hermes-agent.nousresearch.com/) - Hermes AI Agent 官方指南

---

## 💡 學習建議與最佳實務

### 1. 環境管理守則
- **保持 `base` 環境純淨**：切勿在 `(base)` 環境中安裝專案套件；所有課程練習一律在 `conda activate PY310` 後進行。
- **為不同研究主題建立獨立環境**：未來若進行其他專題需使用不同版本的套件，可另建新環境（如 `conda create -n my_research python=3.11`），與本課程的 `PY310` 互不干擾。

### 2. 本課程 Notebook 與專案組織規範
- **統一使用跨環境路徑變數**：在讀取資料或儲存圖表時，統一使用 `pathlib.Path` 物件（如 `OUTPUT_DIR / 'result.csv'`、`FIG_DIR / 'plot.png'`），自動相容 Windows 與 Linux 路徑分隔符。
- **圖表標籤全英文規範**：Matplotlib 與 Seaborn 圖表的標題（`title`）、軸標籤（`xlabel` / `ylabel`）與圖例（`label`）請統一使用英文，避免跨平台缺少中文字型時產生方框亂碼。
- **固定隨機種子（`SEED = 42`）**：涉及隨機抽樣、資料切分或神經網路初始化時，務必固定隨機種子以確保實驗 $100\%$ 可重現。
- **繳交作業前完整重跑驗證**：提交 `.ipynb` 前，請執行 `Kernel → Restart Kernel and Run All Cells...`，確認所有儲存格由 `[1]` 順序執行完畢且無任何報錯。

### 3. 本課程標準目錄架構

```
ChemE-3590/
├── Part_0/                              # 環境設定教學與 PY310_environment.yml
├── Part_1/                              # Unit01–Unit04: AI概論、Python、NumPy/Pandas、視覺化
├── Part_2/                              # Unit05–Unit09: 非監督式學習（分群、降維、異常檢測、關聯規則）
├── Part_3/                              # Unit10–Unit14: 監督式學習（回歸、分類、集成學習、模型評估）
├── Part_4/                              # Unit15–Unit17: 深度學習（DNN、CNN、RNN/LSTM）
└── Part_5/                              # 進階專題（強化學習、生成式AI、大型語言模型）
```

每個單元（以 `Part_X/UnitXX/` 為例）的標準輸出結構：
```
UnitXX/
├── data/{dataset_name}/                 # 單元數據集（如有）
├── outputs/P{X}_Unit{XX}_{Topic}/       # 程式執行產出目錄
│   ├── figs/                            # 高解析度圖表輸出 (.png)
│   └── models/                          # 訓練完成的模型權重與轉換器（如有）
├── Unit{XX}_{Topic}.md                  # 單元教學講義
├── Unit{XX}_{Topic}.ipynb               # 單元範例演練 Notebook
└── Unit{XX}_{Topic}_Homework.ipynb      # 單元課後作業 Notebook
```

---

## ✍️ 學習檢核與下一步

### 完成本單元後，請確認您已達成以下目標：

- [ ] 能夠說明虛擬環境（Virtual Environment）的隔離機制與重要性
- [ ] 已在 Google Colab 或 Windows 本地端建立好可用的課程開發環境
- [ ] 若使用 Colab：已將 `ChemE-3590` 儲存庫 `git clone` 至 Google Drive，並成功執行 `Unit00_Colab_Environment_Setup.ipynb`
- [ ] 若使用本地環境：已成功建立 `PY310` 環境，並執行 `Unit00_Local_Environment_Setup.ipynb` 通過全部 10 項 `[PASS]` 檢核
- [ ] 熟悉本課程 `OUTPUT_DIR` 與 `FIG_DIR` 跨環境路徑設定邏輯
- [ ] 了解遇到套件匯入失敗、路徑錯誤或 GPU 未啟用時的故障排除方法

### 準備進入下一階段：Part 1 AI與機器學習概論

完成環境設定後，您已準備好開始 **Part 1** 的核心課程：

- **Part 1 總覽**：[Part 1: Python基礎與資料處理](../Part_1/README.md)
- **Part 1 - Unit01**：[AI與機器學習概論](../Part_1/Unit01/README.md)
- **Part 1 - Unit02**：[Python程式語言基礎](../Part_1/Unit02/README.md)
- **Part 1 - Unit03**：[NumPy與Pandas模組](../Part_1/Unit03/README.md)
- **Part 1 - Unit04**：[Matplotlib與Seaborn視覺化](../Part_1/Unit04/README.md)

**重要提醒**：
- 每次在本地端開始工作前，請記得先在 Anaconda Prompt 啟動環境：`conda activate PY310`
- 如遇到任何環境問題，請隨時回來查閱本單元的「故障排除指南」或兩份環境設定詳細講義

---

## 📞 技術支援

如有疑問或需要協助，歡迎透過以下方式聯繫：
- **課程討論區**：提出問題與同學交流討論
- **授課教師 Office Hour**：面對面深入討論課程與專題問題
- **課程助教**：協助解決環境安裝與程式技術問題

### 提問時請一併提供以下資訊，以利快速診斷：
1. **使用環境**：Google Colab 或 Windows 10/11 本地環境
2. **Python 與環境路徑**：執行 `import sys; print(sys.version, sys.executable)` 的輸出結果
3. **完整的錯誤訊息**：包含最後一行的錯誤類型（如 `ModuleNotFoundError`、`FileNotFoundError`）與完整 Traceback 截圖或文字
4. **已嘗試過的解決步驟**

---

**課程資訊**
- 課程名稱：AI在化工上之應用 (ChemE 3590)
- 課程單元：Part 0 Python學習環境設定教學
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-10-31

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
