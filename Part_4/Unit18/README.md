# Unit18 圖神經網路 (Graph Neural Networks, GNN)

## 📚 單元簡介

在 Unit15、Unit16、Unit17 中，我們分別學習了處理**結構化數據**的 DNN、處理**影像數據**的 CNN，以及處理**序列數據**的 RNN。這些架構各有擅長的資料形態，但它們都假設輸入具有固定的規則結構：向量、網格或序列。然而在化工領域，大量最重要的資料天生就是**圖 (Graph) 結構**，無法直接套用上述架構。

**圖神經網路 (Graph Neural Networks, GNN)** 是專為圖結構資料設計的深度學習架構，核心思想是讓每個節點透過不斷**聚合 (Aggregate) 鄰居的資訊**來更新自身的向量表示 (Embedding)，使最終表示同時編碼節點自身特徵與其在圖中的結構脈絡。

GNN 的核心創新與優勢：

- **排列不變性 (Permutation Invariance)**：重新編號節點不改變結果
- **處理任意大小與拓撲**：同一個模型可處理不同大小的分子或網路
- **訊息傳遞 (Message Passing)**：節點與鄰居交換資訊並逐層更新表示
- **利用結構資訊**：明確地將連接關係納入學習過程

在化工領域，許多資料本質上是圖：

- **分子結構**：原子是節點，化學鍵是邊，分子性質取決於整體拓撲
- **反應網路**：反應物與產物是節點，反應路徑是邊
- **製程流程圖 (PFD/P&ID)**：單元操作是節點，物流與能流是邊
- **感測器網路**：感測器是節點，空間或製程關聯是邊

本單元涵蓋：

- **GNN 核心原理**：圖的數學表示、訊息傳遞框架、圖卷積運算
- **經典架構**：GCN、GraphSAGE、GAT、MPNN
- **圖層級任務**：讀出 (Readout) 池化、圖分類與圖回歸
- **TensorFlow/Keras 實作**：從零建立圖卷積網路
- **三個實際案例**：分子水溶解度回歸、分子毒性分類、製程故障診斷

本單元是 Part_4 深度學習系列的進階專題，建議在完成 Unit15 (DNN) 基礎後學習。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解圖資料的本質**：認識分子、反應網路、感測器網路都是圖結構資料
2. **掌握圖的數學表示**：鄰接矩陣、度矩陣、特徵矩陣、拉普拉斯矩陣
3. **理解訊息傳遞機制**：節點如何與鄰居交換資訊並更新表示
4. **掌握經典 GNN 架構**：GCN、GraphSAGE、GAT 與 MPNN 的差異與適用場景
5. **解決化工圖問題**：分子性質預測 (QSPR)、毒性篩選、製程故障診斷

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：GNN 基礎理論 ⭐

**檔案**：
- 講義：[Unit18_GNN_Overview.md](Unit18_GNN_Overview.md)
- 範例：[Unit18_GNN_Overview.ipynb](Unit18_GNN_Overview.ipynb)

**內容重點**：
- **為什麼需要 GNN？**：
  - DNN/CNN/RNN 面對不規則、無序圖資料的限制
  - 化工圖資料無所不在：分子、反應網路、製程流程圖、感測器網路
  - GNN 的核心優勢：排列不變性、任意拓撲、結構資訊利用
- **圖的數學表示**：
  - 節點 (Node)、邊 (Edge)、鄰接矩陣 $A$ 、度矩陣 $D$
  - 特徵矩陣 $X$ 、拉普拉斯矩陣 $L = D - A$
  - 正規化鄰接矩陣與自環 (Self-loop)
- **訊息傳遞框架 (Message Passing)**：
  - 聚合 (Aggregate)：收集鄰居資訊
  - 更新 (Update)：結合自身與鄰居表示
  - 多層堆疊：擴大感受野 (K-hop)
- **經典 GNN 架構**：
  - **GCN (Graph Convolutional Network)**：頻譜觀點的圖卷積
  - **GraphSAGE**：鄰居採樣 + 聚合 (mean/max/LSTM)
  - **GAT (Graph Attention Network)**：注意力權重的鄰居聚合
  - **MPNN (Message Passing Neural Network)**：統一框架
- **圖層級任務與池化**：
  - 節點層級 (Node-level)：節點分類、節點回歸
  - 圖層級 (Graph-level)：讀出池化 (sum/mean/max)、圖分類/回歸
- **TensorFlow/Keras 實作**：從零實作 GCN 進行節點分類
- **化工領域應用場景**

**適合讀者**：所有學員，**建議最先閱讀**以建立完整的 GNN 理論基礎

---

### 2️⃣ 實際案例 1：分子水溶解度預測 (Molecular Solubility) ⭐

**檔案**：
- 講義：[Unit18_Example_Molecular_Solubility.md](Unit18_Example_Molecular_Solubility.md)
- 程式範例：[Unit18_Example_Molecular_Solubility.ipynb](Unit18_Example_Molecular_Solubility.ipynb)

**內容重點**：
- **問題背景**：
  - 水溶解度 ( $\log S$ ) 是化工與製藥領域最重要的物性之一
  - 應用於結晶、萃取、吸收、配方設計、環境評估、藥物開發
  - QSPR (定量結構-性質關係) 的代表性任務
- **ESOL (Delaney) 資料集**：
  - 1128 個有機小分子，SMILES 字串輸入
  - 目標：實測水溶解度 $\log S$ （mol/L）
- **關鍵技術**：
  - 以 **RDKit** 將 SMILES 轉換成分子圖（原子為節點、化學鍵為邊）
  - **padding + 節點遮罩** 批次化不同大小分子圖
  - **graph-level regression** 與讀出 (Readout) 池化
  - 與傳統描述子方法比較
- **任務類型**：圖層級回歸 (Graph-level Regression)

**適合場景**：分子性質預測、QSPR 建模、配方設計

---

### 3️⃣ 實際案例 2：分子毒性分類 (Molecular Toxicity) ⭐

**檔案**：
- 講義：[Unit18_Example_Molecular_Toxicity.md](Unit18_Example_Molecular_Toxicity.md)
- 程式範例：[Unit18_Example_Molecular_Toxicity.ipynb](Unit18_Example_Molecular_Toxicity.ipynb)

**內容重點**：
- **問題背景**：
  - 分子毒性篩選是化工、製藥與環境工程的共同關鍵議題
  - 應用於製程安全、綠色化學、環境風險評估、法規遵循
  - 芳香烴受體 (AhR) 是環境毒理學最受關注的標靶之一
- **Tox21 資料集與 NR-AhR 任務**：
  - 約 7800 個化合物在 12 種生物檢測上的毒性標記
  - 本範例預測分子對 AhR 的活化能力（NR-AhR）
- **關鍵技術**：
  - 從零實作 **圖注意力網路 (GAT)** 的注意力機制與多頭設計
  - 處理**類別不平衡**：class weight、ROC-AUC、PR-AUC
  - 與**描述子 + 隨機森林**基線比較
  - 從預測結果解讀化學意義（PAH 類結構辨識）
- **任務類型**：圖層級二元分類 (Graph-level Classification)

**適合場景**：毒性篩選、分子分類、綠色化學、環境風險評估

---

### 4️⃣ 實際案例 3：製程故障診斷 (Process Fault Diagnosis) ⭐

**檔案**：
- 講義：[Unit18_Example_Process_Fault_Diagnosis.md](Unit18_Example_Process_Fault_Diagnosis.md)
- 程式範例：[Unit18_Example_Process_Fault_Diagnosis.ipynb](Unit18_Example_Process_Fault_Diagnosis.ipynb)

**內容重點**：
- **問題背景**：
  - 製程故障診斷是化工廠安全與效益的核心課題
  - 傳統 MSPC（PCA 的 $T^2$ 與 SPE）擅長偵測但分類能力有限
  - 把感測器網路建模為圖，以 GNN 明確利用變數間的關聯結構
- **田納西-伊士曼製程 (TEP)**：
  - Downs & Vogel (1993) 提出的經典基準案例
  - 52 個製程變數，1 個正常狀態 + 21 種預定義故障 (IDV)
- **關鍵技術**：
  - 從**正常操作資料的相關性**建立固定拓撲的**感測器關聯圖**
  - 從零實作 **GraphSAGE**（鄰居聚合 + 自身串接）
  - 「固定拓撲 → 節點對齊 flatten 讀出」vs 分子圖「排列不變 pooling」
  - 與 **MLP、Random Forest** 誠實比較
  - 透過**感測器失效與雜訊**的穩健性測試展現 GNN 價值
- **任務類型**：多類別故障分類 (Multi-class Classification)

**適合場景**：製程監控、故障診斷、感測器網路建模、系統工程

---

## 📊 經典 GNN 架構對比

| 架構 | 聚合方式 | 節點順序 | 適用場景 |
|------|---------|---------|---------|
| **GCN** | 正規化鄰接加權平均 | 無序 | 節點分類、小型圖 |
| **GraphSAGE** | 鄰居採樣 + 聚合 (mean/max) | 無序 | 大型圖、固定拓撲 |
| **GAT** | 注意力加權聚合 | 無序 | 鄰居重要性不均、分子圖 |
| **MPNN** | 自訂訊息函數 | 無序 | 通用框架、分子性質 |

---

## 🧭 GNN 任務類型與讀出策略

| 任務類型 | 預測對象 | 讀出 (Readout) | 範例 |
|---------|---------|---------------|------|
| **節點層級** | 單一節點 | 無（直接輸出節點） | 製程感測器診斷 |
| **圖層級回歸** | 整張圖 | sum/mean/max pooling | 分子溶解度 |
| **圖層級分類** | 整張圖 | sum/mean/max pooling | 分子毒性 |

### 分子圖 vs 製程感測器圖

| 特性 | 分子圖 | 製程感測器圖 |
|------|-------|-------------|
| **拓撲** | 每個分子不同 | 固定（感測器數固定） |
| **大小** | 可變（需 padding） | 固定 |
| **讀出** | 排列不變 pooling | 節點對齊 flatten |
| **架構** | GCN / GAT / MPNN | GraphSAGE |

---

## 💻 實作環境需求

### 必要套件
```python
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0
scikit-learn >= 1.0.0
tensorflow >= 2.10.0             # 深度學習框架
keras >= 2.10.0                  # 高階 API
```

### 選用套件
```python
rdkit >= 2022.9                  # 分子處理（SMILES → 分子圖）
networkx >= 2.8                  # 圖視覺化與分析
```

---

## 📈 學習路徑建議

### 第一階段：理論基礎建立（必讀）
1. 閱讀 [Unit18_GNN_Overview.md](Unit18_GNN_Overview.md)
2. 執行 [Unit18_GNN_Overview.ipynb](Unit18_GNN_Overview.ipynb)
3. 重點掌握：
   - 圖的數學表示（鄰接矩陣、度矩陣、特徵矩陣）
   - 訊息傳遞框架（聚合 + 更新）
   - GCN / GraphSAGE / GAT 的差異
   - 圖層級任務與讀出池化

### 第二階段：從簡單到複雜的案例學習

**建議學習順序**：

1. **分子水溶解度**（建議最先學習，圖層級回歸）
   - 閱讀 [Unit18_Example_Molecular_Solubility.md](Unit18_Example_Molecular_Solubility.md)
   - 執行 [Unit18_Example_Molecular_Solubility.ipynb](Unit18_Example_Molecular_Solubility.ipynb)
   - 重點：SMILES → 分子圖、padding + 遮罩、Readout 池化

2. **分子毒性分類**（GAT 注意力機制、類別不平衡）
   - 閱讀 [Unit18_Example_Molecular_Toxicity.md](Unit18_Example_Molecular_Toxicity.md)
   - 執行 [Unit18_Example_Molecular_Toxicity.ipynb](Unit18_Example_Molecular_Toxicity.ipynb)
   - 重點：GAT 注意力、class weight、ROC-AUC/PR-AUC

3. **製程故障診斷**（GraphSAGE、固定拓撲感測器圖）
   - 閱讀 [Unit18_Example_Process_Fault_Diagnosis.md](Unit18_Example_Process_Fault_Diagnosis.md)
   - 執行 [Unit18_Example_Process_Fault_Diagnosis.ipynb](Unit18_Example_Process_Fault_Diagnosis.ipynb)
   - 重點：感測器關聯圖、GraphSAGE、穩健性測試

---

## 🔍 化工領域核心應用

### 1. 分子性質預測 (QSPR) ⭐
- **目標**：直接從分子結構預測物理化學性質
- **GNN 優勢**：無需人工挑選描述子，自動學習結構特徵
- **典型應用**：溶解度、沸點、黏度、 $\log P$ 、毒性預測
- **關鍵技術**：SMILES → 分子圖、Readout 池化、圖層級預測

### 2. 製程故障診斷
- **目標**：識別製程異常的類型與根因
- **GNN 優勢**：明確利用感測器之間的關聯結構
- **典型應用**：TEP 故障分類、設備診斷、感測器失效檢測
- **關鍵技術**：感測器關聯圖、GraphSAGE、穩健性分析

### 3. 反應網路與製程圖分析
- **目標**：分析反應路徑、製程流程的拓撲結構
- **GNN 優勢**：處理任意拓撲、捕捉全域連接關係
- **典型應用**：反應路徑預測、製程優化、流程圖分析

---

## 🔗 與其他單元的關聯

- **Unit15 (DNN)**：GNN 的讀出層與基線模型皆使用 MLP
- **Unit16 (CNN)**：CNN 處理規則網格，GNN 處理不規則圖，互補關係
- **Unit17 (RNN)**：RNN 處理序列，GNN 處理圖，兩者皆打破固定向量限制
- **Unit09 (異常檢測)**：TEP 資料集與 MSPC 方法在本單元作為故障診斷的對照
- **Unit19 (PINN)**：同為 Part_4 進階深度學習專題，PINN 將物理方程嵌入學習

---

**課程資訊**
- 課程名稱：AI在化工上之應用 (ChemE 3590)
- 課程單元：Unit18 圖神經網路 (Graph Neural Networks, GNN)
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-11-18

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。
