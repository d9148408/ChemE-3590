# Unit19 物理資訊神經網路 (Physics-Informed Neural Networks, PINN)

## 📚 單元簡介

在 Unit15 ~ Unit18 中，我們學習了 DNN、CNN、RNN 與 GNN。這些架構都是**純資料驅動 (Data-Driven)** 的：模型完全從資料中學習規律，當資料稀少或帶有雜訊時容易過擬合，甚至預測出**違反物理定律**的結果（如質量不守恆）。然而化工程序的核心幾乎都由**微分方程**描述，且我們往往已經知道這些控制方程。

**物理資訊神經網路 (Physics-Informed Neural Networks, PINN)** 由 Raissi、Perdikaris 與 Karniadakis 於 2019 年系統性提出，核心思想是把**已知的物理方程直接寫進神經網路的學習目標**：用一個神經網路 $\hat{u}_\theta(x, t)$ 近似微分方程的解，並要求它不僅擬合觀測資料，還要讓**微分方程的殘差 (Residual) 趨近於零**。

PINN 的核心創新與優勢：

- **物理方程當作正則化 (Regularization)**：即使資料稀少，物理殘差仍約束解的形狀
- **自動微分 (Automatic Differentiation)**：精確計算網路輸出對輸入的各階導數，組成微分方程
- **無網格 (Mesh-free)**：以配點法 (Collocation) 在連續域上採樣，不需建立網格
- **統一處理正問題與反問題**：同一框架既能求解方程，也能從資料辨識未知參數

在化工領域，PINN 特別適合：

- **反應動力學**：濃度隨時間的變化 $\frac{dC}{dt} = -kC$ （質量守恆）
- **傳遞現象**：熱傳導、質量擴散、動量傳遞（偏微分方程）
- **程序動態**：反應器、蒸餾塔、換熱器的動態響應
- **參數辨識**：由稀疏實驗資料反推速率常數、活化能、擴散係數

本單元涵蓋：

- **PINN 核心原理**：用神經網路表示解、自動微分、複合損失函數
- **正問題 vs 反問題**：求解方程 vs 參數辨識
- **化工常見控制方程**：ODE、BVP、PDE、耦合非線性系統
- **TensorFlow/Keras 實作**：從零建立 PINN 求解微分方程
- **四個遞進案例**：一階 ODE、觸媒 BVP、暫態擴散 PDE、非等溫反應器系統

本單元是 Part_4 深度學習系列的進階專題，建議在完成 Unit15 (DNN) 基礎後學習。

---

## 🎯 學習目標

完成本單元後，您將能夠：

1. **理解 PINN 的動機**：認識「資料驅動」與「物理驅動」兩種建模範式的互補
2. **掌握自動微分**：了解神經網路如何對輸入變數求偏導，組成微分方程殘差
3. **理解複合損失**：物理殘差、初始/邊界條件與觀測資料如何組成總損失
4. **區分正反問題**：正問題（求解方程）與反問題（參數辨識）的差異
5. **解決化工微分方程**：反應動力學、傳遞現象、耦合非線性系統

---

## 📖 單元內容架構

### 1️⃣ 總覽篇：PINN 基礎理論 ⭐

**檔案**：
- 講義：[Unit19_PINN_Overview.md](Unit19_PINN_Overview.md)
- 範例：[Unit19_PINN_Overview.ipynb](Unit19_PINN_Overview.ipynb)

**內容重點**：
- **為什麼需要 PINN？**：
  - 純數值解法 vs 純資料驅動 DNN 的兩難
  - 化工程序核心由微分方程描述，且常已知機理
  - PINN 的核心優勢：物理正則化、無網格、正反問題統一
- **用神經網路表示解 + 自動微分**：
  - 以 $\hat{u}_\theta(x, t)$ 近似微分方程的解
  - 自動微分計算 $\frac{\partial \hat{u}}{\partial x}$ 、 $\frac{\partial^2 \hat{u}}{\partial x^2}$ 等各階導數
  - 巢狀 GradientTape 的實作技巧
- **複合損失函數**：
  - 物理殘差損失 $L_{\text{PDE}}$ ：方程殘差趨近於零
  - 初始條件損失 $L_{\text{IC}}$ 、邊界條件損失 $L_{\text{BC}}$
  - 觀測資料損失 $L_{\text{data}}$
  - 總損失 $L = L_{\text{PDE}} + L_{\text{IC}} + L_{\text{BC}} + L_{\text{data}}$
- **正問題 vs 反問題**：
  - 正問題 (Forward)：已知參數，求解方程
  - 反問題 (Inverse)：由觀測資料辨識未知參數（設為 `tf.Variable`）
- **化工常見控制方程**：一階 ODE、二階 BVP、拋物線 PDE
- **TensorFlow/Keras 實作**：從零求解一階反應動力學 $\frac{dC}{dt} = -kC$
- **化工領域應用場景**

**適合讀者**：所有學員，**建議最先閱讀**以建立完整的 PINN 理論基礎

---

### 2️⃣ 實際案例 1：觸媒顆粒內擴散-反應 (Catalyst Diffusion-Reaction) ⭐

**檔案**：
- 講義：[Unit19_Example_Catalyst_Diffusion_Reaction.md](Unit19_Example_Catalyst_Diffusion_Reaction.md)
- 程式範例：[Unit19_Example_Catalyst_Diffusion_Reaction.ipynb](Unit19_Example_Catalyst_Diffusion_Reaction.ipynb)

**內容重點**：
- **問題背景**：
  - 多孔觸媒顆粒內的擴散-反應（非均相催化無所不在）
  - 內擴散限制：反應快、擴散慢時，顆粒中心濃度遠低於表面
  - Thiele 模數 $\phi$ 與有效因子 $\eta$ 的物理意義（Thiele 1939）
- **數學模型**：
  - 二階邊界值問題 (BVP)： $\frac{d^2 C}{d\xi^2} = \phi^2 C$
  - Dirichlet (表面) + Neumann (中心對稱) 邊界條件
  - 解析解 $C = \cosh(\phi\xi)/\cosh(\phi)$ ，有效因子 $\eta = \tanh(\phi)/\phi$
- **關鍵技術**：
  - **巢狀自動微分**（兩層 GradientTape）計算二階導數
  - 邊界項加權幫助 BVP 收斂
  - 正問題（求濃度分佈與 $\eta$ ）+ 反問題（辨識 Thiele 模數 $\phi$ ）
- **方程類型**：二階 ODE 邊界值問題 (BVP)

**適合場景**：反應工程、非均相催化、觸媒設計、反應器放大

---

### 3️⃣ 實際案例 2：薄膜內暫態擴散 (Transient Diffusion) ⭐

**檔案**：
- 講義：[Unit19_Example_Transient_Diffusion.md](Unit19_Example_Transient_Diffusion.md)
- 程式範例：[Unit19_Example_Transient_Diffusion.ipynb](Unit19_Example_Transient_Diffusion.ipynb)

**內容重點**：
- **問題背景**：
  - 擴散是化工傳遞現象三大支柱之一
  - 應用於薄膜分離、乾燥、吸附脫附、藥物控釋、熱傳導
  - 擴散係數 $D$ 常需由滲透實驗的稀疏量測反推
- **數學模型**：
  - Fick 第二定律（拋物線型 PDE）： $\frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial x^2}$
  - 初始條件 (IC) + 邊界條件 (BC)
  - 解析解 $C = \sin(\pi x)\exp(-D\pi^2 t)$
- **關鍵技術**：
  - **2 輸入 PINN** $(x, t)$ 於時空域求解 PDE
  - 巢狀自動微分同時計算時間一階導數與空間二階導數
  - **學習率分段衰減**穩定正問題收斂
  - 正問題（求整個時空濃度場）+ 反問題（辨識擴散係數 $D$ ）
- **方程類型**：拋物線型偏微分方程 (PDE)

**適合場景**：傳遞現象、薄膜分離、乾燥程序、材料鑑定

---

### 4️⃣ 進階案例 3：非等溫反應器動態與活化能辨識 (Non-isothermal Reactor) ⭐

**檔案**：
- 講義：[Unit19_Example_Reactor_Kinetics.md](Unit19_Example_Reactor_Kinetics.md)
- 程式範例：[Unit19_Example_Reactor_Kinetics.ipynb](Unit19_Example_Reactor_Kinetics.ipynb)

**內容重點**：
- **問題背景**：
  - 放熱反應的非等溫反應器（聚合、氧化、加氫、硝化）
  - 熱失控 (thermal runaway) 是製程安全的頭號風險
  - 動力學參數估計：由稀疏量測反推活化能
- **數學模型**：
  - 耦合非線性 ODE 系統（質量平衡 + 能量平衡 + Arrhenius 動力學）
  - $\frac{dc}{d\tau} = -Da \cdot c \cdot \exp\left(\frac{\gamma\theta}{1+\theta}\right)$
  - $\frac{d\theta}{d\tau} = B \cdot Da \cdot c \cdot \exp(\cdots) - \beta\theta$
  - 無解析解 → 以高精度 `solve_ivp` 作為驗證基準
- **關鍵技術**：
  - **單一網路、雙輸出** PINN 同時求解濃度 $c$ 與溫度 $\theta$
  - persistent GradientTape 一次取 $\frac{dc}{d\tau}$ 與 $\frac{d\theta}{d\tau}$
  - 學習率分段衰減
  - 正問題（求反應器動態）+ 反問題（辨識活化能 $\gamma$ ）
- **方程類型**：耦合非線性 ODE 系統（進階）

**適合場景**：反應工程、製程安全、動力學參數估計、數位分身

---

## 📊 四個案例的遞進關係

| 案例 | 方程類型 | 輸入 | 輸出 | 辨識參數 | 解析解 |
|------|---------|------|------|---------|--------|
| **總覽** | 一階 ODE (IVP) | $t$ | $C$ | 速率常數 $k$ | 有 |
| **觸媒擴散** | 二階 ODE (BVP) | $\xi$ | $C$ | Thiele 模數 $\phi$ | 有 |
| **暫態擴散** | 拋物線 PDE | $x, t$ | $C$ | 擴散係數 $D$ | 有 |
| **非等溫反應器** | 耦合非線性系統 | $\tau$ | $c, \theta$ | 活化能 $\gamma$ | 無 |

---

## 🧭 PINN 正問題 vs 反問題

| 特性 | 正問題 (Forward) | 反問題 (Inverse) |
|------|-----------------|------------------|
| **已知** | 完整方程 + 參數 | 方程形式 + 稀疏觀測 |
| **求解** | 方程的解 | 未知參數 + 解 |
| **資料需求** | 無需觀測資料 | 少量帶雜訊觀測 |
| **參數設定** | 固定常數 | `tf.Variable`（加入訓練變數） |
| **化工應用** | 製程模擬、數位分身 | 動力學參數估計、物性辨識 |

---

## 💻 實作環境需求

### 必要套件
```python
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
scipy >= 1.7.0                   # solve_ivp 數值基準
tensorflow >= 2.10.0             # 深度學習框架 + 自動微分
keras >= 2.10.0                  # 高階 API
```

> [!NOTE]
> PINN 的核心依賴 **TensorFlow 的自動微分 (GradientTape)**。模型儲存建議使用 TF checkpoint 格式（而非 `.h5`），以避免 h5py 相依性問題。

---

## 📈 學習路徑建議

### 第一階段：理論基礎建立（必讀）
1. 閱讀 [Unit19_PINN_Overview.md](Unit19_PINN_Overview.md)
2. 執行 [Unit19_PINN_Overview.ipynb](Unit19_PINN_Overview.ipynb)
3. 重點掌握：
   - 用神經網路表示解 + 自動微分
   - 複合損失函數（物理殘差 + IC/BC + 觀測資料）
   - 正問題 vs 反問題
   - 巢狀 GradientTape 計算導數

### 第二階段：從簡單到複雜的案例學習

**建議學習順序（由淺入深）**：

1. **觸媒擴散-反應**（二階 BVP、巢狀自動微分）
   - 閱讀 [Unit19_Example_Catalyst_Diffusion_Reaction.md](Unit19_Example_Catalyst_Diffusion_Reaction.md)
   - 執行 [Unit19_Example_Catalyst_Diffusion_Reaction.ipynb](Unit19_Example_Catalyst_Diffusion_Reaction.ipynb)
   - 重點：二階導數、Dirichlet + Neumann BC、Thiele 模數辨識

2. **薄膜暫態擴散**（拋物線 PDE、時空域）
   - 閱讀 [Unit19_Example_Transient_Diffusion.md](Unit19_Example_Transient_Diffusion.md)
   - 執行 [Unit19_Example_Transient_Diffusion.ipynb](Unit19_Example_Transient_Diffusion.ipynb)
   - 重點：2 輸入 PINN、時間一階 + 空間二階導數、擴散係數辨識

3. **非等溫反應器**（耦合非線性系統、進階）
   - 閱讀 [Unit19_Example_Reactor_Kinetics.md](Unit19_Example_Reactor_Kinetics.md)
   - 執行 [Unit19_Example_Reactor_Kinetics.ipynb](Unit19_Example_Reactor_Kinetics.ipynb)
   - 重點：單網路雙輸出、Arrhenius 耦合、活化能辨識、solve_ivp 基準

---

## 🔍 化工領域核心應用

### 1. 反應動力學與參數辨識 ⭐
- **目標**：求解反應器動態並由稀疏資料辨識動力學參數
- **PINN 優勢**：求解軌跡與辨識參數合而為一，天生融入物理定律
- **典型應用**：速率常數、活化能、反應級數估計
- **關鍵技術**：複合損失、反問題 `tf.Variable`、數值基準驗證

### 2. 傳遞現象建模
- **目標**：求解熱傳、質傳、動量傳遞的 PDE
- **PINN 優勢**：無網格、可融合稀疏實驗觀測
- **典型應用**：暫態擴散、熱傳導、薄膜滲透、材料物性鑑定
- **關鍵技術**：時空域配點、巢狀自動微分、邊界/初始條件損失

### 3. 製程數位分身 (Digital Twin)
- **目標**：建立尊重物理定律的製程模型
- **PINN 優勢**：資料稀少時仍保證物理一致性
- **典型應用**：反應器數位分身、製程監控、即時狀態估計

---

## 🔗 與其他單元的關聯

- **Unit15 (DNN)**：PINN 的網路主體即為 MLP，差別在於損失函數加入物理殘差
- **Unit17 (RNN)**：RNN 以資料學習動態，PINN 以物理方程約束動態，互補關係
- **Unit18 (GNN)**：同為 Part_4 進階深度學習專題，GNN 處理圖結構資料
- **Unit04 (反應動力學) / 傳遞現象課程**：PINN 為這些經典化工方程提供新的求解與辨識工具

---

**課程資訊**
- 課程名稱：AI在化工上之應用 (ChemE 3590)
- 課程單元：Unit19 物理資訊神經網路 (Physics-Informed Neural Networks, PINN)
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜禎 助理教授
- 更新日期：2026-11-18

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。
