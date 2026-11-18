# Part 4: 深度學習 (Deep Learning)

## 📚 Part 4 簡介

歡迎來到深度學習的世界！在前面的 Part_3 中，我們學習了傳統機器學習方法（線性模型、非線性模型、集成學習）。這些方法在結構化數據上表現優異，但面對以下場景時，深度學習展現出革命性的優勢：

- **複雜非線性關係**：化學反應網路、多相流動力學、複雜傳遞現象
- **高維數據**：光譜數據、影像數據、高通量實驗數據
- **序列數據**：時間序列製程數據、批次操作軌跡
- **自動特徵學習**：從原始數據自動提取有用特徵，無需人工特徵工程

**深度學習 (Deep Learning)** 透過多層神經網路的堆疊，能夠自動學習數據的階層式特徵表示，從簡單到複雜，層層抽象。在化工領域，深度學習正在革新傳統的建模、控制、優化與診斷方法。

本 Part 涵蓋五個核心深度學習主題，從三大基礎架構到兩個進階專題：
1. **深度神經網路 (DNN/MLP, Unit15)**：處理結構化表格數據，適合軟感測器、製程預測
2. **卷積神經網路 (CNN, Unit16)**：處理影像數據，適合缺陷檢測、顯微影像分析
3. **循環神經網路 (RNN/LSTM/GRU, Unit17)**：處理時間序列數據，適合製程動態建模、故障預測
4. **圖神經網路 (GNN, Unit18)**：處理圖結構數據，適合分子性質預測、製程故障診斷
5. **物理資訊神經網路 (PINN, Unit19)**：將物理方程嵌入學習，適合求解微分方程與參數辨識

基礎三架構 (Unit15-17) 為大學部核心內容；GNN 與 PINN (Unit18-19) 為延伸單元（進階專題，不列入課程週次），適合研究所選修或對深度學習有進一步興趣的學員。

---

## 🎯 學習目標

完成 Part 4 後，您將能夠：

1. **理解深度學習的核心概念**：神經元、前向傳播、反向傳播、激活函數、損失函數
2. **掌握五大深度學習主題**：DNN（表格數據）、CNN（影像數據）、RNN（序列數據）、GNN（圖數據）、PINN（物理方程）
3. **熟練使用 TensorFlow/Keras**：模型建立、訓練、評估、調參、部署、自動微分
4. **解決實際化工問題**：軟感測器設計、缺陷檢測、製程預測、異常診斷、分子性質預測、微分方程求解
5. **優化模型性能**：Early Stopping、Dropout、Batch Normalization、數據增強、複合損失設計

---

## 📖 單元目錄

### [Unit15: 深度神經網路與多層感知機 (Deep Neural Networks & Multi-Layer Perceptron)](Unit15/)

**主題**：深度學習基礎，處理結構化表格數據

**單元概述**：
- [Unit15_DNN_MLP_Overview.md](Unit15/Unit15_DNN_MLP_Overview.md)
- [Unit15_DNN_MLP_Overview.ipynb](Unit15/Unit15_DNN_MLP_Overview.ipynb)

**實際案例**：
- [蒸餾塔控制 (Distillation)](Unit15/Unit15_Example_Distillation.md) ([Notebook](Unit15/Unit15_Example_Distillation.ipynb))
- [燃料氣排放預測 (Fuel Gas Emission)](Unit15/Unit15_Example_FuelGasEmission.md) ([Notebook](Unit15/Unit15_Example_FuelGasEmission.ipynb))
- [採礦過程優化 (Mining)](Unit15/Unit15_Example_Mining.md) ([Notebook](Unit15/Unit15_Example_Mining.ipynb))
- [紅酒品質預測 (Red Wine)](Unit15/Unit15_Example_RedWine.md) ([Notebook](Unit15/Unit15_Example_RedWine.ipynb))

**作業練習**：[Unit15_DNN_MLP_Homework.ipynb](Unit15/Unit15_DNN_MLP_Homework.ipynb)

**核心概念**：
- 神經元數學模型
- 前向傳播與反向傳播
- 激活函數（ReLU、Sigmoid、Tanh）
- 損失函數（MSE、MAE、Cross-Entropy）
- 優化器（SGD、Adam、RMSprop）
- 正則化（Dropout、L1/L2）
- 批次標準化 (Batch Normalization)

**TensorFlow/Keras 技能**：
- Sequential API
- Functional API
- 模型編譯與訓練
- Early Stopping & Callbacks
- 模型儲存與載入

**應用場景**：
- **軟感測器 (Soft Sensor)**：從易測量變數預測難測量變數
- **品質預測**：基於操作條件預測產品品質
- **製程優化**：建立製程模型進行參數優化
- **實時監控**：快速預測與決策支援

---

### [Unit16: 卷積神經網路 (Convolutional Neural Networks, CNN)](Unit16/)

**主題**：專為影像數據設計的深度學習架構

**單元概述**：
- [Unit16_CNN_Overview.md](Unit16/Unit16_CNN_Overview.md)
- [Unit16_CNN_Overview.ipynb](Unit16/Unit16_CNN_Overview.ipynb)

**實際案例**：
- [大腸癌病理影像分類 (Colorectal Cancer)](Unit16/Unit16_Eample_Colorectal_Cancer.md) ([Notebook](Unit16/Unit16_Eample_Colorectal_Cancer.ipynb))
- [中文手寫數字識別 (Chinese MNIST)](Unit16/Unit16_Example_Chinese_MNIST.md) ([Notebook](Unit16/Unit16_Example_Chinese_MNIST.ipynb))
- [工業缺陷檢測 (Defect Detection)](Unit16/Unit16_Example_Defect_Detection.md) ([Notebook](Unit16/Unit16_Example_Defect_Detection.ipynb))
- [遷移學習缺陷檢測 (Transfer Learning)](Unit16/Unit16_Example_Transfer_Learning_Defect_Detection.md) ([Notebook](Unit16/Unit16_Example_Transfer_Learning_Defect_Detection.ipynb))

**作業練習**：[Unit16_CNN_Homework.ipynb](Unit16/Unit16_CNN_Homework.ipynb)

**核心概念**：
- 卷積運算 (Convolution)
- 池化 (Pooling)
- 感受野 (Receptive Field)
- 特徵圖 (Feature Maps)
- 步幅 (Stride) 與填充 (Padding)
- 局部連接與參數共享

**經典架構**：
- LeNet-5（手寫數字識別）
- AlexNet（深度 CNN 的突破）
- VGG（簡單而深的架構）
- ResNet（殘差連接）
- Inception（多尺度特徵）

**TensorFlow/Keras 技能**：
- Conv2D、MaxPooling2D
- Data Augmentation（數據增強）
- Transfer Learning（遷移學習）
- Fine-Tuning（微調）
- 預訓練模型（VGG16、ResNet50、EfficientNet）

**應用場景**：
- **缺陷檢測**：產品表面瑕疵、裂紋、異常斑點識別
- **顯微影像分析**：晶體形態、細胞計數、粒徑分布
- **製程視覺監控**：火焰狀態、液位識別、泡沫特性
- **光譜影像處理**：高光譜影像的成分分佈分析
- **安全監控**：洩漏檢測、人員行為識別

---

### [Unit17: 循環神經網路 (Recurrent Neural Networks, RNN)](Unit17/)

**主題**：專為時間序列數據設計的深度學習架構

**單元概述**：
- [Unit17_RNN_Overview.md](Unit17/Unit17_RNN_Overview.md)
- [Unit17_RNN_Overview.ipynb](Unit17/Unit17_RNN_Overview.ipynb)

**實際案例**：
- [鍋爐 NOx 排放預測 (Boiler)](Unit17/Unit17_Example_Boiler.md) ([Notebook](Unit17/Unit17_Example_Boiler.ipynb))
- [脫丁烷塔控制 (Debutanizer Column)](Unit17/Unit17_Example_debutanizer_column.md) ([Notebook](Unit17/Unit17_Example_debutanizer_column.ipynb))
- [NASA 渦輪風扇 RUL 預測](Unit17/Unit17_Example_RUL_NASA_Turbofan.md) ([Notebook](Unit17/Unit17_Example_RUL_NASA_Turbofan.ipynb))

**進階專題**：[Unit17_RNN_Advanced.md](Unit17/Unit17_RNN_Advanced.md)（雙向 RNN、Seq2Seq、Attention）

**作業練習**：[Unit17_RNN_Homework.ipynb](Unit17/Unit17_RNN_Homework.ipynb)

**核心概念**：
- 循環連接與記憶機制
- 隱藏狀態 (Hidden State)
- 前向傳播與 BPTT（時間反向傳播）
- 梯度消失與梯度爆炸問題
- 長短期記憶 (LSTM)
- 門控循環單元 (GRU)
- 雙向 RNN (BiRNN)
- 序列到序列 (Seq2Seq)
- 注意力機制 (Attention)

**TensorFlow/Keras 技能**：
- SimpleRNN、LSTM、GRU
- return_sequences 參數
- Stateful RNN
- Bidirectional Wrapper
- TimeDistributed Layer
- 多步預測策略

**應用場景**：
- **製程動態建模**：反應器、蒸餾塔動態響應預測
- **故障預測與診斷**：設備健康監測、剩餘壽命 (RUL) 預測
- **品質預測**：基於歷史操作軌跡預測產品品質
- **預測性控制 (MPC)**：多步預測，優化控制策略
- **批次製程監控**：追蹤批次進程，早期異常檢測
- **時間序列異常檢測**：識別製程異常模式

---

### [Unit18: 圖神經網路 (Graph Neural Networks, GNN)](Unit18/)

**主題**：專為圖結構數據設計的深度學習架構（分子、製程感測器網路）

**單元概述**：
- [Unit18_GNN_Overview.md](Unit18/Unit18_GNN_Overview.md)
- [Unit18_GNN_Overview.ipynb](Unit18/Unit18_GNN_Overview.ipynb)

**實際案例**：
- [分子水溶解度預測 (Molecular Solubility)](Unit18/Unit18_Example_Molecular_Solubility.md) ([Notebook](Unit18/Unit18_Example_Molecular_Solubility.ipynb))
- [分子毒性分類 (Molecular Toxicity)](Unit18/Unit18_Example_Molecular_Toxicity.md) ([Notebook](Unit18/Unit18_Example_Molecular_Toxicity.ipynb))
- [製程故障診斷 (Process Fault Diagnosis)](Unit18/Unit18_Example_Process_Fault_Diagnosis.md) ([Notebook](Unit18/Unit18_Example_Process_Fault_Diagnosis.ipynb))

**單元導覽**：[Unit18/README.md](Unit18/README.md)

**核心概念**：
- 圖的數學表示（鄰接矩陣、度矩陣、特徵矩陣、拉普拉斯矩陣）
- 訊息傳遞 (Message Passing)：聚合 + 更新
- 排列不變性 (Permutation Invariance)
- 讀出 (Readout) 池化與圖層級任務

**經典架構**：
- GCN（圖卷積網路）
- GraphSAGE（鄰居採樣 + 聚合）
- GAT（圖注意力網路）
- MPNN（訊息傳遞統一框架）

**TensorFlow/Keras 技能**：
- 從零實作圖卷積層
- 以 RDKit 將 SMILES 轉換為分子圖
- padding + 節點遮罩批次化
- 類別不平衡處理（class weight、ROC-AUC/PR-AUC）

**應用場景**：
- **分子性質預測 (QSPR)**：溶解度、沸點、毒性直接從結構預測
- **製程故障診斷**：感測器網路建模、多類別故障分類
- **反應網路分析**：反應路徑、製程流程圖拓撲分析

---

### [Unit19: 物理資訊神經網路 (Physics-Informed Neural Networks, PINN)](Unit19/)

**主題**：將物理方程嵌入神經網路學習目標，求解微分方程與參數辨識

**單元概述**：
- [Unit19_PINN_Overview.md](Unit19/Unit19_PINN_Overview.md)
- [Unit19_PINN_Overview.ipynb](Unit19/Unit19_PINN_Overview.ipynb)

**實際案例**：
- [觸媒顆粒內擴散-反應 (Catalyst Diffusion-Reaction)](Unit19/Unit19_Example_Catalyst_Diffusion_Reaction.md) ([Notebook](Unit19/Unit19_Example_Catalyst_Diffusion_Reaction.ipynb))
- [薄膜內暫態擴散 (Transient Diffusion)](Unit19/Unit19_Example_Transient_Diffusion.md) ([Notebook](Unit19/Unit19_Example_Transient_Diffusion.ipynb))
- [非等溫反應器與活化能辨識 (Reactor Kinetics)](Unit19/Unit19_Example_Reactor_Kinetics.md) ([Notebook](Unit19/Unit19_Example_Reactor_Kinetics.ipynb))

**單元導覽**：[Unit19/README.md](Unit19/README.md)

**核心概念**：
- 用神經網路表示微分方程的解
- 自動微分 (Automatic Differentiation) 與巢狀 GradientTape
- 複合損失：物理殘差 + 初始/邊界條件 + 觀測資料
- 正問題（求解方程）vs 反問題（參數辨識）
- 無網格 (Mesh-free) 配點法

**四個遞進案例**：
- 一階 ODE（反應動力學，辨識速率常數 $k$ ）
- 二階 BVP（觸媒擴散，辨識 Thiele 模數 $\phi$ ）
- 拋物線 PDE（暫態擴散，辨識擴散係數 $D$ ）
- 耦合非線性系統（非等溫反應器，辨識活化能 $\gamma$ ）

**TensorFlow/Keras 技能**：
- 巢狀 GradientTape 計算各階導數
- 單網路雙輸出求解耦合系統
- `tf.Variable` 作為待辨識參數
- 學習率分段衰減、TF checkpoint 模型儲存

**應用場景**：
- **反應動力學與參數辨識**：速率常數、活化能、反應級數估計
- **傳遞現象建模**：熱傳、質傳、薄膜滲透的 PDE 求解
- **製程數位分身 (Digital Twin)**：資料稀少時仍保證物理一致性

---

## 🗂️ 數據集

本 Part 使用的數據集涵蓋多種化工與工業場景：

### Unit15 (DNN) 數據
- **蒸餾塔數據**：溫度、壓力、流量、組成
- **燃料氣排放數據**：操作條件與排放濃度
- **採礦數據**：礦石性質與品質參數
- **紅酒品質數據**：化學成分與感官評分

### Unit16 (CNN) 數據
- **醫學影像**：病理切片、細胞影像
- **手寫數字**：中文手寫數字資料集
- **工業缺陷影像**：表面瑕疵、裂紋、異常
- **公開影像數據集**：MNIST、CIFAR-10、ImageNet 遷移學習

### Unit17 (RNN) 數據
- **鍋爐監測數據**：多變數時間序列
- **脫丁烷塔數據**：動態操作數據
- **NASA C-MAPSS**：渦輪風扇退化數據
- **製程時間序列**：溫度、壓力、流量隨時間變化

### Unit18 (GNN) 數據
- **ESOL (Delaney) 水溶解度**：1128 個有機小分子的 SMILES 與 $\log S$
- **Tox21 毒理學資料**：約 7800 個化合物的 NR-AhR 活化標記
- **田納西-伊士曼製程 (TEP)**：52 個製程變數、正常 + 21 種故障 (IDV)

### Unit19 (PINN) 數據
- **合成配點資料**：物理殘差配點與稀疏觀測雜訊數據
- **數值基準**：解析解（ODE/BVP/PDE）與 `solve_ivp` 高精度數值解

---

## 💻 環境需求

### 必要套件

```python
# 深度學習框架
tensorflow>=2.8.0  # 或 tensorflow-gpu（若有 GPU）
keras>=2.8.0  # TensorFlow 2.x 已內建 Keras

# 核心數據處理
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0

# 影像處理（Unit16）
opencv-python>=4.5.0  # 影像讀取與預處理
Pillow>=8.3.0  # 影像處理
scikit-image>=0.18.0  # 進階影像處理

# 模型評估與視覺化
scikit-learn>=1.0.0
scipy>=1.7.0

# 進階視覺化
plotly>=5.3.0  # 互動式圖表
tensorboard>=2.8.0  # TensorFlow 訓練監控

# 數據增強（Unit16）
albumentations>=1.1.0  # 強大的影像增強套件
imgaug>=0.4.0  # 影像增強

# 其他工具
h5py>=3.1.0  # 模型儲存
pydot>=1.4.0  # 模型視覺化
graphviz>=0.16  # 模型視覺化

# 圖神經網路（Unit18）
rdkit>=2022.9  # 分子處理（SMILES → 分子圖）
networkx>=2.8  # 圖視覺化與分析
# 物理資訊神經網路（Unit19）依賴 TensorFlow 自動微分與 scipy（solve_ivp，已列於上方）
```

### 安裝指令

```bash
# 基礎安裝（CPU 版本）
pip install tensorflow numpy pandas matplotlib seaborn opencv-python Pillow scikit-image scikit-learn scipy plotly h5py pydot

# GPU 版本（若有 NVIDIA GPU）
pip install tensorflow-gpu numpy pandas matplotlib seaborn opencv-python Pillow scikit-image scikit-learn scipy plotly h5py pydot

# 影像增強套件（Unit16）
pip install albumentations imgaug

# 視覺化工具
pip install graphviz
# Windows 用戶需額外安裝 Graphviz：https://graphviz.org/download/

# 使用 conda（建議）
conda create -n dl_env python=3.9
conda activate dl_env
conda install tensorflow numpy pandas matplotlib seaborn scikit-learn scipy
pip install opencv-python Pillow scikit-image plotly albumentations imgaug h5py pydot graphviz
```

### GPU 加速（可選但強烈建議）

若要使用 GPU 加速訓練，需要安裝：
- **CUDA Toolkit**（NVIDIA 提供）
- **cuDNN**（NVIDIA 提供）
- **tensorflow-gpu** 或 TensorFlow 2.x（自動偵測 GPU）

詳細安裝指南：[TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)

---

## 📊 學習路徑建議

### 1. 標準學習路徑（按順序學習）

**適合對象**：初學深度學習者、需系統性學習

1. **Unit15 DNN/MLP**：建立深度學習基礎概念
   - 理解神經元、前向/反向傳播
   - 掌握 TensorFlow/Keras 基本操作
   - 完成 4 個實際案例
2. **Unit16 CNN**：學習影像處理
   - 理解卷積、池化、特徵提取
   - 掌握數據增強與遷移學習
   - 完成 4 個影像案例
3. **Unit17 RNN**：學習時間序列建模
   - 理解記憶機制、LSTM、GRU
   - 掌握序列預測與異常檢測
   - 完成時間序列案例
4. **Unit18 GNN**（進階專題）：學習圖結構建模
   - 理解訊息傳遞、GCN/GraphSAGE/GAT
   - 掌握分子圖與感測器圖建模
   - 完成分子性質與故障診斷案例
5. **Unit19 PINN**（進階專題）：學習物理資訊建模
   - 理解自動微分、複合損失、正/反問題
   - 掌握以神經網路求解微分方程
   - 完成反應動力學與傳遞現象案例

### 2. 快速應用路徑（針對特定需求）

**適合對象**：有深度學習基礎、需快速解決特定問題

- **需要軟感測器/製程預測**：Unit15 (DNN)
- **需要影像分析/缺陷檢測**：Unit15 (基礎) → Unit16 (CNN)
- **需要時間序列預測/異常檢測**：Unit15 (基礎) → Unit17 (RNN)
- **需要分子性質預測/製程故障診斷**：Unit15 (基礎) → Unit18 (GNN)
- **需要求解微分方程/參數辨識**：Unit15 (基礎) → Unit19 (PINN)
- **需要全面掌握**：Unit15 → Unit16 → Unit17 → Unit18 → Unit19

### 3. 進階研究路徑

**適合對象**：研究生、進階應用者

1. 完成所有單元的基礎學習
2. 深入研究進階架構：
   - ResNet、Inception、EfficientNet（CNN）
   - Transformer、BERT（Attention）
   - Autoencoder、VAE、GAN（生成模型）
3. 探索超參數優化：Keras Tuner、Optuna
4. 學習模型部署：TensorFlow Serving、TensorFlow Lite
5. 將方法應用於自己的研究數據

---

## 🎓 學習建議

### 1. 理論與實務並重
- **理解數學原理**：前向/反向傳播、梯度計算、優化演算法
- **實作加深理解**：透過 Notebook 逐步實作模型
- **視覺化訓練過程**：使用 TensorBoard 監控訓練

### 2. 從小數據開始
- 先在小數據集上實驗（如 MNIST）
- 理解模型行為與超參數影響
- 再應用於大型複雜數據

### 3. 系統性調參
- **網路架構**：層數、神經元數、卷積核大小
- **訓練策略**：學習率、批次大小、優化器
- **正則化**：Dropout 比例、L2 係數
- **數據增強**：旋轉、翻轉、縮放、雜訊

### 4. 避免常見錯誤
- **過擬合**：使用 Dropout、Early Stopping、數據增強
- **梯度消失/爆炸**：使用 Batch Normalization、Gradient Clipping
- **訓練不穩定**：調整學習率、使用適當的初始化
- **記憶體不足**：減少批次大小、使用 Mixed Precision

### 5. 利用遷移學習
- Unit16 (CNN) 大量使用預訓練模型（VGG、ResNet）
- 在小數據集上能顯著提升性能
- 節省訓練時間與計算資源

### 6. 結合領域知識
- 用化工專業知識設計網路架構
- 評估模型預測的物理合理性
- 識別模型的限制與適用範圍

---

## 📝 評估方式

- **單元作業**：每個單元都有對應的作業 Notebook（40%）
- **期中專題**：選擇 Unit15 或 Unit16 完成一個完整專題（30%）
- **期末專題**：使用深度學習解決實際化工問題（30%）

**評估重點**：
- 模型架構設計的合理性
- 訓練策略的有效性
- 過擬合防止的完整性
- 結果解釋的正確性
- 程式碼的可讀性與效率
- 視覺化的清晰度

---

## 🔗 相關資源

### 官方文件
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)

### 推薦閱讀
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning* (MIT Press)
- Chollet, F. (2021). *Deep Learning with Python, 2nd Edition*
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition"
- Hochreiter, S., & Schmidhuber, J. (1997). "Long Short-Term Memory"

### 化工應用文獻
- 深度學習在製程控制的應用
- 軟感測器設計與優化
- 影像分析在化工的應用
- 時間序列預測與異常檢測

### 線上課程與教學
- [Coursera: Deep Learning Specialization (Andrew Ng)](https://www.coursera.org/specializations/deep-learning)
- [Fast.ai: Practical Deep Learning for Coders](https://www.fast.ai/)
- [TensorFlow: Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/)

### 實用工具
- **TensorBoard**：訓練監控與視覺化
- **Keras Tuner**：自動超參數優化
- **TensorFlow Datasets (TFDS)**：大量公開數據集
- **Model Garden**：預訓練模型與範例

---

## ⚙️ 與其他 Part 的連接

### 前置知識（來自 Part_1、Part_2、Part_3）
- **Part_1**：Python 基礎、NumPy、Pandas、Matplotlib
- **Part_2**：非監督式學習（降維、分群、異常檢測）
- **Part_3**：監督式學習（回歸、分類、集成學習）

### 何時使用深度學習 vs. 傳統機器學習？

**使用深度學習（Part_4）的場景**：
- 影像、音訊、文本等非結構化數據
- 數據量大（通常 > 10,000 樣本）
- 複雜的非線性關係
- 需要自動特徵學習
- 時間序列長期依賴關係

**使用傳統機器學習（Part_3）的場景**：
- 結構化表格數據
- 數據量小到中等（< 10,000 樣本）
- 需要高度可解釋性
- 計算資源有限
- 訓練時間要求短

---

## 🚀 進階主題（未來擴展）

本 Part 涵蓋深度學習的三大基礎架構（DNN/CNN/RNN）與兩個進階專題（GNN/PINN）。未來可能擴展的主題包括：

- **Transformer 與自注意力機制**（序列與時空預測的新主流）
- **自編碼器與生成模型 (Autoencoder, VAE, GAN)**
- **深度強化學習 (Deep RL) 在製程控制的應用**（見 Part_5）
- **生成式 AI 與大語言模型 (LLM) 在化工的應用**（見 Part_5）
- **模型部署與邊緣計算 (TensorFlow Serving / Lite)**
- **聯邦學習與隱私保護**

---

## ⚠️ 計算資源建議

### 最低需求（可學習但訓練較慢）
- **CPU**：Intel i5 或 AMD Ryzen 5
- **RAM**：8 GB
- **儲存空間**：10 GB 可用空間

### 建議配置（較佳學習體驗）
- **CPU**：Intel i7 或 AMD Ryzen 7
- **RAM**：16 GB
- **GPU**：NVIDIA GTX 1060 或更好（6 GB VRAM）
- **儲存空間**：20 GB 可用空間（SSD 更佳）

### 理想配置（順暢訓練與實驗）
- **CPU**：Intel i9 或 AMD Ryzen 9
- **RAM**：32 GB
- **GPU**：NVIDIA RTX 3060 或更好（12 GB VRAM）
- **儲存空間**：50 GB 可用空間（NVMe SSD）

### 雲端選項（若本機資源不足）
- **Google Colab**：免費 GPU（T4），適合學習與小專題
- **Kaggle Notebooks**：免費 GPU，每週 30 小時
- **AWS SageMaker**：付費，專業級訓練
- **Azure Machine Learning**：付費，企業級解決方案

---

**課程資訊**
- 課程名稱：AI在化工上之應用 (ChemE 3590)
- 課程單元：Part 4 深度學習 (Deep Learning)
- 課程製作：逢甲大學 化工系 智慧程序系統工程實驗室
- 授課教師：莊曜桽 助理教授
- 更新日期：2026-11-18

**課程授權 [CC BY-NC-SA 4.0]**
 - 本教材遵循 [創用CC 姓名標示-非商業性-相同方式分享 4.0 國際 (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.zh) 授權。

---
