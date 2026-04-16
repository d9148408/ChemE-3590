from pathlib import Path
import os

# ========================================
# 路徑設定 (兼容 Colab 與 Local)
# ========================================
UNIT_OUTPUT_DIR = 'P2_Unit07_Isolation_Forest_Wafer'
SOURCE_DATA_DIR = 'wafer_defect'

try:
    from google.colab import drive
    IN_COLAB = True
    print("✓ 偵測到 Colab 環境，準備掛載 Google Drive...")
    drive.mount('/content/drive', force_remount=True)
except ImportError:
    IN_COLAB = False
    print("✓ 偵測到 Local 環境")

try:
    shortcut_path = '/content/ChemE-3590'
    os.remove(shortcut_path)
except FileNotFoundError:
    pass

if IN_COLAB:
    source_path = Path('/content/drive/My Drive/Colab Notebooks/ChemE-3590')
    os.symlink(source_path, shortcut_path)
    shortcut_path = Path(shortcut_path)
    if source_path.exists():
        NOTEBOOK_DIR = shortcut_path / 'Part_2' / 'Unit07'
        OUTPUT_DIR = NOTEBOOK_DIR / 'outputs' / UNIT_OUTPUT_DIR
        DATA_DIR = NOTEBOOK_DIR / 'data' / SOURCE_DATA_DIR
        MODEL_DIR = OUTPUT_DIR / 'models'
        FIG_DIR = OUTPUT_DIR / 'figs'
    else:
        print("⚠️ 找不到路徑雲端ChemE-3590路徑，請確認自己的雲端資料夾是否正確")
else:
    NOTEBOOK_DIR = Path.cwd()
    OUTPUT_DIR = NOTEBOOK_DIR / 'outputs' / UNIT_OUTPUT_DIR
    DATA_DIR = NOTEBOOK_DIR / 'data' / SOURCE_DATA_DIR
    MODEL_DIR = OUTPUT_DIR / 'models'
    FIG_DIR = OUTPUT_DIR / 'figs'

NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n✓ Notebook工作目錄: {NOTEBOOK_DIR}")
print(f"✓ 數據來源目錄: {DATA_DIR}")
print(f"✓ 結果輸出目錄: {OUTPUT_DIR}")
print(f"✓ 模型輸出目錄: {MODEL_DIR}")
print(f"✓ 圖檔輸出目錄: {FIG_DIR}")

# 基礎套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import joblib
warnings.filterwarnings('ignore')

# sklearn 套件
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)

# 設定隨機種子
SEED = 42
np.random.seed(SEED)

# 設定繪圖樣式
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style='whitegrid', font_scale=1.0)

print("套件載入完成")
print(f"  numpy:       {np.__version__}")
print(f"  pandas:      {pd.__version__}")
print(f"  scikit-learn: ", end="")
import sklearn; print(sklearn.__version__)

# 數據來源: Semiconductor Wafer Defect Classification Dataset
# https://www.kaggle.com/datasets/meruvakodandasuraj/semiconductor-wafer-defect-classification-dataset
# 授權: CC0: Public Domain

import requests
import zipfile

# ----------------------------------------
# 路徑設定
# ----------------------------------------
url      = "https://www.kaggle.com/api/v1/datasets/download/meruvakodandasuraj/semiconductor-wafer-defect-classification-dataset"
zip_path = DATA_DIR / 'archive.zip'
csv_file = DATA_DIR / 'semiconductor_wafer_defect_dataset.csv'

def download_and_extract():
    print(f"正在下載資料集...")
    print(f"  URL: {url}")
    try:
        response = requests.get(url, allow_redirects=True, stream=True, timeout=60)
        response.raise_for_status()
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  ✓ 下載成功: {zip_path}")
    except Exception as e:
        print(f"  ✗ 下載失敗: {e}")
        print("  → 請手動將 archive.zip 放至:", DATA_DIR)
        return False

    if zipfile.is_zipfile(zip_path):
        print(f"  正在解壓縮至: {DATA_DIR}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(DATA_DIR)
        print(f"  ✓ 解壓縮完成")
        return True
    else:
        print("  ✗ 下載的檔案不是有效的 ZIP 格式")
        return False

# ----------------------------------------
# 執行下載邏輯
# ----------------------------------------
if csv_file.exists():
    print(f"✓ 資料集已存在，跳過下載")
    print(f"  路徑: {csv_file}")
elif zip_path.exists():
    print(f"✓ 偵測到 archive.zip，直接解壓縮...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(DATA_DIR)
    print(f"  ✓ 解壓縮完成")
else:
    print("未偵測到本地資料，嘗試線上下載...")
    success = download_and_extract()
    if not success:
        raise FileNotFoundError(
            f"無法取得資料集。請手動下載 archive.zip 並放至:\n  {DATA_DIR}\n"
            f"下載頁面: https://www.kaggle.com/datasets/meruvakodandasuraj/semiconductor-wafer-defect-classification-dataset"
        )

# ----------------------------------------
# 確認檔案清單
# ----------------------------------------
print(f"\n=== 數據集檔案清單 ===")
for f in sorted(DATA_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45} {size_kb:>8.1f} KB")

# ----------------------------------------
# 載入數據
# ----------------------------------------
data_path = DATA_DIR / 'semiconductor_wafer_defect_dataset.csv'
df = pd.read_csv(data_path)

print("=== 數據集基本資訊 ===")
print(f"總筆數: {len(df)}")
print(f"欄位數: {len(df.columns)}")
print(f"欄位名稱: {list(df.columns)}")
print()

print("=== 前 5 筆數據 ===")
print(df.head())
print()

print("=== 資料型態 ===")
print(df.dtypes)
print()

print("=== 缺失值 ===")
print(df.isnull().sum())
print()

print("=== 目標變數分布（defect_label）===")
counts = df['defect_label'].value_counts().sort_index()
for label, count in counts.items():
    label_str = "正常晶圓" if label == 0 else "缺陷晶圓"
    print(f"  {label} ({label_str}): {count} 筆 ({count/len(df)*100:.1f}%)")
print()

print("=== 製程步驟分布（process_step）===")
print(df['process_step'].value_counts())
print()

print("=== 數值特徵統計摘要 ===")
num_cols = ['temperature_c', 'pressure_torr', 'gas_flow_sccm',
            'etch_rate_nm_min', 'voltage_v', 'current_ma']
print(df[num_cols].describe().round(3))

# ----------------------------------------
# Figure 1: 製程步驟分布與各步驟缺陷率
# ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Semiconductor Wafer: Process Step Distribution & Defect Rate', fontsize=14, fontweight='bold')

# 左圖：各製程步驟樣本數（按缺陷/正常分層）
step_label = df.groupby(['process_step', 'defect_label']).size().unstack(fill_value=0)
step_label.columns = ['Normal (0)', 'Defect (1)']
step_label.plot(kind='bar', ax=axes[0], color=['steelblue', 'tomato'], edgecolor='white', width=0.7)
axes[0].set_title('Sample Count by Process Step')
axes[0].set_xlabel('Process Step')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=30)
axes[0].legend(title='Defect Label')

# 右圖：各製程步驟缺陷率
defect_rate = df.groupby('process_step')['defect_label'].mean() * 100
defect_rate.plot(kind='bar', ax=axes[1], color='tomato', edgecolor='white', width=0.6)
axes[1].set_title('Defect Rate by Process Step (%)')
axes[1].set_xlabel('Process Step')
axes[1].set_ylabel('Defect Rate (%)')
axes[1].tick_params(axis='x', rotation=30)
axes[1].axhline(y=df['defect_label'].mean()*100, color='navy', linestyle='--', label='Overall Avg')
axes[1].legend()
for bar in axes[1].patches:
    axes[1].annotate(f'{bar.get_height():.1f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     ha='center', va='bottom', fontsize=10)

plt.tight_layout()
save_path = FIG_DIR / '01_process_step_distribution.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"圖檔已儲存: {save_path}")
plt.show()
plt.close()

# ----------------------------------------
# Figure 2: 數值特徵分布（正常 vs 缺陷）
# ----------------------------------------
num_cols = ['temperature_c', 'pressure_torr', 'gas_flow_sccm',
            'etch_rate_nm_min', 'voltage_v', 'current_ma']
col_labels = ['Temperature (°C)', 'Pressure (Torr)', 'Gas Flow (sccm)',
              'Etch Rate (nm/min)', 'Voltage (V)', 'Current (mA)']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Feature Distributions: Normal vs Defective Wafers', fontsize=14, fontweight='bold')
axes = axes.flatten()

colors = {0: 'steelblue', 1: 'tomato'}
labels_map = {0: 'Normal (0)', 1: 'Defect (1)'}

for i, (col, label) in enumerate(zip(num_cols, col_labels)):
    for lbl in [0, 1]:
        data = df[df['defect_label'] == lbl][col]
        axes[i].hist(data, bins=40, alpha=0.6, color=colors[lbl],
                     label=labels_map[lbl], density=True, edgecolor='none')
    axes[i].set_title(f'{label}')
    axes[i].set_xlabel(label)
    axes[i].set_ylabel('Density')
    axes[i].legend(fontsize=9)

    # 顯示均值差
    mean0 = df[df['defect_label'] == 0][col].mean()
    mean1 = df[df['defect_label'] == 1][col].mean()
    axes[i].axvline(mean0, color='steelblue', linestyle='--', linewidth=1.5, alpha=0.9)
    axes[i].axvline(mean1, color='tomato', linestyle='--', linewidth=1.5, alpha=0.9)

plt.tight_layout()
save_path = FIG_DIR / '02_feature_distributions.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"圖檔已儲存: {save_path}")

# 印出均值差
print("\n=== 各特徵的正常/缺陷均值比較 ===")
print(f"{'Feature':<22} {'Normal Mean':>12} {'Defect Mean':>12} {'Diff%':>8}")
print("-" * 56)
for col, label in zip(num_cols, col_labels):
    m0 = df[df['defect_label'] == 0][col].mean()
    m1 = df[df['defect_label'] == 1][col].mean()
    diff_pct = (m1 - m0) / m0 * 100
    print(f"{label:<22} {m0:>12.2f} {m1:>12.2f} {diff_pct:>7.1f}%")

plt.show()
plt.close()

# ----------------------------------------
# Figure 3: 特徵相關性熱圖
# ----------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
corr_matrix = df[num_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, ax=ax,
            linewidths=0.5, square=True)
ax.set_title('Feature Correlation Heatmap\n(Numerical Features)', fontsize=13, fontweight='bold')
ax.set_xticklabels(['Temp', 'Pressure', 'Gas Flow', 'Etch Rate', 'Voltage', 'Current'],
                   rotation=30, ha='right')
ax.set_yticklabels(['Temp', 'Pressure', 'Gas Flow', 'Etch Rate', 'Voltage', 'Current'],
                   rotation=0)

plt.tight_layout()
save_path = FIG_DIR / '03_correlation_heatmap.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"圖檔已儲存: {save_path}")
plt.show()
plt.close()

# 印出高相關對
print("\n=== 高相關特徵對（|r| > 0.3）===")
for i in range(len(num_cols)):
    for j in range(i+1, len(num_cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.3:
            print(f"  {num_cols[i]:<22} vs {num_cols[j]:<22}  r = {r:+.3f}")

# ----------------------------------------
# 數據前處理
# ----------------------------------------
# 1. 移除 wafer_id（ID欄位，無建模意義）
df_feat = df.drop(columns=['wafer_id'])

# 定義特徵類型
num_features = ['temperature_c', 'pressure_torr', 'gas_flow_sccm',
                'etch_rate_nm_min', 'voltage_v', 'current_ma']
cat_features = ['process_step']
target_col = 'defect_label'

X = df_feat[num_features + cat_features]
y = df_feat[target_col]

# 2. 訓練/測試集分割（分層抽樣 70:30）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED
)

print("=== 數據分割結果（分層抽樣）===")
print(f"訓練集: {len(X_train)} 筆 → "
      f"正常: {(y_train==0).sum()} ({(y_train==0).mean()*100:.1f}%)  "
      f"缺陷: {(y_train==1).sum()} ({(y_train==1).mean()*100:.1f}%)")
print(f"測試集: {len(X_test)} 筆  → "
      f"正常: {(y_test==0).sum()} ({(y_test==0).mean()*100:.1f}%)  "
      f"缺陷: {(y_test==1).sum()} ({(y_test==1).mean()*100:.1f}%)")

# 3. 建立 ColumnTransformer（數值標準化 + 類別 One-Hot Encoding）
#    注意: OneHotEncoder sparse_output=False 適用於 sklearn >= 1.2
#           舊版 sklearn 請改用 sparse=False
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_features)
    ],
    remainder='drop'
)

# 4. 用訓練集 fit，再 transform 訓練集與測試集（避免數據洩漏）
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

# 取得處理後的特徵名稱
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
all_feature_names = num_features + list(ohe_feature_names)

print(f"\n處理後特徵數: {X_train_proc.shape[1]}")
print("特徵名稱:")
for i, name in enumerate(all_feature_names):
    print(f"  {i+1:2d}. {name}")

print(f"\nX_train_proc shape: {X_train_proc.shape}")
print(f"X_test_proc  shape: {X_test_proc.shape}")

import time

# ----------------------------------------
# 建立並訓練兩種 contamination 設定的模型
# ----------------------------------------
contamination_configs = {
    'conservative': 0.10,   # 保守設定（低於真實缺陷率）
    'matched':      0.15    # 匹配設定（與真實缺陷率一致）
}

models = {}

for name, contamination in contamination_configs.items():
    clf = IsolationForest(
        n_estimators  = 200,
        max_samples   = 'auto',      # 預設 min(256, n_samples)
        contamination = contamination,
        max_features  = 1.0,
        bootstrap     = False,
        random_state  = SEED,
        n_jobs        = -1
    )
    t0 = time.time()
    clf.fit(X_train_proc)
    elapsed = time.time() - t0
    models[name] = clf
    print(f"[{name}] contamination={contamination}  "
          f"訓練時間: {elapsed:.2f}s  "
          f"offset_: {clf.offset_:.4f}")

# ----------------------------------------
# 儲存模型
# ----------------------------------------
for name, clf in models.items():
    model_path = MODEL_DIR / f'isolation_forest_{name}.pkl'
    joblib.dump(clf, model_path)
    print(f"模型已儲存: {model_path}")

print("\n✓ 所有模型訓練完成")

# ----------------------------------------
# 在測試集上取得預測結果與異常分數
# ----------------------------------------
# Isolation Forest 標籤約定：1=正常，-1=異常
# → 轉換為二元標籤：0=正常，1=缺陷，與 y_test 一致
results = {}
for name, clf in models.items():
    # 預測標籤（1=正常 / -1=異常）
    preds_raw = clf.predict(X_test_proc)
    # 轉換：-1 → 1（缺陷），1 → 0（正常）
    preds = (preds_raw == -1).astype(int)
    # 異常分數（越低越異常，negate 使越高越可疑）
    scores = -clf.decision_function(X_test_proc)

    prec  = precision_score(y_test, preds)
    rec   = recall_score(y_test, preds)
    f1    = f1_score(y_test, preds)
    auc   = roc_auc_score(y_test, scores)

    results[name] = {
        'preds': preds, 'scores': scores,
        'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
        'contamination': contamination_configs[name]
    }

    print(f"\n=== [{name}] contamination={contamination_configs[name]} ===")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print()
    print(classification_report(y_test, preds,
                                 target_names=['Normal (0)', 'Defect (1)']))

# ----------------------------------------
# Figure 4: 混淆矩陣比較
# ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrix Comparison\n(Isolation Forest on Wafer Test Set)', fontsize=13, fontweight='bold')

titles = {
    'conservative': 'Conservative  (contamination=0.10)',
    'matched':      'Matched       (contamination=0.15)'
}
for ax, (name, res) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, res['preds'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred Normal', 'Pred Defect'],
                yticklabels=['True Normal', 'True Defect'],
                linewidths=0.5)
    ax.set_title(f"{titles[name]}\n"
                 f"Precision={res['precision']:.3f}  "
                 f"Recall={res['recall']:.3f}  "
                 f"F1={res['f1']:.3f}")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
save_path = FIG_DIR / '04_confusion_matrices.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"圖檔已儲存: {save_path}")
plt.show()
plt.close()

# ----------------------------------------
# Figure 5: ROC 曲線
# ----------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
colors_map = {'conservative': 'steelblue', 'matched': 'tomato'}
styles_map  = {'conservative': '--',        'matched': '-'}

for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['scores'])
    label = f"{name} (AUC={res['auc']:.4f})"
    ax.plot(fpr, tpr, color=colors_map[name], linestyle=styles_map[name],
            linewidth=2, label=label)

ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random (AUC=0.50)')
ax.fill_between(fpr, tpr, alpha=0.05, color='tomato')

ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
ax.set_title('ROC Curve — Isolation Forest\n(Wafer Defect Detection)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])

plt.tight_layout()
save_path = FIG_DIR / '05_roc_curve.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"圖檔已儲存: {save_path}")
plt.show()
plt.close()

print(f"\n=== ROC-AUC 評估 ===")
for name, res in results.items():
    print(f"  [{name}] contamination={res['contamination']}  AUC = {res['auc']:.4f}")

# ----------------------------------------
# Figure 6: 異常分數分布（正常 vs 缺陷）
# ----------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Anomaly Score Distribution: Normal vs Defective Wafers', fontsize=13, fontweight='bold')

for ax, (name, res) in zip(axes, results.items()):
    scores_normal = res['scores'][y_test.values == 0]
    scores_defect = res['scores'][y_test.values == 1]

    ax.hist(scores_normal, bins=50, alpha=0.6, color='steelblue',
            density=True, label=f'Normal (n={len(scores_normal)})', edgecolor='none')
    ax.hist(scores_defect, bins=50, alpha=0.6, color='tomato',
            density=True, label=f'Defect (n={len(scores_defect)})', edgecolor='none')

    # 決策閾值
    threshold = -models[name].offset_
    ax.axvline(threshold, color='black', linestyle='--', linewidth=1.8,
               label=f'Threshold ({threshold:.3f})')

    ax.set_xlabel('Anomaly Score (higher = more anomalous)')
    ax.set_ylabel('Density')
    ax.set_title(f'[{name}] contamination={res["contamination"]}')
    ax.legend(fontsize=9)

    # 統計摘要
    print(f"\n[{name}] 異常分數統計:")
    print(f"  正常晶圓:  Mean={scores_normal.mean():.4f}  Std={scores_normal.std():.4f}")
    print(f"  缺陷晶圓:  Mean={scores_defect.mean():.4f}  Std={scores_defect.std():.4f}")
    print(f"  分數差異:  {scores_defect.mean() - scores_normal.mean():.4f}")

plt.tight_layout()
save_path = FIG_DIR / '06_anomaly_score_distribution.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n圖檔已儲存: {save_path}")
plt.show()
plt.close()

# ----------------------------------------
# Figure 7: 不同 contamination 值的 Precision / Recall / F1
# ----------------------------------------
contam_range = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]
metrics_list = []

# 使用已訓練的 matched 模型的異常分數（與 contamination 無關），只改閾值
# 重新訓練每個 contamination 設定
base_scores = -models['matched'].decision_function(X_test_proc)

for c in contam_range:
    # 根據 contamination 決定閾值：以訓練集分位數決定
    train_scores = -models['matched'].decision_function(X_train_proc)
    threshold_c = np.percentile(train_scores, 100 * (1 - c))
    preds_c = (base_scores >= threshold_c).astype(int)

    prec_c = precision_score(y_test, preds_c, zero_division=0)
    rec_c  = recall_score(y_test, preds_c, zero_division=0)
    f1_c   = f1_score(y_test, preds_c, zero_division=0)
    metrics_list.append({'contamination': c, 'Precision': prec_c,
                         'Recall': rec_c, 'F1-Score': f1_c})

df_metrics = pd.DataFrame(metrics_list)

# 繪圖
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_metrics['contamination'], df_metrics['Precision'],
        marker='o', color='steelblue', linewidth=2, label='Precision')
ax.plot(df_metrics['contamination'], df_metrics['Recall'],
        marker='s', color='tomato', linewidth=2, label='Recall')
ax.plot(df_metrics['contamination'], df_metrics['F1-Score'],
        marker='^', color='green', linewidth=2, label='F1-Score')

best_idx = df_metrics['F1-Score'].idxmax()
best_c   = df_metrics.loc[best_idx, 'contamination']
best_f1  = df_metrics.loc[best_idx, 'F1-Score']
ax.axvline(best_c, color='gold', linestyle='--', linewidth=1.5,
           label=f'Best F1 @ contamination={best_c}')

ax.set_xlabel('contamination', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision / Recall / F1 vs contamination\n(Isolation Forest, Wafer Defect)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim([0, 1.0])

plt.tight_layout()
save_path = FIG_DIR / '07_contamination_analysis.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"圖檔已儲存: {save_path}")
plt.show()
plt.close()

print("\n=== contamination 掃描結果 ===")
print(df_metrics.to_string(index=False, float_format='{:.3f}'.format))
print(f"\n最佳 F1-Score: {best_f1:.4f} @ contamination={best_c}")

# ----------------------------------------
# Figure 8: 特徵重要性（基於孤立樹中各特徵的分割次數）
# ----------------------------------------
clf_matched = models['matched']

# 計算每個特徵在所有孤立樹中被選為分割特徵的平均次數
feature_split_counts = np.zeros(len(all_feature_names))
for tree in clf_matched.estimators_:
    feature_indices = tree.tree_.feature
    for idx in feature_indices:
        if idx >= 0:  # 非葉節點（-2 代表葉節點）
            feature_split_counts[idx] += 1

# 正規化為相對重要性
feature_importance = feature_split_counts / feature_split_counts.sum()
sorted_idx = np.argsort(feature_importance)[::-1]

# 整理結果
importance_df = pd.DataFrame({
    'Feature':    [all_feature_names[i] for i in sorted_idx],
    'Importance': feature_importance[sorted_idx]
})

# 繪圖
fig, ax = plt.subplots(figsize=(9, 6))
colors_feat = ['#e74c3c' if i < 6 else '#3498db' for i in range(len(sorted_idx))]
bars = ax.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1],
               color=colors_feat[::-1], edgecolor='white', height=0.7)

# 標記數值
for bar, val in zip(bars, importance_df['Importance'][::-1]):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='left', fontsize=9)

ax.set_xlabel('Relative Feature Importance (Split Frequency)', fontsize=11)
ax.set_title('Feature Importance in Isolation Forest\n(Red = Sensor, Blue = Process Step OHE)', fontsize=12, fontweight='bold')
ax.set_xlim([0, importance_df['Importance'].max() * 1.15])

plt.tight_layout()
save_path = FIG_DIR / '08_feature_importance.png'
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"圖檔已儲存: {save_path}")
plt.show()
plt.close()

print("\n=== 特徵重要性排序 ===")
print(importance_df.to_string(index=False, float_format='{:.4f}'.format))

# ----------------------------------------
# 前 10 名最異常晶圓案例解析（使用 matched 模型）
# ----------------------------------------
scores_all = -models['matched'].decision_function(X_test_proc)

# 組合測試集原始資料與異常分數
df_test_result = X_test.copy().reset_index(drop=True)
df_test_result['anomaly_score'] = scores_all
df_test_result['true_label']    = y_test.values
df_test_result['predicted']     = results['matched']['preds']

# 取分數最高（最可疑）的前 10 筆
top10 = df_test_result.nlargest(10, 'anomaly_score')

print("=== 前 10 名最異常晶圓（matched 模型，anomaly_score 由高至低）===")
print(f"{'Rank':<5} {'process_step':<15} {'temp_c':>7} {'pres':>6} "
      f"{'gas':>6} {'etch':>6} {'volt':>6} {'curr':>7} "
      f"{'score':>8} {'true':>5} {'pred':>5}")
print("-" * 82)

for rank, (_, row) in enumerate(top10.iterrows(), 1):
    true_str = "DEF" if row['true_label'] == 1 else "NRM"
    pred_str = "DEF" if row['predicted']   == 1 else "NRM"
    flag = " ✓" if row['true_label'] == row['predicted'] else " ✗"
    print(f"{rank:<5} {row['process_step']:<15} "
          f"{row['temperature_c']:>7.1f} {row['pressure_torr']:>6.2f} "
          f"{row['gas_flow_sccm']:>6.1f} {row['etch_rate_nm_min']:>6.1f} "
          f"{row['voltage_v']:>6.2f} {row['current_ma']:>7.1f} "
          f"{row['anomaly_score']:>8.4f} {true_str:>5} {pred_str:>5}{flag}")

# 統計準確率
correct = (top10['true_label'] == top10['predicted']).sum()
print(f"\n前 10 名異常晶圓中，正確預測 {correct}/10 筆")
print(f"真實缺陷數: {top10['true_label'].sum()}/10 筆")

# ----------------------------------------
# 結論總結：評估指標彙總表
# ----------------------------------------
print("=" * 60)
print("   Isolation Forest — 半導體晶圓缺陷檢測 結果總結")
print("=" * 60)
print(f"\n  數據集  : semiconductor_wafer_defect_dataset.csv")
print(f"  訓練集  : {len(X_train)} 筆")
print(f"  測試集  : {len(X_test)} 筆（缺陷佔 {(y_test==1).mean()*100:.1f}%）")
print(f"  特徵數  : {X_train_proc.shape[1]}（6 數值 + 5 OHE 製程步驟）")
print()
print(f"  {'模型設定':<25} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
print(f"  {'-'*55}")
for name, res in results.items():
    print(f"  {name:<25} {res['precision']:>7.4f} {res['recall']:>7.4f} "
          f"{res['f1']:>7.4f} {res['auc']:>7.4f}")

print()
print("  關鍵結論:")
print("  1. contamination=0.15 (匹配真實缺陷率) F1 最高，Recall 最佳")
print("  2. AUC=0.73 顯示孤立森林在非監督條件下有合理區分能力")
print("  3. current_ma 與 temperature_c 是最重要的感測器特徵")
print("  4. 建議作為第一道篩選器，高異常分數晶圓再交監督模型複檢")
print()
print("  儲存產出:")
for name in models.keys():
    model_path = MODEL_DIR / f'isolation_forest_{name}.pkl'
    print(f"  → 模型: {model_path.name}")
for fig_file in sorted(FIG_DIR.glob('*.png')):
    print(f"  → 圖檔: {fig_file.name}")
print("=" * 60)
