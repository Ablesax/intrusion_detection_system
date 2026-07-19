
import os, time, warnings, psutil, traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from collections import Counter

warnings.filterwarnings('ignore')


# CONFIGURATION


DATASET_DIR   = r"C:\Users\HP\Downloads\files\NSL-KDD"   
OUTPUT_DIR    = r"C:\Users\HP\Downloads\files\NSL-KDD\RESULTS"  
NUM_FEATURES  = 20          # top-k features after selection
K_FOLDS       = 10          # for cross-validation
TEST_SIZE     = 0.20        # 70/30 split
RANDOM_STATE  = 42
KNN_K         = 5

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR,  exist_ok=True)


# NSL-KDD COLUMN DEFINITIONS

NSL_KDD_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins','logged_in',
    'num_compromised','root_shell','su_attempted','num_root','num_file_creations',
    'num_shells','num_access_files','num_outbound_cmds','is_host_login',
    'is_guest_login','count','srv_count','serror_rate','srv_serror_rate',
    'rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','label','difficulty_level'
]

ATTACK_CATEGORIES = {
    'normal': 'Normal',
    # DoS attacks
    'neptune':'DoS','back':'DoS','land':'DoS','pod':'DoS','smurf':'DoS',
    'teardrop':'DoS','apache2':'DoS','udpstorm':'DoS','processtable':'DoS','mailbomb':'DoS',
    # Probe attacks
    'satan':'Probe','ipsweep':'Probe','nmap':'Probe','portsweep':'Probe',
    'mscan':'Probe','saint':'Probe',
    # R2L attacks
    'guess_passwd':'R2L','ftp_write':'R2L','imap':'R2L','phf':'R2L','multihop':'R2L',
    'warezmaster':'R2L','warezclient':'R2L','spy':'R2L','xlock':'R2L','xsnoop':'R2L',
    'snmpgetattack':'R2L','named':'R2L','sendmail':'R2L','httptunnel':'R2L','snmpguess':'R2L',
    # U2R attacks
    'buffer_overflow':'U2R','loadmodule':'U2R','rootkit':'U2R','perl':'U2R',
    'sqlattack':'U2R','xterm':'U2R','ps':'U2R'
}


# SECTION 1: DATASET DOWNLOAD & LOADING

def download_nsl_kdd():
    """Check that local dataset files exist (no download needed)."""
    files = ["KDDTrain+.txt", "KDDTest+.txt"]
    all_ok = True
    for fname in files:
        fpath = os.path.join(DATASET_DIR, fname)
        if os.path.exists(fpath):
            print(f"  ✓ {fname} found ({os.path.getsize(fpath)//1024} KB)")
        else:
            print(f"  ✗ {fname} NOT FOUND at: {fpath}")
            print(f"    → Update DATASET_DIR at the top of this file to the folder")
            print(f"      containing KDDTrain+.txt and KDDTest+.txt")
            all_ok = False
    return all_ok


def load_nsl_kdd():
    """Load and combine NSL-KDD train/test into one dataframe."""
    train_path = os.path.join(DATASET_DIR, "KDDTrain+.txt")
    test_path  = os.path.join(DATASET_DIR, "KDDTest+.txt")

    train_df = pd.read_csv(train_path, header=None, names=NSL_KDD_COLUMNS)
    test_df  = pd.read_csv(test_path,  header=None, names=NSL_KDD_COLUMNS)

    df = pd.concat([train_df, test_df], ignore_index=True)
    df.drop(columns=['difficulty_level'], inplace=True)

    print(f"  Dataset shape  : {df.shape}")
    print(f"  Label distribution:\n{df['label'].value_counts().head(10).to_string()}")
    return df



# SECTION 2: PREPROCESSING

def preprocess(df):
    """Full preprocessing pipeline: encode, normalize, binary-label."""
    print("\n[2] PREPROCESSING")
    print(f"  Initial shape  : {df.shape}")
    print(f"  Missing values : {df.isnull().sum().sum()}")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {before - len(df)}")

    # Encode categorical features
    cat_cols = ['protocol_type', 'service', 'flag']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Binary label: Normal=0, Attack=1
    df['binary_label'] = df['label'].apply(lambda x: 0 if x.strip() == 'normal' else 1)

    # Multi-class label (attack category)
    df['category'] = df['label'].apply(
        lambda x: ATTACK_CATEGORIES.get(x.strip(), 'Other'))

    print(f"  Binary label dist: {dict(Counter(df['binary_label']))}")
    print(f"  Attack categories: {dict(Counter(df['category']))}")

    # Separate features and labels
    drop_cols = ['label', 'binary_label', 'category']
    X = df.drop(columns=drop_cols).values.astype(float)
    y_binary   = df['binary_label'].values
    y_category = df['category'].values

    # Min-Max normalization
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    feature_names = [c for c in df.columns if c not in drop_cols]
    print(f"  Features after encoding: {X.shape[1]}")
    return X, y_binary, y_category, feature_names



# SECTION 3: FEATURE SELECTION

def select_features(X, y, feature_names, k=NUM_FEATURES):
    """Mutual-information-based feature selection (top-k)."""
    print(f"\n[3] FEATURE SELECTION  (top-{k} by mutual information)")
    selector = SelectKBest(mutual_info_classif, k=k)
    X_sel = selector.fit_transform(X, y)
    
    scores  = selector.scores_
    indices = selector.get_support(indices=True)
    selected_names = [feature_names[i] for i in indices]

    print(f"  Selected features: {selected_names}")

    # Bar chart of feature importance
    sorted_idx = np.argsort(scores[indices])[::-1]
    plt.figure(figsize=(12, 5))
    plt.bar(range(k), scores[indices][sorted_idx], color='#1F4E79', alpha=0.85)
    plt.xticks(range(k), [selected_names[i] for i in sorted_idx], rotation=45, ha='right', fontsize=9)
    plt.title("Feature Importance Scores (Mutual Information)", fontweight='bold', fontsize=13)
    plt.xlabel("Feature")
    plt.ylabel("Mutual Information Score")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
    plt.close()
    print(f"  → Saved: feature_importance.png")
    return X_sel, selected_names



# SECTION 4: MODEL DEFINITIONS

def get_models():
    return {
        "KNN":           KNeighborsClassifier(n_neighbors=KNN_K, metric='euclidean', n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=15,
                                                random_state=RANDOM_STATE),
        "SVM":           SVC(kernel='rbf', C=1.0, gamma='scale',
                             random_state=RANDOM_STATE, probability=True)
    }



# SECTION 5: EVALUATION METRICS

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train, predict, and compute all metrics including system metrics."""
    proc = psutil.Process(os.getpid())

    # ── Train ──
    mem_before_train = proc.memory_info().rss / (1024 ** 2)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - t0
    mem_after_train = proc.memory_info().rss / (1024 ** 2)

    # ── Predict ──
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = (time.perf_counter() - t0) / len(X_test) * 1000  # ms/instance

    # ── Classification metrics ──
    TP = int(np.sum((y_pred == 1) & (y_test == 1)))
    TN = int(np.sum((y_pred == 0) & (y_test == 0)))
    FP = int(np.sum((y_pred == 1) & (y_test == 0)))
    FN = int(np.sum((y_pred == 0) & (y_test == 1)))

    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far       = FP / (FP + TN) if (FP + TN) > 0 else 0
    detection_rate = recall

    mem_used = mem_after_train - mem_before_train

    metrics = {
        "Model":           name,
        "Accuracy":        round(accuracy, 4),
        "Precision":       round(precision, 4),
        "Recall":          round(recall, 4),
        "F1-Score":        round(f1, 4),
        "Detection Rate":  round(detection_rate, 4),
        "False Alarm Rate":round(far, 4),
        "Inference Time (ms/inst)": round(inference_time, 5),
        "Memory Usage (MB)":        round(abs(mem_used), 2),
        "Training Time (s)":        round(train_time, 4),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "model_obj": model,
        "y_pred": y_pred
    }
    return metrics



# SECTION 6: ENSEMBLE MAJORITY VOTING

def ensemble_predict(model_results, X_test, y_test, models_dict):
    """Majority vote across KNN, DT, SVM predictions."""
    print("\n[6] ENSEMBLE MAJORITY VOTING MODEL")
    preds = np.stack([r['y_pred'] for r in model_results], axis=1)
    y_ensemble = np.apply_along_axis(
        lambda row: np.bincount(row).argmax(), axis=1, arr=preds)

    TP = int(np.sum((y_ensemble == 1) & (y_test == 1)))
    TN = int(np.sum((y_ensemble == 0) & (y_test == 0)))
    FP = int(np.sum((y_ensemble == 1) & (y_test == 0)))
    FN = int(np.sum((y_ensemble == 0) & (y_test == 1)))

    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    far       = FP / (FP + TN) if (FP + TN) > 0 else 0

    return {
        "Model": "Ensemble (MV)",
        "Accuracy": round(accuracy, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-Score": round(f1, 4),
        "Detection Rate": round(recall, 4),
        "False Alarm Rate": round(far, 4),
        "Inference Time (ms/inst)": round(
            np.mean([r['Inference Time (ms/inst)'] for r in model_results]), 5),
        "Memory Usage (MB)": round(
            np.sum([r['Memory Usage (MB)'] for r in model_results]), 2),
        "Training Time (s)": round(
            np.sum([r['Training Time (s)'] for r in model_results]), 4),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "confusion_matrix": confusion_matrix(y_test, y_ensemble),
        "y_pred": y_ensemble
    }



# SECTION 7: CROSS-VALIDATION

def run_cross_validation(X, y, models_dict):
    """10-fold stratified cross-validation for each model."""
    print(f"\n[7] {K_FOLDS}-FOLD CROSS-VALIDATION")
    cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    for name, model in models_dict.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
        cv_results[name] = {
            "mean_f1":  round(scores.mean(), 4),
            "std_f1":   round(scores.std(), 4),
            "scores":   scores
        }
        print(f"  {name:15s} | Mean F1: {scores.mean():.4f} ± {scores.std():.4f}")
    return cv_results



# SECTION 8: VISUALIZATION

COLORS = {
    "KNN":          "#1F4E79",
    "Decision Tree":"#2E75B6",
    "SVM":          "#00B0F0",
    "Ensemble (MV)":"#FF6B35"
}

def plot_confusion_matrices(all_results):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("Confusion Matrices — All Models", fontweight='bold', fontsize=14)
    for ax, res in zip(axes, all_results):
        cm = res['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                    cmap='Blues', linewidths=0.5,
                    xticklabels=['Normal','Attack'],
                    yticklabels=['Normal','Attack'])
        ax.set_title(res['Model'], fontweight='bold')
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150)
    plt.close()
    print("  → Saved: confusion_matrices.png")


def plot_metric_comparison(all_results):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "Detection Rate", "False Alarm Rate"]
    models  = [r['Model'] for r in all_results]
    x = np.arange(len(metrics))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))
    for i, res in enumerate(all_results):
        vals = [res[m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width, label=res['Model'],
                      color=list(COLORS.values())[i], alpha=0.88, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, rotation=0)

    ax.set_xlabel("Metric", fontweight='bold')
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics — Model Comparison", fontweight='bold', fontsize=13)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics, rotation=15, ha='right')
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/metric_comparison.png", dpi=150)
    plt.close()
    print("  → Saved: metric_comparison.png")


def plot_system_metrics(all_results):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("System Performance Metrics", fontweight='bold', fontsize=13)

    sys_metrics = ["Inference Time (ms/inst)", "Memory Usage (MB)", "Training Time (s)"]
    ylabels     = ["ms per instance", "Megabytes", "Seconds"]
    colors = list(COLORS.values())

    for ax, metric, ylabel, color in zip(axes, sys_metrics, ylabels, colors):
        models = [r['Model'] for r in all_results]
        vals   = [r[metric] for r in all_results]
        bars = ax.bar(models, vals, color=colors[:len(models)], alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_title(metric, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(models, rotation=20, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/system_metrics.png", dpi=150)
    plt.close()
    print("  → Saved: system_metrics.png")


def plot_cross_validation(cv_results):
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(cv_results.keys())
    means = [cv_results[n]['mean_f1'] for n in names]
    stds  = [cv_results[n]['std_f1']  for n in names]
    colors = [COLORS.get(n, '#888888') for n in names]

    bars = ax.bar(names, means, color=colors, alpha=0.85, edgecolor='white', yerr=stds,
                  capsize=6, error_kw={'elinewidth': 2, 'ecolor': '#333333'})
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.003,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title(f"{K_FOLDS}-Fold Cross-Validation F1-Score", fontweight='bold', fontsize=13)
    ax.set_ylabel("Mean F1-Score")
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cross_validation.png", dpi=150)
    plt.close()
    print("  → Saved: cross_validation.png")


def plot_label_distribution(y):
    unique, counts = np.unique(y, return_counts=True)
    labels = ['Normal', 'Attack']
    fig, ax = plt.subplots(figsize=(6, 5))
    wedges, texts, autotexts = ax.pie(counts, labels=labels,
        autopct='%1.1f%%', colors=['#1F4E79', '#FF6B35'],
        startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
    for t in autotexts:
        t.set_fontsize(12)
        t.set_fontweight('bold')
    ax.set_title("Dataset Label Distribution (Binary)", fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/label_distribution.png", dpi=150)
    plt.close()
    print("  → Saved: label_distribution.png")


def save_results_csv(all_results, cv_results):
    metric_cols = ["Model","Accuracy","Precision","Recall","F1-Score",
                   "Detection Rate","False Alarm Rate",
                   "Inference Time (ms/inst)","Memory Usage (MB)","Training Time (s)",
                   "TP","TN","FP","FN"]
    rows = [{k: r[k] for k in metric_cols} for r in all_results]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_DIR}/results_summary.csv", index=False)
    print("  → Saved: results_summary.csv")

    cv_rows = [{"Model": n, "Mean_F1": v['mean_f1'], "Std_F1": v['std_f1']}
               for n, v in cv_results.items()]
    pd.DataFrame(cv_rows).to_csv(f"{OUTPUT_DIR}/cross_validation_results.csv", index=False)
    print("  → Saved: cross_validation_results.csv")
    return df


def print_full_report(df_results, cv_results):
    print("\n" + "="*80)
    print("  FINAL RESULTS REPORT — IDS FOR WSN USING LIGHTWEIGHT ML")
    print("="*80)
    print("\n┌─ CLASSIFICATION METRICS ─────────────────────────────────────────────────┐")
    display_cols = ["Model","Accuracy","Precision","Recall","F1-Score",
                    "Detection Rate","False Alarm Rate"]
    print(df_results[display_cols].to_string(index=False))
    print("\n┌─ SYSTEM PERFORMANCE METRICS ─────────────────────────────────────────────┐")
    sys_cols = ["Model","Inference Time (ms/inst)","Memory Usage (MB)","Training Time (s)"]
    print(df_results[sys_cols].to_string(index=False))
    print("\n┌─ CONFUSION MATRIX DETAILS ───────────────────────────────────────────────┐")
    for _, row in df_results.iterrows():
        print(f"  {row['Model']:18s}  TP={row['TP']}  TN={row['TN']}  FP={row['FP']}  FN={row['FN']}")
    print("\n┌─ 10-FOLD CROSS-VALIDATION F1 ────────────────────────────────────────────┐")
    for name, v in cv_results.items():
        print(f"  {name:18s}  Mean F1 = {v['mean_f1']:.4f}  ±  {v['std_f1']:.4f}")
    print("="*80)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("="*70)
    print(" IDS FOR WIRELESS SENSOR NETWORKS — LIGHTWEIGHT ML IMPLEMENTATION")
    print("="*70)

    # STEP 1: Download
    print("\n[1] DATASET ACQUISITION — NSL-KDD")
    ok = download_nsl_kdd()
    if not ok:
        print("  ERROR: Could not download dataset. Exiting.")
        return

    # STEP 2: Load & preprocess
    df = load_nsl_kdd()
    X, y_binary, y_category, feature_names = preprocess(df)

    # Plot distribution
    plot_label_distribution(y_binary)

    # STEP 3: Feature selection
    X_sel, selected_features = select_features(X, y_binary, feature_names, k=NUM_FEATURES)

    # STEP 4: Train/test split
    print(f"\n[4] TRAIN/TEST SPLIT  (70% train / 30% test)")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y_binary, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_binary)
    print(f"  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # STEP 5: Train & evaluate individual models
    print("\n[5] MODEL TRAINING & EVALUATION")
    models       = get_models()
    model_results = []
    for name, model in models.items():
        print(f"\n  ▸ {name}")
        res = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        model_results.append(res)
        print(f"    Accuracy={res['Accuracy']:.4f}  Precision={res['Precision']:.4f}  "
              f"Recall={res['Recall']:.4f}  F1={res['F1-Score']:.4f}  FAR={res['False Alarm Rate']:.4f}")
        print(f"    Inference={res['Inference Time (ms/inst)']:.5f}ms  "
              f"Memory={res['Memory Usage (MB)']:.2f}MB  "
              f"TrainTime={res['Training Time (s)']:.4f}s")

    # STEP 6: Ensemble
    ens = ensemble_predict(model_results, X_test, y_test, models)
    print(f"  Ensemble → Accuracy={ens['Accuracy']:.4f}  F1={ens['F1-Score']:.4f}  FAR={ens['False Alarm Rate']:.4f}")
    all_results = model_results + [ens]

    # STEP 7: Cross-validation (individual models only)
    cv_results = run_cross_validation(X_sel, y_binary, get_models())

    # STEP 8: Plots
    print("\n[8] GENERATING VISUALIZATIONS")
    plot_confusion_matrices(all_results)
    plot_metric_comparison(all_results)
    plot_system_metrics(all_results)
    plot_cross_validation(cv_results)

    # STEP 9: Save & report
    print("\n[9] SAVING RESULTS")
    df_results = save_results_csv(all_results, cv_results)
    print_full_report(df_results, cv_results)

    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}/")
    print("="*70)


if __name__ == "__main__":
    main()