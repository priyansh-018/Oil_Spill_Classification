import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import clone

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

# ─────────────────────────────────────────────
# 0. MANUAL SMOTE (no external dependency)
# ─────────────────────────────────────────────
def smote_oversample(X, y, minority_class=1, k=5, random_state=42):
    """
    Synthetic Minority Over-sampling Technique.
    Balances classes by generating synthetic minority samples.
    """
    rng = np.random.RandomState(random_state)
    X_min = X[y == minority_class]
    n_maj  = int((y == 0).sum())
    n_min  = len(X_min)
    n_syn  = n_maj - n_min

    synthetic = []
    for _ in range(n_syn):
        idx    = rng.randint(0, n_min)
        sample = X_min[idx]
        dists  = np.linalg.norm(X_min - sample, axis=1)
        dists[idx] = np.inf
        nn    = X_min[rng.choice(np.argsort(dists)[:k])]
        alpha = rng.random()
        synthetic.append(sample + alpha * (nn - sample))

    X_syn = np.array(synthetic)
    y_syn = np.ones(len(synthetic), dtype=int)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
DATA_PATH = r"C:\Users\Priyansh\Downloads\oil_spill.csv"   # ← update path if needed

df = pd.read_csv(DATA_PATH)
print(f"Dataset shape : {df.shape}")
print(f"Target dist.  :\n{df['target'].value_counts()}")
print(f"Imbalance     : {df['target'].value_counts()[0]/df['target'].value_counts()[1]:.1f}:1\n")

X = df.drop(columns=['target']).values
y = df['target'].values
feature_names = df.drop(columns=['target']).columns.tolist()


# ─────────────────────────────────────────────
# 2. TRAIN / TEST SPLIT  (stratified, 80/20)
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train : {len(X_train)} samples  (oil spill: {y_train.sum()})")
print(f"Test  : {len(X_test)} samples   (oil spill: {y_test.sum()})\n")


# ─────────────────────────────────────────────
# 3. SCALE FEATURES
# ─────────────────────────────────────────────
scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)
X_test_sc    = scaler.transform(X_test)


# ─────────────────────────────────────────────
# 4. HANDLE CLASS IMBALANCE  (SMOTE on train only)
# ─────────────────────────────────────────────
X_train_res, y_train_res = smote_oversample(X_train_sc, y_train, random_state=42)
print(f"After SMOTE → Train: {len(X_train_res)}  "
      f"(oil: {y_train_res.sum()}  non: {(y_train_res==0).sum()})\n")


# ─────────────────────────────────────────────
# 5. DEFINE & TRAIN MODELS
# ─────────────────────────────────────────────
model_defs = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300, class_weight='balanced',
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=1000,
        solver='lbfgs', random_state=42
    ),
    "MLP Neural Network": MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        solver='adam', alpha=0.001, max_iter=500,
        early_stopping=True, validation_fraction=0.1,
        random_state=42
    ),
}

results = {}
for name, clf in model_defs.items():
    print(f"Training  {name} ...")
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]

    results[name] = {
        'model':     clf,
        'y_pred':    y_pred,
        'y_prob':    y_prob,
        'accuracy':  accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall':    recall_score(y_test, y_pred, zero_division=0),
        'f1':        f1_score(y_test, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_test, y_prob),
        'avg_prec':  average_precision_score(y_test, y_prob),
    }


# ─────────────────────────────────────────────
# 6. RESULTS TABLE
# ─────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Model':<25} {'Acc':>7} {'Prec':>7} {'Recall':>8} {'F1':>7} {'ROC-AUC':>9}")
print("="*75)
for name, r in results.items():
    print(f"{name:<25} {r['accuracy']:>7.4f} {r['precision']:>7.4f} "
          f"{r['recall']:>8.4f} {r['f1']:>7.4f} {r['roc_auc']:>9.4f}")
print("="*75)

best_name = max(results, key=lambda n: results[n]['f1'])
best      = results[best_name]
print(f"\nBest model by F1 : {best_name}")
print("\nClassification Report (best model):")
print(classification_report(y_test, best['y_pred'],
      target_names=["Non-Oil Spill", "Oil Spill"]))


# ─────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────
colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Oil Spill Detection — Model Evaluation Dashboard",
             fontsize=16, fontweight='bold', y=1.01)

# (A) Metrics bar chart
ax = axes[0, 0]
metrics_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
x = np.arange(len(metrics_keys))
w = 0.18
for i, (name, r) in enumerate(results.items()):
    ax.bar(x + i*w, [r[m] for m in metrics_keys], w,
           label=name, color=colors[i], alpha=0.85)
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(['Accuracy','Precision','Recall','F1','ROC-AUC'], fontsize=9)
ax.set_ylim(0, 1.15)
ax.set_title("Metrics Comparison (All Models)", fontweight='bold')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.4)

# (B) Confusion matrix
ax = axes[0, 1]
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Oil Spill','Oil Spill'],
            yticklabels=['Non-Oil Spill','Oil Spill'],
            ax=ax, annot_kws={"size": 14})
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title(f"Confusion Matrix\n({best_name})", fontweight='bold')

# (C) ROC curves
ax = axes[0, 2]
for i, (name, r) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f"{name} (AUC={r['roc_auc']:.3f})")
ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Random')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (D) Precision-Recall curves
ax = axes[1, 0]
for i, (name, r) in enumerate(results.items()):
    p_c, r_c, _ = precision_recall_curve(y_test, r['y_prob'])
    ax.plot(r_c, p_c, color=colors[i], lw=2,
            label=f"{name} (AP={r['avg_prec']:.3f})")
ax.axhline(y=y_test.mean(), color='k', linestyle='--',
           alpha=0.4, label='Baseline')
ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# (E) Feature importances
ax = axes[1, 1]
rf = results['Random Forest']['model']
if hasattr(rf, 'feature_importances_'):
    imp    = rf.feature_importances_
    top_ix = np.argsort(imp)[-15:]
    ax.barh([feature_names[i] for i in top_ix], imp[top_ix],
            color='#2196F3', alpha=0.85)
    ax.set_title("Top 15 Feature Importances\n(Random Forest)", fontweight='bold')
    ax.set_xlabel('Importance'); ax.grid(axis='x', alpha=0.3)

# (F) Class distribution
ax = axes[1, 2]
x = np.arange(2)
ax.bar(x - 0.2, [(y_train_res==0).sum(), y_train_res.sum()],
       0.35, label='Train (after SMOTE)', color='#4CAF50', alpha=0.85)
ax.bar(x + 0.2, [(y_test==0).sum(), y_test.sum()],
       0.35, label='Test (original)',     color='#FF9800', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(['Non-Oil Spill', 'Oil Spill'])
ax.set_title('Class Distribution\n(Train vs Test)', fontweight='bold')
ax.set_ylabel('Count'); ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("oil_spill_results.png", dpi=150, bbox_inches='tight')
print("Plot saved → oil_spill_results.png")


# ─────────────────────────────────────────────
# 8. 5-FOLD CROSS-VALIDATION  (best model)
# ─────────────────────────────────────────────
print(f"\n─── 5-Fold CV ({best_name}) ───")
X_sc_all = scaler.fit_transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1, cv_auc = [], []
for fold, (tr, va) in enumerate(skf.split(X_sc_all, y)):
    Xtr, Xva = X_sc_all[tr], X_sc_all[va]
    ytr, yva = y[tr], y[va]
    Xtr_r, ytr_r = smote_oversample(Xtr, ytr, random_state=42)
    clf_cv = clone(model_defs[best_name])
    clf_cv.fit(Xtr_r, ytr_r)
    yp    = clf_cv.predict(Xva)
    yprob = clf_cv.predict_proba(Xva)[:,1]
    cv_f1.append(f1_score(yva, yp, zero_division=0))
    cv_auc.append(roc_auc_score(yva, yprob))
    print(f"  Fold {fold+1}: F1={cv_f1[-1]:.4f}  ROC-AUC={cv_auc[-1]:.4f}")

print(f"  Mean F1     : {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")
print(f"  Mean ROC-AUC: {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")


# ─────────────────────────────────────────────
# 9. INFERENCE HELPER
# ─────────────────────────────────────────────
def predict_new(feature_vector: list, threshold: float = 0.5) -> dict:
    """
    Predict on a single new sample.

    Parameters
    ----------
    feature_vector : list of 49 numeric values (f_1 … f_49)
    threshold      : decision boundary (default 0.5)

    Returns
    -------
    dict with label, probability, confidence
    """
    arr  = scaler.transform(np.array(feature_vector).reshape(1, -1))
    prob = float(best['model'].predict_proba(arr)[0][1])
    label = "Oil Spill" if prob >= threshold else "Non-Oil Spill"
    return {"label": label,
            "probability": round(prob, 4),
            "confidence":  round(max(prob, 1-prob)*100, 2)}


# Example:
# sample = df.drop(columns=['target']).iloc[0].tolist()
# print(predict_new(sample))
