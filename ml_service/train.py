"""
train.py — Colon Cancer ML Pipeline
Stratégie: SelectKBest(20) + Logistic Regression (class_weight='balanced')
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings("ignore")

DATA_PATH = os.environ.get("DATA_PATH", "/data/colon_cancer_dataset.csv")
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")
TOP_N     = int(os.environ.get("TOP_N", 20))

os.makedirs(MODEL_DIR, exist_ok=True)

print("[1/6] Chargement du dataset...")
df = pd.read_csv(DATA_PATH)
X  = df.drop(columns=["Class", df.columns[0]])
y  = (df["Class"] == "Abnormal").astype(int)
feature_names = np.array(X.columns.tolist())
print(f"     Shape: {X.shape}  |  Abnormal={y.sum()}  Normal={len(y)-y.sum()}")

print("[2/6] Normalisation StandardScaler...")
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"[3/6] Sélection des TOP {TOP_N} gènes (SelectKBest + ANOVA)...")
selector = SelectKBest(f_classif, k=TOP_N)
X_sel    = selector.fit_transform(X_scaled, y)
selected_indices  = selector.get_support(indices=True)
selected_features = feature_names[selected_indices]
f_scores          = selector.scores_[selected_indices]
top_genes = sorted(
    [{"rank": 0, "gene": g, "f_score": round(float(s), 4)}
     for g, s in zip(selected_features, f_scores)],
    key=lambda x: -x["f_score"]
)
for i, g in enumerate(top_genes): g["rank"] = i + 1
print(f"     TOP 5 gènes: {[g['gene'] for g in top_genes[:5]]}")

print("[4/6] Split train/test (80/20, stratifié)...")
X_tr, X_te, y_tr, y_te = train_test_split(
    X_sel, y, test_size=0.2, random_state=42, stratify=y
)

print("[5/6] Entraînement Logistic Regression (C=0.01, balanced)...")
model = LogisticRegression(C=0.01, max_iter=1000, random_state=42,
                           class_weight="balanced", solver="lbfgs")
model.fit(X_tr, y_tr)

train_acc = accuracy_score(y_tr, model.predict(X_tr))
test_acc  = accuracy_score(y_te, model.predict(X_te))
cv_scores = cross_val_score(model, X_sel, y, cv=5, scoring="accuracy")
test_auc  = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
cm        = confusion_matrix(y_te, model.predict(X_te))
report    = classification_report(y_te, model.predict(X_te),
                                  target_names=["Normal", "Abnormal"], output_dict=True)

print(f"\n{'='*50}")
print(f"  Train Accuracy : {train_acc:.4f}")
print(f"  Test  Accuracy : {test_acc:.4f}")
print(f"  CV-5  Accuracy : {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
print(f"  Test  AUC-ROC  : {test_auc:.4f}")
print(f"  Confusion matrix:\n{cm}")
print(f"{'='*50}\n")

print("[6/6] Sauvegarde des artefacts...")
with open(f"{MODEL_DIR}/scaler.pkl", "wb")   as f: pickle.dump(scaler, f)
with open(f"{MODEL_DIR}/selector.pkl", "wb") as f: pickle.dump(selector, f)
with open(f"{MODEL_DIR}/model.pkl", "wb")    as f: pickle.dump(model, f)

metadata = {
    "top_n": TOP_N,
    "top_genes": top_genes,
    "feature_names_all": feature_names.tolist(),
    "selected_feature_names": selected_features.tolist(),
    "metrics": {
        "train_accuracy": round(train_acc, 4),
        "test_accuracy":  round(test_acc, 4),
        "cv5_mean":       round(float(cv_scores.mean()), 4),
        "cv5_std":        round(float(cv_scores.std()), 4),
        "test_auc_roc":   round(test_auc, 4),
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    },
    "model_params": {
        "C": 0.01, "class_weight": "balanced", "solver": "lbfgs",
        "strategy": f"StandardScaler + SelectKBest(f_classif, k={TOP_N}) + LogisticRegression"
    }
}
with open(f"{MODEL_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"     ✓ scaler.pkl | selector.pkl | model.pkl | metadata.json")
print("✅ Entraînement terminé avec succès.")
