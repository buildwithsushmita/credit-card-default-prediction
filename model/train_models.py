"""
Train all 5 classification models on UCI Credit Card Default dataset.
Saves trained models as .pkl and test data as test_data.csv
"""

import os
import urllib.request
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# ---- Download dataset ----
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
DATA_FILE = "default_credit.xls"

if not os.path.exists(DATA_FILE):
    print("Downloading Credit Card Default dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print("Done.")
else:
    print("Dataset already downloaded.")

# ---- Load and preprocess ----
df = pd.read_excel(DATA_FILE, header=1)

# drop the ID column
df = df.drop(columns=["ID"])

# target column
target_col = "default payment next month"
X = df.drop(columns=[target_col])
y = df[target_col]

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ---- Train/Test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# save test data for streamlit app
test_df = X_test.copy()
test_df[target_col] = y_test.values
test_df.to_csv("../test_data.csv", index=False)
print(f"\nSaved test_data.csv ({len(test_df)} rows)")

# ---- Scale features (needed for LR and KNN) ----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# save scaler
joblib.dump(scaler, "scaler.pkl")

# ---- Define models ----
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
}

# models that need scaled data
needs_scaling = {"Logistic Regression", "KNN"}

# ---- Train and evaluate ----
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    if name in needs_scaling:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    # compute all 6 metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
    }
    results[name] = metrics

    # save model
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, filename)
    print(f"  Saved {filename}")
    print(f"  Accuracy: {metrics['Accuracy']:.4f} | AUC: {metrics['AUC']:.4f} | "
          f"F1: {metrics['F1']:.4f} | MCC: {metrics['MCC']:.4f}")

# ---- Print comparison table ----
print("\n" + "=" * 80)
print("MODEL COMPARISON TABLE")
print("=" * 80)
print(f"{'Model':<22} {'Accuracy':<10} {'AUC':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'MCC':<10}")
print("-" * 80)
for name, metrics in results.items():
    print(f"{name:<22} {metrics['Accuracy']:<10.4f} {metrics['AUC']:<10.4f} "
          f"{metrics['Precision']:<10.4f} {metrics['Recall']:<10.4f} "
          f"{metrics['F1']:<10.4f} {metrics['MCC']:<10.4f}")

# save results for the app to use
results_df = pd.DataFrame(results).T
results_df.index.name = "Model"
results_df.to_csv("../model_results.csv")
print("\nSaved model_results.csv")
print("\nAll done! Models trained and saved.")
