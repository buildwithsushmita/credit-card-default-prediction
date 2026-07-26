import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---- Paths ----
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_TEST = os.path.join(BASE_DIR, "test_data.csv")
TARGET_COL = "default payment next month"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

NEEDS_SCALING = {"Logistic Regression", "KNN"}


@st.cache_resource
def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, MODEL_FILES[name]))


@st.cache_resource
def load_scaler():
    return joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))


# ========== SIDEBAR ==========
with st.sidebar:
    st.title("ML Assignment 2")
    st.caption("Machine Learning - BITS Pilani (WILP)")
    st.divider()

    st.markdown("**Sushmita Mehta**")
    st.markdown("Roll No: `2025ac05031`")
    st.markdown("M.Tech AI ML")

    st.divider()

    st.subheader("Upload Test Data")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success(f"{uploaded_file.name} ({len(data):,} rows)")
    else:
        data = pd.read_csv(DEFAULT_TEST)
        st.info(f"Default test data ({len(data):,} rows)")

    st.divider()

    st.subheader("Select Model")
    selected_model = st.selectbox(
        "Classification model:",
        list(MODEL_FILES.keys()),
        index=4,
    )

# ========== VALIDATION ==========
if TARGET_COL not in data.columns:
    st.error(f"CSV must contain target column: '{TARGET_COL}'")
    st.stop()

X = data.drop(columns=[TARGET_COL])
y = data[TARGET_COL]

# ========== PREDICT ==========
model = load_model(selected_model)
scaler = load_scaler()

if selected_model in NEEDS_SCALING:
    X_input = scaler.transform(X)
else:
    X_input = X.values

y_pred = model.predict(X_input)
y_proba = model.predict_proba(X_input)[:, 1]

# ========== HEADER ==========
st.title("💳 Credit Card Default Prediction")
st.caption(
    "5 ML classification models on the UCI Default of Credit Card Clients dataset (30,000 instances, 23 features)"
)

# ========== TABS ==========
tab_model, tab_compare, tab_data = st.tabs([
    "Model Results", "All Models Comparison", "Dataset Info"
])

# ==================== TAB 1 ====================
with tab_model:
    st.subheader(f"{selected_model}")

    # Metrics in 3 columns x 2 rows
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
    c2.metric("AUC Score", f"{roc_auc_score(y, y_proba):.4f}")
    c3.metric("Precision", f"{precision_score(y, y_pred):.4f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Recall", f"{recall_score(y, y_pred):.4f}")
    c5.metric("F1 Score", f"{f1_score(y, y_pred):.4f}")
    c6.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")

    st.divider()

    # Confusion Matrix and ROC side by side
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Confusion Matrix**")
        cm = confusion_matrix(y, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3.2))
        sns.heatmap(
            cm, annot=True, fmt=",d", cmap="Blues", ax=ax_cm,
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            linewidths=0.5, annot_kws={"size": 12},
        )
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

    with col_right:
        st.markdown("**ROC Curve**")
        fpr, tpr, _ = roc_curve(y, y_proba)
        fig_roc, ax_roc = plt.subplots(figsize=(4, 3.2))
        ax_roc.plot(fpr, tpr, color="#2563eb", lw=2,
                    label=f"AUC = {roc_auc_score(y, y_proba):.3f}")
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.legend(loc="lower right")
        ax_roc.grid(alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig_roc)
        plt.close()

    # Classification Report
    st.markdown("**Classification Report**")
    report = classification_report(y, y_pred, target_names=["No Default", "Default"])
    st.code(report, language="text")

# ==================== TAB 2 ====================
with tab_compare:
    st.subheader("All Models - Evaluation Metrics")

    # Run all 5 models on the current data
    all_results = {}
    for model_name in MODEL_FILES:
        m = load_model(model_name)
        if model_name in NEEDS_SCALING:
            X_eval = scaler.transform(X)
        else:
            X_eval = X.values
        preds = m.predict(X_eval)
        probas = m.predict_proba(X_eval)[:, 1]
        all_results[model_name] = {
            "Accuracy": accuracy_score(y, preds),
            "AUC": roc_auc_score(y, probas),
            "Precision": precision_score(y, preds),
            "Recall": recall_score(y, preds),
            "F1": f1_score(y, preds),
            "MCC": matthews_corrcoef(y, preds),
        }

    results_df = pd.DataFrame(all_results).T
    results_df.index.name = "Model"

    # Display as a clean markdown table (readable in any theme)
    table_md = "| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |\n"
    table_md += "|-------|----------|-----|-----------|--------|-----|-----|\n"
    for model_name, row in results_df.iterrows():
        table_md += f"| {model_name} | {row['Accuracy']:.4f} | {row['AUC']:.4f} | {row['Precision']:.4f} | {row['Recall']:.4f} | {row['F1']:.4f} | {row['MCC']:.4f} |\n"
    st.markdown(table_md)

    # Find the winner
    winner = results_df["F1"].idxmax()
    st.success(f"**Best Model: {winner}** (F1 = {results_df.loc[winner, 'F1']:.4f}, AUC = {results_df.loc[winner, 'AUC']:.4f})")

    st.divider()

    # Grouped bar chart (vertical)
    st.markdown("**Performance Comparison**")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results_df))
    width = 0.13
    metrics_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    colors = ["#3b82f6", "#f97316", "#8b5cf6", "#06b6d4", "#10b981", "#ec4899"]

    for i, (col, color) in enumerate(zip(metrics_cols, colors)):
        offset = (i - 2.5) * width
        bars = ax_bar.bar(x + offset, results_df[col], width, label=col, color=color)
        for bar in bars:
            height = bar.get_height()
            ax_bar.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=7)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(results_df.index, fontsize=10)
    ax_bar.set_ylim(0, 1.1)
    ax_bar.set_ylabel("Score", fontsize=10)
    ax_bar.legend(fontsize=9, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax_bar.grid(axis="y", alpha=0.2)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close()

    st.divider()

    # Observations
    st.subheader("Observations")

    st.markdown(f"""
| Model | Observation |
|-------|-------------|
| Logistic Regression | Accuracy {results_df.loc['Logistic Regression','Accuracy']:.1%}, Recall {results_df.loc['Logistic Regression','Recall']:.1%}. Conservative, rarely predicts default. Good baseline. |
| Decision Tree | Accuracy {results_df.loc['Decision Tree','Accuracy']:.1%}, AUC {results_df.loc['Decision Tree','AUC']:.3f}. Better recall, interpretable, depth-limited. |
| KNN | Accuracy {results_df.loc['KNN','Accuracy']:.1%}. Struggles with 23 dimensions. More false positives. |
| Naive Bayes | Recall {results_df.loc['Naive Bayes','Recall']:.1%} but accuracy only {results_df.loc['Naive Bayes','Accuracy']:.1%}. Independence assumption fails. |
| Random Forest (Ensemble) | Accuracy {results_df.loc['Random Forest (Ensemble)','Accuracy']:.1%}, AUC {results_df.loc['Random Forest (Ensemble)','AUC']:.3f}. Best overall, captures complex interactions. |
| **Overall Winner** | **{winner}** with best F1 ({results_df.loc[winner,'F1']:.4f}) and MCC ({results_df.loc[winner,'MCC']:.4f}). |
""")

# ==================== TAB 3 ====================
with tab_data:
    st.subheader("Default of Credit Card Clients")
    st.markdown(
        "[UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) "
        "| 30,000 instances | 23 features | Binary classification"
    )

    st.divider()

    col_info, col_pie = st.columns([3, 2])

    with col_info:
        st.markdown("**Features**")
        features = [
            ("LIMIT_BAL", "Credit amount (NT dollar)"),
            ("SEX", "1=male, 2=female"),
            ("EDUCATION", "1=grad, 2=univ, 3=HS, 4=other"),
            ("MARRIAGE", "1=married, 2=single, 3=other"),
            ("AGE", "Age in years"),
            ("PAY_0 - PAY_6", "Repayment status (-1=paid, 1-9=delay months)"),
            ("BILL_AMT1-6", "Bill statement amounts"),
            ("PAY_AMT1-6", "Previous payment amounts"),
        ]
        st.table(pd.DataFrame(features, columns=["Feature", "Description"]))

    with col_pie:
        st.markdown("**Target Distribution**")
        fig_p, ax_p = plt.subplots(figsize=(3.5, 3))
        counts = y.value_counts()
        ax_p.pie(
            counts, labels=["No Default", "Default"],
            autopct="%1.1f%%", colors=["#3b82f6", "#ef4444"],
            startangle=90, textprops={"fontsize": 9},
        )
        plt.tight_layout()
        st.pyplot(fig_p)
        plt.close()

    st.divider()
    with st.expander("Preview test data"):
        st.dataframe(data.head(20), use_container_width=True)
