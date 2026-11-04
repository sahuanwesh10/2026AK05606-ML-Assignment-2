"""Streamlit app - ML Assignment 2 (Anwesh Sahu, BITS ID 2026AK05606)
Loads the 5 saved classifiers and evaluates them on an uploaded TEST csv.
No retraining is done here."""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report)

st.set_page_config(page_title="Classifier Evaluation Lab", page_icon="📊", layout="wide")

MODEL_DIR = "model"

@st.cache_resource
def load_meta():
    return joblib.load(os.path.join(MODEL_DIR, "meta.joblib"))

@st.cache_resource
def load_model(file_name):
    return joblib.load(os.path.join(MODEL_DIR, file_name))

def compute_metrics(model, X_t, y_t):
    y_pred = model.predict(X_t)
    y_proba = model.predict_proba(X_t)
    present = np.unique(y_t)
    if len(present) == 2 and y_proba.shape[1] == 2:
        auc, avg = roc_auc_score(y_t, y_proba[:, 1]), "binary"
    elif y_proba.shape[1] == 2:
        auc, avg = np.nan, "binary"
    else:
        try:
            auc = roc_auc_score(y_t, y_proba, multi_class="ovr", average="macro",
                                labels=list(range(y_proba.shape[1])))
        except ValueError:
            auc = np.nan
        avg = "weighted"
    return {
        "Accuracy": accuracy_score(y_t, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_t, y_pred, average=avg, zero_division=0),
        "Recall": recall_score(y_t, y_pred, average=avg, zero_division=0),
        "F1": f1_score(y_t, y_pred, average=avg, zero_division=0),
        "MCC": matthews_corrcoef(y_t, y_pred),
    }, y_pred

def main():
    st.title("📊 Classification Model Evaluation Lab")
    st.caption("Anwesh Sahu | BITS ID: 2026AK05606 | ML Assignment 2")

    try:
        meta = load_meta()
    except Exception as e:
        st.error(f"Could not load saved models from the '{MODEL_DIR}' folder: {e}")
        st.stop()

    target_col = meta["target_col"]
    features = meta["feature_columns"]
    class_names = meta["class_names"]
    le = meta["label_encoder"]
    model_files = meta["model_files"]

    # ---------------- Sidebar ----------------
    st.sidebar.header("⚙️ Controls")
    uploaded = st.sidebar.file_uploader("1. Upload TEST data (CSV)", type=["csv"])
    model_name = st.sidebar.selectbox("2. Select a model", list(model_files.keys()))
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Target column:** `{target_col}`")
    st.sidebar.markdown(f"**Classes:** {', '.join(map(str, class_names))}")
    st.sidebar.info("Upload only the test file (`test_data.csv`) - the free tier has limited capacity. "
                    "Models are pre-trained and are NOT retrained on your upload.")

    if uploaded is None:
        st.info("👈 Upload `test_data.csv` from the sidebar to begin.")
        with st.expander("Columns the uploaded CSV must contain"):
            st.write(features + [target_col])
        st.stop()

    # ---------------- Validate ----------------
    data = pd.read_csv(uploaded)
    data.columns = [c.strip() for c in data.columns]
    missing = [c for c in features + [target_col] if c not in data.columns]
    if missing:
        st.error(f"Uploaded file is missing required column(s): {missing}")
        st.stop()
    data = data.dropna(subset=features + [target_col])

    unknown = set(data[target_col].astype(str)) - set(map(str, class_names))
    if unknown:
        st.error(f"Target column contains unknown labels: {sorted(unknown)}")
        st.stop()

    X_t = data[features]
    y_t = le.transform(data[target_col].astype(str)) if data[target_col].dtype == object \
        else le.transform(data[target_col])

    st.subheader("Uploaded data preview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(data)); c2.metric("Features", len(features)); c3.metric("Classes", len(class_names))
    st.dataframe(data.head(10), width="stretch")

    tab1, tab2, tab3 = st.tabs(["🔎 Selected model", "🏆 Compare all models", "📝 Predictions"])

    # ---------------- Selected model ----------------
    model = load_model(model_files[model_name])
    metrics, y_pred = compute_metrics(model, X_t, y_t)

    with tab1:
        st.subheader(f"Evaluation metrics - {model_name}")
        cols = st.columns(6)
        for col, (k, v) in zip(cols, metrics.items()):
            col.metric(k, "N/A" if pd.isna(v) else f"{v:.4f}")

        left, right = st.columns(2)
        with left:
            st.markdown("**Confusion matrix**")
            cm = confusion_matrix(y_t, y_pred, labels=list(range(len(class_names))))
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="viridis",
                        xticklabels=class_names, yticklabels=class_names, ax=ax)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            plt.xticks(rotation=45, ha="right"); plt.tight_layout()
            st.pyplot(fig)
        with right:
            st.markdown("**Classification report**")
            rep = classification_report(y_t, y_pred, labels=list(range(len(class_names))),
                                        target_names=[str(c) for c in class_names],
                                        output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

    # ---------------- Compare all ----------------
    with tab2:
        st.subheader("All five models on the uploaded test data")
        rows = {}
        for name, f in model_files.items():
            rows[name], _ = compute_metrics(load_model(f), X_t, y_t)
        comp = pd.DataFrame(rows).T
        comp.index.name = "ML Model Name"
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#2e7d32"),
                     width="stretch")
        st.bar_chart(comp[["Accuracy", "F1", "MCC"]])
        st.success(f"🏆 Best model by MCC on this data: **{comp['MCC'].idxmax()}**")

    # ---------------- Predictions ----------------
    with tab3:
        out = data.copy()
        out["Predicted"] = le.inverse_transform(y_pred)
        out["Correct"] = out["Predicted"].astype(str) == out[target_col].astype(str)
        st.dataframe(out.head(200), width="stretch")
        st.download_button("⬇️ Download predictions (CSV)", out.to_csv(index=False).encode(),
                           file_name=f"predictions_{model_name.replace(' ', '_')}.csv")

main()
