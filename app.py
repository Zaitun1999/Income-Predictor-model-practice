import os
import joblib
import pandas as pd
import streamlit as st
from typing import cast

PROCESSED_PATH =r"C:\Users\Personal\PyCharmMiscProject\PythonProject\data\processed\train_clean.csv"
MODEL_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\outputs\income_model.pkl"
METRICS_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\outputs\model_metrics.txt"
FIGURES_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\outputs\figures"
TARGET = "INCOME-IS->50K"

st.set_page_config(page_title="Income Predictor", page_icon="💰", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return cast(pd.DataFrame,pd.read_csv(PROCESSED_PATH)) #type:ignore


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


df = load_data()
model = load_model()
feature_names = list(model.feature_names_in_)
inputs_cols = [c for c in df.columns if c != TARGET]

st.title("💰 Income Prediction (>50K)")
st.caption("Random Forest model trained on the cleaned adult income dataset.")

tab_predict, tab_explore, tab_perf = st.tabs(["🔮 Predict", "📊 Explore data", "📈 Model performance"])

# ---------------- PREDICT ----------------
with tab_predict:
    st.subheader("Enter a person's details")
    cols = st.columns(3)
    values = {}
    for i, col in enumerate(inputs_cols):
        with cols[i % 3]:
            if pd.api.types.is_numeric_dtype(df[col]):
                values[col] = st.number_input(
                    col,
                    min_value=int(df[col].min()),
                    max_value=int(df[col].max()),
                    value=int(df[col].median()),
                    step=1,
                )
            else:
                options = sorted(df[col].dropna().unique().tolist())
                values[col] = st.selectbox(col, options)

    if st.button("Predict", type="primary"):
        # Build one row matching the training columns (numbers + dummy columns)
        row = pd.DataFrame(0, index=[0], columns=feature_names)
        for col, val in values.items():
            if col in row.columns:
                row.at[0, col] = val
            else:
                dummy = f"{col}_{val}"
                if dummy in row.columns:
                    row.at[0, dummy] = 1 # first category was dropped -> all zeros

        pred = int(model.predict(row)[0])
        prob = float(model.predict_proba(row)[0][1])

        if pred == 1:
            st.success(f"Predicted income: **>50K** (probability {prob:.1%})")
        else:
            st.info(f"Predicted income: **≤50K** (probability of >50K: {prob:.1%})")
        st.progress(prob)

# ---------------- EXPLORE ----------------
with tab_explore:
    st.subheader("Dataset overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("% earning >50K", f"{df[TARGET].mean():.1%}")

    st.dataframe(df.head(50), width=True)

    col = st.selectbox("Pick a column to visualize", inputs_cols)
    if pd.api.types.is_numeric_dtype(df[col]):
        st.write(f"Distribution of {col}")
        counts = df[col].value_counts(bins=20, sort=False)
        counts.index = counts.index.astype(str)
        st.bar_chart(counts)
    else:
        st.write(f"{col}: share earning >50K")
        st.bar_chart(df.groupby(col)[TARGET].mean().sort_values())

# ---------------- PERFORMANCE ----------------
with tab_perf:
    st.subheader("Model metrics")
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            st.code(f.read())

    left, right = st.columns(2)
    cm = os.path.join(FIGURES_PATH, "confusion_matrix.png")
    fi = os.path.join(FIGURES_PATH, "feature_importance.png")
    if os.path.exists(cm):
        left.image(cm, caption="Confusion matrix")
    if os.path.exists(fi):
        right.image(fi, caption="Top 10 feature importances")