"""
HDI Prediction System - Streamlit Web Application

Professional UI for predicting Human Development Index scores
and classifying countries into development categories.
"""

from __future__ import annotations

import json
from datetime import datetime
from io import StringIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from predict import HDIPredictor
from utils import (
    BEST_MODEL_PATH,
    CATEGORY_COLORS,
    DATASET_PATH,
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODELS_DIR,
    PROJECT_ROOT,
    SCALER_PATH,
    TARGET_COLUMN,
    classify_hdi,
    get_recommendation,
)

PLOTS_DIR = PROJECT_ROOT / "screenshots"

# ---------------------------------------------------------------------------
# Page configuration & theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="HDI Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .hdi-badge {
        display: inline-block;
        padding: 0.6rem 1.8rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.2rem;
        color: white;
        margin: 1rem 0;
    }
    .result-box {
        background: rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.15);
        margin-top: 1rem;
    }
    .recommendation {
        background: rgba(102, 126, 234, 0.15);
        border-left: 4px solid #667eea;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
</style>
"""

st.markdown(DARK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_predictor() -> HDIPredictor:
    """Cache predictor instance."""
    return HDIPredictor()


@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load training dataset."""
    if DATASET_PATH.exists():
        return pd.read_csv(DATASET_PATH)
    return pd.DataFrame()


@st.cache_data
def load_metrics() -> dict:
    """Load model evaluation metrics."""
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def models_ready() -> bool:
    return BEST_MODEL_PATH.exists() and SCALER_PATH.exists()


def render_badge(category: str, color: str) -> None:
    st.markdown(
        f'<div class="hdi-badge" style="background-color:{color};">{category}</div>',
        unsafe_allow_html=True,
    )


def add_to_history(result: dict) -> None:
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **result["inputs"],
        "Predicted_HDI": result["hdi_score"],
        "Category": result["category"],
    }
    st.session_state.prediction_history.insert(0, entry)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home() -> None:
    st.markdown('<p class="main-header">Human Development Index Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Machine Learning-powered socio-economic development analysis</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-card"><h3>🎯 Objective</h3><p>Predict HDI scores and classify '
            'countries into development tiers using socio-economic indicators.</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><h3>📊 Features</h3><p>Life Expectancy, Mean & Expected '
            'Years of Schooling, GNI Per Capita</p></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><h3>🤖 Models</h3><p>Linear Regression, Random Forest, '
            'Decision Tree, XGBoost, Gradient Boosting</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Project Overview")
    st.markdown(
        """
        The **Human Development Index (HDI)** is a composite statistic measuring average achievement
        in three key dimensions: **health** (life expectancy), **education** (years of schooling),
        and **standard of living** (GNI per capita).

        This application uses supervised machine learning to predict HDI scores and automatically
        classify development levels:

        | Category | HDI Range |
        |----------|-----------|
        | Very High | ≥ 0.800 |
        | High | 0.700 – 0.799 |
        | Medium | 0.550 – 0.699 |
        | Low | < 0.550 |
        """
    )

    df = load_dataset()
    if not df.empty:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        st.caption(f"Total records: **{len(df)}** countries / profiles")


def page_predict() -> None:
    st.markdown('<p class="main-header">HDI Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Enter socio-economic indicators to predict HDI score</p>',
        unsafe_allow_html=True,
    )

    if not models_ready():
        st.error("Model artifacts not found. Run `python train.py` to train and save models.")
        if st.button("Generate Dataset Only"):
            from generate_dataset import main as gen
            gen()
            st.success("Dataset generated. Please run train.py.")
        return

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            life_exp = st.number_input(
                "Life Expectancy at Birth (years)",
                min_value=40.0,
                max_value=90.0,
                value=72.0,
                step=0.1,
                help="Average number of years a newborn is expected to live.",
            )
            mean_school = st.number_input(
                "Mean Years of Schooling",
                min_value=0.0,
                max_value=20.0,
                value=9.0,
                step=0.1,
                help="Average years of education received by people aged 25+.",
            )
        with col2:
            expected_school = st.number_input(
                "Expected Years of Schooling",
                min_value=0.0,
                max_value=25.0,
                value=12.0,
                step=0.1,
                help="Total years of schooling a child entering school can expect.",
            )
            gni = st.number_input(
                "GNI Per Capita (PPP, USD)",
                min_value=500.0,
                max_value=150000.0,
                value=14000.0,
                step=100.0,
                help="Gross National Income per capita adjusted for purchasing power.",
            )

        submitted = st.form_submit_button("🔮 Predict HDI", use_container_width=True)

    # Quick scenario buttons
    st.markdown("**Quick Examples:**")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    scenario = None
    with ex_col1:
        if st.button("Very High Example"):
            scenario = (82, 13, 18, 60000)
    with ex_col2:
        if st.button("Medium Example"):
            scenario = (72, 9, 12, 14000)
    with ex_col3:
        if st.button("Low Example"):
            scenario = (58, 5, 7, 2500)

    if scenario:
        life_exp, mean_school, expected_school, gni = scenario
        submitted = True

    if submitted:
        with st.spinner("Analyzing socio-economic indicators..."):
            try:
                predictor = load_predictor()
                result = predictor.predict(life_exp, mean_school, expected_school, gni)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        if not result.get("success"):
            st.warning(result.get("error", "Invalid input."))
            return

        add_to_history(result)

        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.metric("Predicted HDI Score", f"{result['hdi_score']:.3f}")
            render_badge(result["category"], result["color"])
        with res_col2:
            st.markdown("#### Input Summary")
            st.write(f"**Life Expectancy:** {life_exp} years")
            st.write(f"**Mean Years of Schooling:** {mean_school}")
            st.write(f"**Expected Years of Schooling:** {expected_school}")
            st.write(f"**GNI Per Capita:** ${gni:,.0f}")

        st.markdown(
            f'<div class="recommendation"><strong>Recommendation:</strong> '
            f'{result["recommendation"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Prediction history
    if st.session_state.prediction_history:
        st.markdown("---")
        st.subheader("📋 Prediction History")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

        csv_buffer = StringIO()
        history_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Predictions as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"hdi_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


def page_visualizations() -> None:
    st.markdown('<p class="main-header">Data Visualizations</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Exploratory analysis and model diagnostics</p>',
        unsafe_allow_html=True,
    )

    df = load_dataset()
    if df.empty:
        st.warning("Dataset not found. Run `python generate_dataset.py` or `python train.py`.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Distribution", "Scatter & Correlation", "Boxplots", "Model Diagnostics"]
    )

    with tab1:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[TARGET_COLUMN], kde=True, bins=25, color="#667eea", ax=ax)
        ax.set_title("HDI Score Distribution")
        ax.set_xlabel("HDI Score")
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(7, 5))
        corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, square=True)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)
        plt.close()

        fig2, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax_i, col in zip(axes.ravel(), FEATURE_COLUMNS):
            sns.scatterplot(data=df, x=col, y=TARGET_COLUMN, ax=ax_i, alpha=0.6)
            ax_i.set_title(f"{col} vs HDI")
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    with tab3:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax_i, col in zip(axes.ravel(), FEATURE_COLUMNS):
            sns.boxplot(y=df[col], ax=ax_i, color="#764ba2")
            ax_i.set_title(col)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab4:
        plot_files = {
            "Prediction vs Actual": PLOTS_DIR / "prediction_vs_actual.png",
            "Residual Plot": PLOTS_DIR / "residual_plot.png",
            "Feature Importance": PLOTS_DIR / "feature_importance.png",
        }
        if not models_ready():
            st.info("Train models first to view diagnostic plots.")
        else:
            for title, path in plot_files.items():
                if path.exists():
                    st.image(str(path), caption=title, use_container_width=True)
                else:
                    st.caption(f"{title}: run `python train.py` to generate.")


def page_metrics() -> None:
    st.markdown('<p class="main-header">Performance Metrics Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Model evaluation and comparison</p>',
        unsafe_allow_html=True,
    )

    metrics_data = load_metrics()
    if not metrics_data:
        st.warning("Metrics not found. Run `python train.py` to evaluate models.")
        return

    best = metrics_data.get("best_model", "Unknown")
    st.success(f"🏆 Best Model: **{best}**")

    metrics = metrics_data.get("metrics", {})
    if metrics:
        rows = []
        for name, vals in metrics.items():
            rows.append({"Model": name, **vals})
        metrics_df = pd.DataFrame(rows).sort_values("R2", ascending=False)
        st.dataframe(
            metrics_df.style.format(
                {"R2": "{:.4f}", "MAE": "{:.4f}", "MSE": "{:.6f}", "RMSE": "{:.4f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(metrics_df))
        width = 0.2
        ax.bar(x - width, metrics_df["R2"], width, label="R²", color="#667eea")
        ax.bar(x, metrics_df["MAE"], width, label="MAE", color="#764ba2")
        ax.bar(x + width, metrics_df["RMSE"], width, label="RMSE", color="#f093fb")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df["Model"], rotation=25, ha="right")
        ax.set_title("Model Performance Comparison")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.subheader("Classification Thresholds")
    thresholds = pd.DataFrame(
        {
            "Category": ["Very High", "High", "Medium", "Low"],
            "HDI Range": ["≥ 0.800", "0.700 – 0.799", "0.550 – 0.699", "< 0.550"],
            "Color": list(CATEGORY_COLORS.values()),
        }
    )
    st.dataframe(thresholds, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Sidebar & routing
# ---------------------------------------------------------------------------

def main() -> None:
    with st.sidebar:
        st.markdown("## 🌍 HDI Predictor")
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["Home", "Predict HDI", "Visualizations", "Metrics Dashboard"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("**Status**")
        st.write("✅ Dataset" if DATASET_PATH.exists() else "❌ Dataset")
        st.write("✅ Model" if BEST_MODEL_PATH.exists() else "❌ Model")
        st.write("✅ Scaler" if SCALER_PATH.exists() else "❌ Scaler")
        st.markdown("---")
        st.caption("Human Development Index Prediction System v1.0")

    pages = {
        "Home": page_home,
        "Predict HDI": page_predict,
        "Visualizations": page_visualizations,
        "Metrics Dashboard": page_metrics,
    }
    pages[page]()


if __name__ == "__main__":
    main()
