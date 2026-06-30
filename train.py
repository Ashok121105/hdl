"""
HDI Prediction System - Model Training Pipeline

Phases covered:
1. Data preprocessing (loading, cleaning, scaling, EDA)
2. Training multiple regression models
3. Model evaluation and selection
4. Saving best model and scaler with joblib
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from generate_dataset import main as generate_dataset
from utils import (
    BEST_MODEL_PATH,
    DATASET_PATH,
    FEATURE_COLUMNS,
    MODELS_DIR,
    SCALER_PATH,
    TARGET_COLUMN,
)

warnings.filterwarnings("ignore")

PLOTS_DIR = Path(__file__).resolve().parent / "screenshots"
METRICS_PATH = MODELS_DIR / "metrics.json"


def ensure_dataset() -> None:
    """Create dataset if it does not exist."""
    if not DATASET_PATH.exists():
        generate_dataset()


def load_data() -> pd.DataFrame:
    """Load CSV dataset."""
    ensure_dataset()
    return pd.read_csv(DATASET_PATH)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or drop missing values."""
    df = df.copy()
    numeric_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on feature columns."""
    subset = FEATURE_COLUMNS + [TARGET_COLUMN]
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Removed {removed} duplicate row(s).")
    return df


def detect_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
    """Cap outliers using IQR method for numeric features."""
    df = df.copy()
    for col in FEATURE_COLUMNS + [TARGET_COLUMN]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[col] = df[col].clip(lower, upper)
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """Full preprocessing pipeline."""
    print("\n=== Phase 1: Data Preprocessing ===\n")
    print("First 5 rows:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    print("\nMissing values:")
    print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().sum())
    print("\nDescriptive statistics:")
    print(df[FEATURE_COLUMNS + [TARGET_COLUMN]].describe())

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = detect_outliers_iqr(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=FEATURE_COLUMNS,
        index=X.index,
    )

    return X_scaled, y, scaler


def run_eda(df: pd.DataFrame) -> None:
    """Generate exploratory data analysis plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="viridis")

    # Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df[FEATURE_COLUMNS + [TARGET_COLUMN]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Histogram of HDI
    plt.figure(figsize=(8, 5))
    sns.histplot(df[TARGET_COLUMN], kde=True, bins=25, color="#2196F3")
    plt.title("HDI Score Distribution")
    plt.xlabel("HDI Score")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hdi_histogram.png", dpi=150)
    plt.close()

    # Scatter plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, col in zip(axes.ravel(), FEATURE_COLUMNS):
        sns.scatterplot(data=df, x=col, y=TARGET_COLUMN, ax=ax, alpha=0.7)
        ax.set_title(f"{col} vs HDI")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scatter_plots.png", dpi=150)
    plt.close()

    # Boxplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, col in zip(axes.ravel(), FEATURE_COLUMNS):
        sns.boxplot(y=df[col], ax=ax, color="#4CAF50")
        ax.set_title(f"{col} Boxplot")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "boxplots.png", dpi=150)
    plt.close()

    print(f"\nEDA plots saved to {PLOTS_DIR}")


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    return {
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
    }


def get_models() -> Dict[str, Any]:
    """Return dictionary of regression models to train."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=42, max_depth=12
        ),
        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=42, max_depth=10
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            objective="reg:squarederror",
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42
        ),
    }


def train_and_evaluate(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[Dict[str, Dict[str, float]], Any, np.ndarray, np.ndarray, np.ndarray, str]:
    """Train all models and return metrics, best model, and test data."""
    print("\n=== Phase 2: Machine Learning ===\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = get_models()
    results: Dict[str, Dict[str, float]] = {}
    trained_models: Dict[str, Any] = {}
    predictions: Dict[str, np.ndarray] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test.values, y_pred)
        results[name] = metrics
        trained_models[name] = model
        predictions[name] = y_pred
        print(
            f"{name}: R²={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}, "
            f"MSE={metrics['MSE']:.6f}, RMSE={metrics['RMSE']:.4f}"
        )

    best_name = max(results, key=lambda k: results[k]["R2"])
    best_model = trained_models[best_name]
    best_pred = predictions[best_name]

    print(f"\nBest model: {best_name} (R² = {results[best_name]['R2']:.4f})")

    return results, best_model, X_test.values, y_test.values, best_pred, best_name


def plot_model_diagnostics(
    y_test: np.ndarray, y_pred: np.ndarray, best_name: str, best_model: Any, feature_names: list
) -> None:
    """Create prediction vs actual, residual, and feature importance plots."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Prediction vs Actual
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors="k", linewidth=0.5)
    plt.plot([0, 1], [0, 1], "r--", label="Perfect Prediction")
    plt.xlabel("Actual HDI")
    plt.ylabel("Predicted HDI")
    plt.title(f"Prediction vs Actual ({best_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "prediction_vs_actual.png", dpi=150)
    plt.close()

    # Residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.7, edgecolors="k", linewidth=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted HDI")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residual_plot.png", dpi=150)
    plt.close()

    # Feature importance
    importance = None
    if hasattr(best_model, "feature_importances_"):
        importance = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        importance = np.abs(best_model.coef_)

    if importance is not None:
        plt.figure(figsize=(8, 5))
        order = np.argsort(importance)
        plt.barh(
            np.array(feature_names)[order],
            importance[order],
            color="#673AB7",
        )
        plt.title(f"Feature Importance ({best_name})")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "feature_importance.png", dpi=150)
        plt.close()


def save_artifacts(
    best_model: Any,
    scaler: StandardScaler,
    metrics: Dict[str, Dict[str, float]],
    best_name: str,
) -> None:
    """Save model, scaler, and metrics."""
    print("\n=== Phase 4: Model Saving ===\n")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    payload = {
        "best_model": best_name,
        "metrics": metrics,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2))

    print(f"Saved best model to {BEST_MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


def main() -> None:
    """Run the complete training pipeline."""
    df = load_data()
    run_eda(df)

    X_scaled, y, scaler = preprocess_data(df)
    results, best_model, X_test, y_test, y_pred, best_name = train_and_evaluate(
        X_scaled, y
    )

    plot_model_diagnostics(y_test, y_pred, best_name, best_model, FEATURE_COLUMNS)
    save_artifacts(best_model, scaler, results, best_name)

    print("\n=== Phase 3: Classification Preview ===\n")
    from utils import classify_hdi, get_recommendation

    sample_pred = float(y_pred[0])
    category = classify_hdi(sample_pred)
    print(f"Sample predicted HDI: {sample_pred:.3f}")
    print(f"Category: {category}")
    print(f"Recommendation: {get_recommendation(category)}")

    print("\nTraining pipeline completed successfully.")


if __name__ == "__main__":
    main()
