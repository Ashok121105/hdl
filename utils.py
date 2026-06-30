"""
Shared utilities for the HDI Prediction System.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "hdi.csv"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

FEATURE_COLUMNS = [
    "Life_Expectancy",
    "Mean_Years_Schooling",
    "Expected_Years_Schooling",
    "GNI_Per_Capita",
]
TARGET_COLUMN = "HDI_Score"

# Valid input ranges for validation
INPUT_RANGES = {
    "Life_Expectancy": (40.0, 90.0),
    "Mean_Years_Schooling": (0.0, 20.0),
    "Expected_Years_Schooling": (0.0, 25.0),
    "GNI_Per_Capita": (500.0, 150000.0),
}

HDI_THRESHOLDS = {
    "Very High": 0.800,
    "High": 0.700,
    "Medium": 0.550,
    "Low": 0.0,
}

CATEGORY_COLORS = {
    "Very High": "#00C853",
    "High": "#2196F3",
    "Medium": "#FF9800",
    "Low": "#F44336",
}

CATEGORY_RECOMMENDATIONS = {
    "Very High": (
        "Maintain strong healthcare, education, and economic policies. "
        "Focus on innovation, sustainability, and reducing inequality."
    ),
    "High": (
        "Strengthen social development policies. Invest in quality education, "
        "healthcare access, and inclusive economic growth."
    ),
    "Medium": (
        "Improve education and healthcare. Increase public investment in schools, "
        "hospitals, and social safety nets to accelerate development."
    ),
    "Low": (
        "Prioritize healthcare, education, and economic development. "
        "Focus on basic infrastructure, literacy programs, and income generation."
    ),
}


def classify_hdi(hdi_score: float) -> str:
    """Classify an HDI score into one of four development categories."""
    score = float(np.clip(hdi_score, 0.0, 1.0))
    if score >= HDI_THRESHOLDS["Very High"]:
        return "Very High"
    if score >= HDI_THRESHOLDS["High"]:
        return "High"
    if score >= HDI_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


def get_recommendation(category: str) -> str:
    """Return a human-readable recommendation for an HDI category."""
    return CATEGORY_RECOMMENDATIONS.get(category, "Review development indicators and policy priorities.")


def validate_inputs(
    life_expectancy: float,
    mean_years_schooling: float,
    expected_years_schooling: float,
    gni_per_capita: float,
) -> Tuple[bool, str]:
    """Validate user inputs against expected socio-economic ranges."""
    checks = {
        "Life Expectancy": (life_expectancy, INPUT_RANGES["Life_Expectancy"]),
        "Mean Years of Schooling": (mean_years_schooling, INPUT_RANGES["Mean_Years_Schooling"]),
        "Expected Years of Schooling": (expected_years_schooling, INPUT_RANGES["Expected_Years_Schooling"]),
        "GNI Per Capita": (gni_per_capita, INPUT_RANGES["GNI_Per_Capita"]),
    }

    for label, (value, (low, high)) in checks.items():
        if value is None or np.isnan(value):
            return False, f"{label} is required."
        if value < low or value > high:
            return False, f"{label} must be between {low} and {high}."

    if expected_years_schooling < mean_years_schooling:
        return False, "Expected Years of Schooling should not be less than Mean Years of Schooling."

    return True, ""
