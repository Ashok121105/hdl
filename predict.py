"""
HDI Prediction System - Inference Module

Loads trained model and scaler to predict HDI scores and categories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from utils import (
    BEST_MODEL_PATH,
    CATEGORY_COLORS,
    FEATURE_COLUMNS,
    SCALER_PATH,
    classify_hdi,
    get_recommendation,
    validate_inputs,
)


class HDIPredictor:
    """Wrapper for HDI regression and classification inference."""

    def __init__(
        self,
        model_path: Path = BEST_MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
    ) -> None:
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model and scaler from disk."""
        if not self.model_path.exists() or not self.scaler_path.exists():
            raise FileNotFoundError(
                "Model artifacts not found. Run `python train.py` first."
            )
        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

    def predict(
        self,
        life_expectancy: float,
        mean_years_schooling: float,
        expected_years_schooling: float,
        gni_per_capita: float,
    ) -> Dict[str, object]:
        """
        Predict HDI score and development category from socio-economic indicators.

        Returns:
            Dictionary with HDI score, category, color, recommendation, and validity.
        """
        is_valid, message = validate_inputs(
            life_expectancy,
            mean_years_schooling,
            expected_years_schooling,
            gni_per_capita,
        )
        if not is_valid:
            return {
                "success": False,
                "error": message,
            }

        features = pd.DataFrame(
            [
                {
                    "Life_Expectancy": life_expectancy,
                    "Mean_Years_Schooling": mean_years_schooling,
                    "Expected_Years_Schooling": expected_years_schooling,
                    "GNI_Per_Capita": gni_per_capita,
                }
            ]
        )

        scaled = self.scaler.transform(features[FEATURE_COLUMNS])
        raw_score = float(self.model.predict(scaled)[0])
        hdi_score = float(np.clip(raw_score, 0.0, 1.0))
        category = classify_hdi(hdi_score)

        return {
            "success": True,
            "hdi_score": round(hdi_score, 3),
            "category": category,
            "color": CATEGORY_COLORS[category],
            "recommendation": get_recommendation(category),
            "inputs": {
                "Life_Expectancy": life_expectancy,
                "Mean_Years_Schooling": mean_years_schooling,
                "Expected_Years_Schooling": expected_years_schooling,
                "GNI_Per_Capita": gni_per_capita,
            },
        }


def predict_hdi(
    life_expectancy: float,
    mean_years_schooling: float,
    expected_years_schooling: float,
    gni_per_capita: float,
    predictor: Optional[HDIPredictor] = None,
) -> Dict[str, object]:
    """Convenience function for single prediction."""
    engine = predictor or HDIPredictor()
    return engine.predict(
        life_expectancy,
        mean_years_schooling,
        expected_years_schooling,
        gni_per_capita,
    )


if __name__ == "__main__":
    scenarios = [
        ("Scenario 1 (Very High)", 82, 13, 18, 60000),
        ("Scenario 2 (Medium)", 72, 9, 12, 14000),
        ("Scenario 3 (Low)", 58, 5, 7, 2500),
    ]

    try:
        predictor = HDIPredictor()
        for name, le, mys, eys, gni in scenarios:
            result = predictor.predict(le, mys, eys, gni)
            print(f"\n{name}")
            print(f"  HDI Score: {result['hdi_score']}")
            print(f"  Category: {result['category']}")
            print(f"  Recommendation: {result['recommendation']}")
    except FileNotFoundError as exc:
        print(exc)
