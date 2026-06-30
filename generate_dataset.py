"""
Generate a realistic HDI dataset for training and demonstration.
Run once to create dataset/hdi.csv if the file does not exist.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import DATASET_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def _health_index(life_expectancy: float) -> float:
    """Approximate UNDP health index from life expectancy."""
    le_min, le_max = 20.0, 85.0
    return (life_expectancy - le_min) / (le_max - le_min)


def _education_index(mean_years: float, expected_years: float) -> float:
    """Approximate UNDP education index."""
    mys = mean_years / 15.0
    eys = expected_years / 18.0
    return np.sqrt(max(mys, 0.0) * max(eys, 0.0))


def _income_index(gni: float) -> float:
    """Approximate UNDP income index using log transformation."""
    gni_min, gni_max = 100.0, 75000.0
    if gni <= gni_min:
        return 0.0
    return (np.log(gni) - np.log(gni_min)) / (np.log(gni_max) - np.log(gni_min))


def compute_hdi(
    life_expectancy: float,
    mean_years: float,
    expected_years: float,
    gni: float,
    noise: float = 0.0,
) -> float:
    """Compute HDI as geometric mean of dimension indices."""
    hi = _health_index(life_expectancy)
    ei = _education_index(mean_years, expected_years)
    ii = _income_index(gni)
    hdi = (hi * ei * ii) ** (1 / 3)
    return float(np.clip(hdi + noise, 0.0, 1.0))


COUNTRY_PROFILES = [
    ("Norway", 83.2, 13.0, 18.2, 68000),
    ("Switzerland", 83.9, 13.4, 16.5, 66000),
    ("Ireland", 82.3, 12.7, 19.6, 62000),
    ("Germany", 81.0, 14.1, 17.0, 54000),
    ("Sweden", 83.0, 12.4, 19.4, 52000),
    ("Australia", 83.4, 13.1, 22.0, 55000),
    ("Netherlands", 82.2, 12.4, 18.0, 53000),
    ("Denmark", 81.6, 12.7, 19.1, 58000),
    ("Finland", 81.9, 12.9, 19.5, 49000),
    ("Canada", 82.3, 13.5, 16.4, 48000),
    ("United States", 78.9, 13.4, 16.5, 63000),
    ("United Kingdom", 81.3, 13.2, 17.4, 46000),
    ("Japan", 84.5, 12.9, 15.2, 42000),
    ("South Korea", 83.6, 12.2, 16.5, 43000),
    ("France", 82.7, 11.6, 16.0, 43000),
    ("Spain", 83.6, 10.0, 18.0, 38000),
    ("Italy", 83.0, 10.4, 16.5, 38000),
    ("Singapore", 83.6, 11.4, 16.3, 88000),
    ("New Zealand", 82.3, 12.5, 20.3, 42000),
    ("Belgium", 81.6, 12.1, 19.8, 47000),
    ("Israel", 82.8, 13.0, 16.0, 42000),
    ("Austria", 81.7, 12.3, 16.0, 51000),
    ("UAE", 78.7, 12.1, 15.1, 67000),
    ("Saudi Arabia", 75.7, 9.9, 15.3, 53000),
    ("Qatar", 80.2, 9.8, 14.0, 88000),
    ("Kuwait", 75.5, 7.2, 14.0, 48000),
    ("Malaysia", 76.2, 10.3, 13.7, 28000),
    ("China", 77.3, 7.6, 14.2, 16100),
    ("Brazil", 75.9, 8.0, 15.4, 15800),
    ("Mexico", 75.1, 9.2, 14.5, 20000),
    ("Argentina", 76.7, 10.2, 18.5, 22000),
    ("Chile", 80.2, 10.4, 16.5, 25000),
    ("Colombia", 77.3, 8.5, 14.4, 15000),
    ("Peru", 76.7, 9.4, 14.6, 13000),
    ("Ecuador", 77.0, 8.8, 14.7, 12000),
    ("Turkey", 76.0, 8.0, 16.4, 28000),
    ("Russia", 73.2, 12.2, 15.5, 26000),
    ("Poland", 78.0, 12.1, 16.4, 32000),
    ("Czech Republic", 79.4, 12.8, 16.9, 38000),
    ("Hungary", 76.9, 12.0, 15.0, 31000),
    ("Romania", 76.3, 11.1, 14.9, 28000),
    ("Bulgaria", 75.0, 11.2, 14.0, 22000),
    ("Ukraine", 72.0, 11.8, 15.0, 12000),
    ("South Africa", 64.1, 10.1, 13.4, 13000),
    ("Egypt", 72.0, 7.2, 13.5, 12000),
    ("Morocco", 76.7, 5.6, 13.4, 8000),
    ("Algeria", 76.9, 8.0, 14.4, 14000),
    ("Tunisia", 76.7, 7.2, 15.2, 11000),
    ("Nigeria", 55.2, 6.2, 10.2, 5300),
    ("Kenya", 66.7, 6.5, 11.5, 4500),
    ("Ethiopia", 66.6, 4.0, 10.8, 2500),
    ("Ghana", 64.1, 7.4, 12.0, 5500),
    ("Tanzania", 66.2, 6.0, 9.8, 2800),
    ("Uganda", 63.7, 6.2, 10.6, 2200),
    ("India", 70.4, 6.7, 12.6, 6600),
    ("Pakistan", 67.3, 5.2, 10.5, 5400),
    ("Bangladesh", 72.4, 6.2, 11.2, 5600),
    ("Nepal", 70.5, 4.9, 12.8, 3700),
    ("Sri Lanka", 77.0, 10.6, 13.3, 13000),
    ("Vietnam", 75.4, 8.1, 12.9, 11000),
    ("Indonesia", 71.7, 8.2, 13.0, 11800),
    ("Philippines", 71.2, 9.1, 13.5, 9500),
    ("Thailand", 77.7, 8.0, 15.1, 18000),
    ("Myanmar", 67.1, 5.0, 11.2, 4500),
    ("Cambodia", 70.0, 4.8, 11.2, 4300),
    ("Laos", 68.7, 5.0, 11.0, 7800),
    ("Afghanistan", 64.8, 3.9, 10.0, 2000),
    ("Iraq", 71.0, 6.8, 11.2, 11000),
    ("Iran", 76.7, 9.4, 14.8, 16000),
    ("Syria", 72.1, 5.1, 9.1, 3700),
    ("Yemen", 66.1, 3.8, 9.9, 2800),
    ("Haiti", 64.0, 5.0, 10.0, 3200),
    ("Honduras", 75.3, 6.9, 12.8, 5800),
    ("Guatemala", 74.3, 6.5, 11.5, 9200),
    ("Nicaragua", 75.0, 6.8, 12.3, 5800),
    ("Bolivia", 71.5, 9.2, 14.7, 8500),
    ("Paraguay", 74.3, 8.4, 13.0, 12000),
    ("Venezuela", 72.1, 9.1, 14.0, 7500),
    ("Cuba", 78.9, 11.8, 14.0, 9600),
    ("Jamaica", 74.5, 9.4, 13.3, 9000),
    ("Dominican Republic", 74.6, 8.1, 14.0, 19000),
    ("Mongolia", 70.2, 9.4, 14.9, 12000),
    ("Kazakhstan", 73.6, 11.7, 15.7, 26000),
    ("Uzbekistan", 71.7, 11.8, 12.0, 8500),
    ("Georgia", 74.2, 13.1, 14.8, 14000),
    ("Armenia", 75.0, 11.3, 13.0, 14000),
    ("Azerbaijan", 73.0, 10.3, 13.0, 14000),
    ("Belarus", 74.8, 12.0, 15.5, 21000),
    ("Serbia", 75.9, 10.6, 14.0, 18000),
    ("Croatia", 78.6, 11.8, 15.0, 28000),
    ("Slovakia", 77.8, 12.3, 15.0, 32000),
    ("Slovenia", 81.2, 12.2, 17.6, 36000),
    ("Estonia", 78.8, 13.1, 16.5, 36000),
    ("Latvia", 75.0, 12.3, 16.0, 30000),
    ("Lithuania", 75.9, 12.4, 16.4, 34000),
    ("Iceland", 83.1, 12.2, 19.7, 52000),
    ("Luxembourg", 82.6, 12.4, 14.9, 72000),
    ("Portugal", 81.8, 9.1, 16.8, 33000),
    ("Greece", 81.4, 10.8, 20.0, 29000),
    ("Cyprus", 81.2, 11.8, 16.4, 34000),
    ("Malta", 82.6, 11.8, 15.8, 39000),
    ("Bahrain", 77.3, 9.6, 15.4, 42000),
    ("Oman", 78.0, 9.7, 14.0, 38000),
    ("Jordan", 74.5, 10.0, 13.6, 10000),
    ("Lebanon", 75.0, 8.7, 14.0, 14000),
    ("Libya", 72.9, 7.3, 12.6, 16000),
    ("Sudan", 65.3, 3.5, 8.0, 4200),
    ("Senegal", 67.9, 3.5, 10.5, 3500),
    ("Mali", 59.3, 2.4, 8.0, 2200),
    ("Burkina Faso", 61.6, 1.5, 10.0, 2100),
    ("Niger", 62.4, 2.0, 8.0, 1200),
    ("Chad", 54.2, 2.3, 8.0, 1600),
    ("Central African Republic", 53.9, 4.3, 7.6, 1100),
    ("Democratic Republic of Congo", 60.7, 6.8, 11.1, 1100),
    ("Mozambique", 61.2, 3.5, 10.0, 1300),
    ("Zambia", 63.9, 7.0, 11.0, 3500),
    ("Zimbabwe", 61.2, 8.5, 11.2, 2800),
    ("Angola", 61.6, 5.4, 11.0, 6500),
    ("Cameroon", 60.3, 6.0, 12.0, 3800),
    ("Ivory Coast", 58.6, 5.0, 10.0, 5500),
    ("Madagascar", 67.0, 6.2, 10.3, 1500),
    ("Rwanda", 69.0, 4.0, 11.0, 2200),
    ("Malawi", 65.3, 4.5, 11.0, 1600),
    ("Botswana", 69.6, 9.5, 13.1, 16000),
    ("Namibia", 64.0, 6.5, 13.0, 9500),
    ("Lesotho", 55.6, 6.5, 11.3, 3200),
    ("Swaziland", 60.2, 6.7, 11.8, 7800),
    ("Papua New Guinea", 65.4, 4.3, 10.0, 4200),
    ("Fiji", 67.4, 10.3, 15.4, 11000),
    ("Samoa", 73.4, 10.6, 13.2, 6000),
    ("Tonga", 71.0, 10.4, 14.0, 6500),
    ("Scenario Very High", 82.0, 13.0, 18.0, 60000),
    ("Scenario Medium", 72.0, 9.0, 12.0, 14000),
    ("Scenario Low", 58.0, 5.0, 7.0, 2500),
]


def generate_dataset() -> pd.DataFrame:
    """Build dataset from country profiles with small random variation."""
    rng = np.random.default_rng(42)
    rows = []

    for country, le, mys, eys, gni in COUNTRY_PROFILES:
        noise = rng.normal(0, 0.008)
        hdi = compute_hdi(le, mys, eys, gni, noise)
        rows.append(
            {
                "Country": country,
                "Life_Expectancy": le,
                "Mean_Years_Schooling": mys,
                "Expected_Years_Schooling": eys,
                "GNI_Per_Capita": gni,
                TARGET_COLUMN: round(hdi, 3),
            }
        )

    # Add synthetic variations for richer training data
    for _ in range(80):
        base = rows[rng.integers(0, len(rows))]
        le = np.clip(base["Life_Expectancy"] + rng.normal(0, 2.5), 45, 88)
        mys = np.clip(base["Mean_Years_Schooling"] + rng.normal(0, 0.8), 1, 16)
        eys = np.clip(base["Expected_Years_Schooling"] + rng.normal(0, 1.0), mys, 22)
        gni = np.clip(base["GNI_Per_Capita"] * rng.lognormal(0, 0.15), 800, 90000)
        noise = rng.normal(0, 0.012)
        hdi = compute_hdi(le, mys, eys, gni, noise)
        rows.append(
            {
                "Country": f"Synthetic_{len(rows)}",
                "Life_Expectancy": round(le, 1),
                "Mean_Years_Schooling": round(mys, 1),
                "Expected_Years_Schooling": round(eys, 1),
                "GNI_Per_Capita": round(gni, 0),
                TARGET_COLUMN: round(hdi, 3),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATASET_PATH, index=False)
    print(f"Dataset saved to {DATASET_PATH} ({len(df)} rows)")


if __name__ == "__main__":
    main()
