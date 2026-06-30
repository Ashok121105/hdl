# Human Development Index (HDI) Prediction Using Machine Learning

A complete end-to-end machine learning system that predicts Human Development Index (HDI) scores and classifies countries into development categories based on socio-economic indicators.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)

---

## Project Description

The Human Development Index (HDI) is a summary measure of average achievement in key dimensions of human development: a long and healthy life, knowledge, and a decent standard of living. This project builds a regression-based ML pipeline to predict HDI from four core indicators and maps predictions to development tiers for actionable policy insights.

## Objectives

- Predict HDI scores from socio-economic indicators
- Classify countries into **Very High**, **High**, **Medium**, and **Low** development categories
- Compare multiple regression algorithms and select the best performer
- Deploy an interactive Streamlit web application for real-time predictions
- Provide visual analytics and model performance dashboards

## Dataset Description

| Feature | Description |
|---------|-------------|
| **Life Expectancy at Birth** | Average years a newborn is expected to live |
| **Mean Years of Schooling** | Average years of education for adults aged 25+ |
| **Expected Years of Schooling** | Total years of schooling a child can expect |
| **GNI Per Capita (PPP)** | Gross National Income per capita in USD (PPP) |
| **HDI Score** (target) | Composite Human Development Index (0–1) |

**Classification Rules:**

| Category | HDI Range |
|----------|-----------|
| Very High | ≥ 0.800 |
| High | 0.700 – 0.799 |
| Medium | 0.550 – 0.699 |
| Low | < 0.550 |

## Technologies Used

- **Python 3.9+**
- **pandas & numpy** — Data manipulation
- **scikit-learn** — Preprocessing, models, metrics
- **XGBoost** — Gradient boosting regressor
- **matplotlib & seaborn** — Visualizations
- **Streamlit** — Web application
- **joblib** — Model persistence

## Project Structure

```
HDI_Prediction/
│
├── dataset/
│   └── hdi.csv
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   └── metrics.json
│
├── notebooks/
│   └── HDI_Prediction.ipynb
│
├── screenshots/
│   └── (EDA and model diagnostic plots)
│
├── app.py                 # Streamlit web application
├── train.py               # Training pipeline
├── predict.py             # Inference module
├── generate_dataset.py    # Dataset generator
├── utils.py               # Shared utilities
├── requirements.txt
└── README.md
```

## Installation

1. **Clone or download** the project:

```bash
cd HDI_Prediction
```

2. **Create a virtual environment** (recommended):

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Generate Dataset (optional — auto-runs during training)

```bash
python generate_dataset.py
```

### Step 2: Train Models

```bash
python train.py
```

This will:
- Preprocess data (missing values, duplicates, outliers, scaling)
- Generate EDA plots in `screenshots/`
- Train 5 regression models and compare metrics
- Save `models/best_model.pkl` and `models/scaler.pkl`

### Step 3: Run Predictions (CLI)

```bash
python predict.py
```

### Step 4: Launch Web Application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Example Scenarios

| Scenario | Life Exp. | Mean School | Expected School | GNI | Expected HDI | Category |
|----------|-----------|-------------|-----------------|-----|--------------|----------|
| 1 | 82 | 13 | 18 | 60,000 | ~0.93 | Very High |
| 2 | 72 | 9 | 12 | 14,000 | ~0.66 | Medium |
| 3 | 58 | 5 | 7 | 2,500 | ~0.47 | Low |

## Screenshots

After running `train.py`, the following plots are generated in `screenshots/`:

- `correlation_heatmap.png` — Feature correlation matrix
- `hdi_histogram.png` — HDI distribution
- `scatter_plots.png` — Feature vs HDI scatter plots
- `boxplots.png` — Feature boxplots
- `prediction_vs_actual.png` — Model accuracy visualization
- `residual_plot.png` — Residual analysis
- `feature_importance.png` — Feature importance chart

> Capture Streamlit UI screenshots after launching the app and save them to `screenshots/` for submission.

## Models Evaluated

| Model | Metrics |
|-------|---------|
| Linear Regression | R², MAE, MSE, RMSE |
| Random Forest Regressor | R², MAE, MSE, RMSE |
| Decision Tree Regressor | R², MAE, MSE, RMSE |
| XGBoost Regressor | R², MAE, MSE, RMSE |
| Gradient Boosting Regressor | R², MAE, MSE, RMSE |

The best model (highest R²) is saved automatically.

## App Features

- Dark-themed responsive UI
- Sidebar navigation (Home, Predict, Visualizations, Metrics)
- Input validation with helpful error messages
- Loading spinner during prediction
- Prediction history with CSV download
- Color-coded HDI category badges
- Policy recommendations per category
- Performance metrics dashboard

## Future Enhancements

- Integrate real UNDP HDI dataset via API
- Add country name lookup and geospatial map visualization
- Implement SHAP explainability for predictions
- Add time-series HDI trend forecasting
- Deploy to cloud (Streamlit Cloud, AWS, Azure)
- Multi-language support for recommendations
- User authentication and saved sessions

## License

This project is intended for academic and educational purposes.

## Author

Machine Learning Engineering Project — Human Development Index Prediction System
