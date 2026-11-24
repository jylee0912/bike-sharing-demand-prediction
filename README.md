# 🚲 Bike Sharing Demand Prediction

An end-to-end machine learning project that predicts hourly bike-sharing rental demand using regression models. This project covers data exploration, feature engineering, model development, hyperparameter tuning, model evaluation, and deployment through an interactive Streamlit application.

**[Live Demo](https://bike-sharing-demand-prediction-2wbcrnbrdf85e7nawv2dr3.streamlit.app/)** | **[GitHub Repository](https://github.com/jylee0912/bike-sharing-demand-prediction)**

---

## 1. Project Overview

Accurately estimating bike-sharing demand helps operators understand rental patterns and make better decisions about resource allocation.

This project uses historical bike rental records and temporal, seasonal, and weather-related features to predict the total number of bike rentals per hour.

Multiple regression models are developed and evaluated. Hyperparameter tuning is performed on XGBoost and LightGBM to investigate potential performance improvements.

An interactive Streamlit application is also deployed, allowing users to generate predictions by specifying date, time, and weather conditions.

## 2. Project Objectives

- Explore temporal, seasonal, and weather-related patterns in bike rental demand.
- Perform data preprocessing and feature engineering.
- Develop and compare multiple machine learning regression models.
- Apply hyperparameter tuning to selected models.
- Evaluate predictive performance using MAE, RMSE, and R².
- Deploy the selected model through an interactive web application.

## 3. Dataset

**Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

| Attribute | Description |
|---|---|
| Dataset | Hourly bike rental records (`hour.csv`) |
| Records | 17,379 |
| Period | 2011–2012 |
| Target variable | `cnt` |
| Target description | Total bike rentals per hour |

The `casual` and `registered` columns were excluded from model training to prevent target leakage, as their sum directly determines the target variable `cnt`.

## 4. Methodology

### 4.1 Exploratory Data Analysis

- Examined data distributions and descriptive statistics.
- Checked missing values and duplicate records.
- Investigated hourly, weekly, seasonal, and weather-related rental patterns.
- Analysed relationships between numerical features and rental demand.

### 4.2 Feature Engineering

Created cyclical features using sine and cosine transformations to represent recurring time patterns:

- Hour of day
- Day of week
- Month of year

These transformations help capture the cyclical nature of time-related variables.

### 4.3 Data Preprocessing

- StandardScaler for numerical features.
- OneHotEncoder for categorical features.
- Excluded `instant`, `casual`, and `registered` from model training.

### 4.4 Train-Test Split

Applied a chronological 80:20 train-test split to preserve the temporal order of observations.

TimeSeriesSplit was also used during hyperparameter tuning to avoid randomly mixing past and future observations across validation folds.

### 4.5 Model Development

The following models were trained and evaluated:

- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

RandomizedSearchCV with TimeSeriesSplit was used to tune XGBoost and LightGBM hyperparameters.

### 4.6 Model Evaluation

Models were evaluated using:

- **MAE (Mean Absolute Error):** Measures the average absolute prediction error.
- **RMSE (Root Mean Squared Error):** Penalises larger prediction errors more heavily.
- **R² Score:** Measures the proportion of target variance explained by the model.

Additional diagnostic analysis included actual-versus-predicted plots, residual analysis, and feature importance.

## 5. Model Performance

The following results were obtained on the held-out test dataset:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| **XGBoost Tuned** | **43.726** | **66.231** | **0.910** |
| XGBoost | 44.946 | 68.084 | 0.905 |
| LightGBM | 45.402 | 68.806 | 0.903 |
| LightGBM Tuned | 49.186 | 72.672 | 0.891 |
| Random Forest | 51.188 | 76.811 | 0.879 |
| Decision Tree | 58.831 | 91.371 | 0.828 |
| Ridge Regression | 120.738 | 164.411 | 0.444 |
| Linear Regression | 120.741 | 164.409 | 0.444 |

### Selected Model: XGBoost Tuned

The tuned XGBoost model achieved the best overall performance among the evaluated models:

- **MAE:** 43.726
- **RMSE:** 66.231
- **R²:** 0.910

The model explains approximately 91% of the variation in bike rental demand within the held-out test dataset.

Hyperparameter tuning reduced XGBoost's test MAE by approximately 2.71% compared with its baseline configuration.

LightGBM tuning did not improve its held-out test performance, highlighting that hyperparameter optimisation does not necessarily translate into better generalisation.

## 6. Model Interpretation

Feature importance analysis was performed using:

- Built-in XGBoost feature importance.
- Permutation importance.

The analysis identified hour-related features (`hour_sin` and `hour_cos`) and `workingday` as particularly influential features for the model's predictions.

Feature importance indicates the model's reliance on input features; it does not establish causal relationships.

## 7. Interactive Streamlit Application

The trained XGBoost model is deployed through an interactive Streamlit application.

**[Launch Bike Sharing Demand Predictor](https://bike-sharing-demand-prediction-2wbcrnbrdf85e7nawv2dr3.streamlit.app/)**

Application features:

- Select a date and hour.
- Specify public holiday status.
- Select weather conditions.
- Enter temperature, feels-like temperature, humidity, and wind speed.
- Generate predicted hourly bike rental demand.
- View estimated demand levels.
- Explore model evaluation metrics.
- Visualise the top 15 model feature importance scores.

## 8. Project Structure

```text
bike-sharing-demand-prediction/
├── .devcontainer/
├── data/
│   ├── day.csv
│   └── hour.csv
├── notebooks/
│   └── Regression.ipynb
├── reports/
│   └── bike_sharing_profiling_report.html
├── saved_models/
│   └── best_model/
│       └── xgboost_tuned_model.pkl
├── .gitignore
├── DATASET_README.txt
├── README.md
├── app.py
└── requirements.txt
```

## 9. Installation and Usage

### Clone the repository

```bash
git clone https://github.com/jylee0912/bike-sharing-demand-prediction.git

cd bike-sharing-demand-prediction
```

### Install dependencies

Python 3.11 is used for the deployed Streamlit application.

```bash
python3 -m pip install -r requirements.txt
```

### Run the Streamlit application

```bash
python3 -m streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## 10. Tools and Technologies

| Category | Tools |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Gradient Boosting | XGBoost, LightGBM |
| Model Persistence | Joblib |
| Interactive Application | Streamlit |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git, GitHub |
| Deployment | Streamlit Community Cloud |

## 11. Limitations and Future Improvements

- Incorporate historical rental demand and lag-based features to support more explicit time-series forecasting.
- Explore rolling-origin evaluation to assess model stability across different time periods.
- Investigate additional feature selection and feature ablation experiments.
- Further optimise model performance and computational efficiency.
- Explore model explainability techniques such as SHAP.

## 12. References

Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*. https://doi.org/10.1007/s13748-013-0040-3

Dataset: UCI Bike Sharing Dataset. https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset