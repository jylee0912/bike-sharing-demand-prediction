# Bike Sharing Demand Prediction

## Project Overview

This project develops a machine learning regression model to predict hourly bike-sharing rental demand based on temporal, seasonal, and weather-related features.

Using the UCI Bike Sharing Dataset, the project explores data patterns, performs feature engineering, compares multiple regression algorithms, and fine-tunes selected models to improve predictive performance.

An interactive Streamlit application is also developed to allow users to generate bike rental demand predictions based on customised input conditions.

## Objectives

- Explore temporal, seasonal, and weather-related patterns in bike rental demand.
- Perform data preprocessing and feature engineering.
- Develop and compare multiple machine learning regression models.
- Fine-tune XGBoost and LightGBM models.
- Evaluate model performance using MAE, RMSE, and R².
- Develop an interactive application for bike rental demand prediction.

## Dataset

**Source:** [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

- Dataset: Hourly bike rental records (`hour.csv`)
- Records: 17,379
- Period: 2011–2012
- Target variable: `cnt` (total bike rentals per hour)

The `casual` and `registered` columns were excluded from model training to prevent target leakage, as their sum directly determines the target variable `cnt`.

## Methodology

1. **Data Exploration:** Examine data distributions, missing values, duplicate records, temporal patterns, and relationships between features.
2. **Feature Engineering:** Create cyclical representations of hour, weekday, and month to capture periodic patterns.
3. **Data Preprocessing:** Apply standardisation to numerical features and one-hot encoding to categorical features.
4. **Data Splitting:** Use chronological 80:20 training and testing split.
5. **Model Development:** Train and compare multiple regression algorithms.
6. **Hyperparameter Tuning:** Apply RandomizedSearchCV with TimeSeriesSplit for XGBoost and LightGBM.
7. **Model Evaluation:** Compare models using MAE, RMSE, and R².
8. **Deployment:** Develop an interactive Streamlit application using the selected model.

## Models

- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- Tuned XGBoost
- Tuned LightGBM

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost Tuned | 43.726 | 66.231 | 0.910 |
| XGBoost | 44.946 | 68.084 | 0.905 |
| LightGBM | 45.402 | 68.806 | 0.903 |
| LightGBM Tuned | 49.186 | 72.672 | 0.891 |
| Random Forest | 51.188 | 76.811 | 0.879 |
| Decision Tree | 58.831 | 91.371 | 0.828 |
| Ridge Regression | 120.738 | 164.411 | 0.444 |
| Linear Regression | 120.741 | 164.409 | 0.444 |

**Selected Model: Tuned XGBoost**

- MAE: 43.726
- RMSE: 66.231
- R²: 0.910

The tuned XGBoost model achieved the best overall performance among the evaluated models on the held-out test dataset.

## Streamlit Application

The project includes an interactive application that allows users to:

- Select a date and hour.
- Specify weather conditions and environmental variables.
- Generate predicted hourly bike rental demand.
- View the estimated demand level.
- Explore model performance metrics and feature importance.

## Project Structure

```text
bike+sharing+dataset/
├── data/
├── notebooks/
├── reports/
├── saved_models/
├── app.py
├── README.md
├── DATASET_README.txt
└── requirements.txt
```

## Installation and Usage

Clone the repository:

```bash
git clone <your-repository-url>
cd bike+sharing+dataset
```

Install the required packages:

```bash
python3 -m pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python3 -m streamlit run app.py
```

## Tools and Technologies

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- XGBoost
- LightGBM
- Streamlit
- Jupyter Notebook
- Joblib

## Dataset Reference

Fanaee-T, H., & Gama, J. (2013). Event labeling combining ensemble detectors and background knowledge. *Progress in Artificial Intelligence*. https://doi.org/10.1007/s13748-013-0040-3
