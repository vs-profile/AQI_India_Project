"""
train_model.py
--------------
Loads data/city_day.csv, cleans + feature-engineers it exactly like the
original analysis notebook, trains a RandomForestRegressor to predict AQI,
evaluates it, and saves everything the dashboard (app.py) needs:

    models/random_forest_aqi_model.pkl   -> trained model
    models/feature_columns.pkl           -> exact column order used for X
    models/cities.pkl                    -> sorted list of city names
    models/metrics.pkl                   -> MAE / R2 + a few chart-ready summaries

Run:
    python src/train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "city_day.csv")
MODEL_DIR = os.path.join(HERE, "..", "models")

BINS = [0, 50, 100, 200, 300, 400, 500]
LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
AQI_MAP = {label: i + 1 for i, label in enumerate(LABELS)}

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
}

POLLUTANT_COLS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
                  "O3", "Benzene", "Toluene", "Xylene", "AQI"]

NUMERICAL_FEATURES = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
                       "O3", "Benzene", "Toluene", "Xylene", "Year", "Month", "Day"]


def load_and_clean(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Fill pollutant/AQI NaNs with column means (same approach as the notebook)
    df[POLLUTANT_COLS] = df[POLLUTANT_COLS].fillna(df[POLLUTANT_COLS].mean().round(2))

    # Rebuild AQI_Bucket from AQI wherever it's missing
    df["AQI_Bucket"] = df["AQI_Bucket"].fillna(pd.cut(df["AQI"], bins=BINS, labels=LABELS))
    df["AQI_Level"] = df["AQI_Bucket"].map(AQI_MAP)

    # Feature engineering
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Season"] = df["Month"].map(SEASON_MAP)

    return df


def build_features(df):
    y = df["AQI"]
    city_encoded = pd.get_dummies(df["City"], prefix="City")
    season_encoded = pd.get_dummies(df["Season"], prefix="Season")
    X = pd.concat([df[NUMERICAL_FEATURES], city_encoded, season_encoded], axis=1)
    return X, y


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading & cleaning dataset...")
    df = load_and_clean(DATA_PATH)

    print("Building model features...")
    X, y = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  X_train: {X_train.shape}   X_test: {X_test.shape}")

    print("Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(
        n_estimators=120, max_depth=14, min_samples_leaf=3, n_jobs=-1, random_state=42
    )
    rf_model.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  R-squared (R2) Score:      {r2:.3f}")

    # Feature importances (for the dashboard "what drives AQI" chart)
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    top_importances = importances.sort_values(ascending=False).head(12)

    # A few chart-ready summaries so the dashboard doesn't need to reprocess
    # the full raw CSV every time it starts
    summaries = {
        "season_boxplot": df[["Season", "AQI"]].copy(),
        "top_cities": df.groupby("City")["AQI"].mean().sort_values(ascending=False),
        "yearly_trend": df.groupby("Year")["AQI"].mean(),
        "correlation": df[POLLUTANT_COLS].corr(),
        "bucket_counts": df["AQI_Bucket"].value_counts(),
        "feature_importance": top_importances,
        "actual_vs_pred": pd.DataFrame({"actual": y_test.values, "predicted": y_pred}),
    }

    metrics = {"mae": mae, "r2": r2, "n_rows": len(df), "n_features": X.shape[1]}

    print("Saving model artifacts to /models ...")
    with open(os.path.join(MODEL_DIR, "random_forest_aqi_model.pkl"), "wb") as f:
        pickle.dump(rf_model, f)
    with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "wb") as f:
        pickle.dump(list(X.columns), f)
    with open(os.path.join(MODEL_DIR, "cities.pkl"), "wb") as f:
        pickle.dump(sorted(df["City"].unique().tolist()), f)
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "wb") as f:
        pickle.dump(metrics, f)
    with open(os.path.join(MODEL_DIR, "summaries.pkl"), "wb") as f:
        pickle.dump(summaries, f)

    print("Done. Model + supporting artifacts saved in the models/ folder.")


if __name__ == "__main__":
    main()
