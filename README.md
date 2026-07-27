# 🌫️ AirIndex India — Air Quality Analysis & AQI Prediction

A complete, self-contained data science project: an **inbuilt synthetic dataset**, a full
**Jupyter analysis notebook**, a **model training pipeline**, and an **interactive Streamlit
dashboard** that predicts the Air Quality Index (AQI) for any Indian city, season, and pollutant
mix — all runnable offline in **VS Code**, no Kaggle account or internet download required.

---

## ✨ What's inside

| Piece | File | What it does |
|---|---|---|
| **Dataset generator** | `src/generate_dataset.py` | Builds a realistic `data/city_day.csv` from scratch (26 Indian cities, 2015–2020, daily pollutant readings) |
| **Analysis notebook** | `notebooks/aqi_analysis.ipynb` | Full EDA → cleaning → feature engineering → visualization → Random Forest modeling → save/load/predict, with outputs already rendered |
| **Training pipeline** | `src/train_model.py` | Cleans data, engineers features, trains & evaluates a `RandomForestRegressor`, saves the model + chart-ready summaries |
| **Interactive dashboard** | `app.py` | A polished Streamlit app: city/season analysis, trends & correlation, a live AQI predictor with a gauge readout, and a data explorer |

---

## 📦 Project structure

```
AQI_India_Project/
├── app.py                      # Streamlit dashboard (the main "UI")
├── requirements.txt
├── README.md
├── data/
│   └── city_day.csv            # generated dataset (created by generate_dataset.py)
├── models/                      # generated model artifacts (created by train_model.py)
│   ├── random_forest_aqi_model.pkl
│   ├── feature_columns.pkl
│   ├── cities.pkl
│   ├── metrics.pkl
│   └── summaries.pkl
├── notebooks/
│   └── aqi_analysis.ipynb      # full step-by-step analysis notebook
└── src/
    ├── generate_dataset.py     # builds the inbuilt dataset 
    └── train_model.py          # trains & saves the model
```

> **Note:** `data/city_day.csv` and everything in `models/` are already generated for you in this
> folder — you can open the notebook or run the dashboard immediately. The commands below are for
> when you want to regenerate them (e.g. with different random seeds, or after editing the
> generator).

---

## 🚀 Running it in VS Code

### 1. Open the folder
Open this entire `AQI_India_Project` folder in VS Code (`File → Open Folder…`).

### 2. Create a virtual environment (recommended)
Open a VS Code terminal (`` Ctrl+` ``) and run:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Re)generate the dataset — optional, already included

```bash
python src/generate_dataset.py
```

This writes `data/city_day.csv` — ~52,000 rows across 26 cities, complete with realistic
seasonal pollution spikes, city-level baselines, and a few percent of missing sensor readings
(just like the real CPCB dataset).

### 5. (Re)train the model — optional, already included

```bash
python src/train_model.py
```

This trains a `RandomForestRegressor` (R² ≈ 0.93, MAE ≈ 14 AQI points on held-out data) and saves
everything the dashboard needs into `models/`.

### 6. Explore the notebook

Open `notebooks/aqi_analysis.ipynb` in VS Code (with the Jupyter extension installed), pick your
Python kernel, and run the cells top to bottom — or just read it, since outputs and charts are
already saved in the file.

### 7. Launch the dashboard

```bash
streamlit run app.py
```

VS Code will show a clickable `http://localhost:8501` link in the terminal — open it in your
browser to use the full interactive dashboard.

---

## 🖥️ Dashboard tour

- **📊 City & Season** — top 10 most polluted cities, AQI distribution by season (boxplot), and
  the overall share of each AQI category.
- **📈 Trends & Correlation** — yearly national AQI trend, a pollutant correlation heatmap, and
  the model's feature-importance ranking.
- **🔮 Predict AQI** — pick a city, season, date, and pollutant levels with sliders, then get a
  live AQI prediction shown on a color-coded gauge (Good → Severe), powered by the trained Random
  Forest model.
- **🗂️ Data Explorer** — filter the raw daily records by city and date range.
- **ℹ️ About** — project background, pipeline summary, and notes on the dataset.

---

## 🧠 About the data

To make this project **100% self-contained and runnable anywhere**, `data/city_day.csv` is
generated synthetically to statistically resemble the real-world CPCB *"Air Quality Data in
India"* dataset (city rankings, seasonal cycles, pollutant correlations, missing-value patterns).
If you'd like to use the real dataset instead, just replace `data/city_day.csv` with the real
`city_day.csv` (same column names) — every script and the dashboard will work unchanged.

**Pollutants tracked:** PM2.5, PM10, NO, NO₂, NOx, NH₃, CO, SO₂, O₃, Benzene, Toluene, Xylene, AQI.

**AQI categories:** Good · Satisfactory · Moderate · Poor · Very Poor · Severe.

---

## 🛠️ Tech stack

- **Python**, **pandas**, **NumPy** — data wrangling & feature engineering
- **scikit-learn** — `RandomForestRegressor` for AQI prediction
- **Matplotlib / Seaborn** — static charts in the notebook
- **Plotly** — interactive charts in the dashboard
- **Streamlit** — the interactive web dashboard

---

## 📌 Ideas for extending this project

- Swap in the real CPCB dataset for production-grade analysis
- Add a time-series forecasting model (e.g. Prophet or an LSTM) for future AQI prediction
- Deploy the dashboard publicly with Streamlit Community Cloud
- Add city-level maps (e.g. with `pydeck` or `folium`) for a geographic view of pollution
