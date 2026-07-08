"""
generate_dataset.py
--------------------
Generates a realistic, fully synthetic "city_day.csv" dataset that mirrors the
structure of the real-world CPCB "Air Quality Data in India" dataset
(City, Date, pollutant concentrations, AQI, AQI_Bucket) — WITHOUT needing an
internet connection or a Kaggle account.

This makes the whole project self-contained: anyone can clone the folder,
run this one script, and immediately have data/city_day.csv ready to use
in the notebook, the training script, and the dashboard.

Run:
    python src/generate_dataset.py
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. Cities & their relative pollution "baseline" (mirrors real-world ranking)
#    1.0 = average, >1 = more polluted, <1 = cleaner
# ---------------------------------------------------------------------------
CITY_PROFILE = {
    "Delhi":              1.85,
    "Patna":               1.65,
    "Lucknow":             1.55,
    "Gurugram":            1.50,
    "Jaipur":              1.30,
    "Talcher":             1.35,
    "Brajrajnagar":        1.30,
    "Jorapokhar":          1.25,
    "Amritsar":            1.20,
    "Chandigarh":          1.05,
    "Kolkata":             1.15,
    "Ahmedabad":           1.10,
    "Bhopal":              0.95,
    "Guwahati":            1.00,
    "Mumbai":              0.90,
    "Hyderabad":           0.85,
    "Bengaluru":           0.70,
    "Chennai":             0.75,
    "Coimbatore":          0.65,
    "Amaravati":           0.80,
    "Visakhapatnam":       0.75,
    "Ernakulam":           0.55,
    "Kochi":               0.55,
    "Thiruvananthapuram":  0.50,
    "Shillong":            0.45,
    "Aizawl":              0.40,
}

CITIES = list(CITY_PROFILE.keys())

START_DATE = "2015-01-01"
END_DATE = "2020-06-30"

# Baseline mean concentrations at profile == 1.0 (roughly realistic, µg/m3 or mg/m3)
BASELINES = {
    "PM2.5":   85.0,
    "PM10":    150.0,
    "NO":      15.0,
    "NO2":     35.0,
    "NOx":     40.0,
    "NH3":     20.0,
    "CO":      1.2,
    "SO2":     15.0,
    "O3":      35.0,
    "Benzene": 3.0,
    "Toluene": 8.0,
    "Xylene":  2.0,
}

# Seasonal multipliers by month (winter = worse due to inversion & crop burning,
# monsoon = better due to rain washout) — applied mainly to particulate matter & NOx family
SEASON_MULT = {
    1: 1.55, 2: 1.35, 3: 1.10, 4: 0.95, 5: 0.90,
    6: 0.70, 7: 0.60, 8: 0.60, 9: 0.75,
    10: 1.15, 11: 1.50, 12: 1.60,
}

AQI_BINS = [0, 50, 100, 200, 300, 400, 500]
AQI_LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def compute_aqi_from_pm25_pm10(pm25, pm10):
    """A simplified sub-index style AQI approximation driven by PM2.5 / PM10,
    which are the dominant pollutants in the real Indian AQI calculation."""
    # simplified linear breakpoint mapping for PM2.5 (India CPCB-like breakpoints)
    def sub_index_pm25(c):
        bps = [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200),
               (90, 120, 200, 300), (120, 250, 300, 400), (250, 500, 400, 500)]
        for lo, hi, ilo, ihi in bps:
            if lo <= c <= hi:
                return ilo + (c - lo) * (ihi - ilo) / (hi - lo)
        return 500

    def sub_index_pm10(c):
        bps = [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200),
               (250, 350, 200, 300), (350, 430, 300, 400), (430, 600, 400, 500)]
        for lo, hi, ilo, ihi in bps:
            if lo <= c <= hi:
                return ilo + (c - lo) * (ihi - ilo) / (hi - lo)
        return 500

    i25 = np.array([sub_index_pm25(c) for c in pm25])
    i10 = np.array([sub_index_pm10(c) for c in pm10])
    return np.maximum(i25, i10)


def generate():
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    rows = []

    for city in CITIES:
        profile = CITY_PROFILE[city]
        n = len(dates)
        months = np.asarray(dates.month)
        years = np.asarray(dates.year)
        season_factor = np.array([SEASON_MULT[m] for m in months])

        # slow multi-year improvement/worsening trend + random walk noise for realism
        year_trend = 1.0 - 0.01 * (years - years.min())  # slight improvement over years
        noise = RNG.normal(1.0, 0.18, size=n).clip(0.5, 1.8)

        factor = np.asarray(profile * season_factor * year_trend * noise, dtype=float)

        pm25 = np.clip(BASELINES["PM2.5"] * factor + RNG.normal(0, 8, n), 5, 500)
        pm10 = np.clip(pm25 * RNG.uniform(1.4, 1.9, n) + RNG.normal(0, 10, n), 10, 600)
        no = np.clip(BASELINES["NO"] * factor * RNG.uniform(0.7, 1.3, n), 0.1, 100)
        no2 = np.clip(BASELINES["NO2"] * factor * RNG.uniform(0.7, 1.3, n), 1, 150)
        nox = np.clip(no + no2 * RNG.uniform(0.6, 1.1, n), 1, 200)
        nh3 = np.clip(BASELINES["NH3"] * factor * RNG.uniform(0.6, 1.4, n), 1, 120)
        co = np.clip(BASELINES["CO"] * factor * RNG.uniform(0.6, 1.5, n), 0.05, 15)
        so2 = np.clip(BASELINES["SO2"] * factor * RNG.uniform(0.6, 1.4, n), 1, 100)
        o3 = np.clip(BASELINES["O3"] * RNG.uniform(0.6, 1.5, n) * (2 - factor.clip(0, 1.5) / 1.5),
                     5, 180)
        benzene = np.clip(BASELINES["Benzene"] * factor * RNG.uniform(0.4, 1.6, n), 0, 30)
        toluene = np.clip(BASELINES["Toluene"] * factor * RNG.uniform(0.4, 1.6, n), 0, 60)
        xylene = np.clip(BASELINES["Xylene"] * factor * RNG.uniform(0.4, 1.6, n), 0, 20)

        aqi = compute_aqi_from_pm25_pm10(pm25, pm10)
        aqi += RNG.normal(0, 6, n)
        aqi = np.clip(aqi, 15, 500)

        # Inject some realistic missingness (sensors go offline sometimes)
        missing_mask_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
                              "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI"]
        data = {
            "City": city,
            "Date": dates,
            "PM2.5": pm25.round(2),
            "PM10": pm10.round(2),
            "NO": no.round(2),
            "NO2": no2.round(2),
            "NOx": nox.round(2),
            "NH3": nh3.round(2),
            "CO": co.round(2),
            "SO2": so2.round(2),
            "O3": o3.round(2),
            "Benzene": benzene.round(2),
            "Toluene": toluene.round(2),
            "Xylene": xylene.round(2),
            "AQI": aqi.round(0),
        }
        city_df = pd.DataFrame(data)

        # randomly null out ~6% of pollutant/AQI cells to mimic real sensor gaps
        for col in missing_mask_cols:
            mask = RNG.random(n) < 0.06
            city_df.loc[mask, col] = np.nan

        city_df["AQI_Bucket"] = pd.cut(city_df["AQI"], bins=AQI_BINS, labels=AQI_LABELS)
        # also null bucket wherever AQI itself is null, and a few extra random spots
        city_df.loc[city_df["AQI"].isna(), "AQI_Bucket"] = np.nan

        rows.append(city_df)

    full_df = pd.concat(rows, ignore_index=True)
    full_df["Date"] = full_df["Date"].dt.strftime("%Y-%m-%d")

    col_order = ["City", "Date", "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO",
                 "SO2", "O3", "Benzene", "Toluene", "Xylene", "AQI", "AQI_Bucket"]
    full_df = full_df[col_order]
    return full_df


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "city_day.csv")

    df = generate()
    df.to_csv(out_path, index=False)
    print(f"Generated synthetic dataset with {len(df):,} rows and {len(df.columns)} columns.")
    print(f"Saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
