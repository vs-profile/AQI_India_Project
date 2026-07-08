"""
app.py — AirIndex India Dashboard
---------------------------------
An interactive Streamlit dashboard for the "Air Quality in India" project.

Run:
    streamlit run app.py

Tabs:
    Overview            -> headline KPIs + AQI scale legend
    City & Season       -> most polluted cities, seasonal boxplot, data explorer
    Trends & Correlation-> yearly AQI trend, pollutant correlation heatmap, feature importance
    Predict AQI         -> interactive form -> RandomForest prediction with a gauge readout
    About               -> project / dataset / model notes
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AirIndex India",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "city_day.csv")
MODEL_DIR = os.path.join(HERE, "models")

BINS = [0, 50, 100, 200, 300, 400, 500]
LABELS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

AQI_COLORS = {
    "Good": "#4CAF50",
    "Satisfactory": "#9ACD32",
    "Moderate": "#FFC107",
    "Poor": "#FF7043",
    "Very Poor": "#E53935",
    "Severe": "#7B1E3A",
}

SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Summer", 4: "Summer", 5: "Summer",
    6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
    10: "Post-Monsoon", 11: "Post-Monsoon",
}

# ---------------------------------------------------------------------------
# Theming — dusk-sky / smog palette, Space Grotesk + Inter + JetBrains Mono
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

:root{
  --bg-deep:#0E141B;
  --bg-panel:#161F29;
  --bg-panel-2:#1C2732;
  --text-primary:#EAF1F5;
  --text-muted:#8CA0AC;
  --accent-sky:#4FA8D8;
  --hairline:#26323D;
}

html, body, [class*="css"]  { font-family:'Inter', sans-serif; }
.stApp{ background:var(--bg-deep); color:var(--text-primary); }

h1,h2,h3, .hero-title { font-family:'Space Grotesk', sans-serif; letter-spacing:-0.01em; }

section[data-testid="stSidebar"]{ background:var(--bg-panel); }

.block-container{ padding-top:1.6rem; max-width:1200px; }

/* Hero */
.hero-wrap{
  background:linear-gradient(135deg, #101823 0%, #0E141B 55%, #12212B 100%);
  border:1px solid var(--hairline);
  border-radius:18px;
  padding:2.2rem 2.4rem;
  margin-bottom:1.4rem;
  position:relative;
  overflow:hidden;
}
.hero-eyebrow{
  font-family:'JetBrains Mono', monospace;
  color:var(--accent-sky);
  font-size:0.78rem;
  letter-spacing:0.12em;
  text-transform:uppercase;
  margin-bottom:0.5rem;
}
.hero-title{ font-size:2.3rem; font-weight:700; margin:0 0 0.5rem 0; color:var(--text-primary); }
.hero-sub{ color:var(--text-muted); font-size:1.02rem; max-width:640px; line-height:1.55; }

/* KPI cards */
.kpi-card{
  background:var(--bg-panel);
  border:1px solid var(--hairline);
  border-radius:14px;
  padding:1.1rem 1.3rem;
  height:100%;
}
.kpi-label{ font-family:'JetBrains Mono', monospace; font-size:0.72rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.08em;}
.kpi-value{ font-family:'Space Grotesk', sans-serif; font-size:1.8rem; font-weight:700; color:var(--text-primary); margin-top:0.15rem;}

/* AQI legend chips */
.aqi-chip-row{ display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.8rem;}
.aqi-chip{
  font-family:'JetBrains Mono', monospace;
  font-size:0.74rem;
  padding:0.35rem 0.75rem;
  border-radius:999px;
  color:#0E141B;
  font-weight:600;
}

/* Panels */
.panel{
  background:var(--bg-panel);
  border:1px solid var(--hairline);
  border-radius:16px;
  padding:1.4rem 1.5rem;
  margin-bottom:1.2rem;
}
.panel-title{ font-family:'Space Grotesk', sans-serif; font-weight:600; font-size:1.15rem; margin-bottom:0.2rem;}
.panel-sub{ color:var(--text-muted); font-size:0.88rem; margin-bottom:0.9rem;}

/* Tabs */
button[data-baseweb="tab"]{ font-family:'Space Grotesk', sans-serif; font-weight:600; }

/* Gauge number */
.gauge-value{ font-family:'JetBrains Mono', monospace; }

hr{ border-color:var(--hairline); }

/* Metric override */
[data-testid="stMetricValue"]{ font-family:'Space Grotesk', sans-serif; }

footer {visibility:hidden;}

header[data-testid="stHeader"]{
  background:var(--bg-deep) !important;
}

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#EAF1F5", family="Inter"),
        xaxis=dict(gridcolor="#1F2A34", zerolinecolor="#1F2A34"),
        yaxis=dict(gridcolor="#1F2A34", zerolinecolor="#1F2A34"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
)


# ---------------------------------------------------------------------------
# Data & model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    pollutant_cols = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2",
                      "O3", "Benzene", "Toluene", "Xylene", "AQI"]
    df[pollutant_cols] = df[pollutant_cols].fillna(df[pollutant_cols].mean().round(2))
    df["AQI_Bucket"] = df["AQI_Bucket"].fillna(pd.cut(df["AQI"], bins=BINS, labels=LABELS))
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Season"] = df["Month"].map(SEASON_MAP)
    return df


@st.cache_resource(show_spinner="Loading trained model...")
def load_model_artifacts():
    with open(os.path.join(MODEL_DIR, "random_forest_aqi_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_columns.pkl"), "rb") as f:
        feature_columns = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "cities.pkl"), "rb") as f:
        cities = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.pkl"), "rb") as f:
        metrics = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "summaries.pkl"), "rb") as f:
        summaries = pickle.load(f)
    return model, feature_columns, cities, metrics, summaries


def bucket_for(aqi_value):
    for lo, hi, label in zip(BINS[:-1], BINS[1:], LABELS):
        if lo < aqi_value <= hi or (lo == 0 and aqi_value <= hi):
            return label
    return "Severe"


artifacts_ok = os.path.exists(os.path.join(MODEL_DIR, "random_forest_aqi_model.pkl"))
data_ok = os.path.exists(DATA_PATH)

if not data_ok:
    st.error(
        "Dataset not found. Run `python src/generate_dataset.py` from the project "
        "root first, then restart the app."
    )
    st.stop()

df = load_data()

if not artifacts_ok:
    st.warning(
        "Trained model not found yet. Run `python src/train_model.py` from the "
        "project root to enable the Predict AQI tab and the model-driven charts.",
        icon="⚠️",
    )
    model = feature_columns = cities = metrics = summaries = None
else:
    model, feature_columns, cities, metrics, summaries = load_model_artifacts()


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div class="hero-wrap">
      <div class="hero-eyebrow">CPCB-style · Multi-City · 2015 – 2020</div>
      <div class="hero-title">🌫️ AirIndex India</div>
      <div class="hero-sub">
        Pollution trends, city comparisons, and a live AQI predictor built on
        daily air-quality readings across {df['City'].nunique()} Indian cities.
        Explore where the air is worst, how it shifts with the seasons, and what
        a Random Forest model expects the AQI to be for any city, season, and
        pollutant mix you choose.
      </div>
      <div class="aqi-chip-row">
        {''.join(f'<span class="aqi-chip" style="background:{c}">{lbl}</span>' for lbl, c in AQI_COLORS.items())}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPI row
k1, k2, k3, k4 = st.columns(4)
kpis = [
    ("Records", f"{len(df):,}"),
    ("Cities Tracked", f"{df['City'].nunique()}"),
    ("Date Range", f"{df['Date'].dt.year.min()}–{df['Date'].dt.year.max()}"),
    ("Model R² Score", f"{metrics['r2']:.2f}" if metrics else "—"),
]
for col, (label, value) in zip([k1, k2, k3, k4], kpis):
    with col:
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div></div>""",
            unsafe_allow_html=True,
        )

st.write("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_city, tab_trend, tab_predict, tab_about = st.tabs(
    ["📊 City & Season", "📈 Trends & Correlation", "🔮 Predict AQI", "🗂️ Data Explorer", "ℹ️ About"]
)

# ===================== TAB: City & Season =====================
with tab_overview:
    c1, c2 = st.columns([1.1, 1])

    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Top 10 Most Polluted Cities</div>'
                     '<div class="panel-sub">Ranked by average AQI across the full period</div>',
                     unsafe_allow_html=True)
        top_cities = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)
        bucket_labels = [bucket_for(v) for v in top_cities.values]
        colors = [AQI_COLORS[b] for b in bucket_labels]
        fig = go.Figure(go.Bar(
            x=top_cities.values, y=top_cities.index, orientation="h",
            marker_color=colors,
            text=[f"{v:.0f}" for v in top_cities.values], textposition="outside",
        ))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=420,
                           xaxis_title="Average AQI")
        fig.update_yaxes(autorange="reversed", gridcolor="#1F2A34")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">AQI Distribution by Season</div>'
                     '<div class="panel-sub">Winter inversion vs. monsoon washout</div>',
                     unsafe_allow_html=True)
        season_order = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
        fig2 = px.box(df, x="Season", y="AQI", category_orders={"Season": season_order},
                       color="Season",
                       color_discrete_sequence=["#4FA8D8", "#FFC107", "#4CAF50", "#FF7043"])
        fig2.update_layout(**PLOTLY_TEMPLATE["layout"], height=420, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">AQI Bucket Share</div>'
                 '<div class="panel-sub">Share of all daily readings in each AQI category</div>',
                 unsafe_allow_html=True)
    bucket_counts = df["AQI_Bucket"].value_counts().reindex(LABELS).fillna(0)
    fig3 = go.Figure(go.Bar(
        x=bucket_counts.index, y=bucket_counts.values,
        marker_color=[AQI_COLORS[l] for l in bucket_counts.index],
        text=bucket_counts.values, textposition="outside",
    ))
    fig3.update_layout(**PLOTLY_TEMPLATE["layout"], height=340, xaxis_title="", yaxis_title="Days")
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== TAB: Trends & Correlation =====================
with tab_trend:
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Average AQI Over Years</div>'
                     '<div class="panel-sub">National average across all tracked cities</div>',
                     unsafe_allow_html=True)
        yearly = df.groupby("Year")["AQI"].mean().reset_index()
        fig4 = px.line(yearly, x="Year", y="AQI", markers=True)
        fig4.update_traces(line_color="#4FA8D8", marker=dict(size=9, color="#4FA8D8"))
        fig4.update_layout(**PLOTLY_TEMPLATE["layout"], height=380)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">What Drives the Prediction</div>'
                     '<div class="panel-sub">Top feature importances from the Random Forest</div>',
                     unsafe_allow_html=True)
        if summaries is not None:
            fi = summaries["feature_importance"].sort_values(ascending=True)
            fig5 = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h",
                                     marker_color="#4FA8D8"))
            fig5.update_layout(**PLOTLY_TEMPLATE["layout"], height=380, xaxis_title="Importance")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("Train the model to see feature importances here.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Correlation Between Pollutants</div>'
                 '<div class="panel-sub">Pearson correlation across all pollutant readings + AQI</div>',
                 unsafe_allow_html=True)
    pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3",
                  "Benzene", "Toluene", "Xylene", "AQI"]
    corr = df[pollutants].corr()
    fig6 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig6.update_layout(**PLOTLY_TEMPLATE["layout"], height=520)
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== TAB: Predict AQI =====================
with tab_predict:
    if model is None:
        st.info("Run `python src/train_model.py` first to enable predictions.")
    else:
        left, right = st.columns([1, 1])

        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Configure a Scenario</div>'
                         '<div class="panel-sub">Pick a city, season, date, and pollutant levels</div>',
                         unsafe_allow_html=True)

            city_choice = st.selectbox("City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
            season_choice = st.selectbox("Season", ["Winter", "Summer", "Monsoon", "Post-Monsoon"])
            date_choice = st.date_input("Date", value=pd.Timestamp("2023-01-15"))

            st.markdown("**Pollutant levels**")
            p1, p2 = st.columns(2)
            with p1:
                pm25 = st.slider("PM2.5 (µg/m³)", 0.0, 500.0, 90.0)
                pm10 = st.slider("PM10 (µg/m³)", 0.0, 600.0, 160.0)
                no = st.slider("NO (µg/m³)", 0.0, 100.0, 15.0)
                no2 = st.slider("NO2 (µg/m³)", 0.0, 150.0, 35.0)
                nox = st.slider("NOx (µg/m³)", 0.0, 200.0, 40.0)
                nh3 = st.slider("NH3 (µg/m³)", 0.0, 120.0, 20.0)
            with p2:
                co = st.slider("CO (mg/m³)", 0.0, 15.0, 1.2)
                so2 = st.slider("SO2 (µg/m³)", 0.0, 100.0, 15.0)
                o3 = st.slider("O3 (µg/m³)", 0.0, 180.0, 35.0)
                benzene = st.slider("Benzene (µg/m³)", 0.0, 30.0, 3.0)
                toluene = st.slider("Toluene (µg/m³)", 0.0, 60.0, 8.0)
                xylene = st.slider("Xylene (µg/m³)", 0.0, 20.0, 2.0)

            predict_clicked = st.button("Predict AQI", type="primary", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Prediction</div>'
                         '<div class="panel-sub">Random Forest Regressor output</div>',
                         unsafe_allow_html=True)

            if predict_clicked:
                sample = pd.DataFrame(0, index=[0], columns=feature_columns)
                values = {
                    "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2, "NOx": nox,
                    "NH3": nh3, "CO": co, "SO2": so2, "O3": o3, "Benzene": benzene,
                    "Toluene": toluene, "Xylene": xylene,
                    "Year": date_choice.year, "Month": date_choice.month, "Day": date_choice.day,
                }
                for k, v in values.items():
                    if k in sample.columns:
                        sample[k] = v
                city_col = f"City_{city_choice}"
                season_col = f"Season_{season_choice}"
                if city_col in sample.columns:
                    sample[city_col] = 1
                if season_col in sample.columns:
                    sample[season_col] = 1

                pred_aqi = float(model.predict(sample)[0])
                pred_bucket = bucket_for(pred_aqi)
                pred_color = AQI_COLORS[pred_bucket]

                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(pred_aqi, 1),
                    number={"suffix": "", "font": {"size": 46, "color": pred_color}},
                    gauge={
                        "axis": {"range": [0, 500], "tickcolor": "#8CA0AC"},
                        "bar": {"color": pred_color},
                        "bgcolor": "rgba(0,0,0,0)",
                        "steps": [
                            {"range": [0, 50], "color": "rgba(76,175,80,0.25)"},
                            {"range": [50, 100], "color": "rgba(154,205,50,0.25)"},
                            {"range": [100, 200], "color": "rgba(255,193,7,0.25)"},
                            {"range": [200, 300], "color": "rgba(255,112,67,0.25)"},
                            {"range": [300, 400], "color": "rgba(229,57,53,0.25)"},
                            {"range": [400, 500], "color": "rgba(123,30,58,0.25)"},
                        ],
                    },
                ))
                gauge.update_layout(**PLOTLY_TEMPLATE["layout"], height=320)
                st.plotly_chart(gauge, use_container_width=True)

                st.markdown(
                    f"""<div style="text-align:center;">
                    <span class="aqi-chip" style="background:{pred_color}; font-size:1rem; padding:0.5rem 1.1rem;">
                    {pred_bucket}
                    </span>
                    </div>""",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Predicted for **{city_choice}**, **{season_choice}** season, "
                    f"on **{date_choice.strftime('%d %b %Y')}**."
                )
            else:
                st.info("Set the pollutant levels on the left and click **Predict AQI**.")
            st.markdown('</div>', unsafe_allow_html=True)

            if metrics:
                m1, m2 = st.columns(2)
                m1.metric("Model MAE", f"{metrics['mae']:.2f} AQI pts")
                m2.metric("Model R²", f"{metrics['r2']:.3f}")

# ===================== TAB: Data Explorer =====================
with tab_city:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">Browse the Raw Data</div>'
                 '<div class="panel-sub">Filter by city and date range</div>',
                 unsafe_allow_html=True)

    fc1, fc2 = st.columns([1, 2])
    with fc1:
        city_filter = st.multiselect("City", sorted(df["City"].unique()),
                                      default=sorted(df["City"].unique())[:3])
    with fc2:
        min_d, max_d = df["Date"].min(), df["Date"].max()
        date_range = st.slider("Date range", min_value=min_d.to_pydatetime(),
                                max_value=max_d.to_pydatetime(),
                                value=(min_d.to_pydatetime(), max_d.to_pydatetime()))

    filtered = df.copy()
    if city_filter:
        filtered = filtered[filtered["City"].isin(city_filter)]
    filtered = filtered[(filtered["Date"] >= date_range[0]) & (filtered["Date"] <= date_range[1])]

    st.dataframe(
        filtered[["City", "Date", "PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "AQI", "AQI_Bucket"]]
        .sort_values("Date"),
        use_container_width=True, height=420,
    )
    st.caption(f"Showing {len(filtered):,} of {len(df):,} total records.")
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== TAB: About =====================
with tab_about:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">About This Project</div>', unsafe_allow_html=True)
    st.markdown(
        """
This dashboard analyzes daily air-quality readings (PM2.5, PM10, NO₂, SO₂, CO, O₃, and more)
across Indian cities and predicts the **Air Quality Index (AQI)** using a
**Random Forest Regressor**.

**Pipeline**
1. `src/generate_dataset.py` — builds a realistic, fully synthetic `data/city_day.csv`
   (no internet or Kaggle account needed), modeled on the structure and behavior of the
   CPCB "Air Quality Data in India" dataset — including seasonal winter spikes, monsoon
   washouts, and city-level pollution baselines.
2. `src/train_model.py` — cleans the data, engineers Year/Month/Day/Season features,
   one-hot encodes City & Season, trains a Random Forest, and saves the model + supporting
   summaries to `models/`.
3. `app.py` (this dashboard) — loads the data and trained model to power every chart and
   the live predictor.
4. `notebooks/aqi_analysis.ipynb` — the full step-by-step analysis notebook (EDA, feature
   engineering, modeling, evaluation, save/load) for anyone who wants to read or re-run the
   analysis in Jupyter / VS Code.

**Note on the data:** to keep this project runnable anywhere with zero setup friction,
the dataset is synthetically generated to statistically resemble real Indian air-quality
patterns (city rankings, seasonal cycles, pollutant correlations). Swap in the real CPCB
`city_day.csv` at any time — the same scripts will work unchanged.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)
