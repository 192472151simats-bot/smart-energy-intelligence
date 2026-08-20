import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# CONFIG
st.set_page_config(
    page_title="Smart Energy Intelligence",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
.hero {
    padding:25px;
    border-radius:20px;
    background:linear-gradient(135deg,#0b2447,#19376d);
    color:white;
    margin-bottom:20px;
}
.hero h1 {margin:0;font-size:36px;}
.hero p {opacity:.85;}
</style>
""", unsafe_allow_html=True)


# DATA
@st.cache_data
def load_data(file=None):

    df = pd.read_csv(
        file if file else "data/energydata_complete.csv",
        sep="\t"
    )

    df.columns = df.columns.str.strip()

    return df


# PREPROCESSING
@st.cache_data
def preprocess(df):

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"],
        dayfirst=True,
        errors="coerce"
    )

    df["Energy_kWh"] = df["Appliances"] / 1000

    df["Hour"] = df.date.dt.hour
    df["Day"] = df.date.dt.day
    df["Month"] = df.date.dt.month
    df["Day_of_Week"] = df.date.dt.dayofweek
    df["Day_Name"] = df.date.dt.day_name()

    df["Is_Weekend"] = (
        df.Day_of_Week >= 5
    ).astype(int)

    df["Is_Peak_Hour"] = (
        df.Hour.between(18, 21)
    ).astype(int)

    temps = [f"T{i}" for i in range(1, 10)]
    hums = [f"RH_{i}" for i in range(1, 10)]

    df["Temperature_Avg"] = df[temps].mean(axis=1)
    df["Humidity_Avg"] = df[hums].mean(axis=1)

    df["Energy_Lag_1"] = df.Energy_kWh.shift(1)
    df["Energy_Lag_6"] = df.Energy_kWh.shift(6)
    df["Energy_Rolling_6"] = df.Energy_kWh.rolling(6).mean()

    num = df.select_dtypes("number").columns

    df[num] = (
        df[num]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(df[num].median())
    )

    return df


# HELPERS
def metrics(df):

    cols = st.columns(4)

    values = [
        ("Records", f"{len(df):,}"),
        ("Average", f"{df.Energy_kWh.mean():.3f} kWh"),
        ("Peak", f"{df.Energy_kWh.max():.3f} kWh"),
        ("Missing", f"{df.isna().sum().sum():,}")
    ]

    for col, (name, value) in zip(cols, values):
        col.metric(name, value)


def chart(fig):
    st.plotly_chart(
        fig,
        use_container_width=True
    )


def features(df):

    return [
        c for c in [
            "lights",
            "T_out",
            "RH_out",
            "Windspeed",
            "Visibility",
            "Tdewpoint",
            "Temperature_Avg",
            "Humidity_Avg",
            "Hour",
            "Day_of_Week",
            "Is_Weekend",
            "Is_Peak_Hour",
            "Energy_Lag_1",
            "Energy_Lag_6",
            "Energy_Rolling_6"
        ]
        if c in df.columns
    ]


# MACHINE LEARNING
@st.cache_resource
def train(df, selected):

    X = df[selected]
    y = df.Energy_kWh

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=.2,
        random_state=42
    )

    models = {

        "Linear Regression":
            make_pipeline(
                StandardScaler(),
                LinearRegression()
            ),

        "Decision Tree":
            DecisionTreeRegressor(
                max_depth=10,
                random_state=42
            ),

        "KNN":
            make_pipeline(
                StandardScaler(),
                KNeighborsRegressor(
                    n_neighbors=7
                )
            )
    }

    results = []
    fitted = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mse = mean_squared_error(y_test, pred)

        results.append([
            name,
            mean_absolute_error(y_test, pred),
            mse,
            np.sqrt(mse),
            r2_score(y_test, pred)
        ])

        fitted[name] = model

    results = pd.DataFrame(
        results,
        columns=[
            "Model",
            "MAE",
            "MSE",
            "RMSE",
            "R²"
        ]
    ).sort_values("RMSE")

    return results, fitted, X_test, y_test


# HEADER
st.markdown("""
<div class="hero">
<h1>⚡ Smart Energy Intelligence</h1>
<p>
Smart Energy Consumption Prediction and Analytics
Using Data Science Techniques
</p>
</div>
""", unsafe_allow_html=True)


# SIDEBAR
st.sidebar.title("⚡ Smart Energy")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Dashboard",
        "📥 Module 1 - Data Management",
        "🔎 Module 2 - EDA",
        "🧠 Module 3 - Statistics & Features",
        "🤖 Module 4 - Prediction",
        "📊 Module 5 - Evaluation",
        "🚨 Anomaly Detection",
        "🎯 What-If Simulator",
        "✨ AI Energy Advisor"
    ]
)

upload = st.sidebar.file_uploader(
    "Upload Energy CSV",
    type="csv"
)

try:
    df = preprocess(load_data(upload))
except Exception as e:
    st.error(f"Dataset error: {e}")
    st.stop()

st.sidebar.info(
    f"Records: {len(df):,}\n\n"
    f"Target: Energy_kWh"
)


# DASHBOARD
if page == "🏠 Executive Dashboard":

    st.header("🏠 Executive Energy Dashboard")

    metrics(df)

    daily = (
        df.set_index("date")
        .Energy_kWh
        .resample("D")
        .mean()
        .reset_index()
    )

    chart(
        px.area(
            daily,
            x="date",
            y="Energy_kWh",
            title="Daily Average Energy Consumption"
        )
    )

    c1, c2 = st.columns(2)

    with c1:

        hourly = (
            df.groupby("Hour")
            .Energy_kWh
            .mean()
            .reset_index()
        )

        chart(
            px.line(
                hourly,
                x="Hour",
                y="Energy_kWh",
                markers=True,
                title="Average Consumption by Hour"
            )
        )

    with c2:

        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]

        weekly = (
            df.groupby("Day_Name")
            .Energy_kWh
            .mean()
            .reindex(days)
            .reset_index()
        )

        chart(
            px.bar(
                weekly,
                x="Day_Name",
                y="Energy_kWh",
                title="Average Consumption by Day"
            )
        )

    latest = df.Energy_kWh.iloc[-1]
    avg = df.Energy_kWh.mean()

    if latest > avg * 1.25:
        st.error("🚨 Current consumption is significantly above average.")
    elif latest > avg:
        st.warning("🟡 Current consumption is above average.")
    else:
        st.success("🟢 Current consumption is within the normal range.")


# MODULE 1
elif page == "📥 Module 1 - Data Management":

    st.header("📥 Module 1 — Data Acquisition and Management")

    metrics(df)

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Data Quality")

    profile = pd.DataFrame({
        "Column": df.columns,
        "Type": [str(df[c].dtype) for c in df.columns],
        "Missing": [df[c].isna().sum() for c in df.columns],
        "Unique": [df[c].nunique() for c in df.columns]
    })

    st.dataframe(
        profile,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Statistics")

    st.dataframe(
        df.describe().T,
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Processed Dataset",
        df.to_csv(index=False).encode(),
        "processed_energy_dataset.csv",
        "text/csv"
    )


# MODULE 2
elif page == "🔎 Module 2 - EDA":

    st.header("🔎 Module 2 — Preprocessing and EDA")

    metrics(df)

    c1, c2 = st.columns(2)

    with c1:
        chart(
            px.histogram(
                df,
                x="Energy_kWh",
                nbins=50,
                title="Energy Distribution"
            )
        )

    with c2:
        chart(
            px.box(
                df,
                y="Energy_kWh",
                points="outliers",
                title="Energy Box Plot"
            )
        )

    q1 = df.Energy_kWh.quantile(.25)
    q3 = df.Energy_kWh.quantile(.75)
    iqr = q3 - q1

    outliers = df[
        (df.Energy_kWh < q1 - 1.5 * iqr) |
        (df.Energy_kWh > q3 + 1.5 * iqr)
    ]

    st.subheader("IQR Outlier Analysis")

    st.metric(
        "Potential Outliers",
        f"{len(outliers):,}"
    )

    st.subheader("Correlation Heatmap")

    chart(
        px.imshow(
            df.select_dtypes("number").corr(),
            text_auto=".2f",
            aspect="auto",
            title="Feature Correlation"
        )
    )

    choices = [
        c for c in [
            "lights",
            "T1", "T2", "T3", "T4", "T5",
            "T6", "T7", "T8", "T9",
            "RH_1", "RH_2", "RH_3", "RH_4",
            "RH_5", "RH_6", "RH_7", "RH_8", "RH_9",
            "T_out", "Press_mm_hg",
            "RH_out", "Windspeed",
            "Visibility", "Tdewpoint"
        ]
        if c in df.columns
    ]

    choice = st.selectbox(
        "Select Feature",
        choices
    )

    chart(
        px.scatter(
            df,
            x=choice,
            y="Energy_kWh",
            opacity=.45,
            title=f"{choice} vs Energy"
        )
    )


# MODULE 3
elif page == "🧠 Module 3 - Statistics & Features":

    st.header(
        "🧠 Module 3 — Statistical Analysis and Feature Engineering"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Mean", f"{df.Energy_kWh.mean():.3f}")
    c2.metric("Median", f"{df.Energy_kWh.median():.3f}")
    c3.metric("Std Dev", f"{df.Energy_kWh.std():.3f}")
    c4.metric(
        "95th Percentile",
        f"{df.Energy_kWh.quantile(.95):.3f}"
    )

    engineered = [
        "Hour",
        "Day",
        "Month",
        "Day_of_Week",
        "Is_Weekend",
        "Is_Peak_Hour",
        "Energy_Lag_1",
        "Energy_Lag_6",
        "Energy_Rolling_6",
        "Temperature_Avg",
        "Humidity_Avg"
    ]

    st.subheader("Engineered Features")

    st.dataframe(
        df[["date", "Energy_kWh"] + engineered].head(20),
        use_container_width=True
    )

    corr = (
        df.select_dtypes("number")
        .corr()["Energy_kWh"]
        .drop("Energy_kWh")
        .sort_values(key=abs, ascending=False)
        .head(15)
        .reset_index()
    )

    corr.columns = ["Feature", "Correlation"]

    chart(
        px.bar(
            corr,
            x="Correlation",
            y="Feature",
            orientation="h",
            title="Top Correlated Features"
        )
    )


# MODULE 4
elif page == "🤖 Module 4 - Prediction":

    st.header(
        "🤖 Module 4 — Energy Consumption Prediction"
    )

    selected = st.multiselect(
        "Prediction Features",
        features(df),
        default=features(df)
    )

    if not selected:
        st.warning("Select at least one feature.")
        st.stop()

    results, models, X_test, y_test = train(
        df,
        selected
    )

    st.dataframe(
        results.style.format({
            "MAE": "{:.4f}",
            "MSE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    best = results.iloc[0].Model

    st.success(f"🏆 Best Model: {best}")

    pred = models[best].predict(X_test)

    chart(
        px.line(
            pd.DataFrame({
                "Actual": y_test.values[:200],
                "Predicted": pred[:200]
            }),
            title="Actual vs Predicted Energy"
        )
    )

    st.subheader("Quick Prediction")

    values = {}
    cols = st.columns(3)

    for i, f in enumerate(selected):

        with cols[i % 3]:

            values[f] = st.number_input(
                f,
                value=float(df.iloc[-1][f])
            )

    if st.button("⚡ Predict Energy", type="primary"):

        result = models[best].predict(
            pd.DataFrame([values])
        )[0]

        st.metric(
            "Predicted Consumption",
            f"{result:.3f} kWh"
        )


# MODULE 5
elif page == "📊 Module 5 - Evaluation":

    st.header(
        "📊 Module 5 — Evaluation and Decision Support"
    )

    results, _, _, _ = train(
        df,
        features(df)
    )

    c1, c2 = st.columns(2)

    with c1:
        chart(
            px.bar(
                results,
                x="Model",
                y="RMSE",
                text_auto=".3f",
                title="RMSE Comparison"
            )
        )

    with c2:
        chart(
            px.bar(
                results,
                x="Model",
                y="R²",
                text_auto=".3f",
                title="R² Comparison"
            )
        )

    st.dataframe(
        results.style.format({
            "MAE": "{:.4f}",
            "MSE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"🏆 Best Model: {results.iloc[0].Model}"
    )


# ANOMALY DETECTION
elif page == "🚨 Anomaly Detection":

    st.header("🚨 Energy Anomaly Detection")

    cols = [
        "Energy_kWh",
        "Temperature_Avg",
        "Humidity_Avg",
        "Hour"
    ]

    temp = df.copy()

    temp["Anomaly"] = IsolationForest(
        contamination=.02,
        random_state=42
    ).fit_predict(temp[cols])

    anomalies = temp[temp.Anomaly == -1]

    c1, c2 = st.columns(2)

    c1.metric(
        "Total Records",
        f"{len(temp):,}"
    )

    c2.metric(
        "Detected Anomalies",
        f"{len(anomalies):,}"
    )

    chart(
        px.scatter(
            temp,
            x="date",
            y="Energy_kWh",
            color=temp.Anomaly.astype(str),
            title="Energy Anomalies"
        )
    )

    st.dataframe(
        anomalies[
            [
                "date",
                "Energy_kWh",
                "Temperature_Avg",
                "Humidity_Avg"
            ]
        ].head(50),
        use_container_width=True
    )


# WHAT-IF SIMULATOR
elif page == "🎯 What-If Simulator":

    st.header("🎯 What-If Energy Simulator")

    selected = features(df)

    results, models, _, _ = train(
        df,
        selected
    )

    best = results.iloc[0].Model

    values = {}
    cols = st.columns(3)

    for i, f in enumerate(selected):

        with cols[i % 3]:

            values[f] = st.number_input(
                f,
                value=float(df.iloc[-1][f])
            )

    if st.button(
        "🔮 Simulate Energy",
        type="primary"
    ):

        prediction = models[best].predict(
            pd.DataFrame([values])
        )[0]

        st.metric(
            "Predicted Energy",
            f"{prediction:.3f} kWh"
        )


# AI ENERGY ADVISOR
elif page == "✨ AI Energy Advisor":

    st.header("✨ AI Energy Advisor")

    question = st.text_area(
        "Ask your question",
        "How can I reduce peak energy consumption?"
    )

    if st.button(
        "Generate Advice",
        type="primary"
    ):

        hourly = (
            df.groupby("Hour")
            .Energy_kWh
            .mean()
        )

        peak_hour = hourly.idxmax()
        peak_value = hourly.max()

        corr = (
            df.select_dtypes("number")
            .corr()["Energy_kWh"]
            .drop("Energy_kWh")
            .sort_values(
                key=abs,
                ascending=False
            )
        )

        strongest = corr.index[0]

        st.info(
            f"Peak demand occurs around "
            f"{peak_hour}:00 with "
            f"{peak_value:.3f} kWh."
        )

        st.info(
            f"Strongest correlated feature: "
            f"{strongest}."
        )

        st.success(
            "Recommended actions:\n\n"
            "• Shift flexible appliance usage away from peak hours.\n\n"
            "• Investigate unusual consumption spikes.\n\n"
            "• Monitor environmental conditions.\n\n"
            "• Use the What-If Simulator for scenarios."
        )

        key = os.getenv("GEMINI_API_KEY")

        if key:

            try:

                from google import genai

                client = genai.Client(
                    api_key=key
                )

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"""
User question:
{question}

Average energy:
{df.Energy_kWh.mean():.3f} kWh

Peak hour:
{peak_hour}

Strongest feature:
{strongest}

Give practical energy-saving recommendations.
"""
                )

                st.subheader("🤖 AI Analysis")

                st.write(response.text)

            except Exception as e:

                st.warning(
                    f"AI service unavailable: {e}"
                )

        else:

            st.caption(
                "Gemini is optional. Add GEMINI_API_KEY "
                "to enable generative AI."
            )


st.markdown("---")

st.caption(
    "Smart Energy Consumption Prediction and Analytics • "
    "Fundamentals of Data Science"
)