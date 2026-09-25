import streamlit as st
import pandas as pd
import joblib


# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load("air_quality_model.pkl")


# ==========================================================
# AQI CATEGORY
# ==========================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# ==========================================================
# AQI DESCRIPTION
# ==========================================================

def get_aqi_description(category):

    descriptions = {

        "Good":
            "Air quality is considered good.",

        "Satisfactory":
            "Air quality is generally acceptable.",

        "Moderate":
            "Air quality may affect sensitive individuals.",

        "Poor":
            "Air pollution may cause discomfort and health effects.",

        "Very Poor":
            "Air pollution can cause noticeable health effects.",

        "Severe":
            "Air pollution represents a serious health concern."
    }

    return descriptions[category]


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Air Quality Prediction",
    page_icon="🌫️",
    layout="wide"
)


# ==========================================================
# TITLE
# ==========================================================

st.title("🌫️ Air Quality Prediction System")

st.write(
    "A machine-learning application that predicts "
    "Air Quality Index (AQI) from pollutant concentrations."
)

st.divider()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.header("📊 Model Information")

    st.write("**Machine Learning Model:**")
    st.write("Random Forest Regressor")

    st.write("**Prediction Type:**")
    st.write("AQI Regression")

    st.write("**Input Parameters:**")
    st.write("12 pollutant parameters")

    st.write("**Model R² Score:**")
    st.write("0.908")

    st.divider()

    st.write("### AQI Categories")

    st.write("0–50 → Good")
    st.write("51–100 → Satisfactory")
    st.write("101–200 → Moderate")
    st.write("201–300 → Poor")
    st.write("301–400 → Very Poor")
    st.write("401+ → Severe")


# ==========================================================
# INPUT SECTION
# ==========================================================

st.subheader("Enter Air Quality Parameters")

st.write(
    "Enter the measured concentration of each pollutant."
)


col1, col2, col3 = st.columns(3)


# -------------------------
# Column 1
# -------------------------

with col1:

    pm25 = st.number_input(
        "PM2.5",
        min_value=0.0,
        value=50.0
    )

    pm10 = st.number_input(
        "PM10",
        min_value=0.0,
        value=80.0
    )

    no = st.number_input(
        "NO",
        min_value=0.0,
        value=20.0
    )

    no2 = st.number_input(
        "NO2",
        min_value=0.0,
        value=30.0
    )


# -------------------------
# Column 2
# -------------------------

with col2:

    nox = st.number_input(
        "NOx",
        min_value=0.0,
        value=40.0
    )

    nh3 = st.number_input(
        "NH3",
        min_value=0.0,
        value=15.0
    )

    co = st.number_input(
        "CO",
        min_value=0.0,
        value=0.8
    )

    so2 = st.number_input(
        "SO2",
        min_value=0.0,
        value=10.0
    )


# -------------------------
# Column 3
# -------------------------

with col3:

    o3 = st.number_input(
        "O3",
        min_value=0.0,
        value=40.0
    )

    benzene = st.number_input(
        "Benzene",
        min_value=0.0,
        value=2.0
    )

    toluene = st.number_input(
        "Toluene",
        min_value=0.0,
        value=5.0
    )

    xylene = st.number_input(
        "Xylene",
        min_value=0.0,
        value=1.0
    )


st.divider()


# ==========================================================
# PREDICTION
# ==========================================================

if st.button(
    "🔮 Predict AQI",
    use_container_width=True
):

    # Create input dataframe

    input_data = pd.DataFrame([{

        "PM2.5": pm25,
        "PM10": pm10,
        "NO": no,
        "NO2": no2,
        "NOx": nox,
        "NH3": nh3,
        "CO": co,
        "SO2": so2,
        "O3": o3,
        "Benzene": benzene,
        "Toluene": toluene,
        "Xylene": xylene

    }])


    # Predict AQI

    predicted_aqi = model.predict(input_data)[0]

    category = get_aqi_category(predicted_aqi)

    description = get_aqi_description(category)


    # ======================================================
    # RESULT
    # ======================================================

    st.subheader("Prediction Result")


    result1, result2 = st.columns(2)


    with result1:

        st.metric(
            "Predicted AQI",
            f"{predicted_aqi:.2f}"
        )


    with result2:

        st.metric(
            "Air Quality",
            category
        )


    st.info(description)


    # ======================================================
    # INPUT SUMMARY
    # ======================================================

    st.subheader("Pollutant Values Used")

    display_data = pd.DataFrame({
        "Pollutant": [
            "PM2.5",
            "PM10",
            "NO",
            "NO2",
            "NOx",
            "NH3",
            "CO",
            "SO2",
            "O3",
            "Benzene",
            "Toluene",
            "Xylene"
        ],

        "Value": [
            pm25,
            pm10,
            no,
            no2,
            nox,
            nh3,
            co,
            so2,
            o3,
            benzene,
            toluene,
            xylene
        ]
    })

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )


    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    st.subheader("Model Feature Importance")

    features = [
        "PM2.5",
        "PM10",
        "NO",
        "NO2",
        "NOx",
        "NH3",
        "CO",
        "SO2",
        "O3",
        "Benzene",
        "Toluene",
        "Xylene"
    ]

    importance = pd.DataFrame({
        "Pollutant": features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Pollutant")
    )
    # ==========================================================
# HISTORICAL AQI ANALYSIS
# ==========================================================

st.divider()

st.header("📈 Historical AQI Analysis")

st.write(
    "Explore historical Air Quality Index values by city and year."
)

# Load historical dataset
historical_df = pd.read_csv("city_day.csv")

# Convert Date column
historical_df["Date"] = pd.to_datetime(
    historical_df["Date"],
    errors="coerce"
)

# Remove rows without valid dates or AQI
historical_df = historical_df.dropna(
    subset=["Date", "AQI"]
)

# Create Year column
historical_df["Year"] = historical_df["Date"].dt.year


# ==========================================================
# CITY SELECTION
# ==========================================================

cities = sorted(
    historical_df["City"].dropna().unique()
)

selected_city = st.selectbox(
    "Select City",
    cities
)


# Filter selected city
city_data = historical_df[
    historical_df["City"] == selected_city
]


# ==========================================================
# YEAR SELECTION
# ==========================================================

years = sorted(
    city_data["Year"].dropna().unique()
)

selected_year = st.selectbox(
    "Select Year",
    years
)


# Filter city and year
historical_data = city_data[
    city_data["Year"] == selected_year
].copy()


# ==========================================================
# AQI TREND
# ==========================================================

st.subheader(
    f"AQI Trend - {selected_city} ({selected_year})"
)

chart_data = historical_data[
    ["Date", "AQI"]
].sort_values("Date")

chart_data = chart_data.set_index("Date")

st.line_chart(chart_data)


# ==========================================================
# STATISTICS
# ==========================================================

st.subheader("Historical AQI Statistics")

stat1, stat2, stat3 = st.columns(3)

with stat1:
    st.metric(
        "Average AQI",
        f"{historical_data['AQI'].mean():.2f}"
    )

with stat2:
    st.metric(
        "Maximum AQI",
        f"{historical_data['AQI'].max():.2f}"
    )

with stat3:
    st.metric(
        "Minimum AQI",
        f"{historical_data['AQI'].min():.2f}"
    )