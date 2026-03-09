import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------------------
# 1. Synthetic Data Generation
# -------------------------------

def generate_synthetic_data(n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)

    # Crude properties
    api = rng.uniform(20, 45, n_samples)             # API gravity
    sulfur = rng.uniform(0.1, 4.0, n_samples)        # wt% sulfur
    t10 = rng.uniform(120, 200, n_samples)           # T10 (°C)
    t50 = rng.uniform(200, 320, n_samples)           # T50 (°C)
    t90 = rng.uniform(320, 420, n_samples)           # T90 (°C)

    # Operating conditions
    furnace_inlet_temp = rng.uniform(320, 420, n_samples)  # °C
    top_temp = rng.uniform(100, 160, n_samples)            # °C
    bottom_temp = rng.uniform(250, 360, n_samples)         # °C
    column_pressure = rng.uniform(1.0, 2.5, n_samples)     # bar
    reflux_ratio = rng.uniform(0.5, 3.0, n_samples)        # dimensionless
    feed_rate = rng.uniform(100, 500, n_samples)           # t/h

    # Heuristic base fractions: just to get plausible tendencies
    light_factor = (
        0.015 * (api - 30) +
        0.02 * (furnace_inlet_temp - 350) / 50 -
        0.01 * (sulfur - 1.5)
    )

    mid_factor = (
        0.01 * (t50 - 250) / 50 +
        0.02 * (reflux_ratio - 1.5)
    )

    heavy_factor = (
        -0.015 * (api - 30) +
        0.01 * (bottom_temp - 300) / 40 +
        0.01 * (column_pressure - 1.5)
    )

    # Base yields before normalization
    lpg = 0.05 + 0.05 * light_factor
    naphtha = 0.20 + 0.20 * light_factor + 0.05 * mid_factor
    kerosene = 0.15 + 0.20 * mid_factor
    diesel = 0.25 + 0.10 * mid_factor - 0.05 * light_factor
    gas_oil = 0.15 + 0.10 * heavy_factor
    residue = 0.20 + 0.20 * heavy_factor

    yields = np.vstack([lpg, naphtha, kerosene, diesel, gas_oil, residue]).T
    noise = rng.normal(0, 0.01, size=yields.shape)
    yields = yields + noise

    # Non-negative
    yields = np.clip(yields, 0.001, None)

    # Normalize rows to sum = 1
    row_sums = yields.sum(axis=1, keepdims=True)
    yields = yields / row_sums

    df = pd.DataFrame(
        {
            "API": api,
            "Sulfur": sulfur,
            "T10": t10,
            "T50": t50,
            "T90": t90,
            "Furnace_T": furnace_inlet_temp,
            "Top_T": top_temp,
            "Bottom_T": bottom_temp,
            "Column_P": column_pressure,
            "Reflux_Ratio": reflux_ratio,
            "Feed_Rate": feed_rate,
            "LPG_yield": yields[:, 0],
            "Naphtha_yield": yields[:, 1],
            "Kerosene_yield": yields[:, 2],
            "Diesel_yield": yields[:, 3],
            "GasOil_yield": yields[:, 4],
            "Residue_yield": yields[:, 5],
        }
    )

    return df


# -------------------------------
# 2. Model Training
# -------------------------------

@st.cache_resource(show_spinner=True)
def train_model():
    df = generate_synthetic_data()

    feature_cols = [
        "API",
        "Sulfur",
        "T10",
        "T50",
        "T90",
        "Furnace_T",
        "Top_T",
        "Bottom_T",
        "Column_P",
        "Reflux_Ratio",
        "Feed_Rate",
    ]
    target_cols = [
        "LPG_yield",
        "Naphtha_yield",
        "Kerosene_yield",
        "Diesel_yield",
        "GasOil_yield",
        "Residue_yield",
    ]

    X = df[feature_cols].values
    y = df[target_cols].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2_per_fraction = []
    mae_per_fraction = []

    for i, name in enumerate(target_cols):
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2_per_fraction.append((name, r2))
        mae_per_fraction.append((name, mae))

    metrics_df = pd.DataFrame(
        {
            "Fraction": [n for n, _ in r2_per_fraction],
            "R2": [v for _, v in r2_per_fraction],
            "MAE": [v for _, v in mae_per_fraction],
        }
    )

    return model, feature_cols, target_cols, metrics_df


# -------------------------------
# 3. Helper for prediction
# -------------------------------

def predict_yields(model, input_vector, target_cols):
    """Return normalized yields and a DataFrame."""
    yields_pred = model.predict(input_vector)[0]
    yields_pred = np.clip(yields_pred, 0.0, None)
    total = yields_pred.sum()
    if total <= 0:
        total = 1.0
    yields_pred = yields_pred / total

    fractions = ["LPG", "Naphtha", "Kerosene", "Diesel", "Gas Oil", "Residue"]
    wt_percent = yields_pred * 100.0

    result_df = pd.DataFrame(
        {
            "Fraction": fractions,
            "Yield (fraction of feed)": yields_pred,
            "Yield (wt%)": wt_percent,
        }
    )
    return yields_pred, wt_percent, result_df


# -------------------------------
# 4. Streamlit UI
# -------------------------------

def main():
    st.set_page_config(
        page_title="Mini Smart Crude Distillation Predictor",
        layout="wide",
    )

    st.title("🧠 Mini Smart Crude Oil Distillation Predictor")
    st.markdown(
        """
This app is a **virtual crude distillation column** powered by a machine learning model.

- Choose **crude properties** and **operating conditions**
- Predict **product yields** (LPG, naphtha, kerosene, diesel, gas oil, residue)
- Compare two scenarios and run a simple **yield optimization**
"""
    )

    # Train/load model
    with st.spinner("Training model (synthetic plant data)..."):
        model, feature_cols, target_cols, metrics_df = train_model()

    with st.expander("Model performance on synthetic test data", expanded=False):
        st.dataframe(
            metrics_df.style.format({"R2": "{:.3f}", "MAE": "{:.3f}"}),
            use_container_width=True,
        )

    tab1, tab2 = st.tabs(["Single Scenario", "Compare & Optimize"])

    # Ranges used both for sliders and optimization
    ranges = {
        "API": (20.0, 45.0),
        "Sulfur": (0.1, 4.0),
        "T10": (120.0, 200.0),
        "T50": (200.0, 320.0),
        "T90": (320.0, 420.0),
        "Furnace_T": (320.0, 420.0),
        "Top_T": (100.0, 160.0),
        "Bottom_T": (250.0, 360.0),
        "Column_P": (1.0, 2.5),
        "Reflux_Ratio": (0.5, 3.0),
        "Feed_Rate": (100.0, 500.0),
    }

    # ---------------------------------------------
    # TAB 1 – Single Scenario Prediction
    # ---------------------------------------------
    with tab1:
        st.subheader("Single Scenario Prediction")

        st.sidebar.header("Inputs – Single Scenario")

        api = st.sidebar.slider("API Gravity", *ranges["API"], 32.0, 0.5)
        sulfur = st.sidebar.slider("Sulfur (wt%)", *ranges["Sulfur"], 1.5, 0.1)
        t10 = st.sidebar.slider("T10 (°C)", *ranges["T10"], 160.0, 1.0)
        t50 = st.sidebar.slider("T50 (°C)", *ranges["T50"], 260.0, 1.0)
        t90 = st.sidebar.slider("T90 (°C)", *ranges["T90"], 370.0, 1.0)

        furnace_t = st.sidebar.slider("Furnace Inlet T (°C)", *ranges["Furnace_T"], 360.0, 1.0)
        top_t = st.sidebar.slider("Column Top T (°C)", *ranges["Top_T"], 130.0, 1.0)
        bottom_t = st.sidebar.slider("Column Bottom T (°C)", *ranges["Bottom_T"], 310.0, 1.0)
        column_p = st.sidebar.slider("Column Pressure (bar)", *ranges["Column_P"], 1.5, 0.05)
        reflux = st.sidebar.slider("Reflux Ratio", *ranges["Reflux_Ratio"], 1.5, 0.1)
        feed_rate = st.sidebar.slider("Feed Rate (t/h)", *ranges["Feed_Rate"], 300.0, 5.0)

        input_vector = np.array(
            [
                api,
                sulfur,
                t10,
                t50,
                t90,
                furnace_t,
                top_t,
                bottom_t,
                column_p,
                reflux,
                feed_rate,
            ]
        ).reshape(1, -1)

        if st.button("Predict Product Yields", key="single_predict"):
            yields_pred, wt_percent, result_df = predict_yields(
                model, input_vector, target_cols
            )

            st.subheader("Predicted Product Slate")
            st.dataframe(
                result_df.style.format(
                    {"Yield (fraction of feed)": "{:.3f}", "Yield (wt%)": "{:.1f}"}
                ),
                use_container_width=True,
            )

            st.subheader("Product Yield Distribution")
            chart_data = pd.DataFrame(
                {
                    "Fraction": result_df["Fraction"],
                    "Yield_wt_percent": result_df["Yield (wt%)"],
                }
            )
            st.bar_chart(chart_data.set_index("Fraction"), height=400)

            st.caption(
                f"Sum of predicted yields: **{wt_percent.sum():.1f} wt%** (should be ~100 wt%)."
            )
        else:
            st.info("Set the input conditions and click **Predict Product Yields**.")

    # ---------------------------------------------
    # TAB 2 – Scenario Comparison & Optimization
    # ---------------------------------------------
    with tab2:
        col_left, col_right = st.columns(2)

        # Scenario A
        with col_left:
            st.subheader("Scenario A")
            api_a = st.slider("API Gravity (A)", *ranges["API"], 32.0, 0.5)
            sulfur_a = st.slider("Sulfur (wt%) (A)", *ranges["Sulfur"], 1.5, 0.1)
            t10_a = st.slider("T10 (°C) (A)", *ranges["T10"], 160.0, 1.0)
            t50_a = st.slider("T50 (°C) (A)", *ranges["T50"], 260.0, 1.0)
            t90_a = st.slider("T90 (°C) (A)", *ranges["T90"], 370.0, 1.0)
            furnace_a = st.slider("Furnace T (°C) (A)", *ranges["Furnace_T"], 360.0, 1.0)
            top_a = st.slider("Top T (°C) (A)", *ranges["Top_T"], 130.0, 1.0)
            bottom_a = st.slider("Bottom T (°C) (A)", *ranges["Bottom_T"], 310.0, 1.0)
            colp_a = st.slider("Column P (bar) (A)", *ranges["Column_P"], 1.5, 0.05)
            reflux_a = st.slider("Reflux Ratio (A)", *ranges["Reflux_Ratio"], 1.5, 0.1)
            feed_a = st.slider("Feed Rate (t/h) (A)", *ranges["Feed_Rate"], 300.0, 5.0)

        # Scenario B
        with col_right:
            st.subheader("Scenario B")
            api_b = st.slider("API Gravity (B)", *ranges["API"], 34.0, 0.5)
            sulfur_b = st.slider("Sulfur (wt%) (B)", *ranges["Sulfur"], 1.0, 0.1)
            t10_b = st.slider("T10 (°C) (B)", *ranges["T10"], 150.0, 1.0)
            t50_b = st.slider("T50 (°C) (B)", *ranges["T50"], 255.0, 1.0)
            t90_b = st.slider("T90 (°C) (B)", *ranges["T90"], 365.0, 1.0)
            furnace_b = st.slider("Furnace T (°C) (B)", *ranges["Furnace_T"], 370.0, 1.0)
            top_b = st.slider("Top T (°C) (B)", *ranges["Top_T"], 125.0, 1.0)
            bottom_b = st.slider("Bottom T (°C) (B)", *ranges["Bottom_T"], 315.0, 1.0)
            colp_b = st.slider("Column P (bar) (B)", *ranges["Column_P"], 1.6, 0.05)
            reflux_b = st.slider("Reflux Ratio (B)", *ranges["Reflux_Ratio"], 1.8, 0.1)
            feed_b = st.slider("Feed Rate (t/h) (B)", *ranges["Feed_Rate"], 320.0, 5.0)

        if st.button("Compare Scenarios A vs B", key="compare"):
            input_a = np.array(
                [
                    api_a,
                    sulfur_a,
                    t10_a,
                    t50_a,
                    t90_a,
                    furnace_a,
                    top_a,
                    bottom_a,
                    colp_a,
                    reflux_a,
                    feed_a,
                ]
            ).reshape(1, -1)

            input_b = np.array(
                [
                    api_b,
                    sulfur_b,
                    t10_b,
                    t50_b,
                    t90_b,
                    furnace_b,
                    top_b,
                    bottom_b,
                    colp_b,
                    reflux_b,
                    feed_b,
                ]
            ).reshape(1, -1)

            _, wt_a, df_a = predict_yields(model, input_a, target_cols)
            _, wt_b, df_b = predict_yields(model, input_b, target_cols)

            st.subheader("Scenario Comparison (wt%)")
            comp_df = pd.DataFrame(
                {
                    "Fraction": df_a["Fraction"],
                    "Scenario A (wt%)": df_a["Yield (wt%)"],
                    "Scenario B (wt%)": df_b["Yield (wt%)"],
                    "Difference (B - A) (wt%)": df_b["Yield (wt%)"] - df_a["Yield (wt%)"],
                }
            )
            st.dataframe(
                comp_df.style.format(
                    {"Scenario A (wt%)": "{:.1f}", "Scenario B (wt%)": "{:.1f}",
                     "Difference (B - A) (wt%)": "{:.1f}"}
                ),
                use_container_width=True,
            )

            chart_df = comp_df.melt(
                id_vars="Fraction",
                value_vars=["Scenario A (wt%)", "Scenario B (wt%)"],
                var_name="Scenario",
                value_name="Yield_wt_percent",
            )
            st.bar_chart(
                chart_df.pivot(index="Fraction", columns="Scenario", values="Yield_wt_percent"),
                height=400,
            )

        st.markdown("---")
        st.subheader("Simple Optimization: Maximize a Product Yield")

        target_map = {
            "LPG": 0,
            "Naphtha": 1,
            "Kerosene": 2,
            "Diesel": 3,
            "Gas Oil": 4,
            "Residue": 5,
        }
        target_fraction = st.selectbox(
            "Select product to maximize",
            list(target_map.keys()),
            index=3,  # default Diesel
        )
        n_samples = st.slider(
            "Number of random trials (higher = slower but better)",
            200,
            5000,
            1000,
            step=200,
        )

        if st.button("Run Optimization", key="optimize"):
            rng = np.random.default_rng(123)
            # Randomly sample within ranges
            samples = np.column_stack(
                [
                    rng.uniform(*ranges["API"], n_samples),
                    rng.uniform(*ranges["Sulfur"], n_samples),
                    rng.uniform(*ranges["T10"], n_samples),
                    rng.uniform(*ranges["T50"], n_samples),
                    rng.uniform(*ranges["T90"], n_samples),
                    rng.uniform(*ranges["Furnace_T"], n_samples),
                    rng.uniform(*ranges["Top_T"], n_samples),
                    rng.uniform(*ranges["Bottom_T"], n_samples),
                    rng.uniform(*ranges["Column_P"], n_samples),
                    rng.uniform(*ranges["Reflux_Ratio"], n_samples),
                    rng.uniform(*ranges["Feed_Rate"], n_samples),
                ]
            )

            preds = model.predict(samples)
            # Normalize each row
            preds = np.clip(preds, 0.0, None)
            row_sums = preds.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            preds = preds / row_sums

            idx = target_map[target_fraction]
            target_yields = preds[:, idx]
            best_idx = np.argmax(target_yields)

            best_conditions = samples[best_idx]
            best_yields = preds[best_idx]

            fractions = ["LPG", "Naphtha", "Kerosene", "Diesel", "Gas Oil", "Residue"]
            wt_percent_best = best_yields * 100.0

            st.markdown(
                f"**Best predicted {target_fraction} yield: {wt_percent_best[idx]:.1f} wt%** "
                f"found after {n_samples} random trials."
            )

            cond_names = [
                "API",
                "Sulfur (wt%)",
                "T10 (°C)",
                "T50 (°C)",
                "T90 (°C)",
                "Furnace T (°C)",
                "Top T (°C)",
                "Bottom T (°C)",
                "Column P (bar)",
                "Reflux Ratio",
                "Feed Rate (t/h)",
            ]
            cond_df = pd.DataFrame(
                {
                    "Variable": cond_names,
                    "Optimal Value": best_conditions,
                }
            )
            st.subheader("Optimal Operating Conditions (within chosen bounds)")
            st.dataframe(
                cond_df.style.format({"Optimal Value": "{:.3f}"}),
                use_container_width=True,
            )

            result_df = pd.DataFrame(
                {
                    "Fraction": fractions,
                    "Yield (wt%)": wt_percent_best,
                }
            )
            st.subheader("Resulting Product Slate at Optimum")
            st.dataframe(
                result_df.style.format({"Yield (wt%)": "{:.1f}"}),
                use_container_width=True,
            )

            st.bar_chart(
                result_df.set_index("Fraction"),
                height=350,
            )

            st.caption(
                f"Sum of predicted yields at optimum: **{wt_percent_best.sum():.1f} wt%**."
            )


if __name__ == "__main__":
    main()
