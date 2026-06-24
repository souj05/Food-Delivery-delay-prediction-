import streamlit as st
import pandas as pd
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Food Delivery Delay Predictor", page_icon="🍕", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_model():
    if os.path.exists("delay_model.pkl") and os.path.exists("feature_columns.pkl"):
        return joblib.load("delay_model.pkl"), joblib.load("feature_columns.pkl")
    return None, None

model, feature_columns = load_model()

st.title("🍕 Food Delivery Delay Predictor")
st.caption("ML-powered prediction using Random Forest · ~90% accuracy · Trained on 15,000 orders")

# Sidebar
st.sidebar.header("📋 Order Details")

st.sidebar.subheader("👤 Customer Info")
customer_age           = st.sidebar.slider("Customer Age", 18, 80, 30)
customer_loyalty_score = st.sidebar.slider("Loyalty Score (1–10)", 1.0, 10.0, 5.0, 0.1)
premium_customer_flag  = st.sidebar.selectbox("Premium Customer?", [False, True])
customer_rating        = st.sidebar.slider("Past Customer Rating (1–5)", 1.0, 5.0, 4.0, 0.1)

st.sidebar.subheader("📦 Order Info")
order_value       = st.sidebar.number_input("Order Value (₹)", 50, 5000, 500)
number_of_items   = st.sidebar.slider("Number of Items", 1, 20, 3)
promo_code_used   = st.sidebar.selectbox("Promo Code Used?", [False, True])
discount_amount   = st.sidebar.number_input("Discount Amount (₹)", 0, 500, 0)
delivery_fee      = st.sidebar.number_input("Delivery Fee (₹)", 0, 200, 30)
tip_amount        = st.sidebar.number_input("Tip Amount (₹)", 0.0, 300.0, 20.0)
final_amount_paid = st.sidebar.number_input("Final Amount Paid (₹)", 50, 5000, 490)
refund_requested  = st.sidebar.selectbox("Refund Requested?", [False, True])

st.sidebar.subheader("🚗 Delivery Info")
delivery_distance_km              = st.sidebar.slider("Distance (km)", 0.5, 30.0, 5.0, 0.5)
estimated_delivery_time           = st.sidebar.slider("Estimated Delivery (min)", 10, 120, 45)
delivery_partner_experience_years = st.sidebar.slider("Partner Experience (yrs)", 0, 15, 3)
delivery_partner_rating           = st.sidebar.slider("Partner Rating (1–5)", 1.0, 5.0, 4.0, 0.1)

st.sidebar.subheader("🌦️ Conditions")
weather_severity_score = st.sidebar.slider("Weather Severity (1–10)", 1, 10, 5)
traffic_level_score    = st.sidebar.slider("Traffic Level (1–10)", 1, 10, 5)
city_tier              = st.sidebar.selectbox("City Tier", [1, 2, 3])

st.sidebar.subheader("📅 Timing")
order_hour               = st.sidebar.slider("Order Hour (0–23)", 0, 23, 12)
order_day_of_week        = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)
order_month              = st.sidebar.slider("Month (1–12)", 1, 12, 6)
restaurant_rating        = st.sidebar.slider("Restaurant Rating (1–5)", 1.0, 5.0, 4.0, 0.1)
cancellation_flag        = st.sidebar.selectbox("Cancelled?", [False, True])
festival_or_weekend_flag = st.sidebar.selectbox("Festival / Weekend?", [False, True])

def build_input():
    return pd.DataFrame([{
        "order_id": 0, "customer_age": customer_age, "customer_loyalty_score": customer_loyalty_score,
        "premium_customer_flag": int(premium_customer_flag), "order_value": order_value,
        "number_of_items": number_of_items, "promo_code_used": int(promo_code_used),
        "discount_amount": discount_amount, "delivery_fee": delivery_fee, "tip_amount": tip_amount,
        "final_amount_paid": final_amount_paid, "delivery_distance_km": delivery_distance_km,
        "estimated_delivery_time": estimated_delivery_time, "weather_severity_score": weather_severity_score,
        "traffic_level_score": traffic_level_score,
        "delivery_partner_experience_years": delivery_partner_experience_years,
        "delivery_partner_rating": delivery_partner_rating, "customer_rating": customer_rating,
        "restaurant_rating": restaurant_rating, "city_tier": city_tier, "order_hour": order_hour,
        "order_day_of_week": order_day_of_week, "order_month": order_month,
        "cancellation_flag": int(cancellation_flag), "festival_or_weekend_flag": int(festival_or_weekend_flag),
        "refund_requested": int(refund_requested),
        "traffic_weather_score": traffic_level_score * weather_severity_score,
        "avg_item_value": order_value / max(number_of_items, 1),
        "discount_percentage": (discount_amount / (order_value + 1)) * 100,
        "fee_per_km": delivery_fee / max(delivery_distance_km, 0.1),
        "delivery_gap": 0,
    }])

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Insights", "ℹ️ About"])

with tab1:
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("Order Summary")
        r1, r2, r3 = st.columns(3)
        r1.metric("Order Value", f"₹{order_value}")
        r2.metric("Distance", f"{delivery_distance_km} km")
        r3.metric("Est. Time", f"{estimated_delivery_time} min")

        r4, r5, r6 = st.columns(3)
        tier_labels = {1: "Metro", 2: "Tier-2", 3: "Tier-3"}
        r4.metric("Traffic", f"{traffic_level_score}/10")
        r5.metric("Weather", f"{weather_severity_score}/10")
        r6.metric("City Tier", tier_labels[city_tier])

        st.subheader("Computed Features")
        st.dataframe(pd.DataFrame({
            "Feature": ["Traffic × Weather", "Avg Item Value", "Discount %", "Fee per km"],
            "Value": [
                f"{traffic_level_score * weather_severity_score:.1f}",
                f"₹{order_value / max(number_of_items, 1):.1f}",
                f"{(discount_amount / (order_value + 1)) * 100:.1f}%",
                f"₹{delivery_fee / max(delivery_distance_km, 0.1):.1f}"
            ]
        }), hide_index=True, use_container_width=True)

    with col_right:
        st.subheader("Prediction")

        if model is None:
            st.warning("⚠️ No model found. Place `delay_model.pkl` and `feature_columns.pkl` next to `app.py`.")
            st.info("Running in **demo mode** (rule-based heuristic).")
            risk = sum([traffic_level_score >= 7, weather_severity_score >= 7,
                        delivery_distance_km > 15, estimated_delivery_time > 60,
                        delivery_partner_experience_years < 2, refund_requested * 3])
            delayed = risk >= 4
            prob_delayed = min(0.95, risk * 0.12)
        else:
            input_df = build_input()
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[feature_columns]
            delayed      = bool(model.predict(input_df)[0])
            prob_delayed = model.predict_proba(input_df)[0][1]

        if delayed:
            st.error("## ⚠️ DELAYED\nThis order is likely to be delayed.")
        else:
            st.success("## ✅ ON TIME\nThis order is expected to arrive on time.")

        for label, prob in [("Delay Probability", prob_delayed), ("On-time Probability", 1 - prob_delayed)]:
            c1, c2 = st.columns([3, 1])
            c1.progress(prob)
            c2.write(f"**{prob*100:.1f}%**")
            st.caption(label)

        st.subheader("Risk Factors")
        risks = []
        if traffic_level_score >= 7:              risks.append("🚗 High traffic")
        if weather_severity_score >= 7:           risks.append("🌧️ Severe weather")
        if delivery_distance_km > 15:             risks.append("📍 Long distance")
        if delivery_partner_experience_years < 2: risks.append("🔰 Inexperienced partner")
        if refund_requested:                      risks.append("↩️ Refund requested")
        if tip_amount < 10:                       risks.append("💸 Low tip")
        if estimated_delivery_time > 60:          risks.append("⏱️ Long estimated window")
        for r in risks:
            st.write(f"- {r}")
        if not risks:
            st.success("No major risk factors!")

with tab2:
    st.subheader("📊 Model & Feature Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏆 Top Feature Importances")
        fi = pd.DataFrame({
            "Feature":    ["refund_requested", "tip_amount", "delivery_distance_km", "customer_loyalty_score", "final_amount_paid"],
            "Importance": [0.142, 0.128, 0.115, 0.098, 0.091]
        })
        st.bar_chart(fi.set_index("Feature"))
    with c2:
        st.markdown("#### 📈 Model Performance")
        st.dataframe(pd.DataFrame({
            "Metric": ["Accuracy", "Cross-Val (5-fold)", "Precision", "Recall", "F1-Score"],
            "Value":  ["~90%", "~90%", "~0.90", "~0.90", "~0.90"]
        }), hide_index=True, use_container_width=True)

        st.markdown("#### ⚙️ Models Compared")
        st.dataframe(pd.DataFrame({
            "Model":    ["Logistic Regression", "Random Forest ✅", "AdaBoost", "XGBoost"],
            "Accuracy": ["~90%", "~90%", "~90%", "~90%"],
            "Chosen":   ["", "Most robust", "", ""]
        }), hide_index=True, use_container_width=True)

with tab3:
    st.markdown("---")
    st.markdown("Developed with ❤️ by **Sowjanya Tadimarri** · Made using Streamlit")
