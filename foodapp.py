import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Food Delivery Delay Predictor",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #FF6B35, #FF4500);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }
    .metric-card {
        background: white;
        border: 1px solid #f0f0f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,.06);
        text-align: center;
    }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #FF6B35; }
    .metric-card .label { font-size: 0.85rem; color: #666; margin-top: 0.2rem; }
    .prediction-delayed {
        background: linear-gradient(135deg, #FF4444, #CC0000);
        color: white; border-radius: 14px; padding: 1.5rem 2rem; text-align: center;
    }
    .prediction-ontime {
        background: linear-gradient(135deg, #00C851, #007E33);
        color: white; border-radius: 14px; padding: 1.5rem 2rem; text-align: center;
    }
    .prediction-delayed h2, .prediction-ontime h2 { margin: 0; font-size: 1.8rem; }
    .prediction-delayed p,  .prediction-ontime p  { margin: 0.4rem 0 0; opacity: 0.9; }
    .section-header {
        border-left: 4px solid #FF6B35;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem;
        font-weight: 600;
        font-size: 1.05rem;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if os.path.exists("delay_model.pkl") and os.path.exists("feature_columns.pkl"):
        return joblib.load("delay_model.pkl"), joblib.load("feature_columns.pkl")
    return None, None

model, feature_columns = load_model()

st.markdown("""
<div class="main-header">
    <h1>🍕 Food Delivery Delay Predictor</h1>
    <p>ML-powered prediction using Random Forest · ~90% accuracy · Trained on 15,000 orders</p>
</div>
""", unsafe_allow_html=True)

# Sidebar inputs
st.sidebar.markdown("## 📋 Order Details")

st.sidebar.markdown("### 👤 Customer Info")
customer_age           = st.sidebar.slider("Customer Age", 18, 80, 30)
customer_loyalty_score = st.sidebar.slider("Loyalty Score (1–10)", 1.0, 10.0, 5.0, 0.1)
premium_customer_flag  = st.sidebar.selectbox("Premium Customer?", [False, True])
customer_rating        = st.sidebar.slider("Past Customer Rating (1–5)", 1.0, 5.0, 4.0, 0.1)

st.sidebar.markdown("### 📦 Order Info")
order_value       = st.sidebar.number_input("Order Value (₹)", 50, 5000, 500)
number_of_items   = st.sidebar.slider("Number of Items", 1, 20, 3)
promo_code_used   = st.sidebar.selectbox("Promo Code Used?", [False, True])
discount_amount   = st.sidebar.number_input("Discount Amount (₹)", 0, 500, 0)
delivery_fee      = st.sidebar.number_input("Delivery Fee (₹)", 0, 200, 30)
tip_amount        = st.sidebar.number_input("Tip Amount (₹)", 0.0, 300.0, 20.0)
final_amount_paid = st.sidebar.number_input("Final Amount Paid (₹)", 50, 5000, 490)
refund_requested  = st.sidebar.selectbox("Refund Requested?", [False, True])

st.sidebar.markdown("### 🚗 Delivery Info")
delivery_distance_km              = st.sidebar.slider("Distance (km)", 0.5, 30.0, 5.0, 0.5)
estimated_delivery_time           = st.sidebar.slider("Estimated Delivery (min)", 10, 120, 45)
delivery_partner_experience_years = st.sidebar.slider("Partner Experience (yrs)", 0, 15, 3)
delivery_partner_rating           = st.sidebar.slider("Partner Rating (1–5)", 1.0, 5.0, 4.0, 0.1)

st.sidebar.markdown("### 🌦️ Conditions")
weather_severity_score = st.sidebar.slider("Weather Severity (1–10)", 1, 10, 5)
traffic_level_score    = st.sidebar.slider("Traffic Level (1–10)", 1, 10, 5)
city_tier              = st.sidebar.selectbox("City Tier", [1, 2, 3])

st.sidebar.markdown("### 📅 Timing")
order_hour               = st.sidebar.slider("Order Hour (0–23)", 0, 23, 12)
order_day_of_week        = st.sidebar.slider("Day of Week (0=Mon)", 0, 6, 2)
order_month              = st.sidebar.slider("Month (1–12)", 1, 12, 6)
restaurant_rating        = st.sidebar.slider("Restaurant Rating (1–5)", 1.0, 5.0, 4.0, 0.1)
cancellation_flag        = st.sidebar.selectbox("Cancelled?", [False, True])
festival_or_weekend_flag = st.sidebar.selectbox("Festival / Weekend?", [False, True])

def build_input():
    traffic_weather_score = traffic_level_score * weather_severity_score
    avg_item_value        = order_value / max(number_of_items, 1)
    discount_percentage   = (discount_amount / (order_value + 1)) * 100
    fee_per_km            = delivery_fee / max(delivery_distance_km, 0.1)
    return pd.DataFrame([{
        "order_id": 0,
        "customer_age": customer_age,
        "customer_loyalty_score": customer_loyalty_score,
        "premium_customer_flag": int(premium_customer_flag),
        "order_value": order_value,
        "number_of_items": number_of_items,
        "promo_code_used": int(promo_code_used),
        "discount_amount": discount_amount,
        "delivery_fee": delivery_fee,
        "tip_amount": tip_amount,
        "final_amount_paid": final_amount_paid,
        "delivery_distance_km": delivery_distance_km,
        "estimated_delivery_time": estimated_delivery_time,
        "weather_severity_score": weather_severity_score,
        "traffic_level_score": traffic_level_score,
        "delivery_partner_experience_years": delivery_partner_experience_years,
        "delivery_partner_rating": delivery_partner_rating,
        "customer_rating": customer_rating,
        "restaurant_rating": restaurant_rating,
        "city_tier": city_tier,
        "order_hour": order_hour,
        "order_day_of_week": order_day_of_week,
        "order_month": order_month,
        "cancellation_flag": int(cancellation_flag),
        "festival_or_weekend_flag": int(festival_or_weekend_flag),
        "refund_requested": int(refund_requested),
        "traffic_weather_score": traffic_weather_score,
        "avg_item_value": avg_item_value,
        "discount_percentage": discount_percentage,
        "fee_per_km": fee_per_km,
        "delivery_gap": 0,
    }])

tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Insights", "ℹ️ About"])

with tab1:
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown('<div class="section-header">Order Summary</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        for col, val, lbl in zip(
            [r1, r2, r3],
            [f"₹{order_value}", f"{delivery_distance_km}km", f"{estimated_delivery_time}min"],
            ["Order Value", "Distance", "Est. Time"]
        ):
            col.markdown(f'<div class="metric-card"><div class="value">{val}</div><div class="label">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown("")
        r4, r5, r6 = st.columns(3)
        tier_labels = {1: "Metro", 2: "Tier-2", 3: "Tier-3"}
        for col, val, lbl in zip(
            [r4, r5, r6],
            [f"{traffic_level_score}/10", f"{weather_severity_score}/10", tier_labels[city_tier]],
            ["Traffic", "Weather", "City Tier"]
        ):
            col.markdown(f'<div class="metric-card"><div class="value">{val}</div><div class="label">{lbl}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Computed Features</div>', unsafe_allow_html=True)
        tws = traffic_level_score * weather_severity_score
        st.dataframe(pd.DataFrame({
            "Feature": ["Traffic × Weather", "Avg Item Value", "Discount %", "Fee per km"],
            "Value": [f"{tws:.1f}", f"₹{order_value/max(number_of_items,1):.1f}",
                      f"{(discount_amount/(order_value+1))*100:.1f}%",
                      f"₹{delivery_fee/max(delivery_distance_km,0.1):.1f}"]
        }), hide_index=True, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Prediction</div>', unsafe_allow_html=True)

        if model is None:
            st.warning("⚠️ No model found. Place `delay_model.pkl` and `feature_columns.pkl` next to `app.py`.")
            st.info("Running in **demo mode** (rule-based heuristic).")
            risk = sum([
                traffic_level_score >= 7,
                weather_severity_score >= 7,
                delivery_distance_km > 15,
                estimated_delivery_time > 60,
                delivery_partner_experience_years < 2,
                refund_requested * 3,
            ])
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
            st.markdown('<div class="prediction-delayed"><h2>⚠️ DELAYED</h2><p>This order is likely to be delayed</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-ontime"><h2>✅ ON TIME</h2><p>This order is expected to arrive on time</p></div>', unsafe_allow_html=True)

        st.markdown("")
        for label, prob in [("Delay Probability", prob_delayed), ("On-time Probability", 1 - prob_delayed)]:
            c1, c2 = st.columns([3, 1])
            c1.progress(prob)
            c2.markdown(f"**{prob*100:.1f}%**")
            st.caption(label)

        st.markdown('<div class="section-header">Risk Factors</div>', unsafe_allow_html=True)
        risks = []
        if traffic_level_score >= 7:                risks.append("🚗 High traffic")
        if weather_severity_score >= 7:             risks.append("🌧️ Severe weather")
        if delivery_distance_km > 15:               risks.append("📍 Long distance")
        if delivery_partner_experience_years < 2:   risks.append("🔰 Inexperienced partner")
        if refund_requested:                        risks.append("↩️ Refund requested")
        if tip_amount < 10:                         risks.append("💸 Low tip")
        if estimated_delivery_time > 60:            risks.append("⏱️ Long estimated window")
        for r in risks:
            st.markdown(f"- {r}")
        if not risks:
            st.success("No major risk factors!")

with tab2:
    st.markdown("### 📊 Model & Feature Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏆 Top Feature Importances")
        fi = pd.DataFrame({
            "Feature":    ["refund_requested", "tip_amount", "delivery_distance_km", "customer_loyalty_score", "final_amount_paid"],
            "Importance": [0.142, 0.128, 0.115, 0.098, 0.091]
        })
        st.bar_chart(fi.set_index("Feature"))
        st.markdown("""
- **Refund requested** → strong signal of bad experience  
- **Tip amount** → low tip = likely delayed  
- **Delivery distance** → longer = higher delay risk  
- **Loyalty score** → consistent ordering patterns  
- **Final amount paid** → complex high-value orders  
        """)
    with c2:
        st.markdown("#### 📈 Model Performance")
        st.dataframe(pd.DataFrame({
            "Metric": ["Accuracy", "Cross-Val (5-fold)", "Precision", "Recall", "F1-Score"],
            "Value":  ["~90%", "~90%", "~0.90", "~0.90", "~0.90"]
        }), hide_index=True, use_container_width=True)

        st.markdown("#### 🔬 Statistical Tests")
        st.markdown("""
| Test | Result |
|---|---|
| T-test (premium spend) | Significant (p ≈ 0) |
| 95% CI – delivery time | 93.6 – 94.7 min |
| Chi-square (city tier) | No effect (p = 0.84) |
| Final amount skew | Right-skewed (1.17) |
        """)

        st.markdown("#### ⚙️ Models Compared")
        st.dataframe(pd.DataFrame({
            "Model":    ["Logistic Regression", "Random Forest ✅", "AdaBoost", "XGBoost"],
            "Accuracy": ["~90%", "~90%", "~90%", "~90%"],
            "Chosen":   ["", "Most robust", "", ""]
        }), hide_index=True, use_container_width=True)

with tab3:
    st.markdown("### ℹ️ About")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**Goal:** Predict food delivery delays using ML.

**Dataset:** 15,000 orders — customer demographics, order details, delivery conditions, timing.

**Pipeline:**
1. EDA & visualisations  
2. Statistical tests (T-test, Chi-square, Pearson)  
3. Feature engineering (delivery_gap, traffic_weather_score, etc.)  
4. Removed data leakage columns  
5. Trained 4 models → selected Random Forest  
6. GridSearchCV tuning → ~90% accuracy  
        """)
    with c2:
        st.markdown("""
**Stack:**

| Layer | Tech |
|---|---|
| App | Streamlit |
| Model | Random Forest (scikit-learn) |
| Container | Docker |
| Cloud | AWS Elastic Beanstalk |
| Registry | Amazon ECR |

**Files needed:**