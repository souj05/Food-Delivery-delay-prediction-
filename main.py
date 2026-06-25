from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Food Delivery Delay Predictor API", version="1.0.0")

model = joblib.load("delay_model.pkl") if os.path.exists("delay_model.pkl") else None
feature_columns = joblib.load("feature_columns.pkl") if os.path.exists("feature_columns.pkl") else None


class OrderInput(BaseModel):
    customer_age: int
    customer_loyalty_score: float
    premium_customer_flag: bool
    order_value: float
    number_of_items: int
    promo_code_used: bool
    discount_amount: float
    delivery_fee: float
    tip_amount: float
    final_amount_paid: float
    delivery_distance_km: float
    estimated_delivery_time: int
    weather_severity_score: int
    traffic_level_score: int
    delivery_partner_experience_years: int
    delivery_partner_rating: float
    customer_rating: float
    restaurant_rating: float
    city_tier: int
    order_hour: int
    order_day_of_week: int
    order_month: int
    cancellation_flag: bool
    festival_or_weekend_flag: bool
    refund_requested: bool


class PredictionResponse(BaseModel):
    delayed: bool
    delay_probability: float
    ontime_probability: float
    risk_factors: list[str]


def build_features(data: OrderInput) -> pd.DataFrame:
    d = data.dict()
    d["order_id"] = 0
    d["premium_customer_flag"] = int(d["premium_customer_flag"])
    d["promo_code_used"] = int(d["promo_code_used"])
    d["cancellation_flag"] = int(d["cancellation_flag"])
    d["festival_or_weekend_flag"] = int(d["festival_or_weekend_flag"])
    d["refund_requested"] = int(d["refund_requested"])
    d["traffic_weather_score"] = d["traffic_level_score"] * d["weather_severity_score"]
    d["avg_item_value"] = d["order_value"] / max(d["number_of_items"], 1)
    d["discount_percentage"] = (d["discount_amount"] / (d["order_value"] + 1)) * 100
    d["fee_per_km"] = d["delivery_fee"] / max(d["delivery_distance_km"], 0.1)
    d["delivery_gap"] = 0
    return pd.DataFrame([d])


@app.get("/")
def root():
    return {"message": "Food Delivery Delay Predictor API", "status": "running"}


@app.get("/health")
def health():
    return {"model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(order: OrderInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    df = build_features(order)
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_columns]

    delayed = bool(model.predict(df)[0])
    prob_delayed = float(model.predict_proba(df)[0][1])

    risks = []
    if order.traffic_level_score >= 7:              risks.append("High traffic")
    if order.weather_severity_score >= 7:           risks.append("Severe weather")
    if order.delivery_distance_km > 15:             risks.append("Long distance")
    if order.delivery_partner_experience_years < 2: risks.append("Inexperienced partner")
    if order.refund_requested:                      risks.append("Refund requested")
    if order.tip_amount < 10:                       risks.append("Low tip")
    if order.estimated_delivery_time > 60:          risks.append("Long estimated window")

    return PredictionResponse(
        delayed=delayed,
        delay_probability=round(prob_delayed, 4),
        ontime_probability=round(1 - prob_delayed, 4),
        risk_factors=risks
    )
