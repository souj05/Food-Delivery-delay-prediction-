# 🍕 Food Delivery Delay Prediction

A Machine Learning project that predicts whether a food delivery order will be delayed or not  with an interactive Streamlit dashboard and a FastAPI REST API.

🔗 **Live App:** [https://huggingface.co/spaces/souj05/food-delivery-delay-predictor](https://huggingface.co/spaces/souj05/food-delivery-delay-predictor)

---

## 📌 Problem Statement

Food delivery platforms need to identify orders that are likely to be delayed so that they can take proactive measures and improve customer satisfaction.

---

## 🧠 Machine Learning Pipeline

### 1. 📥 Data Loading
- Loaded the food delivery dataset with 15,000 orders

### 2. 🧹 Data Cleaning
- Handled missing values using median imputation
- Removed unnecessary identifier columns

### 3. ⚙️ Feature Engineering
- `traffic_weather_score` = traffic level × weather severity
- `avg_item_value` = order value / number of items
- `discount_percentage` = discount / order value × 100
- `fee_per_km` = delivery fee / distance (km)

### 4. 🚫 Data Leakage Detection & Removal
- Identified and removed target-dependent variables (`delivery_time_minutes`, `delivery_efficiency_score`)

### 5. 🔀 Data Preprocessing
- Train-test split (80/20)
- Prepared features for machine learning models

### 6. 🤖 Model Training
- Logistic Regression
- Random Forest ✅ *(Best Model)*
- AdaBoost
- XGBoost

### 7. 📊 Model Evaluation

| Metric | Score |
|---|---|
| Accuracy | ~90% |
| Cross-Validation (5-fold) | ~90% |
| Precision | ~0.90 |
| Recall | ~0.90 |
| F1-Score | ~0.90 |

### 8. ✅ Model Validation
- 5-Fold Cross-Validation → ~90%

### 9. 🔧 Hyperparameter Tuning
- GridSearchCV for Random Forest optimization

### 10. 🏆 Feature Importance & Business Insights

| Rank | Feature | Business Meaning |
|---|---|---|
| #1 | Refund Requested | If a customer requested a refund, the delivery was likely delayed or incorrect — strongest delay signal |
| #2 | Tip Amount | Happy customers tip more; low tip = bad experience = likely delayed |
| #3 | Delivery Distance (km) | Longer distances mean higher chances of delay due to traffic and logistics |
| #4 | Customer Loyalty Score | Loyal customers order frequently — the model learned their patterns to detect anomalies |
| #5 | Final Amount Paid | Higher order values suggest more complex orders, increasing delay risk |

---

## 🖥️ Streamlit Dashboard

Interactive dashboard for real-time delay prediction with order summary, risk factors, and probability scores.

> 

🔗 **Live App:** [https://huggingface.co/spaces/souj05/food-delivery-delay-predictor](https://huggingface.co/spaces/souj05/food-delivery-delay-predictor)

---

## ⚡ FastAPI

REST API for serving model predictions programmatically.

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Home — project info |
| GET | `/api/status` | API & model health check |
| POST | `/api/predict-delay` | Predict delivery delay |

The `/api/predict-delay` endpoint returns:
- Delay Prediction (true/false)
- Delay Probability
- On-Time Probability
- Risk Factors

> 📸FastApi
![FastAPI ](FastApi%201.png)
> ![FastAPI ](FastApi%202.png)
> ![FastAPI Endpoint](FastApi%203.png)

### Run Locally
```bash
python -m uvicorn main:app --reload
```
🔗 **Swagger Docs:** `http://127.0.0.1:8000/docs`

---

## 🐳 Docker

```bash
docker build -t food-delay-predictor .
docker run -p 7860:7860 food-delay-predictor
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| API | FastAPI + Uvicorn |
| Container | Docker |
| Deployment | Hugging Face Spaces |
| Serialization | Joblib |

---

## 📂 Project Structure

```
Food-Delivery-delay-prediction/
│
├── app.py                              # Streamlit dashboard
├── main.py                             # FastAPI backend
├── train_model.py                      # Model training script
├── Dockerfile
├── requirements.txt
├── delay_model.pkl                     # Trained Random Forest model
├── feature_columns.pkl                 # Feature column list
├── food_delivery_analytics_cleaned.csv
├── Food_Delivery_Delay_Prediction.ipynb
└── README.md
```

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/souj05/Food-Delivery-delay-prediction-.git
cd Food-Delivery-delay-prediction-

pip install -r requirements.txt

# Streamlit
streamlit run app.py

# FastAPI
python -m uvicorn main:app --reload
```
