# 🍕 Food Delivery Delay Prediction

## Problem Statement

Food delivery platforms need to identify orders that are likely to be delayed so that they can take proactive measures and improve customer satisfaction. This project builds an ML pipeline to predict delivery delays in real time.

---

## Machine Learning Pipeline

### 1. Data Loading
- Loaded the food delivery dataset with 15,000 orders

### 2. Data Cleaning
- Handled missing values using median imputation
- Removed unnecessary identifier columns

### 3. Feature Engineering
- `traffic_weather_score` = traffic level × weather severity
- `avg_item_value` = order value / number of items
- `discount_percentage` = discount / order value × 100
- `fee_per_km` = delivery fee / distance (km)

### 4. Data Leakage Detection & Removal
- Identified leakage-related features
- Removed target-dependent variables (`delivery_time_minutes`, `delivery_efficiency_score`) and retrained models

### 5. Data Preprocessing
- Train-test split (80/20)
- Prepared features for machine learning models

### 6. Model Training
- Logistic Regression
- Random Forest ✅ *(Best Model)*
- AdaBoost
- XGBoost

### 7. Model Evaluation

| Metric | Score |
|---|---|
| Accuracy | ~90% |
| Precision | ~0.90 |
| Recall | ~0.90 |
| F1-Score | ~0.90 |

### 8. Model Validation
- 5-Fold Cross-Validation → ~90%

### 9. Hyperparameter Tuning
- GridSearchCV for Random Forest optimization

### 10. Feature Importance Analysis
Top features influencing delivery delays:
1. Refund Requested
2. Tip Amount
3. Delivery Distance (km)
4. Customer Loyalty Score
5. Final Amount Paid

### 11. Business Insights
- High traffic + severe weather significantly increases delay risk
- Low tip amount is a strong indicator of delayed delivery
- Longer delivery distances correlate with higher delay probability
- Inexperienced delivery partners contribute to delays

---

## Streamlit Application

Interactive dashboard for real-time delivery delay prediction.

> 📸 *(Add Streamlit screenshots here)*

🔗 **Live App:** [https://huggingface.co/spaces/souj05/food-delivery-delay-predictor](https://huggingface.co/spaces/souj05/food-delivery-delay-predictor)

---


## Docker

```bash
docker build -t food-delay-predictor .
docker run -p 7860:7860 food-delay-predictor
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| ML | Scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Container | Docker |
| Deployment | Hugging Face Spaces |

---

## Project Structure

```
Food-Delivery-delay-prediction/
│
├── app.py                              # Streamlit dashboard
├── train_model.py                      # Model training script
├── Dockerfile
├── requirements.txt
├── delay_model.pkl                     # Trained Random Forest model
├── feature_columns.pkl                 # Feature column list
├── Food_Delivery_Delay_Prediction.ipynb
└── README.md
```

---

## How to Run Locally

```bash
git clone https://github.com/souj05/Food-Delivery-delay-prediction-.git
cd Food-Delivery-delay-prediction-

pip install -r requirements.txt

# Streamlit
streamlit run app.py
```
