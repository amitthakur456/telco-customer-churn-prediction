
# Telco Customer Churn Prediction & Analytics Dashboard

An interactive customer churn analytics dashboard built with Python, Pandas, Plotly, and Streamlit. The project analyzes historical telecom customer data, identifies churn patterns, highlights high-risk customer segments, and provides actionable business recommendations.

## Project Overview

Customer churn is a major challenge for subscription-based businesses. This project explores customer behavior and helps identify segments with higher churn rates.

The project includes:
- Interactive churn analytics dashboard
- Customer-level churn prediction using a trained machine learning pipeline
- Contract, tenure, and payment-method analysis
- High-risk segment identification
- Business insights and retention recommendations
- Filtered data export to CSV

## Features

### Interactive Dashboard
- KPI cards for total customers, churned customers, churn rate, and average monthly charges
- Interactive filters for contract type, internet service, and gender
- Plotly charts for churn distribution, contract type, tenure, and monthly charges
- Filtered customer data table

### Advanced Analytics
- Churn rate by tenure group
- Churn rate by payment method
- Contract-level churn analysis
- Adjustable high-risk threshold
- Download filtered data as CSV

### Business Insights
The dashboard generates observations from the selected historical data and suggests possible retention actions.

Example observations from one dashboard view:
- Selected-data churn rate: 26.58%
- Highest churn by tenure group: 0–12 months
- Highest churn by payment method: Electronic check
- Highest churn by contract: Month-to-month

These values depend on the active dashboard filters and may change.

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly
- Streamlit
- Joblib

## Project Structure

```text
CUSTOMER CHURN PREDICTION/
├── app.py
├── dashboard.py
├── requirements.txt
├── telco_churn_pipeline.joblib
├── 01_Data_Understanding.ipynb
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── LICENSE
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/amitthakur456/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

Activate it on macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the analytics dashboard

```bash
streamlit run dashboard.py
```

### 5. Run the individual customer prediction app

```bash
streamlit run app.py
```

## Dataset

The project uses the Telco Customer Churn dataset. It contains customer demographics, service information, account details, charges, and churn status.

The dataset is used for exploratory analysis, churn pattern visualization, and model-based prediction.

## Business Value

The dashboard can help teams:
- Understand which customer groups have higher churn rates
- Investigate churn patterns across contract and payment types
- Prioritize retention experiments
- Monitor churn metrics under different customer filters

The recommendations are analytical hypotheses and should be validated through business experiments.

## Limitations

- The dashboard analyzes historical data; it does not establish causation.
- Churn patterns may change over time.
- The prediction output depends on the trained model and its input features.
- The high-risk threshold is configurable and is not a universal industry standard.

## Author

**Amit Thakur**

Computer Science & Engineering | Artificial Intelligence & Machine Learning

GitHub: https://github.com/amitthakur456

