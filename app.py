import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# LOAD SAVED MODEL
# =========================================================

MODEL_PATH = Path(__file__).parent / "telco_churn_pipeline.joblib"


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)

    pipeline = bundle["pipeline"]
    threshold = bundle.get("threshold", 0.60)
    feature_columns = bundle["feature_columns"]

    return pipeline, threshold, feature_columns


try:
    pipeline, threshold, feature_columns = load_model()

except FileNotFoundError:
    st.error(
        "Model file nahi mili. "
        "telco_churn_pipeline.joblib ko app.py ke same folder mein rakho."
    )
    st.stop()

except Exception as error:
    st.error(f"Model load karte waqt error: {error}")
    st.stop()


# =========================================================
# APP HEADER
# =========================================================

st.title("📊 Telco Customer Churn Prediction")

st.write(
    "Customer details enter karo aur churn probability "
    "aur customer risk category predict karo."
)

st.divider()


# =========================================================
# INPUT OPTIONS
# =========================================================

categorical_options = {
    "gender": ["Female", "Male"],

    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],

    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],

    "Contract": [
        "Month-to-month",
        "One year",
        "Two year"
    ],

    "PaperlessBilling": ["Yes", "No"],

    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
}


# =========================================================
# CUSTOMER INPUT FORM
# =========================================================

st.subheader("Customer Information")

customer_data = {}

with st.form("customer_input_form"):

    col1, col2, col3 = st.columns(3)

    for index, feature in enumerate(feature_columns):

        current_column = [col1, col2, col3][index % 3]

        with current_column:

            # Categorical features
            if feature in categorical_options:

                options = categorical_options[feature]

                customer_data[feature] = st.selectbox(
                    label=feature,
                    options=options,
                    key=f"input_{feature}"
                )

            # SeniorCitizen: commonly 0 or 1
            elif feature == "SeniorCitizen":

                customer_data[feature] = st.selectbox(
                    "Senior Citizen",
                    options=[0, 1],
                    format_func=lambda x: (
                        "Yes" if x == 1 else "No"
                    ),
                    key=f"input_{feature}"
                )

            # Tenure
            elif feature == "tenure":

                customer_data[feature] = st.slider(
                    "Tenure (Months)",
                    min_value=0,
                    max_value=72,
                    value=12,
                    key=f"input_{feature}"
                )

            # Monthly Charges
            elif feature == "MonthlyCharges":

                customer_data[feature] = st.number_input(
                    "Monthly Charges ($)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=70.0,
                    step=1.0,
                    key=f"input_{feature}"
                )

            # Total Charges
            elif feature == "TotalCharges":

                customer_data[feature] = st.number_input(
                    "Total Charges ($)",
                    min_value=0.0,
                    max_value=100000.0,
                    value=840.0,
                    step=10.0,
                    key=f"input_{feature}"
                )

            # Other numeric features
            else:

                customer_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    key=f"input_{feature}"
                )

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if submitted:

    # Convert user input to DataFrame
    customer_df = pd.DataFrame(
        [customer_data],
        columns=feature_columns
    )

    try:

        # Predict churn probability
        churn_probability = pipeline.predict_proba(
            customer_df
        )[0][1]

        # Threshold-based prediction
        churn_prediction = (
            churn_probability >= threshold
        )

        # Risk category
        if churn_probability < 0.30:
            risk_level = "Low Risk"
            risk_color = "🟢"

        elif churn_probability < 0.60:
            risk_level = "Medium Risk"
            risk_color = "🟡"

        else:
            risk_level = "High Risk"
            risk_color = "🔴"

        # =================================================
        # DISPLAY RESULTS
        # =================================================

        st.divider()
        st.subheader("Prediction Result")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:
            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        with metric2:
            prediction_label = (
                "Churn" if churn_prediction else "No Churn"
            )

            st.metric(
                "Prediction",
                prediction_label
            )

        with metric3:
            st.metric(
                "Risk Level",
                f"{risk_color} {risk_level}"
            )

        # Probability bar
        st.write("Churn Probability")

        st.progress(
            float(churn_probability)
        )

        # =================================================
        # BUSINESS RECOMMENDATION
        # =================================================

        st.subheader("Recommended Action")

        if risk_level == "High Risk":

            st.error(
                "High-risk customer: retention team ko contact "
                "karna chahiye. Personalized offer, service review "
                "ya customer feedback call consider karo."
            )

        elif risk_level == "Medium Risk":

            st.warning(
                "Medium-risk customer: customer satisfaction "
                "check karo. Relevant service benefits aur "
                "support options communicate karo."
            )

        else:

            st.success(
                "Low-risk customer: existing service experience "
                "maintain karo aur regular engagement continue karo."
            )

        # =================================================
        # SHOW CUSTOMER INPUT
        # =================================================

        with st.expander("View Customer Input"):

            st.dataframe(
                customer_df,
                use_container_width=True
            )

    except Exception as error:

        st.error(
            f"Prediction ke waqt error aaya: {error}"
        )