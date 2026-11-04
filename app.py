import joblib
import pandas as pd
import streamlit as st

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# MODEL PATH
# Keep model file in the same folder as app.py
# =========================================================

MODEL_PATH = Path(__file__).parent / "telco_churn_pipeline.joblib"


# =========================================================
# SIMPLEIMPUTER COMPATIBILITY PATCH
# =========================================================

def fix_imputer_attributes(estimator):
    """
    Compatibility workaround for loaded SimpleImputer objects.
    Restore _fill_dtype if it is missing.
    """

    # SimpleImputer
    if isinstance(estimator, SimpleImputer):

        if (
            not hasattr(estimator, "_fill_dtype")
            and hasattr(estimator, "statistics_")
        ):
            estimator._fill_dtype = estimator.statistics_.dtype

    # Pipeline
    if isinstance(estimator, Pipeline):

        for _, step in estimator.steps:

            if step is not None:
                fix_imputer_attributes(step)

    # ColumnTransformer
    if isinstance(estimator, ColumnTransformer):

        if hasattr(estimator, "transformers_"):
            transformers = estimator.transformers_

        else:
            transformers = estimator.transformers

        for _, transformer, _ in transformers:

            if isinstance(transformer, str):
                continue

            fix_imputer_attributes(transformer)

    # FeatureUnion and similar nested estimators
    if hasattr(estimator, "transformer_list"):

        for _, transformer in estimator.transformer_list:

            fix_imputer_attributes(transformer)


# =========================================================
# LOAD SAVED MODEL
# =========================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "telco_churn_pipeline.joblib ko app.py ke same "
            "folder mein rakho."
        )

    bundle = joblib.load(MODEL_PATH)

    if not isinstance(bundle, dict) or "pipeline" not in bundle:

        raise ValueError(
            "Model bundle mein 'pipeline' key nahi mili."
        )

    pipeline = bundle["pipeline"]

    threshold = float(
        bundle.get("threshold", 0.60)
    )

    feature_columns = bundle.get("feature_columns")

    if not feature_columns:

        raise ValueError(
            "Model bundle mein 'feature_columns' nahi mile."
        )

    # Apply compatibility patch
    fix_imputer_attributes(pipeline)

    return pipeline, threshold, feature_columns


# =========================================================
# LOAD MODEL SAFELY
# =========================================================

try:

    pipeline, threshold, feature_columns = load_model()

except Exception as error:

    st.error(
        f"Model load karte waqt error aaya: {error}"
    )

    st.stop()


# =========================================================
# APP HEADER
# =========================================================

st.title(
    "📊 Telco Customer Churn Prediction"
)

st.write(
    "Customer details enter karo aur churn probability, "
    "prediction aur risk category dekho."
)

st.divider()


# =========================================================
# INPUT OPTIONS
# =========================================================

categorical_options = {

    "gender": [
        "Female",
        "Male"
    ],

    "Partner": [
        "Yes",
        "No"
    ],

    "Dependents": [
        "Yes",
        "No"
    ],

    "PhoneService": [
        "Yes",
        "No"
    ],

    "MultipleLines": [
        "No",
        "Yes",
        "No phone service"
    ],

    "InternetService": [
        "DSL",
        "Fiber optic",
        "No"
    ],

    "OnlineSecurity": [
        "Yes",
        "No",
        "No internet service"
    ],

    "OnlineBackup": [
        "Yes",
        "No",
        "No internet service"
    ],

    "DeviceProtection": [
        "Yes",
        "No",
        "No internet service"
    ],

    "TechSupport": [
        "Yes",
        "No",
        "No internet service"
    ],

    "StreamingTV": [
        "Yes",
        "No",
        "No internet service"
    ],

    "StreamingMovies": [
        "Yes",
        "No",
        "No internet service"
    ],

    "Contract": [
        "Month-to-month",
        "One year",
        "Two year"
    ],

    "PaperlessBilling": [
        "Yes",
        "No"
    ],

    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ],
}


# =========================================================
# CUSTOMER INPUT FORM
# =========================================================

st.subheader(
    "Customer Information"
)

customer_data = {}


with st.form("customer_input_form"):

    col1, col2, col3 = st.columns(3)

    for index, feature in enumerate(feature_columns):

        current_column = [
            col1,
            col2,
            col3
        ][index % 3]

        with current_column:

            # ---------------------------------------------
            # CATEGORICAL FEATURES
            # ---------------------------------------------

            if feature in categorical_options:

                customer_data[feature] = st.selectbox(
                    label=feature,
                    options=categorical_options[feature],
                    key=f"input_{feature}",
                )

            # ---------------------------------------------
            # SENIOR CITIZEN
            # ---------------------------------------------

            elif feature == "SeniorCitizen":

                customer_data[feature] = st.selectbox(
                    "Senior Citizen",
                    options=[0, 1],
                    format_func=lambda value: (
                        "Yes" if value == 1 else "No"
                    ),
                    key=f"input_{feature}",
                )

            # ---------------------------------------------
            # TENURE
            # ---------------------------------------------

            elif feature == "tenure":

                customer_data[feature] = st.slider(
                    "Tenure (Months)",
                    min_value=0,
                    max_value=72,
                    value=12,
                    key=f"input_{feature}",
                )

            # ---------------------------------------------
            # MONTHLY CHARGES
            # ---------------------------------------------

            elif feature == "MonthlyCharges":

                customer_data[feature] = st.number_input(
                    "Monthly Charges ($)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=70.0,
                    step=1.0,
                    key=f"input_{feature}",
                )

            # ---------------------------------------------
            # TOTAL CHARGES
            # ---------------------------------------------

            elif feature == "TotalCharges":

                customer_data[feature] = st.number_input(
                    "Total Charges ($)",
                    min_value=0.0,
                    max_value=100000.0,
                    value=840.0,
                    step=10.0,
                    key=f"input_{feature}",
                )

            # ---------------------------------------------
            # OTHER NUMERIC FEATURES
            # ---------------------------------------------

            else:

                customer_data[feature] = st.number_input(
                    feature,
                    value=0.0,
                    key=f"input_{feature}",
                )

    # ---------------------------------------------
    # SUBMIT BUTTON
    # ---------------------------------------------

    submitted = st.form_submit_button(
        "Predict Churn Risk",
        use_container_width=True,
    )


# =========================================================
# PREDICTION
# =========================================================

if submitted:

    # Create DataFrame in the exact feature order
    customer_df = pd.DataFrame(
        [customer_data],
        columns=feature_columns,
    )

    try:

        # ---------------------------------------------
        # PREDICT CHURN PROBABILITY
        # ---------------------------------------------

        churn_probability = float(
            pipeline.predict_proba(
                customer_df
            )[0][1]
        )

        # ---------------------------------------------
        # THRESHOLD-BASED PREDICTION
        # ---------------------------------------------

        churn_prediction = (
            churn_probability >= threshold
        )

        # ---------------------------------------------
        # RISK CATEGORY
        # ---------------------------------------------

        if churn_probability < 0.30:

            risk_level = "Low Risk"
            risk_emoji = "🟢"

        elif churn_probability < 0.60:

            risk_level = "Medium Risk"
            risk_emoji = "🟡"

        else:

            risk_level = "High Risk"
            risk_emoji = "🔴"

        # =================================================
        # DISPLAY RESULTS
        # =================================================

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        metric1, metric2, metric3 = st.columns(3)

        # ---------------------------------------------
        # CHURN PROBABILITY
        # ---------------------------------------------

        with metric1:

            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%",
            )

        # ---------------------------------------------
        # PREDICTION
        # ---------------------------------------------

        with metric2:

            prediction_label = (
                "Churn"
                if churn_prediction
                else "No Churn"
            )

            st.metric(
                "Prediction",
                prediction_label,
            )

        # ---------------------------------------------
        # RISK LEVEL
        # ---------------------------------------------

        with metric3:

            st.metric(
                "Risk Level",
                f"{risk_emoji} {risk_level}",
            )

        # ---------------------------------------------
        # PROBABILITY BAR
        # ---------------------------------------------

        st.write(
            "Churn Probability"
        )

        st.progress(
            min(
                max(churn_probability, 0.0),
                1.0
            )
        )

        # =================================================
        # BUSINESS RECOMMENDATION
        # =================================================

        st.subheader(
            "Recommended Action"
        )

        if risk_level == "High Risk":

            st.error(
                "High-risk customer: retention team customer "
                "se contact kare. Personalized offer, service "
                "review ya customer feedback call consider kare."
            )

        elif risk_level == "Medium Risk":

            st.warning(
                "Medium-risk customer: customer satisfaction "
                "check kare. Relevant service benefits aur "
                "support options communicate kare."
            )

        else:

            st.success(
                "Low-risk customer: existing service experience "
                "maintain kare aur regular engagement continue kare."
            )

        # =================================================
        # SHOW CUSTOMER INPUT
        # =================================================

        with st.expander(
            "View Customer Input"
        ):

            st.dataframe(
                customer_df,
                use_container_width=True,
            )

    except Exception as error:

        st.error(
            f"Prediction ke waqt error aaya: {error}"
        )