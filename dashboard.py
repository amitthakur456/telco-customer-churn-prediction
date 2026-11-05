import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)
# -----------------------------
# Professional UI Styling
# -----------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 16px;
        border-radius: 12px;
    }

    div[data-testid="stPlotlyChart"] {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 8px;
    }

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    h1, h2, h3 {
        letter-spacing: -0.4px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📊 Customer Churn Analytics")
st.markdown(
    "Explore churn patterns, identify high-risk customer segments, "
    "and discover actionable retention opportunities."
)

st.caption(
    "TELCO CUSTOMER CHURN | HISTORICAL DATA ANALYSIS"
)
# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = Path(__file__).parent / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Remove rows with missing TotalCharges
    df = df.dropna(subset=["TotalCharges"])

    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error(
        "CSV file nahi mili. CSV ko dashboard.py ke same folder mein rakho."
    )
    st.stop()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Dashboard Filters")

contract_options = sorted(df["Contract"].dropna().unique())

selected_contracts = st.sidebar.multiselect(
    "Contract Type",
    options=contract_options,
    default=contract_options
)

internet_options = sorted(df["InternetService"].dropna().unique())

selected_internet = st.sidebar.multiselect(
    "Internet Service",
    options=internet_options,
    default=internet_options
)

gender_options = sorted(df["gender"].dropna().unique())

selected_gender = st.sidebar.multiselect(
    "Gender",
    options=gender_options,
    default=gender_options
)

# -----------------------------
# Apply filters
# -----------------------------
filtered_df = df[
    df["Contract"].isin(selected_contracts)
    & df["InternetService"].isin(selected_internet)
    & df["gender"].isin(selected_gender)
]

if filtered_df.empty:
    st.warning("Selected filters ke liye koi customer data nahi mila.")
    st.stop()

# -----------------------------
# Calculate KPIs
# -----------------------------
total_customers = len(filtered_df)

churned_customers = (
    filtered_df["Churn"].astype(str).str.strip().str.lower() == "yes"
).sum()

churn_rate = (churned_customers / total_customers) * 100

average_monthly_charges = filtered_df["MonthlyCharges"].mean()

# -----------------------------
# KPI cards
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Churned Customers",
    f"{churned_customers:,}"
)

col3.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)

col4.metric(
    "Avg. Monthly Charges",
    f"${average_monthly_charges:.2f}"
)

st.divider()

# -----------------------------
# Churn distribution
# -----------------------------
st.subheader("Customer Churn Distribution")

churn_counts = (
    filtered_df["Churn"]
    .value_counts()
    .rename_axis("Churn")
    .reset_index(name="Customers")
)

fig_churn = px.pie(
    churn_counts,
    names="Churn",
    values="Customers",
    hole=0.45,
    title="Churn vs No Churn"
)

st.plotly_chart(
    fig_churn,
    use_container_width=True
)

# -----------------------------
# Churn by contract
# -----------------------------
st.subheader("Churn by Contract Type")

contract_churn = (
    filtered_df.groupby(["Contract", "Churn"])
    .size()
    .reset_index(name="Customers")
)

fig_contract = px.bar(
    contract_churn,
    x="Contract",
    y="Customers",
    color="Churn",
    barmode="group",
    title="Customer Churn Across Contract Types"
)

st.plotly_chart(
    fig_contract,
    use_container_width=True
)

# -----------------------------
# Tenure distribution
# -----------------------------
st.subheader("Customer Tenure Distribution")

fig_tenure = px.histogram(
    filtered_df,
    x="tenure",
    color="Churn",
    nbins=30,
    title="Tenure Distribution by Churn Status",
    labels={
        "tenure": "Tenure (Months)",
        "count": "Number of Customers"
    }
)

st.plotly_chart(
    fig_tenure,
    use_container_width=True
)

# -----------------------------
# Monthly charges comparison
# -----------------------------
st.subheader("Monthly Charges vs Churn")

fig_charges = px.box(
    filtered_df,
    x="Churn",
    y="MonthlyCharges",
    color="Churn",
    title="Monthly Charges Distribution",
    labels={
        "MonthlyCharges": "Monthly Charges ($)",
        "Churn": "Churn Status"
    }
)

st.plotly_chart(
    fig_charges,
    use_container_width=True
)

# -----------------------------
# Customer data table
# -----------------------------
st.subheader("Filtered Customer Data")

columns_to_show = [
    "customerID",
    "gender",
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

st.dataframe(
    filtered_df[columns_to_show],
    use_container_width=True
)

st.caption(
    "Dashboard historical CSV data ko analyze karta hai. "
    "Yeh model ki individual customer prediction se alag hai."
)
# -----------------------------
# Advanced Analytics
# -----------------------------

st.divider()
st.header("🔍 Advanced Churn Analytics")

# Create tenure groups
tenure_bins = [0, 12, 24, 48, 72]
tenure_labels = [
    "0–12 Months",
    "13–24 Months",
    "25–48 Months",
    "49–72 Months"
]

filtered_df = filtered_df.copy()

filtered_df["TenureGroup"] = pd.cut(
    filtered_df["tenure"],
    bins=tenure_bins,
    labels=tenure_labels,
    include_lowest=True
)

# Convert Churn to a consistent Yes/No format
filtered_df["ChurnStatus"] = (
    filtered_df["Churn"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": "Yes", "no": "No"})
)

# -----------------------------
# Churn rate by tenure group
# -----------------------------

st.subheader("Churn Rate by Tenure Group")

tenure_analysis = (
    filtered_df.groupby("TenureGroup", observed=False)
    .agg(
        TotalCustomers=("ChurnStatus", "count"),
        ChurnedCustomers=(
            "ChurnStatus",
            lambda x: (x == "Yes").sum()
        )
    )
    .reset_index()
)

tenure_analysis["ChurnRate"] = (
    tenure_analysis["ChurnedCustomers"]
    / tenure_analysis["TotalCustomers"].replace(0, pd.NA)
    * 100
)

fig_tenure_churn = px.bar(
    tenure_analysis,
    x="TenureGroup",
    y="ChurnRate",
    text="ChurnRate",
    title="Churn Rate Across Tenure Groups",
    labels={
        "TenureGroup": "Customer Tenure",
        "ChurnRate": "Churn Rate (%)"
    }
)

fig_tenure_churn.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

st.plotly_chart(
    fig_tenure_churn,
    use_container_width=True
)

# -----------------------------
# Churn rate by payment method
# -----------------------------

st.subheader("Churn Rate by Payment Method")

payment_analysis = (
    filtered_df.groupby("PaymentMethod")
    .agg(
        TotalCustomers=("ChurnStatus", "count"),
        ChurnedCustomers=(
            "ChurnStatus",
            lambda x: (x == "Yes").sum()
        )
    )
    .reset_index()
)

payment_analysis["ChurnRate"] = (
    payment_analysis["ChurnedCustomers"]
    / payment_analysis["TotalCustomers"].replace(0, pd.NA)
    * 100
)

fig_payment = px.bar(
    payment_analysis,
    x="PaymentMethod",
    y="ChurnRate",
    color="PaymentMethod",
    title="Churn Rate by Payment Method",
    labels={
        "PaymentMethod": "Payment Method",
        "ChurnRate": "Churn Rate (%)"
    }
)

st.plotly_chart(
    fig_payment,
    use_container_width=True
)

# -----------------------------
# High-risk customer segments
# -----------------------------

st.subheader("⚠️ High-Risk Customer Segments")

risk_analysis = (
    filtered_df.groupby("Contract")
    .agg(
        TotalCustomers=("ChurnStatus", "count"),
        ChurnedCustomers=(
            "ChurnStatus",
            lambda x: (x == "Yes").sum()
        )
    )
    .reset_index()
)

risk_analysis["ChurnRate"] = (
    risk_analysis["ChurnedCustomers"]
    / risk_analysis["TotalCustomers"].replace(0, pd.NA)
    * 100
)

# Segments with churn rate >= 40% are flagged
risk_threshold = st.slider(
    "High-Risk Churn Rate Threshold (%)",
    min_value=10,
    max_value=80,
    value=40,
    step=5
)

risk_analysis["RiskLevel"] = risk_analysis["ChurnRate"].apply(
    lambda rate: "High Risk"
    if pd.notna(rate) and rate >= risk_threshold
    else "Lower Risk"
)

high_risk_segments = risk_analysis[
    risk_analysis["RiskLevel"] == "High Risk"
].sort_values(
    by="ChurnRate",
    ascending=False
)

high_risk_segments = risk_analysis[
    risk_analysis["RiskLevel"] == "High Risk"
].sort_values(
    by="ChurnRate",
    ascending=False
)

if high_risk_segments.empty:
    st.info(
        "Current filters ke under 40% ya usse zyada churn-rate wala "
        "contract segment nahi mila."
    )
else:
    st.dataframe(
        high_risk_segments,
        use_container_width=True
    )

# -----------------------------
# Download filtered data
# -----------------------------

st.subheader("📥 Download Filtered Customer Data")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv_data,
    file_name="filtered_churn_data.csv",
    mime="text/csv"
)
# -----------------------------
# Business Insights
# -----------------------------

st.divider()
st.header("💡 Business Insights & Recommendations")

# Calculate churn rates
overall_churn_rate = (
    filtered_df["ChurnStatus"].eq("Yes").mean() * 100
)

# Highest churn tenure group
valid_tenure = tenure_analysis.dropna(subset=["ChurnRate"])

# Highest churn payment method
valid_payment = payment_analysis.dropna(subset=["ChurnRate"])

# Highest churn contract
valid_contract = risk_analysis.dropna(subset=["ChurnRate"])

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    st.subheader("📌 Key Insights")

    st.metric(
        "Churn Rate in Selected Data",
        f"{overall_churn_rate:.2f}%"
    )

    if not valid_tenure.empty:
        highest_tenure = valid_tenure.loc[
            valid_tenure["ChurnRate"].idxmax()
        ]

        st.write(
            f"**Tenure Insight:** "
            f"{highest_tenure['TenureGroup']} has the highest "
            f"churn rate at {highest_tenure['ChurnRate']:.2f}%."
        )

    if not valid_payment.empty:
        highest_payment = valid_payment.loc[
            valid_payment["ChurnRate"].idxmax()
        ]

        st.write(
            f"**Payment Insight:** "
            f"{highest_payment['PaymentMethod']} has the highest "
            f"churn rate at {highest_payment['ChurnRate']:.2f}%."
        )

    if not valid_contract.empty:
        highest_contract = valid_contract.loc[
            valid_contract["ChurnRate"].idxmax()
        ]

        st.write(
            f"**Contract Insight:** "
            f"{highest_contract['Contract']} has the highest "
            f"churn rate at {highest_contract['ChurnRate']:.2f}%."
        )

with insight_col2:
    st.subheader("🛠️ Recommended Actions")

    st.write(
        "1. **Early-tenure customers:** Improve onboarding, "
        "welcome support, and early engagement."
    )

    st.write(
        "2. **High-churn payment methods:** Review payment "
        "friction and offer simple payment instructions."
    )

    st.write(
        "3. **High-churn contract segments:** Test retention "
        "offers and explain longer-term contract benefits."
    )

    st.write(
        "4. **At-risk customers:** Contact customers with "
        "complaints or service issues before renewal."
    )

st.caption(
    "Insights are based on the selected historical data. "
    "Recommendations are hypotheses to test, not guaranteed outcomes."
)