from pathlib import Path
import sqlite3
import hashlib
import hmac
import secrets
import os

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📊",
    layout="wide",
)

# =============================================================
# CONSTANTS
# =============================================================

BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "telco_churn_pipeline.joblib"
DATA_PATH = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

DEFAULT_THRESHOLD = 0.60
LOW_RISK_LIMIT = 0.30
HIGH_RISK_LIMIT = 0.60

COLOR_CHURN = "#E5484D"
COLOR_STAY = "#3E63DD"
COLOR_MAP = {"Yes": COLOR_CHURN, "No": COLOR_STAY}
RISK_COLORS = {"Low Risk": "#30A46C", "Medium Risk": "#F5A524", "High Risk": COLOR_CHURN}
RISK_EMOJI = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}

YES_NO = ["Yes", "No"]
YES_NO_NO_INTERNET = ["Yes", "No", "No internet service"]

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": YES_NO_NO_INTERNET,
    "OnlineBackup": YES_NO_NO_INTERNET,
    "DeviceProtection": YES_NO_NO_INTERNET,
    "TechSupport": YES_NO_NO_INTERNET,
    "StreamingTV": YES_NO_NO_INTERNET,
    "StreamingMovies": YES_NO_NO_INTERNET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": YES_NO,
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

FORM_GROUPS = {
    "👤 Customer Profile": ["gender", "SeniorCitizen", "Partner", "Dependents", "tenure"],
    "📞 Services": [
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies",
    ],
    "💳 Billing & Contract": [
        "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges",
    ],
}

TENURE_BINS = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0–12 Months", "13–24 Months", "25–48 Months", "49–72 Months"]

CSS = """
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1300px; }
h1, h2, h3 { letter-spacing: -0.4px; }

.hero {
    background: linear-gradient(120deg, #3E63DD 0%, #7C3AED 55%, #E5484D 120%);
    border-radius: 18px; padding: 28px 32px; color: #fff; margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(62, 99, 221, 0.25);
}
.hero h1 { color: #fff; margin: 0; font-size: 2rem; }
.hero p  { color: rgba(255,255,255,0.88); margin: 6px 0 0 0; font-size: 1rem; }

.kpi {
    background: rgba(128,128,128,0.07);
    border: 1px solid rgba(128,128,128,0.2);
    border-left: 5px solid var(--accent, #3E63DD);
    border-radius: 14px; padding: 16px 18px;
}
.kpi .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.6px; opacity: 0.7; }
.kpi .value { font-size: 1.8rem; font-weight: 700; margin-top: 2px; }

.badge {
    display: inline-block; padding: 6px 16px; border-radius: 999px;
    color: #fff; font-weight: 600; font-size: 0.95rem;
}

div[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(128,128,128,0.18); border-radius: 14px; padding: 8px;
}
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
section[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,0.2); }
button[data-baseweb="tab"] { font-weight: 600; }
</style>
"""


# =============================================================
# HELPERS
# =============================================================

def fix_imputer_attributes(estimator):
    """Restore `_fill_dtype` on SimpleImputers loaded from older sklearn versions."""

    if isinstance(estimator, SimpleImputer):
        if not hasattr(estimator, "_fill_dtype") and hasattr(estimator, "statistics_"):
            estimator._fill_dtype = estimator.statistics_.dtype

    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            if step is not None:
                fix_imputer_attributes(step)

    if isinstance(estimator, ColumnTransformer):
        transformers = getattr(estimator, "transformers_", estimator.transformers)
        for _, transformer, _ in transformers:
            if not isinstance(transformer, str):
                fix_imputer_attributes(transformer)

    if hasattr(estimator, "transformer_list"):
        for _, transformer in estimator.transformer_list:
            fix_imputer_attributes(transformer)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "telco_churn_pipeline.joblib ko app.py ke same folder mein rakho."
        )
    bundle = joblib.load(MODEL_PATH)
    if not isinstance(bundle, dict) or "pipeline" not in bundle:
        raise ValueError("Model bundle mein 'pipeline' key nahi mili.")

    pipeline = bundle["pipeline"]
    threshold = float(bundle.get("threshold", DEFAULT_THRESHOLD))
    feature_columns = bundle.get("feature_columns")
    if not feature_columns:
        raise ValueError("Model bundle mein 'feature_columns' nahi mile.")

    fix_imputer_attributes(pipeline)
    return pipeline, threshold, feature_columns


@st.cache_data
def load_data(data_path: str):
    df = pd.read_csv(data_path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()

    df["ChurnStatus"] = (
        df["Churn"].astype(str).str.strip().str.lower().map({"yes": "Yes", "no": "No"})
    )
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS, include_lowest=True
    )
    return df


def get_risk_level(probability):
    if probability < LOW_RISK_LIMIT:
        return "Low Risk"
    if probability < HIGH_RISK_LIMIT:
        return "Medium Risk"
    return "High Risk"


def churn_rate_by(df, column):
    summary = (
        df.groupby(column, observed=False)["ChurnStatus"]
        .agg(TotalCustomers="count", ChurnedCustomers=lambda s: (s == "Yes").sum())
        .reset_index()
    )
    summary["ChurnRate"] = (
        summary["ChurnedCustomers"]
        / summary["TotalCustomers"].where(summary["TotalCustomers"] > 0)
        * 100
    )
    return summary


def style_fig(fig):
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    return fig


def show_chart(fig):
    st.plotly_chart(style_fig(fig), use_container_width=True)


def kpi_card(label, value, accent="#3E63DD"):
    st.markdown(
        f"""
        <div class="kpi" style="--accent:{accent}">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def gauge_chart(probability, threshold):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Churn Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1F2937"},
                "steps": [
                    {"range": [0, LOW_RISK_LIMIT * 100], "color": "#BFE8D0"},
                    {"range": [LOW_RISK_LIMIT * 100, HIGH_RISK_LIMIT * 100], "color": "#FCE3AE"},
                    {"range": [HIGH_RISK_LIMIT * 100, 100], "color": "#F7C2C4"},
                ],
                "threshold": {
                    "line": {"color": COLOR_CHURN, "width": 4},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# =============================================================
# TAB 1: PREDICTION
# =============================================================

def render_input_field(feature):
    key = f"input_{feature}"

    if feature in CATEGORICAL_OPTIONS:
        return st.selectbox(feature, CATEGORICAL_OPTIONS[feature], key=key)
    if feature == "SeniorCitizen":
        return st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda v: "Yes" if v == 1 else "No", key=key,
        )
    if feature == "tenure":
        return st.slider("Tenure (Months)", 0, 72, 12, key=key)
    if feature == "MonthlyCharges":
        return st.number_input(
            "Monthly Charges ($)", 0.0, 1000.0, 70.0, step=1.0, key=key
        )
    if feature == "TotalCharges":
        return st.number_input(
            "Total Charges ($)", 0.0, 100000.0, 840.0, step=10.0, key=key
        )
    return st.number_input(feature, value=0.0, key=key)


def render_form(feature_columns):
    """Render grouped inputs inside a form; return (submitted, customer_data)."""

    grouped = {title: [f for f in feats if f in feature_columns]
               for title, feats in FORM_GROUPS.items()}
    used = {f for feats in grouped.values() for f in feats}
    leftovers = [f for f in feature_columns if f not in used]
    if leftovers:
        grouped["⚙️ Other Features"] = leftovers

    customer_data = {}

    with st.form("customer_input_form"):
        for title, features in grouped.items():
            if not features:
                continue
            with st.container(border=True):
                st.markdown(f"**{title}**")
                columns = st.columns(2)
                for index, feature in enumerate(features):
                    with columns[index % 2]:
                        customer_data[feature] = render_input_field(feature)

        submitted = st.form_submit_button(
            "🔮 Predict Churn Risk", use_container_width=True, type="primary"
        )

    return submitted, customer_data


def render_recommendation(risk_level):
    if risk_level == "High Risk":
        st.error(
            "High-risk customer: retention team customer se contact kare. "
            "Personalized offer, service review ya customer feedback call "
            "consider kare."
        )
    elif risk_level == "Medium Risk":
        st.warning(
            "Medium-risk customer: customer satisfaction check kare. "
            "Relevant service benefits aur support options communicate kare."
        )
    else:
        st.success(
            "Low-risk customer: existing service experience maintain kare "
            "aur regular engagement continue kare."
        )


def render_result(pipeline, threshold, customer_df, full_df):
    try:
        probability = float(pipeline.predict_proba(customer_df)[0][1])
    except Exception as error:
        st.error(f"Prediction karte waqt error aaya: {error}")
        return

    is_churn = probability >= threshold
    risk_level = get_risk_level(probability)

    st.plotly_chart(gauge_chart(probability, threshold), use_container_width=True)

    st.markdown(
        f'<span class="badge" style="background:{RISK_COLORS[risk_level]}">'
        f"{RISK_EMOJI[risk_level]} {risk_level}</span>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.metric("Prediction", "Churn" if is_churn else "No Churn")
    c2.metric("Probability", f"{probability * 100:.2f}%")

    # Link the prediction to the historical data
    if full_df is not None and "Contract" in customer_df.columns:
        contract = customer_df["Contract"].iloc[0]
        segment = full_df[full_df["Contract"] == contract]
        if not segment.empty:
            rate = (segment["ChurnStatus"] == "Yes").mean() * 100
            st.info(
                f"📈 Historically, **{contract}** customers churn at "
                f"**{rate:.1f}%** (dataset average: "
                f"{(full_df['ChurnStatus'] == 'Yes').mean() * 100:.1f}%)."
            )

    st.subheader("Recommended Action")
    render_recommendation(risk_level)

    with st.expander("View Customer Input"):
        st.dataframe(customer_df, use_container_width=True)


def render_prediction_tab(full_df):
    try:
        pipeline, threshold, feature_columns = load_model()
    except Exception as error:
        st.error(f"Model load karte waqt error aaya: {error}")
        return

    st.write(
        "Customer details enter karo aur churn probability, "
        "prediction aur risk category dekho."
    )

    form_col, result_col = st.columns([3, 2], gap="large")

    with form_col:
        submitted, customer_data = render_form(feature_columns)

    with result_col:
        if submitted:
            customer_df = pd.DataFrame([customer_data], columns=feature_columns)
            render_result(pipeline, threshold, customer_df, full_df)
        else:
            st.info("👈 Fill the form and click **Predict Churn Risk** to see results here.")


# =============================================================
# TAB 2: ANALYTICS
# =============================================================

def render_sidebar_filters(df):
    st.sidebar.header("🎛️ Dashboard Filters")
    st.sidebar.caption("Applies to the Analytics and Insights tabs.")

    def multiselect_filter(label, column):
        options = sorted(df[column].dropna().unique())
        return st.sidebar.multiselect(label, options=options, default=options)

    contracts = multiselect_filter("Contract Type", "Contract")
    internet = multiselect_filter("Internet Service", "InternetService")
    genders = multiselect_filter("Gender", "gender")

    return df[
        df["Contract"].isin(contracts)
        & df["InternetService"].isin(internet)
        & df["gender"].isin(genders)
    ].copy()


def render_kpis(df):
    total = len(df)
    churned = int((df["ChurnStatus"] == "Yes").sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Customers", f"{total:,}", "#3E63DD")
    with c2:
        kpi_card("Churned Customers", f"{churned:,}", COLOR_CHURN)
    with c3:
        kpi_card("Churn Rate", f"{churned / total * 100:.2f}%", "#F5A524")
    with c4:
        kpi_card("Avg. Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}", "#30A46C")


def render_analytics_tab(df, tenure_analysis, payment_analysis, risk_analysis):
    render_kpis(df)
    st.write("")

    left, right = st.columns(2)

    with left:
        churn_counts = (
            df["Churn"].value_counts().rename_axis("Churn").reset_index(name="Customers")
        )
        show_chart(px.pie(
            churn_counts, names="Churn", values="Customers", hole=0.55,
            title="Churn vs No Churn", color="Churn", color_discrete_map=COLOR_MAP,
        ))

    with right:
        contract_churn = df.groupby(["Contract", "Churn"]).size().reset_index(name="Customers")
        show_chart(px.bar(
            contract_churn, x="Contract", y="Customers", color="Churn", barmode="group",
            title="Customer Churn Across Contract Types", color_discrete_map=COLOR_MAP,
        ))

    left, right = st.columns(2)

    with left:
        show_chart(px.histogram(
            df, x="tenure", color="Churn", nbins=30,
            title="Tenure Distribution by Churn Status", color_discrete_map=COLOR_MAP,
            labels={"tenure": "Tenure (Months)", "count": "Number of Customers"},
        ))

    with right:
        show_chart(px.box(
            df, x="Churn", y="MonthlyCharges", color="Churn",
            title="Monthly Charges Distribution", color_discrete_map=COLOR_MAP,
            labels={"MonthlyCharges": "Monthly Charges ($)", "Churn": "Churn Status"},
        ))

    st.divider()
    st.subheader("🔍 Advanced Churn Analytics")

    left, right = st.columns(2)

    with left:
        fig = px.bar(
            tenure_analysis, x="TenureGroup", y="ChurnRate", text="ChurnRate",
            title="Churn Rate Across Tenure Groups",
            labels={"TenureGroup": "Tenure Group", "ChurnRate": "Churn Rate (%)"},
            color_discrete_sequence=[COLOR_STAY],
        )
        fig.update_traces(texttemplate="%{text:.1f}%")
        show_chart(fig)

    with right:
        show_chart(px.bar(
            payment_analysis, x="PaymentMethod", y="ChurnRate", color="PaymentMethod",
            title="Churn Rate by Payment Method",
            labels={"PaymentMethod": "Payment Method", "ChurnRate": "Churn Rate (%)"},
        ))

    st.subheader("⚠️ High-Risk Customer Segments")

    risk_threshold = st.slider(
        "High-Risk Churn Rate Threshold (%)", 10, 80, 40, step=5
    )

    risk_table = risk_analysis.copy()
    risk_table["RiskLevel"] = risk_table["ChurnRate"].apply(
        lambda r: "High Risk" if pd.notna(r) and r >= risk_threshold else "Lower Risk"
    )
    high_risk = risk_table[risk_table["RiskLevel"] == "High Risk"].sort_values(
        by="ChurnRate", ascending=False
    )

    if high_risk.empty:
        st.info(
            "Current filters ke under selected threshold se "
            "zyada churn-rate wala contract segment nahi mila."
        )
    else:
        st.dataframe(high_risk, use_container_width=True)

    st.subheader("Filtered Customer Data")
    columns_to_show = [
        "customerID", "gender", "tenure", "Contract",
        "MonthlyCharges", "TotalCharges", "Churn",
    ]
    st.dataframe(df[columns_to_show], use_container_width=True)

    st.download_button(
        "📥 Download Filtered Data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_churn_data.csv",
        mime="text/csv",
    )

    st.caption(
        "Dashboard historical CSV data ko analyze karta hai. "
        "Yeh model ki individual customer prediction se alag hai."
    )


# =============================================================
# TAB 3: INSIGHTS
# =============================================================

def render_insights_tab(df, tenure_analysis, payment_analysis, risk_analysis):
    overall = df["ChurnStatus"].eq("Yes").mean() * 100

    def top_row(summary):
        valid = summary.dropna(subset=["ChurnRate"])
        return None if valid.empty else valid.loc[valid["ChurnRate"].idxmax()]

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.subheader("📌 Key Insights")
            st.metric("Churn Rate in Selected Data", f"{overall:.2f}%")

            for title, summary, column in [
                ("Tenure", tenure_analysis, "TenureGroup"),
                ("Payment", payment_analysis, "PaymentMethod"),
                ("Contract", risk_analysis, "Contract"),
            ]:
                row = top_row(summary)
                if row is not None:
                    st.write(
                        f"**{title} Insight:** {row[column]} has the highest "
                        f"churn rate at {row['ChurnRate']:.2f}%."
                    )

    with col2:
        with st.container(border=True):
            st.subheader("🛠️ Recommended Actions")
            st.write("1. **Early-tenure customers:** Improve onboarding, welcome support, and early engagement.")
            st.write("2. **High-churn payment methods:** Review payment friction and offer simple payment instructions.")
            st.write("3. **High-churn contract segments:** Test retention offers and explain longer-term contract benefits.")
            st.write("4. **At-risk customers:** Contact customers with complaints or service issues before renewal.")

    st.caption(
        "Insights are based on the selected historical data. "
        "Recommendations are hypotheses to test, not guaranteed outcomes."
    )



# =============================================================
# SAAS AUTHENTICATION + WORKSPACE FOUNDATION
# =============================================================

DB_PATH = BASE_DIR / "customer_retention_saas.db"
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


class HybridRow(dict):
    """Dictionary row that also supports SQLite-style integer indexing."""
    def __init__(self, values, ordered_values):
        super().__init__(values)
        self._ordered_values = tuple(ordered_values)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._ordered_values[key]
        return super().__getitem__(key)


def postgres_row(cursor):
    description = cursor.description

    if description is None:
        names = []
    else:
        names = [column.name for column in description]

    def make_row(row):
        return HybridRow(dict(zip(names, row)), row)

    return make_row

class PostgresConnectionAdapter:
    """Small adapter so existing parameterized SQL works on PostgreSQL."""
    def __init__(self, connection):
        self.connection = connection

    def execute(self, query, params=()):
        # App SQL uses ? placeholders; psycopg uses %s placeholders.
        return self.connection.execute(query.replace("?", "%s"), params)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.connection.commit()
            else:
                self.connection.rollback()
        finally:
            self.connection.close()
        return False


def get_db():
    if DATABASE_URL:
        import psycopg
        from psycopg.rows import tuple_row
        url = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        connection = psycopg.connect(url, row_factory=postgres_row)
        return PostgresConnectionAdapter(connection)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_auth_db():
    with get_db() as conn:
        if DATABASE_URL:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id BIGSERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id BIGSERIAL PRIMARY KEY,
                    workspace_id BIGINT NOT NULL REFERENCES workspaces(id),
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'Admin' CHECK (role IN ('Admin','Analyst','Viewer')),
                    is_active SMALLINT NOT NULL DEFAULT 1 CHECK (is_active IN (0,1)),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS is_active SMALLINT NOT NULL DEFAULT 1")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id BIGSERIAL PRIMARY KEY,
                    workspace_id BIGINT NOT NULL REFERENCES workspaces(id),
                    actor_user_id BIGINT,
                    action TEXT NOT NULL,
                    details TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
        else:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workspaces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id INTEGER NOT NULL,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'Admin',
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(workspace_id) REFERENCES workspaces(id)
                )
            """)
            user_columns = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
            if "is_active" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workspace_id INTEGER NOT NULL,
                    actor_user_id INTEGER,
                    action TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(workspace_id) REFERENCES workspaces(id)
                )
            """)


def write_audit_log(workspace_id, actor_user_id, action, details=""):
    with get_db() as conn:
        conn.execute(
            "INSERT INTO audit_logs (workspace_id, actor_user_id, action, details) VALUES (?, ?, ?, ?)",
            (workspace_id, actor_user_id, action, details[:500]),
        )


def hash_password(password, salt=None):
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, 310000
    )
    return salt.hex() + "$" + digest.hex()


def verify_password(password, stored_hash):
    try:
        salt_hex, digest_hex = stored_hash.split("$", 1)
        salt = bytes.fromhex(salt_hex)
        candidate = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), salt, 310000
        ).hex()
        return hmac.compare_digest(candidate, digest_hex)
    except (ValueError, TypeError):
        return False


def create_workspace(workspace_name, full_name, email, password):
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM users WHERE email = ?", (email.lower().strip(),)
        ).fetchone()
        if existing:
            return False, "This email is already registered."

        if DATABASE_URL:
            created = conn.execute(
                "INSERT INTO workspaces (name) VALUES (?) RETURNING id",
                (workspace_name.strip(),),
            ).fetchone()
            workspace_id = created["id"]
        else:
            cursor = conn.execute(
                "INSERT INTO workspaces (name) VALUES (?)",
                (workspace_name.strip(),)
            )
            workspace_id = cursor.lastrowid
        conn.execute(
            """INSERT INTO users
               (workspace_id, full_name, email, password_hash, role)
               VALUES (?, ?, ?, ?, 'Admin')""",
            (
                workspace_id,
                full_name.strip(),
                email.lower().strip(),
                hash_password(password),
            )
        )
    write_audit_log(workspace_id, None, "workspace_created", "Initial Admin account created")
    return True, "Workspace created successfully."


def authenticate_user(email, password):
    with get_db() as conn:
        row = conn.execute(
            """SELECT users.id, users.workspace_id, users.full_name,
                      users.email, users.password_hash, users.role,
                      users.is_active, workspaces.name AS workspace_name
               FROM users
               JOIN workspaces ON workspaces.id = users.workspace_id
               WHERE users.email = ? AND users.is_active = 1""",
            (email.lower().strip(),)
        ).fetchone()

    if row and verify_password(password, row["password_hash"]):
        return dict(row)
    return None


def render_auth_screen():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <h1>🔐 Customer Retention Intelligence</h1>
            <p>Create a company workspace or sign in to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    login_tab, signup_tab = st.tabs(["Sign in", "Create workspace"])

    with login_tab:
        with st.form("saas_login_form"):
            email = st.text_input("Work email", key="login_email")
            password = st.text_input(
                "Password", type="password", key="login_password"
            )
            submitted = st.form_submit_button(
                "Sign in", type="primary", use_container_width=True
            )

        if submitted:
            user = authenticate_user(email, password)
            if user:
                write_audit_log(user["workspace_id"], user["id"], "login_success", "User signed in")
                st.session_state["saas_user"] = user
                st.rerun()
            else:
                st.error("Invalid email or password.")

    with signup_tab:
        st.info(
            "The first account created for a new workspace becomes its Admin."
        )
        with st.form("saas_signup_form"):
            workspace = st.text_input("Company / workspace name")
            full_name = st.text_input("Your full name")
            signup_email = st.text_input("Work email", key="signup_email")
            new_password = st.text_input(
                "Password (minimum 12 characters)",
                type="password",
                key="signup_password",
            )
            confirm_password = st.text_input(
                "Confirm password",
                type="password",
                key="confirm_password",
            )
            submitted = st.form_submit_button(
                "Create workspace", type="primary", use_container_width=True
            )

        if submitted:
            if not all([workspace.strip(), full_name.strip(),
                        signup_email.strip(), new_password]):
                st.error("Please complete all fields.")
            elif "@" not in signup_email or "." not in signup_email.split("@")[-1]:
                st.error("Please enter a valid email address.")
            elif len(new_password) < 12:
                st.error("Password must contain at least 12 characters.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                ok, message = create_workspace(
                    workspace, full_name, signup_email, new_password
                )
                if ok:
                    st.success(message + " Please sign in.")
                else:
                    st.error(message)


def workspace_data_path(user):
    """Return a data file path scoped to the authenticated workspace."""
    return BASE_DIR / "workspace_data" / str(user["workspace_id"]) / "customers.csv"


def get_fresh_session_user():
    """Reload role and active status from DB on every rerun."""
    session_user = st.session_state.get("saas_user")
    if not session_user:
        return None
    with get_db() as conn:
        row = conn.execute(
            """SELECT users.id, users.workspace_id, users.full_name, users.email,
                      users.role, users.is_active, workspaces.name AS workspace_name
               FROM users JOIN workspaces ON workspaces.id = users.workspace_id
               WHERE users.id = ?""",
            (session_user["id"],),
        ).fetchone()
    if not row or not row["is_active"]:
        st.session_state.pop("saas_user", None)
        return None
    fresh = dict(row)
    st.session_state["saas_user"] = fresh
    return fresh


def create_employee(user, full_name, email, password, role):
    if user.get("role") != "Admin":
        return False, "Unauthorized: Only Admin can create employees."

    workspace_id = user.get("workspace_id")

    if not workspace_id:
        return False, "Invalid workspace."

    if role not in ("Admin", "Analyst", "Viewer"):
        return False, "Invalid employee role."

    if len(password) < 12:
        return False, "Password must contain at least 12 characters."

    with get_db() as conn:
        workspace = conn.execute(
            "SELECT id FROM workspaces WHERE id = ?",
            (workspace_id,)
        ).fetchone()

        if not workspace:
            return False, "Workspace does not exist."

        exists = conn.execute(
            "SELECT id FROM users WHERE email = ?",
            (email.lower().strip(),)
        ).fetchone()

        if exists:
            return False, "This email is already registered."

        conn.execute(
            """INSERT INTO users
               (workspace_id, full_name, email, password_hash, role, is_active)
               VALUES (?, ?, ?, ?, ?, 1)""",
            (
                workspace_id,
                full_name.strip(),
                email.lower().strip(),
                hash_password(password),
                role
            ),
        )

    write_audit_log(
        workspace_id,
        user["id"],
        "employee_created",
        f"Role={role}; email={email.lower().strip()}"
    )

    return True, "Employee account created successfully."

def render_admin_panel(user):
    if user.get("role") != "Admin":
        st.error("You are not authorized to manage workspace employees.")
        return
    st.subheader("👥 Employee Management")
    st.caption("Manage users who belong to your company workspace.")

    with get_db() as conn:
        employees = conn.execute(
            """SELECT id, full_name, email, role, is_active, created_at
               FROM users WHERE workspace_id = ? ORDER BY created_at DESC""",
            (user["workspace_id"],),
        ).fetchall()

    active_count = sum(int(row["is_active"]) for row in employees)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total employees", len(employees))
    c2.metric("Active accounts", active_count)
    c3.metric("Inactive accounts", len(employees) - active_count)

    with st.expander("➕ Add new employee", expanded=False):
        with st.form("add_employee_form", clear_on_submit=True):
            full_name = st.text_input("Employee full name")
            email = st.text_input("Employee work email")
            password = st.text_input("Temporary password (minimum 12 characters)", type="password")
            role = st.selectbox("Assign role", ["Analyst", "Viewer", "Admin"])
            submitted = st.form_submit_button("Create employee", type="primary")
        if submitted:
            if not full_name.strip() or not email.strip() or not password:
                st.error("Please complete all fields.")
            elif "@" not in email or "." not in email.split("@")[-1]:
                st.error("Enter a valid email address.")
            elif len(password) < 12:
                st.error("Password must contain at least 12 characters.")
            else:
                ok, message = create_employee(
                user, full_name, email, password, role

                )
                (st.success if ok else st.error)(message)
                if ok:
                    st.rerun()

    with st.expander("📋 Workspace audit log", expanded=False):
        with get_db() as conn:
            logs = conn.execute(
                "SELECT action, details, created_at FROM audit_logs WHERE workspace_id = ? ORDER BY id DESC LIMIT 100",
                (user["workspace_id"],),
            ).fetchall()
        if logs:
            st.dataframe(pd.DataFrame([dict(row) for row in logs]), use_container_width=True, hide_index=True)
        else:
            st.info("No activity recorded yet.")

    st.markdown("#### Workspace employees")
    for employee in employees:
        with st.container(border=True):
            left, middle, right = st.columns([3, 2, 2])
            with left:
                st.markdown(f"**{employee['full_name']}**")
                st.caption(employee["email"])
                st.caption("Joined: " + str(employee["created_at"]))
            with middle:
                st.write("Role: " + employee["role"])
                st.write("Status: " + ("Active" if employee["is_active"] else "Inactive"))
            with right:
                if employee["id"] == user["id"]:
                    st.caption("Current account")
                else:
                    with st.form(f"employee_manage_{employee['id']}"):
                        new_role = st.selectbox(
                            "Change role",
                            ["Admin", "Analyst", "Viewer"],
                            index=["Admin", "Analyst", "Viewer"].index(employee["role"]),
                            key=f"role_{employee['id']}",
                        )
                        save_role = st.form_submit_button("Save role")
                        toggle_label = "Deactivate" if employee["is_active"] else "Activate"
                        toggle = st.form_submit_button(toggle_label)
                    if save_role:
                        with get_db() as conn:
                            conn.execute(
                                "UPDATE users SET role = ? WHERE id = ? AND workspace_id = ?",
                                (new_role, employee["id"], user["workspace_id"]),
                            )
                        write_audit_log(user["workspace_id"], user["id"], "employee_role_changed", f"target_user_id={employee['id']}; role={new_role}")
                        st.success("Employee role updated.")
                        st.rerun()
                    if toggle:
                        if employee["role"] == "Admin" and employee["is_active"]:
                            with get_db() as conn:
                                admins = conn.execute(
                                    """SELECT COUNT(*) FROM users
                                       WHERE workspace_id = ? AND role = 'Admin' AND is_active = 1""",
                                    (user["workspace_id"],),
                                ).fetchone()[0]
                            if admins <= 1:
                                st.error("You cannot deactivate the last active Admin.")
                                continue
                        new_status = 0 if employee["is_active"] else 1
                        with get_db() as conn:
                            conn.execute(
                                "UPDATE users SET is_active = ? WHERE id = ? AND workspace_id = ?",
                                (new_status, employee["id"], user["workspace_id"]),
                            )
                        write_audit_log(user["workspace_id"], user["id"], "employee_status_changed", f"target_user_id={employee['id']}; active={new_status}")
                        st.success("Employee status updated.")
                        st.rerun()


def render_app_for_authenticated_user(user):
    data_path = workspace_data_path(user)
    with st.sidebar:
        st.markdown(f"### 🏢 {user['workspace_name']}")
        st.caption(f"{user['full_name']} · {user['role']}")
        st.caption("Database: PostgreSQL" if DATABASE_URL else "Database: Local SQLite (development)")
        if st.button("Log out", use_container_width=True):
            st.session_state.pop("saas_user", None)
            st.rerun()
        st.divider()
        if user["role"] == "Admin":
            st.subheader("Company data")
            uploaded_file = st.file_uploader(
                "Upload this workspace's customer CSV", type=["csv"],
                key=f"workspace_csv_{user['workspace_id']}",
                help="Only Admins can replace the workspace dataset.",
            )
            if uploaded_file is not None:
                try:
                    preview = pd.read_csv(uploaded_file, nrows=5)
                    required = {"Churn", "TotalCharges", "tenure"}
                    missing = sorted(required - set(preview.columns))
                    if missing:
                        st.error("CSV missing required columns: " + ", ".join(missing))
                    else:
                        data_path.parent.mkdir(parents=True, exist_ok=True)
                        data_path.write_bytes(uploaded_file.getvalue())
                        write_audit_log(user["workspace_id"], user["id"], "dataset_uploaded", f"filename={uploaded_file.name}; bytes={uploaded_file.size}")
                        st.success("Customer dataset saved to this workspace.")
                        st.cache_data.clear()
                except Exception as exc:
                    st.error(f"Could not read CSV: {exc}")
        else:
            st.info("Dataset management is available to Admins only.")
        if data_path.exists():
            st.caption("Workspace dataset: uploaded")
        else:
            st.warning("No customer CSV uploaded for this workspace.")
    return data_path


# =============================================================
# MAIN
# =============================================================

init_auth_db()

current_user = get_fresh_session_user()
if current_user is None:
    render_auth_screen()
    st.stop()

current_workspace_data = render_app_for_authenticated_user(current_user)

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
        <h1>📊 Telco Churn Intelligence</h1>
        <p>Predict individual customer churn, explore historical patterns and
        act on retention insights — all in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    full_df = (
        load_data(str(current_workspace_data))
        if current_workspace_data.exists()
        else None
    )
except FileNotFoundError:
    full_df = None

filtered_df = None
if full_df is not None:
    filtered_df = render_sidebar_filters(full_df)

tab_names = ["📈 Analytics Dashboard", "💡 Insights"]
if current_user["role"] in ("Admin", "Analyst"):
    tab_names.insert(0, "🔮 Predict Churn")
if current_user["role"] == "Admin":
    tab_names.append("👥 Admin Panel")
created_tabs = st.tabs(tab_names)
tab_map = dict(zip(tab_names, created_tabs))

if "🔮 Predict Churn" in tab_map:
    with tab_map["🔮 Predict Churn"]:
        render_prediction_tab(full_df)

# Analytics + Insights share the same filtered data and summaries
if full_df is None:
    for tab_name in ("📈 Analytics Dashboard", "💡 Insights"):
        with tab_map[tab_name]:
            st.error("Is workspace ka customer CSV upload nahi hua. Admin se company CSV upload karwayein.")
elif filtered_df.empty:
    for tab_name in ("📈 Analytics Dashboard", "💡 Insights"):
        with tab_map[tab_name]:
            st.warning("Selected filters ke liye koi customer data nahi mila.")
else:
    tenure_analysis = churn_rate_by(filtered_df, "TenureGroup")
    payment_analysis = churn_rate_by(filtered_df, "PaymentMethod")
    risk_analysis = churn_rate_by(filtered_df, "Contract")

    with tab_map["📈 Analytics Dashboard"]:
        render_analytics_tab(filtered_df, tenure_analysis, payment_analysis, risk_analysis)

    with tab_map["💡 Insights"]:
        render_insights_tab(filtered_df, tenure_analysis, payment_analysis, risk_analysis)

if "👥 Admin Panel" in tab_map:
    with tab_map["👥 Admin Panel"]:
        render_admin_panel(current_user)
