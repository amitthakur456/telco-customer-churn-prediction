"""Retention Action Planner

Batch-score customers, estimate revenue at risk, pick who is worth contacting
(based on offer cost and success rate) and explain why each customer is risky.

Usage in app.py:
    from retention_planner import prepare_features, render_retention_planner

    render_retention_planner(
        full_df, pipeline, threshold, feature_columns,
        audit_callback=lambda: write_audit_log(
            user["workspace_id"], user["id"], "retention_list_exported"
        ),
    )
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Must match the feature engineering used in the training notebook
TENURE_BINS = [-1, 12, 24, 48, 72]
TENURE_GROUP_LABELS = [
    "New Customer", "Early Customer", "Established Customer", "Loyal Customer",
]
CHARGE_BINS = [0, 35, 70, 120]
CHARGE_LABELS = ["Low Charges", "Medium Charges", "High Charges"]

RISK_BINS = [0.0, 0.30, 0.60, 1.0]
RISK_LABELS = ["Low Risk", "Medium Risk", "High Risk"]
RISK_COLORS = {"Low Risk": "#30A46C", "Medium Risk": "#F5A524", "High Risk": "#E5484D"}

MAX_EXPLAINED_CUSTOMERS = 50

# First matching reason (in SHAP order) decides the suggested action
ACTION_RULES = [
    ("Contract = Month-to-month", "Offer a discounted 1-year contract"),
    ("TechSupport = No", "Offer a free 3-month tech support trial"),
    ("OnlineSecurity = No", "Offer a free online security add-on"),
    ("PaymentMethod = Electronic check", "Incentivise switching to auto-pay"),
    ("InternetService = Fiber optic", "Run a service-quality check / plan review"),
    ("TenureGroup = New Customer", "Early-life onboarding call and welcome offer"),
    ("tenure", "Early-life onboarding call and welcome offer"),
    ("ChargeCategory = High Charges", "Review pricing and offer a bundle"),
    ("MonthlyCharges", "Review pricing and offer a bundle"),
]
DEFAULT_ACTION = "Retention call from the customer success team"


# =============================================================
# CORE LOGIC (no Streamlit UI calls, easy to test)
# =============================================================

def prepare_features(df, feature_columns):
    """Rebuild the exact feature set the model was trained on."""

    X = df.copy()

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce").fillna(0)

    # Always recompute engineered columns: the dashboard may already hold a
    # different "TenureGroup" with other labels.
    if "TenureGroup" in feature_columns and "tenure" in X.columns:
        X["TenureGroup"] = pd.cut(
            pd.to_numeric(X["tenure"], errors="coerce"),
            bins=TENURE_BINS, labels=TENURE_GROUP_LABELS,
        )

    if "ChargeCategory" in feature_columns and "MonthlyCharges" in X.columns:
        X["ChargeCategory"] = pd.cut(
            pd.to_numeric(X["MonthlyCharges"], errors="coerce"),
            bins=CHARGE_BINS, labels=CHARGE_LABELS, include_lowest=True,
        )

    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        raise ValueError("Data is missing required columns: " + ", ".join(missing))

    return X[feature_columns]


def score_customers(df, pipeline, threshold, feature_columns):
    """Return (scored dataframe, model-ready features)."""

    X = prepare_features(df, feature_columns)
    probability = pipeline.predict_proba(X)[:, 1]

    scored = df.copy()
    scored["MonthlyCharges"] = pd.to_numeric(
        scored["MonthlyCharges"], errors="coerce"
    ).fillna(0)
    scored["ChurnProbability"] = probability
    scored["PredictedChurn"] = probability >= threshold
    scored["RiskLevel"] = pd.cut(
        probability, bins=RISK_BINS, labels=RISK_LABELS, include_lowest=True
    )
    return scored, X


def add_financials(scored, offer_cost, success_rate, horizon_months):
    """Revenue at risk and net benefit of a retention offer, per customer."""

    out = scored.copy()
    customer_value = out["MonthlyCharges"] * horizon_months

    out["RevenueAtRisk"] = out["ChurnProbability"] * customer_value
    out["ExpectedRetained"] = out["RevenueAtRisk"] * success_rate
    out["NetBenefit"] = out["ExpectedRetained"] - offer_cost
    out["WorthContacting"] = out["NetBenefit"] > 0
    return out


def build_contact_list(financials, max_customers):
    contact = financials[financials["WorthContacting"]]
    return contact.sort_values("NetBenefit", ascending=False).head(int(max_customers))


def summarise(financials, contact, offer_cost):
    programme_cost = len(contact) * offer_cost
    retained = float(contact["ExpectedRetained"].sum())
    net = retained - programme_cost
    return {
        "customers": len(financials),
        "high_risk": int(financials["PredictedChurn"].sum()),
        "revenue_at_risk": float(financials["RevenueAtRisk"].sum()),
        "contacts": len(contact),
        "programme_cost": programme_cost,
        "retained": retained,
        "net_benefit": net,
        "roi": (net / programme_cost * 100) if programme_cost > 0 else 0.0,
    }


def readable_reason(feature_name, raw_row):
    """Turn a transformed feature name into text like 'Contract = Month-to-month'.

    Works with any ColumnTransformer prefix ("num__", "cat__", ...) because it
    matches against the real column names instead of assuming the prefix.
    """

    name = feature_name.split("__", 1)[-1]

    if name in raw_row.index:  # numeric feature
        value = raw_row[name]
        if isinstance(value, (float, np.floating)):
            value = round(float(value), 1)
        return f"{name} = {value}"

    # one-hot feature: "<column>_<category>"; longest column name wins
    for column in sorted(raw_row.index, key=len, reverse=True):
        if name.startswith(f"{column}_"):
            return f"{column} = {name[len(column) + 1:]}"

    return name


def suggest_action(reasons):
    for reason in reasons:
        for key, action in ACTION_RULES:
            if reason.startswith(key):
                return action
    return DEFAULT_ACTION


@st.cache_resource(show_spinner=False)
def _get_explainer(_classifier):
    import shap

    return shap.TreeExplainer(_classifier)


def explain_rows(pipeline, X_rows, top_k=3):
    """Top reasons per row via SHAP.

    Returns (reasons, error): reasons is a list of lists (or None on failure)
    and error is a short message explaining why SHAP could not run.
    """

    try:
        import shap  # noqa: F401  (checked early for a clear message)
    except ImportError:
        return None, "The 'shap' package is not installed (pip install shap)."

    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]

        transformed = preprocessor.transform(X_rows)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        names = preprocessor.get_feature_names_out()
        values = _get_explainer(classifier).shap_values(transformed)
        if isinstance(values, list):
            values = values[1]
        values = np.asarray(values)

        all_reasons = []
        for i in range(len(X_rows)):
            order = np.argsort(values[i])[::-1][:top_k]
            raw_row = X_rows.iloc[i]
            all_reasons.append([
                readable_reason(names[j], raw_row) for j in order if values[i][j] > 0
            ])
        return all_reasons, None

    except Exception as error:
        return None, f"{type(error).__name__}: {error}"


# =============================================================
# STREAMLIT UI
# =============================================================

@st.cache_data(show_spinner=False)
def cached_score(df, _pipeline, threshold, feature_columns):
    """Scoring only depends on the data, so sliders do not trigger a re-score."""

    return score_customers(df, _pipeline, threshold, list(feature_columns))


def _load_source(full_df):
    source = st.radio(
        "Customers to score",
        ["Workspace dataset", "Upload new CSV"],
        horizontal=True,
    )

    if source == "Workspace dataset":
        return full_df

    uploaded = st.file_uploader("Customer CSV", type=["csv"], key="planner_csv")
    if uploaded is None:
        return None
    try:
        return pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read CSV: {error}")
        return None


def _style(fig):
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_kpis(stats, horizon):
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers scored", f"{stats['customers']:,}")
    k2.metric("Predicted to churn", f"{stats['high_risk']:,}")
    k3.metric(f"Revenue at risk ({horizon} mo)", f"${stats['revenue_at_risk']:,.0f}")
    k4.metric("Customers worth contacting", f"{stats['contacts']:,}")

    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Programme cost", f"${stats['programme_cost']:,.0f}")
    k6.metric("Expected revenue retained", f"${stats['retained']:,.0f}")
    k7.metric("Net benefit", f"${stats['net_benefit']:,.0f}")
    k8.metric("ROI", f"{stats['roi']:.0f}%")


def _render_charts(financials):
    left, right = st.columns(2)

    with left:
        if "Contract" in financials.columns:
            by_contract = (
                financials.groupby("Contract", observed=False)["RevenueAtRisk"]
                .sum().reset_index()
            )
            st.plotly_chart(
                _style(px.bar(
                    by_contract, x="Contract", y="RevenueAtRisk",
                    title="Revenue at risk by contract",
                    labels={"RevenueAtRisk": "Revenue at risk ($)"},
                )),
                use_container_width=True,
            )

    with right:
        risk_counts = (
            financials["RiskLevel"].value_counts().reindex(RISK_LABELS).reset_index()
        )
        risk_counts.columns = ["RiskLevel", "Customers"]
        fig = px.bar(
            risk_counts, x="RiskLevel", y="Customers", color="RiskLevel",
            title="Customers by risk level", color_discrete_map=RISK_COLORS,
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(_style(fig), use_container_width=True)


def _add_explanations(result, pipeline, X):
    """Fill Top Reasons / Suggested Action for the first customers in `result`."""

    count = min(MAX_EXPLAINED_CUSTOMERS, len(result))
    reasons, error = explain_rows(pipeline, X.loc[result.index[:count]])

    if reasons is None:
        st.info(
            "SHAP reasons are unavailable, so the default action is shown. "
            "Install it with `pip install shap`."
        )
        with st.expander("Technical details"):
            st.code(error)
        return

    result.loc[result.index[:count], "Top Reasons"] = ["; ".join(r) for r in reasons]
    result.loc[result.index[:count], "Suggested Action"] = [
        suggest_action(r) for r in reasons
    ]


def render_retention_planner(
    full_df, pipeline, threshold, feature_columns, audit_callback=None
):
    st.subheader("🎯 Retention Action Planner")
    st.caption(
        "Score customers, see how much revenue is at risk, and find who is "
        "actually worth contacting given the cost of a retention offer."
    )

    df = _load_source(full_df)
    if df is None or df.empty:
        st.info("Upload a customer CSV or ask an Admin to upload the workspace dataset.")
        return

    if "Churn" in df.columns:
        already_churned = df["Churn"].astype(str).str.strip().str.lower() == "yes"
        if st.checkbox(
            f"Exclude customers who already churned ({int(already_churned.sum()):,})",
            value=True,
        ):
            df = df[~already_churned]

    with st.container(border=True):
        st.markdown("**Business assumptions**")
        c1, c2, c3, c4 = st.columns(4)
        offer_cost = c1.number_input("Offer cost per customer ($)", 0.0, 1000.0, 20.0, step=5.0)
        success_rate = c2.slider("Offer success rate (%)", 1, 100, 30) / 100
        horizon = c3.slider("Revenue horizon (months)", 1, 36, 12)
        max_contacts = c4.number_input("Max customers to contact", 1, 100000, 100, step=10)

    try:
        scored, X = cached_score(df, pipeline, threshold, tuple(feature_columns))
    except Exception as error:
        st.error(f"Scoring failed: {error}")
        return

    financials = add_financials(scored, offer_cost, success_rate, horizon)
    contact = build_contact_list(financials, max_contacts)
    stats = summarise(financials, contact, offer_cost)

    _render_kpis(stats, horizon)
    _render_charts(financials)

    st.subheader("📞 Priority contact list")

    if contact.empty:
        st.warning(
            "No customer is profitable to contact with these assumptions. "
            "Lower the offer cost or raise the success rate."
        )
        return

    result = contact.copy()
    result["Churn Probability (%)"] = (result["ChurnProbability"] * 100).round(1)
    result["Revenue At Risk ($)"] = result["RevenueAtRisk"].round(0)
    result["Net Benefit ($)"] = result["NetBenefit"].round(0)
    result["Top Reasons"] = ""
    result["Suggested Action"] = DEFAULT_ACTION

    if st.checkbox(
        f"Explain each customer with SHAP (top {MAX_EXPLAINED_CUSTOMERS})", value=True
    ):
        _add_explanations(result, pipeline, X)

    columns = [
        c for c in [
            "customerID", "Contract", "tenure", "MonthlyCharges",
            "Churn Probability (%)", "RiskLevel", "Revenue At Risk ($)",
            "Net Benefit ($)", "Top Reasons", "Suggested Action",
        ] if c in result.columns
    ]

    st.dataframe(
        result[columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Churn Probability (%)": st.column_config.ProgressColumn(
                "Churn Probability", min_value=0, max_value=100, format="%.1f%%"
            ),
            "MonthlyCharges": st.column_config.NumberColumn("Monthly ($)", format="$%.2f"),
            "Revenue At Risk ($)": st.column_config.NumberColumn(format="$%.0f"),
            "Net Benefit ($)": st.column_config.NumberColumn(format="$%.0f"),
        },
    )

    download_options = {"on_click": audit_callback} if audit_callback else {}
    st.download_button(
        "📥 Download contact list (CSV)",
        data=result[columns].to_csv(index=False).encode("utf-8"),
        file_name="retention_contact_list.csv",
        mime="text/csv",
        **download_options,
    )

    st.caption(
        "Revenue figures are estimates: churn probability × monthly charges × horizon. "
        "Offer cost and success rate are your own assumptions and should be validated "
        "with an A/B test. Probabilities come from a class-weighted model, so treat "
        "them as risk rankings rather than exact chances."
    )