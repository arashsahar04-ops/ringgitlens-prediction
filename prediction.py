import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Income Group Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# DESIGN TOKENS
# =====================================================
# Palette:
#   Ink        #0F1B17  (headings)
#   Slate      #475A53  (body text)
#   Canvas     #F6F8F4  (app background)
#   Surface    #FFFFFF  (cards)
#   Hairline   #E1E8DE  (borders)
#   Evergreen  #14524A  (primary / brand)
#   Evergreen+ #0D3D38  (primary hover/active)
#   Gold       #C99A3C  (accent — used sparingly, for the signature gauge + highlights)
#   B40 Coral  #D86B5C
#   M40 Amber  #D9A23B
#   T20 Teal   #1F8A70
#
# Type:
#   Display — "Fraunces" (serif, used only for the big result number — gives the
#              one "stated with confidence" moment instead of a generic sans stat)
#   Body/UI — "Inter" (everything else: labels, inputs, copy)
#   Data    — "IBM Plex Mono" (RM figures, table-like numbers — ledger feel)

CLASS_COLORS = {
    "B40": "#D86B5C",
    "M40": "#D9A23B",
    "T20": "#1F8A70",
}

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --ink: #0F1B17;
    --slate: #475A53;
    --canvas: #F6F8F4;
    --surface: #FFFFFF;
    --hairline: #E1E8DE;
    --evergreen: #14524A;
    --evergreen-deep: #0D3D38;
    --gold: #C99A3C;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* =========================
   MAIN APP
========================= */

.stApp {
    background-color: var(--canvas);
}

.block-container {
    max-width: 1100px !important;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    margin: 0 auto;
}

/* =========================
   TEXT
========================= */

h1, h2, h3, h4, h5, h6 {
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif;
    letter-spacing: -0.01em;
}

p, label, .stMarkdown, span {
    color: var(--slate);
}

/* =========================
   HERO HEADER
========================= */

.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--evergreen);
    margin-bottom: 6px;
}

.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 34px;
    font-weight: 700;
    color: var(--ink);
    margin-bottom: 6px;
    line-height: 1.15;
}

.hero-sub {
    font-size: 15px;
    color: var(--slate);
    max-width: 640px;
    line-height: 1.5;
}

.hero-divider {
    height: 1px;
    background: var(--hairline);
    margin: 28px 0 30px 0;
    border: none;
}

/* =========================
   SECTION LABEL
========================= */

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--evergreen);
    margin-bottom: 10px;
    display: block;
}

/* =========================
   INPUT CARD
========================= */

.input-card {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 16px;
    padding: 28px 28px 8px 28px;
    box-shadow: 0 1px 2px rgba(15,27,23,0.04);
}

/* =========================
   RADIO (Malaysia / State)
========================= */

div[role="radiogroup"] {
    gap: 8px;
}

div[role="radiogroup"] label {
    background: var(--canvas);
    border: 1px solid var(--hairline);
    border-radius: 10px;
    padding: 8px 16px !important;
    transition: 0.15s ease;
}

div[role="radiogroup"] label:hover {
    border-color: var(--evergreen);
}

/* =========================
   NUMBER INPUT
========================= */

[data-testid="stNumberInput"] input {
    background-color: var(--canvas) !important;
    color: var(--ink) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    border: 1px solid var(--hairline) !important;
    padding: 10px 14px !important;
}

[data-testid="stNumberInput"] input:focus {
    border-color: var(--evergreen) !important;
    box-shadow: 0 0 0 3px rgba(20,82,74,0.12) !important;
}

[data-testid="stNumberInput"] button {
    background-color: var(--canvas) !important;
    border: 1px solid var(--hairline) !important;
}

/* =========================
   SELECTBOX
========================= */

div[data-baseweb="select"] > div {
    background-color: var(--canvas) !important;
    border-radius: 10px !important;
    border: 1px solid var(--hairline) !important;
}

div[data-baseweb="select"] span {
    color: var(--ink) !important;
    font-weight: 500 !important;
}

div[data-baseweb="popover"] {
    background-color: var(--surface) !important;
}

div[data-baseweb="popover"] ul {
    background-color: var(--surface) !important;
}

div[role="option"] {
    background-color: var(--surface) !important;
    color: var(--ink) !important;
    font-weight: 500 !important;
}

div[role="option"] * {
    color: var(--ink) !important;
}

div[role="option"]:hover {
    background-color: #EAF1E8 !important;
}

div[role="option"]:hover * {
    color: var(--evergreen-deep) !important;
}

div[aria-selected="true"] {
    background-color: #EAF1E8 !important;
}

div[aria-selected="true"] * {
    color: var(--evergreen-deep) !important;
}

/* =========================
   BUTTON
========================= */

.stButton > button {
    width: 100%;
    background-color: var(--evergreen) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 22px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em;
    transition: 0.15s ease;
    margin-top: 18px;
}

.stButton > button:hover {
    background-color: var(--evergreen-deep) !important;
    color: var(--gold) !important;
}

.stButton button * {
    color: inherit !important;
}

/* =========================
   RESULT HEADER CARD
========================= */

.result-hero {
    background: var(--ink);
    border-radius: 18px;
    padding: 32px 32px;
    margin-top: 8px;
    position: relative;
    overflow: hidden;
}

.result-hero::before {
    content: "";
    position: absolute;
    top: -40%;
    right: -10%;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(201,154,60,0.18), transparent 70%);
}

.result-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.55);
    margin-bottom: 10px;
}

.result-group {
    font-family: 'Fraunces', serif;
    font-size: 64px;
    font-weight: 600;
    line-height: 1;
    margin-bottom: 4px;
}

.result-area {
    font-size: 14px;
    color: rgba(255,255,255,0.7);
    font-family: 'IBM Plex Mono', monospace;
}

/* =========================
   STAT GRID
========================= */

.stat-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
    margin-top: 18px;
}

.stat-card {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 14px;
    padding: 18px 20px;
}

.stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--slate);
    margin-bottom: 8px;
}

.stat-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 22px;
    font-weight: 600;
    color: var(--ink);
}

.stat-value.positive { color: #1F8A70; }
.stat-value.negative { color: #D86B5C; }

/* =========================
   ADVICE BOX
========================= */

.advice-box {
    border-radius: 14px;
    padding: 20px 22px;
    margin-top: 16px;
    border-left: 4px solid var(--evergreen);
    background: #EEF3EC;
}

.advice-box.warn {
    border-left-color: #D9A23B;
    background: #FBF3E2;
}

.advice-box.danger {
    border-left-color: #D86B5C;
    background: #FBEAE7;
}

.advice-title {
    font-weight: 700;
    font-size: 14px;
    color: var(--ink);
    margin-bottom: 6px;
}

.advice-body {
    font-size: 14px;
    color: var(--slate);
    line-height: 1.5;
}

/* =========================
   CHART CARD
========================= */

.chart-card {
    background: var(--surface);
    border: 1px solid var(--hairline);
    border-radius: 16px;
    padding: 22px 24px 8px 24px;
    margin-top: 18px;
}

/* =========================
   MISC
========================= */

footer, header {
    visibility: hidden;
}

[data-testid="stHorizontalBlock"] {
    gap: 24px;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("income_group_master_fyp_enriched.csv")

df["state"] = df["state"].astype(str).str.upper()

# =====================================================
# CALCULATE STATE THRESHOLDS
# =====================================================
group_reference = (
    df.groupby("income_group")["income_mean"]
      .mean()
      .to_dict()
)
ref_vals = [
    group_reference.get("B40", 0),
    group_reference.get("M40", 0),
    group_reference.get("T20", 0)
]

# =====================================================
# CLEAN DATA
# =====================================================

df["state"] = df["state"].astype(str).str.upper()
df["scope"] = df["scope"].astype(str).str.upper()
df["income_group"] = df["income_group"].astype(str).str.upper()

# Rename columns
df = df.rename(columns={
    "mean": "income",
    "estimated_expenditure": "expenditure"
})

# =====================================================
# ENCODING
# =====================================================

state_encoder = LabelEncoder()
scope_encoder = LabelEncoder()
target_encoder = LabelEncoder()

df["state_encoded"] = state_encoder.fit_transform(df["state"])
df["scope_encoded"] = scope_encoder.fit_transform(df["scope"])

df["target"] = target_encoder.fit_transform(
    df["income_group"]
)
# =====================================================
# FEATURES & TARGET
# =====================================================

X = df[
    [
        "scope_encoded",
        "state_encoded",
        "income_mean",
        "expenditure"
    ]
]

y = df["target"]

# =====================================================
# TRAIN MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# =====================================================
# HERO HEADER
# =====================================================

st.markdown("""
<div class="hero-eyebrow">Household Finance · Malaysia</div>
<div class="hero-title">Income Group Prediction</div>
<div class="hero-sub">
    Enter your monthly income and expenditure to see whether your household
    sits in B40, M40, or T20 — benchmarked against national or state-level
    income data — plus a clear read on your saving position.
</div>
<hr class="hero-divider" />
""", unsafe_allow_html=True)

# =====================================================
# USER INPUT
# =====================================================

left, right = st.columns([1, 1], gap="large")

with left:

    area_type = st.radio(
        "Choose prediction level:",
        ["Malaysia", "State"],
        horizontal=True
    )

    if area_type == "Malaysia":
        selected_scope = "MALAYSIA"
        selected_state = "MALAYSIA"
    else:
        selected_scope = "STATE"
        state_list = sorted(
            df[df["state"] != "MALAYSIA"]["state"].unique()
        )
        selected_state = st.selectbox(
            "Choose your state:",
            state_list
        )

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:

    income = st.number_input(
        "Estimated monthly income (RM)",
        min_value=0,
        step=100,
        format="%d",
    )

    expenditure = st.number_input(
        "Estimated monthly household expenditure (RM)",
        min_value=0,
        step=100,
        format="%d",
    )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
predict_clicked = st.button(
    "Predict Income Group",
    use_container_width=True
)

# =====================================================
# PREDICTION
# =====================================================

if predict_clicked:

# ===============================================
# RANDOM FOREST PREDICTION
# ===============================================

        encoded_scope = scope_encoder.transform(
        [selected_scope]
        )[0]

        encoded_state = state_encoder.transform(
        [selected_state]
        )[0]

        user_data = pd.DataFrame(
        {
            "scope_encoded": [encoded_scope],
            "state_encoded": [encoded_state],
            "income_mean": [income_mean],
            "expenditure": [expenditure]
        }
        )

        prediction = model.predict(user_data)

        predicted_group = target_encoder.inverse_transform(
        prediction
        )[0]

        balance = income - expenditure

        group_color = CLASS_COLORS.get(
        predicted_group,
        "#14524A"
        )

        # ===============================================
        # RESULT HERO
        # ===============================================

        st.markdown(f"""
        <div class="result-hero">
            <div class="result-eyebrow">Prediction Result</div>
            <div class="result-group" style="color:{group_color};">{predicted_group}</div>
            <div class="result-area">{selected_state} · Closest match by mean income</div>
        </div>
        """, unsafe_allow_html=True)

        balance_class = "positive" if balance >= 0 else "negative"
        balance_sign = "" if balance >= 0 else "−"

        st.markdown(f"""
        <div class="stat-grid">
            <div class="stat-card">
                <div class="stat-label">Monthly Income</div>
                <div class="stat-value">RM {income:,.2f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Monthly Expenditure</div>
                <div class="stat-value">RM {expenditure:,.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ===============================================
        # CHART — income vs. group reference means
        # ===============================================

        st.markdown('<div style="height:4px"></div>', unsafe_allow_html=True)
        st.markdown('<span class="section-label" style="margin-top:24px; display:block;">Where you fall</span>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)

        groups_order = ["B40", "M40", "T20"]
        ref_vals = [group_reference.get(g) for g in groups_order]

        fig = go.Figure()

        # Reference bars (mean income per class)
        fig.add_trace(go.Bar(
            x=groups_order,
            y=[v if v is not None else 0 for v in ref_vals],
            name="Reference mean income",
            marker_color=[CLASS_COLORS[g] for g in groups_order],
            opacity=0.35,
            width=0.55,
            hovertemplate="%{x} reference mean: RM %{y:,.0f}<extra></extra>"
        ))

        # User's income as a marker line
        fig.add_trace(go.Scatter(
            x=groups_order,
            y=[income] * 3,
            mode="lines+markers",
            name="Your income",
            line=dict(color="#0F1B17", width=2, dash="dot"),
            marker=dict(size=9, color="#C99A3C", line=dict(width=2, color="#0F1B17")),
            hovertemplate="Your income: RM %{y:,.0f}<extra></extra>"
        ))

        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#475A53", size=13),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            yaxis=dict(title="RM / month", gridcolor="#E1E8DE", zeroline=False),
            xaxis=dict(title=None, gridcolor="rgba(0,0,0,0)"),
            bargap=0.4,
        )

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ===============================================
        # FINANCIAL SUGGESTION
        # ===============================================

        st.markdown('<span class="section-label" style="margin-top:24px; display:block;">Financial Suggestion</span>', unsafe_allow_html=True)

        if expenditure > income:
            st.markdown("""
            <div class="advice-box danger">
                <div class="advice-title">Your expenditure is higher than your income</div>
                <div class="advice-body">
                    You may need to reduce unnecessary spending and focus on essential
                    expenses first. Creating a monthly budget may help improve financial
                    stability.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif expenditure == income:
            st.markdown("""
            <div class="advice-box warn">
                <div class="advice-title">Your income and expenditure are equal</div>
                <div class="advice-body">
                    You currently have no remaining savings. Consider reducing smaller
                    expenses to build emergency savings gradually.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif balance <= income * 0.10:
            st.markdown("""
            <div class="advice-box warn">
                <div class="advice-title">Your remaining balance is still low</div>
                <div class="advice-body">
                    Your spending is close to your income. Try to save at least
                    10% to 20% of your income.
                </div>
            </div>
            """, unsafe_allow_html=True)

        elif balance <= income * 0.30:
            st.markdown("""
            <div class="advice-box">
                <div class="advice-title">You have a healthy financial balance</div>
                <div class="advice-body">
                    Your income is higher than your expenditure. Continue building
                    savings and emergency funds.
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="advice-box">
                <div class="advice-title">You have strong saving potential</div>
                <div class="advice-body">
                    Your financial condition appears stable. Consider long-term
                    savings and investment planning.
                </div>
            </div>
            """, unsafe_allow_html=True)
