import streamlit as st
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Income Group Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* =========================
   MAIN APP
========================= */

.stApp {
    background-color: #f8fafc;
}

/* =========================
   FULL WIDTH LAYOUT
========================= */

.block-container {
    max-width: 100% !important;
    padding-top: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    padding-bottom: 2rem;
}

section.main > div {
    max-width: 100% !important;
}

/* =========================
   TEXT
========================= */

h1, h2, h3, h4, h5, h6, p, label {
    color: #0f172a !important;
}

/* =========================
   NUMBER INPUT
========================= */

[data-testid="stNumberInput"] input {
    background-color: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

/* =========================
   SELECTBOX
========================= */

/* Main box */
div[data-baseweb="select"] > div {
    background-color: #1e293b !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

/* Selected text */
div[data-baseweb="select"] span {
    color: white !important;
    font-weight: 500 !important;
}

/* Dropdown background */
div[data-baseweb="popover"] {
    background-color: #1e293b !important;
}

/* Dropdown list */
div[data-baseweb="popover"] ul {
    background-color: #1e293b !important;
}

/* Options */
div[role="option"] {
    background-color: #1e293b !important;
    color: white !important;
    font-weight: 500 !important;
}

/* Option text */
div[role="option"] * {
    color: white !important;
}

/* Hover */
div[role="option"]:hover {
    background-color: #0D3D3A !important;
}

/* Hover text */
div[role="option"]:hover * {
    color: #E8B84B !important;
}

/* Selected option */
div[aria-selected="true"] {
    background-color: #0D3D3A !important;
}

/* Selected option text */
div[aria-selected="true"] * {
    color: #E8B84B !important;
}

/* =========================
   BUTTON
========================= */

.stButton > button {
    width: 100%;
    background-color: #0D3D3A !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 22px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    transition: 0.2s ease;
}

/* Button hover */
.stButton > button:hover {
    background-color: #1A6B65 !important;
    color: #E8B84B !important;
}

/* Button text */
.stButton button * {
    color: inherit !important;
}

/* =========================
   RESULT BOX
========================= */

.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-top: 20px;
}

/* =========================
   ADVICE BOX
========================= */

.advice-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #dbeafe;
    margin-top: 15px;
    color: #0f172a !important;
}

/* =========================
   HIDE STREAMLIT BRANDING
========================= */

footer {
    visibility: hidden;
}

header {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATASET
# =====================================================

df = pd.read_csv("income_group_master_fyp_enriched.csv")

df["state"] = df["state"].astype(str).str.upper()
df["income_class"] = df["income_class"].astype(str).str.upper()

# =====================================================
# CALCULATE STATE THRESHOLDS
# =====================================================

state_income_reference = (
    df.groupby(["state", "income_class"])["income_mean"]
      .mean()
      .reset_index()
)

# National reference
malaysia_reference = (
    df.groupby("income_class")["income_mean"]
      .mean()
      .to_dict()
)

# =====================================================
# CLEAN DATA
# =====================================================

df["state"] = df["state"].astype(str).str.upper()
df["scope"] = df["scope"].astype(str).str.upper()
df["income_group"] = df["income_group"].astype(str).str.upper()

# Rename columns
df = df.rename(columns={
    "income_mean": "income",
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
df["income_group_encoded"] = target_encoder.fit_transform(df["income_group"])

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df[[
    "scope_encoded",
    "state_encoded",
    "income",
    "expenditure"
]]

y = df["income_group_encoded"]

# =====================================================
# TRAIN MODEL
# =====================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# =====================================================
# TITLE
# =====================================================

st.title("Income Group Prediction")

st.write(
    "Predict whether a household belongs to "
    "B40, M40, or T20 based on income and expenditure."
)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# USER INPUT
# =====================================================

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

income = st.number_input(
    "Please state your estimated monthly income (RM)",
    min_value=0.0,
    step=100.0
)

expenditure = st.number_input(
    "Please state your estimated monthly household expenditure (RM)",
    min_value=0.0,
    step=100.0
)

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict Income Group"):

    # ===============================================
    # GET REFERENCE INCOME VALUES
    # ===============================================

    if selected_scope == "MALAYSIA":

        reference = malaysia_reference

    else:

        state_data = state_income_reference[
            state_income_reference["state"] == selected_state
        ]

        reference = {
            row["income_class"]: row["income_mean"]
            for _, row in state_data.iterrows()
        }

    # ===============================================
    # FIND CLOSEST INCOME GROUP
    # ===============================================

    differences = {}

    for group in ["B40", "M40", "T20"]:

        if group in reference:

            differences[group] = abs(
                income - reference[group]
            )

    predicted_group = min(
        differences,
        key=differences.get
    )

    balance = income - expenditure

    # ===============================================
    # RESULT
    # ===============================================

    st.markdown(
        '<div class="result-box">',
        unsafe_allow_html=True
    )

    st.subheader("Prediction Result")

    st.success(
        f"Predicted Income Group: {predicted_group}"
    )

    st.write(
        f"Selected Area: {selected_state}"
    )

    st.write(
        f"Monthly Income: RM {income:,.2f}"
    )

    st.write(
        f"Monthly Expenditure: RM {expenditure:,.2f}"
    )

    st.write(
        f"Remaining Balance: RM {balance:,.2f}"
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # ===============================================
    # FINANCIAL SUGGESTION
    # ===============================================

    st.subheader("Financial Suggestion")

    if expenditure > income:

        st.error(
            "Your expenditure is higher than your income."
        )

        st.markdown("""
        <div class="advice-box">
        You may need to reduce unnecessary spending and
        focus on essential expenses first. Creating a
        monthly budget may help improve financial stability.
        </div>
        """, unsafe_allow_html=True)

    elif expenditure == income:

        st.warning(
            "Your income and expenditure are equal."
        )

        st.markdown("""
        <div class="advice-box">
        You currently have no remaining savings.
        Consider reducing smaller expenses to build
        emergency savings gradually.
        </div>
        """, unsafe_allow_html=True)

    elif balance <= income * 0.10:

        st.warning(
            "Your remaining balance is still low."
        )

        st.markdown("""
        <div class="advice-box">
        Your spending is close to your income.
        Try to save at least 10% to 20% of your income.
        </div>
        """, unsafe_allow_html=True)

    elif balance <= income * 0.30:

        st.success(
            "You have a healthy financial balance."
        )

        st.markdown("""
        <div class="advice-box">
        Your income is higher than your expenditure.
        Continue building savings and emergency funds.
        </div>
        """, unsafe_allow_html=True)

    else:

        st.success(
            "You have strong saving potential."
        )

        st.markdown("""
        <div class="advice-box">
        Your financial condition appears stable.
        Consider long-term savings and investment planning.
        </div>
        """, unsafe_allow_html=True)
