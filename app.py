import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# LOAN APPROVAL ANALYTICS - STREAMLIT APPLICATION
# ============================================================


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = Path(
    r"D:\AI-Projects\Loan_Approval_Analytics"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "loan_approval_model.joblib"
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "loan_approval_cleaned.csv"
)

REPORTS_DIR = (
    BASE_DIR
    / "reports"
)


# ============================================================
# 2. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Approval Analytics",
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

if not DATA_PATH.exists():

    st.error(
        f"Dataset not found:\n{DATA_PATH}"
    )

    st.stop()


if not MODEL_PATH.exists():

    st.error(
        f"Trained model not found:\n{MODEL_PATH}"
    )

    st.info(
        "Run the model training script first:\n"
        "python .\\src\\model.py"
    )

    st.stop()


# ============================================================
# 4. LOAD DATA AND MODEL
# ============================================================

@st.cache_data
def load_data():

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


df = load_data()

model = load_model()


# ============================================================
# 5. APPLICATION TITLE
# ============================================================

st.title("🏦 Loan Approval Analytics")

st.write(
    "Machine Learning based system for analyzing "
    "and predicting loan approval status."
)

st.divider()


# ============================================================
# 6. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard",
        "Loan Prediction",
        "EDA",
        "Model Performance"
    ]
)


# ============================================================
# 7. DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header("📊 Project Dashboard")

    total_applications = len(df)

    approved_count = (
        df["loan_status"]
        .eq("Approved")
        .sum()
    )

    rejected_count = (
        df["loan_status"]
        .eq("Rejected")
        .sum()
    )

    approval_rate = (
        approved_count
        / total_applications
        * 100
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Applications",
            f"{total_applications:,}"
        )

    with col2:

        st.metric(
            "Approved",
            f"{approved_count:,}"
        )

    with col3:

        st.metric(
            "Rejected",
            f"{rejected_count:,}"
        )

    with col4:

        st.metric(
            "Approval Rate",
            f"{approval_rate:.2f}%"
        )


    st.divider()


    # --------------------------------------------------------
    # DATASET PREVIEW
    # --------------------------------------------------------

    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(10),
        width="stretch"
    )


    # --------------------------------------------------------
    # LOAN STATUS CHART
    # --------------------------------------------------------

    st.subheader(
        "📈 Loan Status Distribution"
    )

    status_counts = (
        df["loan_status"]
        .value_counts()
    )

    st.bar_chart(
        status_counts,
        width="stretch"
    )


# ============================================================
# 8. LOAN PREDICTION
# ============================================================

elif page == "Loan Prediction":

    st.header("🔮 Loan Approval Prediction")

    st.write(
        "Enter applicant information below to "
        "generate a machine-learning prediction."
    )


    # --------------------------------------------------------
    # APPLICANT INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "Applicant Information"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        no_of_dependents = st.number_input(
            "Number of Dependents",
            min_value=0,
            max_value=20,
            value=2,
            step=1
        )

        education = st.selectbox(
            "Education",
            [
                "Graduate",
                "Not Graduate"
            ]
        )

        self_employed = st.selectbox(
            "Self Employed",
            [
                "Yes",
                "No"
            ]
        )

        income_annum = st.number_input(
            "Annual Income",
            min_value=0,
            value=5000000,
            step=100000
        )


    with col2:

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0,
            value=15000000,
            step=100000
        )

        loan_term = st.number_input(
            "Loan Term",
            min_value=1,
            max_value=50,
            value=10,
            step=1
        )

        cibil_score = st.number_input(
            "CIBIL Score",
            min_value=300,
            max_value=900,
            value=700,
            step=1
        )

        residential_assets_value = st.number_input(
            "Residential Assets Value",
            min_value=0,
            value=10000000,
            step=100000
        )


    with col3:

        commercial_assets_value = st.number_input(
            "Commercial Assets Value",
            min_value=0,
            value=5000000,
            step=100000
        )

        luxury_assets_value = st.number_input(
            "Luxury Assets Value",
            min_value=0,
            value=10000000,
            step=100000
        )

        bank_asset_value = st.number_input(
            "Bank Asset Value",
            min_value=0,
            value=5000000,
            step=100000
        )


    st.divider()


    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        {
            "no_of_dependents": [
                no_of_dependents
            ],

            "education": [
                education
            ],

            "self_employed": [
                self_employed
            ],

            "income_annum": [
                income_annum
            ],

            "loan_amount": [
                loan_amount
            ],

            "loan_term": [
                loan_term
            ],

            "cibil_score": [
                cibil_score
            ],

            "residential_assets_value": [
                residential_assets_value
            ],

            "commercial_assets_value": [
                commercial_assets_value
            ],

            "luxury_assets_value": [
                luxury_assets_value
            ],

            "bank_asset_value": [
                bank_asset_value
            ]
        }
    )


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Loan Status",
        width="stretch"
    ):

        prediction = model.predict(
            input_data
        )[0]


        st.subheader(
            "Prediction Result"
        )


        if prediction == "Approved":

            st.success(
                "✅ Loan Prediction: APPROVED"
            )

        else:

            st.error(
                "❌ Loan Prediction: REJECTED"
            )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probabilities = (
                model
                .predict_proba(
                    input_data
                )[0]
            )

            classes = model.classes_


            probability_df = pd.DataFrame(
                {
                    "Status": classes,

                    "Probability (%)":
                        probabilities * 100
                }
            )


            probability_df[
                "Probability (%)"
            ] = (
                probability_df[
                    "Probability (%)"
                ].round(2)
            )


            st.subheader(
                "📊 Prediction Probability"
            )


            st.dataframe(
                probability_df,
                width="stretch"
            )


# ============================================================
# 9. EXPLORATORY DATA ANALYSIS
# ============================================================

elif page == "EDA":

    st.header(
        "📈 Exploratory Data Analysis"
    )

    st.write(
        "Visual analysis of the loan approval dataset."
    )


    chart_files = [

        (
            "Loan Status Distribution",
            "loan_status_distribution.png"
        ),

        (
            "CIBIL Score Distribution",
            "cibil_score_distribution.png"
        ),

        (
            "Annual Income by Loan Status",
            "income_by_loan_status.png"
        ),

        (
            "Loan Amount by Loan Status",
            "loan_amount_by_status.png"
        ),

        (
            "Education vs Loan Status",
            "education_vs_loan_status.png"
        ),

        (
            "Self Employment vs Loan Status",
            "self_employed_vs_loan_status.png"
        ),

        (
            "Loan Term Distribution",
            "loan_term_distribution.png"
        ),

        (
            "CIBIL Score vs Loan Amount",
            "cibil_vs_loan_amount.png"
        ),

        (
            "Correlation Heatmap",
            "correlation_heatmap.png"
        )
    ]


    for title, filename in chart_files:

        image_path = (
            REPORTS_DIR
            / filename
        )


        if image_path.exists():

            st.subheader(title)

            st.image(
                str(image_path),
                width="stretch"
            )

        else:

            st.warning(
                f"Chart not found: {filename}"
            )


# ============================================================
# 10. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header(
        "🤖 Model Performance"
    )

    comparison_path = (
        REPORTS_DIR
        / "model_comparison.csv"
    )

    confusion_path = (
        REPORTS_DIR
        / "confusion_matrix.png"
    )


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    if comparison_path.exists():

        results = pd.read_csv(
            comparison_path
        )


        st.subheader(
            "📊 Model Comparison"
        )


        st.dataframe(
            results,
            width="stretch"
        )


        # ----------------------------------------------------
        # PERFORMANCE CHART
        # ----------------------------------------------------

        st.subheader(
            "📈 Performance Comparison"
        )


        metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1_Score"
        ]


        available_metrics = [
            column
            for column in metric_columns
            if column in results.columns
        ]


        if available_metrics:

            chart_data = (
                results
                .set_index("Model")[
                    available_metrics
                ]
            )


            st.bar_chart(
                chart_data,
                width="stretch"
            )


    else:

        st.warning(
            "Model comparison file was not found."
        )

        st.code(
            "python .\\src\\model.py"
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    if confusion_path.exists():

        st.subheader(
            "🎯 Confusion Matrix"
        )

        st.image(
            str(confusion_path),
            width="stretch"
        )

    else:

        st.warning(
            "Confusion matrix was not found."
        )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    "Loan Approval Analytics | "
    "Machine Learning Project"
)