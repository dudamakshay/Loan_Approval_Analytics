# Loan Approval Prediction and Analytics Using Machine Learning

## Project Overview

This project develops a machine learning-based system for **loan approval prediction and analytics**.

The system analyzes applicant information such as income, loan amount, loan term, CIBIL score, education, employment status, dependents, and asset values to predict whether a loan application is likely to be **Approved** or **Rejected**.

The project includes:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Data visualization
- Machine learning model development
- Model comparison
- Model evaluation
- Loan approval prediction
- Trained model serialization
- Streamlit-based prediction application

---

## Dataset

The dataset used in this project is the **Loan Approval Prediction Dataset** available on Kaggle.

**Dataset Source:**

https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset

**Dataset file used:**

`loan_approval_dataset.csv`

**Dataset size:**

- Rows: 4,269
- Columns: 13

### Dataset Features

| Feature | Description |
|---|---|
| `loan_id` | Unique loan application identifier |
| `no_of_dependents` | Number of dependents |
| `education` | Applicant education status |
| `self_employed` | Whether the applicant is self-employed |
| `income_annum` | Annual income |
| `loan_amount` | Requested loan amount |
| `loan_term` | Loan repayment term |
| `cibil_score` | Applicant's CIBIL credit score |
| `residential_assets_value` | Value of residential assets |
| `commercial_assets_value` | Value of commercial assets |
| `luxury_assets_value` | Value of luxury assets |
| `bank_asset_value` | Value of bank assets |
| `loan_status` | Target variable: Approved or Rejected |

---

## Project Objectives

The main objectives of this project are:

1. Analyze the loan approval dataset.
2. Identify and handle data-quality issues.
3. Perform exploratory data analysis.
4. Visualize important relationships within the dataset.
5. Prepare the data for machine learning.
6. Train multiple classification algorithms.
7. Compare model performance using standard evaluation metrics.
8. Select a model based on the evaluation results.
9. Save the trained machine learning model.
10. Provide a prediction interface using Streamlit.

---

## Technologies Used

### Programming Language

- Python 3.13

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

### Development Tools

- Visual Studio Code / Python environment
- PowerShell / Command Prompt
- Git
- GitHub

---

## Project Structure

```text
Loan_Approval_Analytics/
│
├── Loan_Approval_Analytics_Project.py
├── app.py
├── inspect_data.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── loan_approval_dataset.csv
│   └── loan_approval_cleaned.csv
│
├── models/
│   └── loan_approval_model.joblib
│
├── reports/
│   ├── cibil_score_distribution.png
│   ├── cibil_vs_loan_amount.png
│   ├── confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── education_vs_loan_status.png
│   ├── income_by_loan_status.png
│   ├── loan_amount_by_status.png
│   ├── loan_status_distribution.png
│   ├── loan_term_distribution.png
│   ├── model_comparison.csv
│   └── self_employed_vs_loan_status.png
│
└── src/
    ├── data_cleaning.py
    ├── eda.py
    └── model.py