"""
LOAN APPROVAL PREDICTION AND ANALYTICS USING MACHINE LEARNING
IBM SkillsBuild Data Analytics with AI Internship Project

This file contains the complete project workflow:
1. Data loading and cleaning
2. Exploratory Data Analysis
3. Machine learning model training and evaluation
4. Loan approval prediction
"""
# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import os
import warnings

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

DATA_FILE = os.path.join(DATA_DIR, "loan_approval_dataset.csv")
CLEANED_FILE = os.path.join(DATA_DIR, "loan_approval_cleaned.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(DATA_FILE)

# Remove unwanted whitespace from column names
df.columns = df.columns.str.strip()

print("\n========== DATASET INFORMATION ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
for column in df.columns:
    print(column)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Remove duplicate records
df = df.drop_duplicates()

# Remove unnecessary whitespace from string columns
for column in df.select_dtypes(include="object").columns:
    df[column] = df[column].str.strip()

# Save cleaned dataset
df.to_csv(CLEANED_FILE, index=False)

print("\n========== CLEANED DATA ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Cleaned dataset saved to:")
print(CLEANED_FILE)


# ============================================================
# 5. TARGET DISTRIBUTION
# ============================================================

print("\n========== TARGET DISTRIBUTION ==========")
print(df["loan_status"].value_counts())


# ============================================================
# 6. EXPLORATORY DATA ANALYSIS
# ============================================================

# Loan status distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="loan_status")
plt.title("Loan Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applications")
plt.tight_layout()
plt.savefig(
    os.path.join(REPORTS_DIR, "loan_status_distribution.png"),
    dpi=150
)
plt.close()


# CIBIL score distribution
plt.figure(figsize=(7, 5))
sns.histplot(data=df, x="cibil_score", kde=True)
plt.title("CIBIL Score Distribution")
plt.xlabel("CIBIL Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(
    os.path.join(REPORTS_DIR, "cibil_score_distribution.png"),
    dpi=150
)
plt.close()


# Loan term distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="loan_term")
plt.title("Loan Term Distribution")
plt.xlabel("Loan Term")
plt.ylabel("Number of Applications")
plt.tight_layout()
plt.savefig(
    os.path.join(REPORTS_DIR, "loan_term_distribution.png"),
    dpi=150
)
plt.close()


# Education vs loan status
plt.figure(figsize=(7, 5))
sns.countplot(
    data=df,
    x="education",
    hue="loan_status"
)
plt.title("Education vs Loan Status")
plt.xlabel("Education")
plt.ylabel("Number of Applications")
plt.tight_layout()
plt.savefig(
    os.path.join(REPORTS_DIR, "education_vs_loan_status.png"),
    dpi=150
)
plt.close()


# Self-employed vs loan status
plt.figure(figsize=(7, 5))
sns.countplot(
    data=df,
    x="self_employed",
    hue="loan_status"
)
plt.title("Self Employed vs Loan Status")
plt.xlabel("Self Employed")
plt.ylabel("Number of Applications")
plt.tight_layout()
plt.savefig(
    os.path.join(REPORTS_DIR, "self_employed_vs_loan_status.png"),
    dpi=150
)
plt.close()


# Income by loan status
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="loan_status",
    y="income_annum"
)
plt.title("Income by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Annual Income")
plt.tight_layout()
plt.savefig(
    os.path.join(REPORTS_DIR, "income_by_loan_status.png"),
    dpi=150
)
plt.close()


# Loan amount by status
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="loan_status",
    y="loan_amount"
)
plt.title("Loan Amount by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Loan Amount")
plt.tight_layout()
plt.savefig(
    os.path.join(REPORTS_DIR, "loan_amount_by_status.png"),
    dpi=150
)
plt.close()


# CIBIL score vs loan amount
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="cibil_score",
    y="loan_amount",
    hue="loan_status"
)
plt.title("CIBIL Score vs Loan Amount")
plt.xlabel("CIBIL Score")
plt.ylabel("Loan Amount")
plt.tight_layout()
plt.savefig(
    os.path.join(REPORTS_DIR, "cibil_vs_loan_amount.png"),
    dpi=150
)
plt.close()


# Correlation heatmap
numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(
    os.path.join(REPORTS_DIR, "correlation_heatmap.png"),
    dpi=150
)
plt.close()


print("\n========== EDA COMPLETE ==========")
print("EDA reports saved in:")
print(REPORTS_DIR)

# ============================================================
# 7. MACHINE LEARNING
# ============================================================

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import joblib


# ============================================================
# 8. PREPARE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["loan_status", "loan_id"])
y = df["loan_status"].str.strip()


# ============================================================
# 9. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ============================================================
# 10. PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# ============================================================
# 11. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN / TEST SPLIT ==========")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 12. DEFINE MACHINE LEARNING MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ============================================================
# 13. TRAIN AND EVALUATE MODELS
# ============================================================

results = []
trained_models = {}

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test,
        y_pred,
        pos_label="Approved"
    )
    recall = recall_score(
        y_test,
        y_pred,
        pos_label="Approved"
    )
    f1 = f1_score(
        y_test,
        y_pred,
        pos_label="Approved"
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1
    })

    trained_models[model_name] = pipeline

    print("\n========================================")
    print(model_name)
    print("========================================")
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_path = os.path.join(
    REPORTS_DIR,
    "model_comparison.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))


# ============================================================
# 15. SELECT MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["F1_Score"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print("\nSelected model:", best_model_name)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

best_predictions = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["Approved", "Rejected"]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Approved", "Rejected"],
    yticklabels=["Approved", "Rejected"]
)

plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()

plt.savefig(
    os.path.join(REPORTS_DIR, "confusion_matrix.png"),
    dpi=150
)

plt.close()


# ============================================================
# 17. CLASSIFICATION REPORT
# ============================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        best_predictions
    )
)


# ============================================================
# 18. SAVE TRAINED MODEL
# ============================================================

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "loan_approval_model.joblib"
)

joblib.dump(
    best_model,
    MODEL_FILE
)

print("\n========== MODEL SAVED ==========")
print(MODEL_FILE)


print("\n========================================")
print("MACHINE LEARNING PIPELINE COMPLETE")
print("========================================")

# ============================================================
# 19. SAMPLE LOAN APPROVAL PREDICTION
# ============================================================

def predict_loan_approval(
    no_of_dependents,
    education,
    self_employed,
    income_annum,
    loan_amount,
    loan_term,
    cibil_score,
    residential_assets_value,
    commercial_assets_value,
    luxury_assets_value,
    bank_asset_value
):
    """
    Predict whether a loan application is Approved or Rejected.
    """

    input_data = pd.DataFrame({
        "no_of_dependents": [no_of_dependents],
        "education": [education],
        "self_employed": [self_employed],
        "income_annum": [income_annum],
        "loan_amount": [loan_amount],
        "loan_term": [loan_term],
        "cibil_score": [cibil_score],
        "residential_assets_value": [residential_assets_value],
        "commercial_assets_value": [commercial_assets_value],
        "luxury_assets_value": [luxury_assets_value],
        "bank_asset_value": [bank_asset_value]
    })

    prediction = best_model.predict(input_data)[0]

    if hasattr(best_model, "predict_proba"):
        probabilities = best_model.predict_proba(input_data)[0]
        confidence = max(probabilities)
    else:
        confidence = None

    return prediction, confidence


# ============================================================
# 20. SAMPLE PREDICTION
# ============================================================

sample_prediction, sample_confidence = predict_loan_approval(
    no_of_dependents=2,
    education="Graduate",
    self_employed="No",
    income_annum=5000000,
    loan_amount=15000000,
    loan_term=10,
    cibil_score=750,
    residential_assets_value=20000000,
    commercial_assets_value=5000000,
    luxury_assets_value=10000000,
    bank_asset_value=8000000
)

print("\n========== SAMPLE LOAN PREDICTION ==========")
print("Prediction:", sample_prediction)

if sample_confidence is not None:
    print("Confidence:", round(sample_confidence * 100, 2), "%")


# ============================================================
# 21. PROJECT SUMMARY
# ============================================================

print("\n========================================")
print("LOAN APPROVAL PREDICTION PROJECT")
print("========================================")

print("Dataset rows:", len(df))
print("Dataset columns:", len(df.columns))
print("Best model:", best_model_name)

best_row = results_df[
    results_df["Model"] == best_model_name
].iloc[0]

print(
    "Accuracy:",
    round(best_row["Accuracy"] * 100, 2),
    "%"
)

print(
    "Precision:",
    round(best_row["Precision"] * 100, 2),
    "%"
)

print(
    "Recall:",
    round(best_row["Recall"] * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(best_row["F1_Score"] * 100, 2),
    "%"
)

print("\nProject execution completed successfully.")