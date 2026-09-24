# ============================================================
# LOAN APPROVAL ANALYTICS
# Exploratory Data Analysis (EDA)
# ============================================================

# Use a non-GUI backend so Matplotlib does not require Tkinter/Tcl
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ============================================================
# 1. FILE PATHS
# ============================================================

DATA_PATH = Path(
    r"D:\AI-Projects\Loan_Approval_Analytics\data\loan_approval_dataset.csv"
)

REPORTS_DIR = Path(
    r"D:\AI-Projects\Loan_Approval_Analytics\reports"
)

# Create reports folder if it does not exist
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)


# ============================================================
# 3. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# 4. CLEAN TEXT VALUES
# ============================================================

categorical_columns = [
    "education",
    "self_employed",
    "loan_status"
]

for column in categorical_columns:
    df[column] = df[column].astype(str).str.strip()


# ============================================================
# 5. BASIC DATASET INFORMATION
# ============================================================

print("\n========================================")
print("DATASET INFORMATION")
print("========================================")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumn names:")
for column in df.columns:
    print("-", column)


# ============================================================
# 6. DATA TYPES
# ============================================================

print("\n========================================")
print("DATA TYPES")
print("========================================")

print(df.dtypes)


# ============================================================
# 7. MISSING VALUES
# ============================================================

print("\n========================================")
print("MISSING VALUES")
print("========================================")

missing_values = df.isnull().sum()

print(missing_values)


# ============================================================
# 8. DUPLICATE ROWS
# ============================================================

print("\n========================================")
print("DUPLICATE ROWS")
print("========================================")

print("Duplicate rows:", df.duplicated().sum())


# ============================================================
# 9. STATISTICAL SUMMARY
# ============================================================

print("\n========================================")
print("STATISTICAL SUMMARY")
print("========================================")

print(df.describe())


# ============================================================
# 10. TARGET DISTRIBUTION
# ============================================================

print("\n========================================")
print("LOAN STATUS DISTRIBUTION")
print("========================================")

target_counts = df["loan_status"].value_counts()

print(target_counts)

print("\nTarget percentages:")

target_percentages = (
    df["loan_status"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print(target_percentages)


# ============================================================
# 11. EDUCATION DISTRIBUTION
# ============================================================

print("\n========================================")
print("EDUCATION DISTRIBUTION")
print("========================================")

print(df["education"].value_counts())


# ============================================================
# 12. SELF-EMPLOYED DISTRIBUTION
# ============================================================

print("\n========================================")
print("SELF-EMPLOYED DISTRIBUTION")
print("========================================")

print(df["self_employed"].value_counts())


# ============================================================
# 13. AVERAGE FINANCIAL VALUES BY LOAN STATUS
# ============================================================

print("\n========================================")
print("AVERAGE VALUES BY LOAN STATUS")
print("========================================")

financial_columns = [
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]

average_values = df.groupby("loan_status")[financial_columns].mean()

print(average_values)


# ============================================================
# 14. LOAN STATUS COUNT CHART
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="loan_status"
)

plt.title("Loan Approval Status Distribution")
plt.xlabel("Loan Status")
plt.ylabel("Number of Applications")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "loan_status_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 15. CIBIL SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="cibil_score",
    hue="loan_status",
    kde=True
)

plt.title("CIBIL Score Distribution by Loan Status")
plt.xlabel("CIBIL Score")
plt.ylabel("Number of Applicants")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "cibil_score_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. ANNUAL INCOME BY LOAN STATUS
# ============================================================

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="loan_status",
    y="income_annum"
)

plt.title("Annual Income by Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Annual Income")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "income_by_loan_status.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. LOAN AMOUNT BY LOAN STATUS
# ============================================================

plt.figure(figsize=(8, 5))

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
    REPORTS_DIR / "loan_amount_by_status.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. EDUCATION VS LOAN STATUS
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="education",
    hue="loan_status"
)

plt.title("Loan Approval by Education")
plt.xlabel("Education")
plt.ylabel("Number of Applicants")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "education_vs_loan_status.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 19. SELF-EMPLOYED VS LOAN STATUS
# ============================================================

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="self_employed",
    hue="loan_status"
)

plt.title("Loan Approval by Employment Status")
plt.xlabel("Self Employed")
plt.ylabel("Number of Applicants")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "self_employed_vs_loan_status.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 20. LOAN TERM DISTRIBUTION
# ============================================================

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="loan_term"
)

plt.title("Loan Term Distribution")
plt.xlabel("Loan Term")
plt.ylabel("Number of Applications")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "loan_term_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. CIBIL SCORE VS LOAN AMOUNT
# ============================================================

plt.figure(figsize=(9, 6))

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
    REPORTS_DIR / "cibil_vs_loan_amount.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. CORRELATION HEATMAP
# ============================================================

numeric_df = df.select_dtypes(include="number")

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 9))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)

plt.title("Correlation Matrix of Numerical Variables")

plt.tight_layout()

plt.savefig(
    REPORTS_DIR / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 23. SAVE CLEANED DATA
# ============================================================

cleaned_data_path = (
    Path(r"D:\AI-Projects\Loan_Approval_Analytics\data")
    / "loan_approval_cleaned.csv"
)

df.to_csv(
    cleaned_data_path,
    index=False
)


# ============================================================
# 24. FINAL MESSAGE
# ============================================================

print("\n========================================")
print("EDA COMPLETE")
print("========================================")

print("\nCharts saved in:")
print(REPORTS_DIR)

print("\nCleaned dataset saved as:")
print(cleaned_data_path)

print("\nGenerated files:")

for file in sorted(REPORTS_DIR.glob("*.png")):
    print("-", file.name)

print("\nProject EDA completed successfully.")