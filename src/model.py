# ============================================================
# LOAN APPROVAL ANALYTICS
# Machine Learning Model
# ============================================================

import pandas as pd
import matplotlib

# Prevent Tkinter/Tcl errors
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = Path(
    r"D:\AI-Projects\Loan_Approval_Analytics\data\loan_approval_cleaned.csv"
)

MODEL_DIR = Path(
    r"D:\AI-Projects\Loan_Approval_Analytics\models"
)

REPORTS_DIR = Path(
    r"D:\AI-Projects\Loan_Approval_Analytics\reports"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n========================================")
print("LOADING DATA")
print("========================================")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

# loan_id is only an identifier and is excluded
X = df.drop(columns=["loan_id", "loan_status"])

y = df["loan_status"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 4. IDENTIFY COLUMN TYPES
# ============================================================

categorical_features = [
    "education",
    "self_employed"
]

numerical_features = [
    column
    for column in X.columns
    if column not in categorical_features
]

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========================================")
print("TRAIN / TEST SPLIT")
print("========================================")

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ============================================================
# 6. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 7. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=42
    )
}


# ============================================================
# 8. TRAIN AND EVALUATE MODELS
# ============================================================

results = []

trained_pipelines = {}


for model_name, model in models.items():

    print("\n========================================")
    print(model_name)
    print("========================================")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
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

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1
        }
    )

    trained_pipelines[model_name] = pipeline


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1_Score",
    ascending=False
)

print("\n========================================")
print("MODEL COMPARISON")
print("========================================")

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 10. SAVE MODEL COMPARISON
# ============================================================

results_path = REPORTS_DIR / "model_comparison.csv"

results_df.to_csv(
    results_path,
    index=False
)

print("\nModel comparison saved to:")
print(results_path)


# ============================================================
# 11. SELECT MODEL
# ============================================================

best_model_name = results_df.iloc[0]["Model"]

best_pipeline = trained_pipelines[
    best_model_name
]

print("\n========================================")
print("SELECTED MODEL")
print("========================================")

print("Model:", best_model_name)


# ============================================================
# 12. SAVE BEST MODEL
# ============================================================

model_path = MODEL_DIR / "loan_approval_model.joblib"

joblib.dump(
    best_pipeline,
    model_path
)

print("\nModel saved to:")
print(model_path)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

best_predictions = best_pipeline.predict(X_test)

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

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

confusion_path = (
    REPORTS_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. FINAL MODEL SUMMARY
# ============================================================

best_row = results_df.iloc[0]

print("\n========================================")
print("FINAL MODEL PERFORMANCE")
print("========================================")

print(
    "Model:",
    best_row["Model"]
)

print(
    "Accuracy:",
    round(best_row["Accuracy"], 4)
)

print(
    "Precision:",
    round(best_row["Precision"], 4)
)

print(
    "Recall:",
    round(best_row["Recall"], 4)
)

print(
    "F1 Score:",
    round(best_row["F1_Score"], 4)
)

print("\nConfusion matrix saved to:")
print(confusion_path)

print("\n========================================")
print("MACHINE LEARNING COMPLETE")
print("========================================")