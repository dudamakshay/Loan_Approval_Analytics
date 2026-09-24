\# Loan Approval Prediction and Analytics Using Machine Learning



\## Project Overview



This project develops a machine learning-based system for predicting loan approval status and performing analytics on loan application data.



The project uses a dataset containing 4,269 loan application records and 13 columns. The target variable is `loan\_status`, containing two classes: Approved and Rejected.



The project includes data cleaning, exploratory data analysis, machine learning model training, model evaluation, model comparison, model saving, and loan prediction.



\## Objectives



\- Analyze the loan approval dataset.

\- Clean and preprocess the dataset.

\- Perform exploratory data analysis.

\- Analyze relationships between applicant, financial and credit attributes.

\- Train multiple machine learning classification models.

\- Compare model performance using standard evaluation metrics.

\- Save the trained machine learning model.

\- Generate loan approval predictions.

\- Provide a Streamlit application for interactive use.



\## Dataset



Dataset file:



`loan\_approval\_dataset.csv`



Dataset size:



\- Rows: 4,269

\- Columns: 13

\- Missing values: 0

\- Duplicate rows: 0



\### Features



\- `loan\_id`

\- `no\_of\_dependents`

\- `education`

\- `self\_employed`

\- `income\_annum`

\- `loan\_amount`

\- `loan\_term`

\- `cibil\_score`

\- `residential\_assets\_value`

\- `commercial\_assets\_value`

\- `luxury\_assets\_value`

\- `bank\_asset\_value`



\### Target Variable



`loan\_status`



Classes:



\- Approved

\- Rejected



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Seaborn

\- Joblib

\- Streamlit



\## Machine Learning Models



The project trains and evaluates:



1\. Logistic Regression

2\. Random Forest

3\. Gradient Boosting



\## Model Performance



| Model | Accuracy | Precision | Recall | F1 Score |

|---|---:|---:|---:|---:|

| Logistic Regression | 92.27% | 92.66% | 95.10% | 93.87% |

| Random Forest | 98.24% | 97.78% | 99.44% | 98.60% |

| Gradient Boosting | 98.24% | 97.96% | 99.25% | 98.60% |



The executed project selected \*\*Random Forest\*\* as the final model.



\### Final Random Forest Results



\- Accuracy: 98.24%

\- Precision: 97.78%

\- Recall: 99.44%

\- F1 Score: 98.60%



\## Project Workflow



```text

Dataset

&#x20;  ↓

Data Inspection

&#x20;  ↓

Data Cleaning

&#x20;  ↓

Exploratory Data Analysis

&#x20;  ↓

Feature Preparation

&#x20;  ↓

Train / Test Split

&#x20;  ↓

Model Training

&#x20;  ↓

Model Evaluation

&#x20;  ↓

Model Comparison

&#x20;  ↓

Random Forest Selection

&#x20;  ↓

Model Saving

&#x20;  ↓

Loan Approval Prediction

