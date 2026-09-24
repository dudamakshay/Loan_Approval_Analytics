import pandas as pd

file_path = r"D:\AI-Projects\Loan_Approval_Analytics\data\loan_approval_dataset.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET SHAPE ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n========== COLUMN NAMES ==========")
for column in df.columns:
    print(repr(column))

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== TARGET DISTRIBUTION ==========")
print(df[" loan_status"].value_counts())