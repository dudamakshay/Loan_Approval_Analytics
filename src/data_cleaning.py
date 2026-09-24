import pandas as pd


def load_data(file_path):
    """Load the raw loan dataset."""
    return pd.read_csv(file_path)


def clean_data(df):
    """Clean and prepare the loan dataset."""

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove duplicate rows
    df = df.drop_duplicates().copy()

    # Clean text columns
    categorical_columns = [
        "education",
        "self_employed",
        "loan_status"
    ]

    for column in categorical_columns:
        df[column] = df[column].astype(str).str.strip()

    return df


def prepare_features(df):
    """Separate input features from the target."""

    # loan_id is an identifier and should not be used for prediction
    X = df.drop(columns=["loan_id", "loan_status"])

    y = df["loan_status"]

    return X, y


if __name__ == "__main__":

    file_path = r"D:\AI-Projects\Loan_Approval_Analytics\data\loan_approval_dataset.csv"

    df = load_data(file_path)

    print("\n========== ORIGINAL DATA ==========")
    print("Shape:", df.shape)

    df = clean_data(df)

    print("\n========== CLEANED DATA ==========")
    print("Shape:", df.shape)

    print("\n========== CLEANED COLUMNS ==========")
    print(df.columns.tolist())

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATES ==========")
    print(df.duplicated().sum())

    print("\n========== TARGET DISTRIBUTION ==========")
    print(df["loan_status"].value_counts())

    X, y = prepare_features(df)

    print("\n========== FEATURES ==========")
    print(X.columns.tolist())

    print("\n========== TARGET ==========")
    print(y.name)