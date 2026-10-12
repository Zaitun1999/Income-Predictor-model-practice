import os
import random
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from typing import cast
pd.set_option('display.max_columns', None)
# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

RAW_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\data\raw\train.csv"
PROCESSED_PATH = r"C:\Users\Personal\PyCharmMiscProject\PythonProject\data\processed\train_clean.csv"

# Columns to rename (applied AFTER uppercasing). Change the right-hand sides if you want other names.
RENAME_MAP = {
    'FNLWGT': 'FINAL-WEIGHT',
    'EDUCATIONAL-NUM': 'EDUCATION-NUM',
    'WORKCLASS' : 'WORK-CLASS',
    'INCOME_>50K' : 'INCOME-IS->50K',
}
def load_raw(path: str) -> pd.DataFrame:
    """Load the CSV with basic error checking."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at {path}")

    df = cast(pd.DataFrame, pd.read_csv(path)) # type: ignore[call-overload]
    if df.empty:
        raise ValueError("The file is empty")
    return df


def profile(df: pd.DataFrame, title: str) -> None:
    """Print a summary of the dataset."""
    print(f"\n=== PROFILE: {title} ===")
    print(df.head(20))
    print("Shape:", df.shape)
    df.info()
    print("\nMissing values:\n", df.isna().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    print("\nUnique values:\n", df.nunique())
    print("\nStatistics:\n", df.describe().T)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data: uppercase + renamed columns, whitespace/'?' handling,
    KNN imputation only where values are missing, and no decimals in the output.
    """
    print("\n--- Starting Data Cleaning ---")
    df_clean = df.copy()

    # 1. Uppercase column names, then rename the requested ones
    df_clean.columns = df_clean.columns.str.strip().str.upper()
    df_clean = df_clean.rename(columns=RENAME_MAP)
    print("Columns:", list(df_clean.columns))

    # 2. Tidy text columns and turn placeholders like '?' into real missing values
    cat_cols = list(df_clean.select_dtypes(exclude=['number']).columns)
    num_cols = list(df_clean.select_dtypes(include=['number']).columns)

    for col in cat_cols:
        df_clean[col] = df_clean[col].astype('string').str.strip()
        df_clean[col] = df_clean[col].replace({'?': pd.NA, '': pd.NA})
        df_clean[col] = df_clean[col].astype(object).where(df_clean[col].notna(), np.nan)

    # 3. Remove exact duplicate rows
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    print(f"Removed {before - len(df_clean)} duplicate rows.")

    missing_total = int(df_clean.isna().sum().sum())
    print("Missing values before imputation:", missing_total)

    # 4. KNN imputation - ONLY if something is actually missing
    if missing_total > 0:
        # 4a. Encode categories as integer codes (NaN stays NaN)
        mappings = {}
        for col in cat_cols:
            codes, uniques = pd.factorize(df_clean[col]) # -1 means NaN
            mappings[col] = dict(enumerate(uniques))
            df_clean[col] = pd.Series(codes, index=df_clean.index).replace(-1, np.nan)

        # 4b. Scale so big columns (e.g. FINAL-WEIGHT) don't dominate the distance
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_clean) # NaNs are ignored by the scaler

        imputer = KNNImputer(n_neighbors=5)
        imputed = imputer.fit_transform(scaled)
        df_imputed = pd.DataFrame(scaler.inverse_transform(imputed), columns=df_clean.columns)

        # 4c. Round back to whole numbers and map category codes back to text
        for col in cat_cols:
            max_code = len(mappings[col]) - 1
            codes = df_imputed[col].round().clip(0, max_code).astype(int)
            df_imputed[col] = codes.map(mappings[col])
        print("KNN imputation applied.")
    else:
        df_imputed = df_clean
        print("No missing values found, KNN imputation not needed.")

    # 5. No decimals: every numeric column is whole numbers in the original data
    for col in num_cols:
        col = RENAME_MAP.get(col.upper(), col.upper())
        if col in df_imputed.columns:
            df_imputed[col] = df_imputed[col].round().astype(int)

    # 6. Final check
    print("\nRemaining missing values in the entire dataset:")
    print(df_imputed.isna().sum().sum())

    return df_imputed

def save_data(df: pd.DataFrame, path: str) -> None:
    """Saves the processed DataFrame to a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSuccessfully saved cleaned data to: {path}")


if __name__ == "__main__":
    try:
        # Step 1: Load raw data
        raw_df = load_raw(RAW_PATH)
        profile(raw_df, title="RAW")

        # Step 2: Clean and impute data
        cleaned_df = clean_data(raw_df)

        # Step 3: Save cleaned data to the processed folder
        save_data(cleaned_df, PROCESSED_PATH)

        # Step 4: Show the first 20 rows of the cleaned data
        print("\n--- First 20 rows of Cleaned Data ---")
        print(cleaned_df.head(20))

    except Exception as e:
        print(f"\nAn error occurred: {e}")



