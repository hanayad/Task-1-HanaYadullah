import os
import re
import sys
from pathlib import Path

import pandas as pd


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def pick_dataset_file(datasets_dir: Path) -> Path:
    supported_patterns = ("*.xlsx", "*.xls", "*.csv")
    files = []
    for pattern in supported_patterns:
        files.extend(datasets_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No supported dataset file found in: {datasets_dir}\n"
            "Supported formats: .xlsx, .xls, .csv"
        )

    # Pick the newest file so it works even if user adds new datasets later.
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_dataset(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series

    as_text = series.astype("string")
    # Remove common currency symbols, commas, and spaces.
    normalized = as_text.str.replace(r"[,\$\u20b9€£\s]", "", regex=True)
    converted = pd.to_numeric(normalized, errors="coerce")

    # Keep conversion only when enough values are actually numeric.
    ratio_numeric = converted.notna().mean()
    if ratio_numeric >= 0.6:
        return converted
    return series


def clean_date_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series

    as_datetime = pd.to_datetime(series, errors="coerce", dayfirst=False)
    ratio_dates = as_datetime.notna().mean()
    if ratio_dates >= 0.6:
        return as_datetime
    return series


def normalize_text_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return series

    text = series.astype("string")
    text = text.str.strip()
    # Collapse repeated spaces inside text.
    text = text.str.replace(r"\s+", " ", regex=True)

    # Treat empty strings as missing.
    text = text.replace("", pd.NA)
    return text


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        series = df[col]
        if series.isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(series):
            fill_value = series.median()
            df[col] = series.fillna(fill_value)
        elif pd.api.types.is_datetime64_any_dtype(series):
            # Forward then backward fill to avoid leaving gaps.
            df[col] = series.ffill().bfill()
        else:
            mode = series.mode(dropna=True)
            if not mode.empty:
                df[col] = series.fillna(mode.iloc[0])
            else:
                df[col] = series.fillna("Unknown")
    return df


def print_missing_summary(df: pd.DataFrame, heading: str) -> None:
    print_section(heading)
    missing_count = df.isna().sum()
    missing_only = missing_count[missing_count > 0]
    if missing_only.empty:
        print("No missing values found.")
    else:
        print("Columns with missing values:")
        print(missing_only.sort_values(ascending=False))


def main() -> None:
    internship_dir = Path(r"C:\Users\Hana\OneDrive\Desktop\internship")
    datasets_dir = internship_dir / "datasets"

    print_section("Task 1 - Data Cleaning Script Started")
    print(f"Working folder: {internship_dir}")
    print(f"Datasets folder: {datasets_dir}")

    if not datasets_dir.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")

    dataset_file = pick_dataset_file(datasets_dir)
    print(f"Selected dataset file: {dataset_file.name}")

    df = load_dataset(dataset_file)
    print_section("Initial Dataset Snapshot")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nData types before cleaning:")
    print(df.dtypes)

    # 1) Identify missing/null values
    print_missing_summary(df, "Step 1 - Missing Value Identification (Before)")

    # 2) Remove duplicates
    print_section("Step 2 - Duplicate Removal")
    duplicates_before = df.duplicated().sum()
    print(f"Duplicate rows found: {duplicates_before}")
    df = df.drop_duplicates().copy()
    duplicates_after = df.duplicated().sum()
    print(f"Duplicate rows after removal: {duplicates_after}")
    print(f"Rows after duplicate removal: {df.shape[0]}")

    # 3) Correct data formats
    print_section("Step 3 - Data Format Correction")
    original_dtypes = df.dtypes.astype(str)

    for col in df.columns:
        lower_col = col.lower()
        series = df[col]

        # Keep already clean numeric/date columns untouched.
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            df[col] = series
            continue

        # Normalize text fields.
        series = normalize_text_series(series)

        # Parse dates only for clearly date-like columns.
        if re.search(r"(date|dob|day|month|year|time)", lower_col):
            series = clean_date_series(series)
        else:
            # Try numeric conversion for non-date text columns.
            series = clean_numeric_series(series)

        df[col] = series

    print("Data types before correction:")
    print(original_dtypes)
    print("\nData types after correction:")
    print(df.dtypes)

    # Fill missing values after type correction.
    print_section("Step 4 - Missing Value Handling")
    df = fill_missing_values(df)
    print_missing_summary(df, "Missing Value Identification (After)")

    output_file = internship_dir / "task1_cleaned_output.xlsx"
    df.to_excel(output_file, index=False)

    print_section("Cleaning Completed Successfully")
    print(f"Final shape: {df.shape}")
    print(f"Cleaned file saved to: {output_file}")
    print("\nPreview of cleaned data (first 10 rows):")
    print(df.head(10))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR: Data cleaning failed.")
        print(f"Reason: {exc}")
        sys.exit(1)
