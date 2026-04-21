"""
data_processing.py
------------------
Handles raw data ingestion, cleaning, type fixing, and storage.
All cleaning logic is centralized here and reused across all pages.
"""

import pandas as pd
import numpy as np
import re
import os

RAW_PATH = os.path.join("datasets", "raw", "Data_center_dataset.csv")
PROCESSED_PATH = os.path.join("datasets", "processed", "cleaned_data.csv")


# ─────────────────────────────────────────────
# HELPER: Parse messy numeric strings
# ─────────────────────────────────────────────

def _parse_numeric(value):
    """
    Convert messy string values like '~12,000+', '40%+', '0.45', '96%'
    into clean float numbers. Returns NaN if unparseable.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    # Remove leading ~ and trailing + characters
    s = s.replace('~', '').replace('+', '').replace(',', '').strip()
    # Remove trailing % sign and convert percentage strings
    if s.endswith('%'):
        s = s[:-1].strip()
        try:
            val = float(s)
            # Normalize: if value > 1, treat as actual percentage (e.g. 27 → 0.27)
            return val / 100.0 if val > 1 else val
        except ValueError:
            return np.nan
    # Handle decimal fractions already in 0.x format
    try:
        val = float(s)
        # If it looks like a fraction (0 < val <= 1), keep as-is (already a ratio)
        return val
    except ValueError:
        return np.nan


# ─────────────────────────────────────────────
# CORE: Load & Clean Dataset
# ─────────────────────────────────────────────

def load_and_clean_data(force_reprocess=False):
    """
    Load the raw dataset, clean it, and save the processed version.

    Steps:
      1. Load raw CSV
      2. Print profile (shape, dtypes, nulls, duplicates)
      3. Drop unnamed/junk columns
      4. Fix data types (parse numeric strings)
      5. Impute missing values
      6. Handle outliers via IQR
      7. Save to processed/cleaned_data.csv

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for analysis.
    """
    # Return cached processed file if it exists and re-processing not forced
    if os.path.exists(PROCESSED_PATH) and not force_reprocess:
        print(f"[INFO] Loading cached cleaned data from {PROCESSED_PATH}")
        return pd.read_csv(PROCESSED_PATH)

    print(f"[INFO] Loading raw data from {RAW_PATH} ...")
    try:
        df = pd.read_csv(RAW_PATH)
    except FileNotFoundError:
        raise FileNotFoundError(f"Raw dataset not found at: {RAW_PATH}")

    # ── Step 1: Profile ──────────────────────────────────────────────
    print(f"\n[PROFILE] Shape: {df.shape}")
    print(f"[PROFILE] Columns: {df.columns.tolist()}")
    print(f"[PROFILE] Null counts:\n{df.isnull().sum()}")
    print(f"[PROFILE] Duplicate rows: {df.duplicated().sum()}")

    # ── Step 2: Drop unnamed / junk columns ─────────────────────────
    unnamed_cols = [c for c in df.columns if 'Unnamed' in c]
    df.drop(columns=unnamed_cols, inplace=True)
    print(f"[CLEAN] Dropped unnamed columns: {unnamed_cols}")

    # ── Step 3: Remove duplicates ────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"[CLEAN] Removed {before - len(df)} duplicate rows.")

    # ── Step 4: Rename columns for consistency ───────────────────────
    df.rename(columns={
        'power_capacity_MW_total': 'power_capacity_MW',
        'average_renewable_energy_usage_percent': 'avg_renewable_energy_pct',
        'growth_rate_of_data_centers_percent_per_year': 'growth_rate_pct',
        'Climate': 'Climate_Type',
        'internet_penetration_percent': 'internet_penetration_pct',
        'avg_latency_to_global_hubs_ms': 'avg_latency_ms',
    }, inplace=True)

    # ── Step 5: Fix numeric types ────────────────────────────────────
    numeric_string_cols = [
        'power_capacity_MW',
        'avg_renewable_energy_pct',
        'growth_rate_pct',
        'colocation_data_centers',
        'hyperscale_data_centers',
        'internet_penetration_pct',
        'number_of_fiber_connections',
        'floor_space_sqft_total',
    ]

    for col in numeric_string_cols:
        if col in df.columns:
            df[col] = df[col].apply(_parse_numeric)
            print(f"[CLEAN] Parsed numeric column: {col}")

    # avg_latency_ms is already numeric strings without symbols
    if 'avg_latency_ms' in df.columns:
        df['avg_latency_ms'] = pd.to_numeric(df['avg_latency_ms'], errors='coerce')

    # total_data_centers is already int64 — no action needed

    # ── Step 6: Impute missing values ────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[IMPUTE] {col}: filled {missing} NaN with median={median_val:.4f}")

    categorical_cols = df.select_dtypes(include=['object', 'str']).columns.tolist()
    for col in categorical_cols:
        missing = df[col].isnull().sum()
        if missing > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            # Sanitize mode_val for safe printing (remove non-ASCII)
            safe_mode = str(mode_val).encode('ascii', errors='replace').decode('ascii')
            print(f"[IMPUTE] {col}: filled {missing} NaN with mode='{safe_mode}'")

    # ── Step 7: Outlier handling — only for percentage cols, skip ranking cols ─
    # NOTE: DO NOT clip total_data_centers or power_capacity_MW — doing so
    # collapses top-ranked countries (US, Germany, etc.) to identical ceiling
    # values, making all bar charts look the same.
    pct_clip_cols = ['avg_renewable_energy_pct', 'growth_rate_pct']
    for col in pct_clip_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 3.0 * IQR   # relaxed multiplier (3×) to preserve real range
            upper = Q3 + 3.0 * IQR
            outliers = ((df[col] < lower) | (df[col] > upper)).sum()
            if outliers > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                print(f"[OUTLIER] {col}: clipped {outliers} outliers to [{lower:.2f}, {upper:.2f}]")

    # ── Step 8: Save cleaned dataset ─────────────────────────────────
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\n[DONE] Cleaned data saved to {PROCESSED_PATH}")
    print(f"[DONE] Final shape: {df.shape}")

    return df


def get_data_profile(df):
    """
    Return a dictionary with key profiling stats for display in the UI.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        dict: Profile summary.
    """
    return {
        "total_countries": df['country'].nunique(),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "climate_types": df['Climate_Type'].nunique(),
        "avg_power_capacity_mw": round(df['power_capacity_MW'].mean(), 2),
        # avg_renewable_energy_pct is stored as 0-100 (e.g. 44.15 = 44.15%)
        "avg_renewable_pct": round(df['avg_renewable_energy_pct'].mean(), 2),
        # growth_rate_pct is stored as raw % (e.g. 5.2 = 5.2%/yr)
        "avg_growth_rate": round(df['growth_rate_pct'].mean(), 2),
        "top_country": df.loc[df['total_data_centers'].idxmax(), 'country'],
        "max_data_centers": int(df['total_data_centers'].max()),
    }
