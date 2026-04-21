"""
preprocessing.py
----------------
Feature engineering, encoding, and scaling logic.
Produces model-ready feature matrices from the cleaned DataFrame.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

def engineer_features(df):
    """
    Create new derived features from the cleaned DataFrame.

    New features:
        - renewable_capacity_MW: actual renewable MW (power * renewable %)
        - colocation_ratio: colocation / total data centers
        - hyperscale_ratio: hyperscale / total data centers

    Args:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: DataFrame with new engineered features appended.
    """
    df = df.copy()

    # Renewable actual capacity
    if 'power_capacity_MW' in df.columns and 'avg_renewable_energy_pct' in df.columns:
        df['renewable_capacity_MW'] = (
            df['power_capacity_MW'] * df['avg_renewable_energy_pct']
        ).round(4)

    # Colocation ratio
    if 'colocation_data_centers' in df.columns and 'total_data_centers' in df.columns:
        df['colocation_ratio'] = (
            df['colocation_data_centers'] / df['total_data_centers'].replace(0, np.nan)
        ).round(4)

    # Hyperscale ratio
    if 'hyperscale_data_centers' in df.columns and 'total_data_centers' in df.columns:
        df['hyperscale_ratio'] = (
            df['hyperscale_data_centers'] / df['total_data_centers'].replace(0, np.nan)
        ).round(4)

    df.fillna(0, inplace=True)
    print("[FEATURE] Engineered: renewable_capacity_MW, colocation_ratio, hyperscale_ratio")
    return df


# ─────────────────────────────────────────────
# COUNTRY → CLIMATE LOOKUP
# ─────────────────────────────────────────────

def build_country_climate_map(df):
    """
    Build a dictionary mapping each country to its Climate_Type.
    Used by Model 2 to auto-determine climate from country input.

    Args:
        df (pd.DataFrame): Cleaned DataFrame with 'country' and 'Climate_Type'.

    Returns:
        dict: {country_name: climate_type}
    """
    if 'country' not in df.columns or 'Climate_Type' not in df.columns:
        return {}
    return dict(zip(df['country'], df['Climate_Type']))


# ─────────────────────────────────────────────
# PREPARE FEATURES FOR MODELING
# ─────────────────────────────────────────────

NUMERIC_FEATURES = [
    'total_data_centers',
    'power_capacity_MW',
    'avg_renewable_energy_pct',
    'growth_rate_pct',
    'avg_latency_ms',
    'internet_penetration_pct',
    'renewable_capacity_MW',
    'colocation_ratio',
    'hyperscale_ratio',
]

CATEGORICAL_FEATURES = ['Climate_Type']

TARGET_COLUMN = 'power_capacity_MW'  # Primary target for regression


def get_feature_matrix(df, target_col=None):
    """
    Extract feature matrix (X) and target vector (y) from the DataFrame.
    Applies label encoding to 'country' and one-hot encoding to 'Climate_Type'.

    Args:
        df (pd.DataFrame): Engineered DataFrame.
        target_col (str): Column name to use as prediction target.
                          Defaults to 'power_capacity_MW'.

    Returns:
        tuple: (X_df, y_series, feature_names)
    """
    if target_col is None:
        target_col = TARGET_COLUMN

    df = df.copy()

    # Label encode country
    le = LabelEncoder()
    if 'country' in df.columns:
        df['country_encoded'] = le.fit_transform(df['country'])
        joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder_country.pkl"))

    # Select numeric features (available in this df)
    available_numerics = [c for c in NUMERIC_FEATURES if c in df.columns and c != target_col]
    available_numerics.append('country_encoded')

    # One-hot encode Climate_Type
    climate_dummies = pd.get_dummies(df['Climate_Type'], prefix='climate')
    feature_df = pd.concat([df[available_numerics].reset_index(drop=True),
                             climate_dummies.reset_index(drop=True)], axis=1)

    y = df[target_col].copy()
    feature_names = feature_df.columns.tolist()

    return feature_df, y, feature_names


def get_scaler_pipeline(numeric_cols):
    """
    Build a column transformer that scales numeric columns and
    passes through already-encoded dummy columns.

    Args:
        numeric_cols (list): List of numeric column names to scale.

    Returns:
        ColumnTransformer: Fitted-ready preprocessor.
    """
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_cols)
    ], remainder='passthrough')
    return preprocessor
