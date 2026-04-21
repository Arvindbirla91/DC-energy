"""
model_training.py
-----------------
Trains, evaluates, compares, and saves the three regression models.
Run this script independently to retrain models: python model_training.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_processing import load_and_clean_data
from preprocessing import engineer_features, get_feature_matrix, get_scaler_pipeline

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────

def evaluate_model(name, model, X_test, y_test):
    """
    Compute and print MAE, MSE, RMSE, R² for a fitted model.

    Args:
        name (str): Model display name.
        model: Fitted sklearn model/pipeline.
        X_test: Test feature matrix.
        y_test: True target values.

    Returns:
        dict: Metric results.
    """
    y_pred = model.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  ── {name} ──────────────────────")
    print(f"     MAE  : {mae:.4f}")
    print(f"     MSE  : {mse:.4f}")
    print(f"     RMSE : {rmse:.4f}")
    print(f"     R²   : {r2:.4f}")

    return {"model": name, "MAE": round(mae, 4), "MSE": round(mse, 4),
            "RMSE": round(rmse, 4), "R2": round(r2, 4)}


# ─────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────

def train_models(target_col='power_capacity_MW'):
    """
    Full ML training pipeline:
      1. Load & clean data
      2. Engineer features
      3. Prepare X, y
      4. Train/test split (80/20)
      5. Train all three models in sklearn Pipelines
      6. Evaluate and compare
      7. Select best by lowest RMSE
      8. Tune best model with RandomizedSearchCV (5-fold CV)
      9. Save best model and metadata

    Args:
        target_col (str): Target column for prediction.

    Returns:
        tuple: (best_pipeline, results_df, feature_names)
    """
    # ── Load & prepare data ──────────────────────────────────────────
    print("\n[STEP 1] Loading and cleaning data...")
    df = load_and_clean_data()

    print("[STEP 2] Engineering features...")
    df = engineer_features(df)

    print("[STEP 3] Preparing feature matrix...")
    X, y, feature_names = get_feature_matrix(df, target_col=target_col)

    # Drop rows where target is 0 or NaN
    mask = y.notna() & (y > 0)
    X, y = X[mask], y[mask]

    print(f"         Features: {feature_names}")
    print(f"         X shape: {X.shape} | y shape: {y.shape}")

    # ── Train/test split ─────────────────────────────────────────────
    print("[STEP 4] Splitting data: 80% train / 20% test (random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify numeric cols (non-climate-dummy columns)
    numeric_cols = [c for c in X.columns if not c.startswith('climate_')]
    preprocessor = get_scaler_pipeline(numeric_cols)

    # ── Define three model pipelines ─────────────────────────────────
    models = {
        "Linear Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('model', LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ('preprocessor', preprocessor),
            ('model', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ]),
    }

    # ── Train and evaluate all three ─────────────────────────────────
    print("\n[STEP 5] Training all three models...")
    results = []
    fitted_models = {}

    for name, pipeline in models.items():
        print(f"\n  Training {name}...")
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline
        result = evaluate_model(name, pipeline, X_test, y_test)
        results.append(result)

    results_df = pd.DataFrame(results).sort_values('RMSE')
    print("\n[STEP 6] Model comparison (sorted by RMSE):")
    print(results_df.to_string(index=False))

    # ── Select best model ─────────────────────────────────────────────
    best_name = results_df.iloc[0]['model']
    print(f"\n[STEP 7] Best model: {best_name}")
    best_pipeline = fitted_models[best_name]

    # ── Hyperparameter tuning ─────────────────────────────────────────
    print(f"\n[STEP 8] Tuning {best_name} with RandomizedSearchCV (5-fold)...")
    param_grids = {
        "Random Forest": {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [None, 5, 10, 20],
            'model__min_samples_split': [2, 5, 10],
        },
        "Gradient Boosting": {
            'model__n_estimators': [100, 200, 300],
            'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'model__max_depth': [3, 5, 7],
        },
        "Linear Regression": {},
    }

    param_grid = param_grids.get(best_name, {})
    if param_grid:
        search = RandomizedSearchCV(
            best_pipeline, param_grid,
            n_iter=10, cv=5, scoring='neg_root_mean_squared_error',
            random_state=42, n_jobs=-1, verbose=1,
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        print(f"  Best params: {search.best_params_}")
        evaluate_model(f"{best_name} (Tuned)", best_pipeline, X_test, y_test)
    else:
        print("  Linear Regression has no hyperparameters to tune. Skipping.")

    # ── Save model & metadata ─────────────────────────────────────────
    print("\n[STEP 9] Saving best model...")
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best_pipeline, model_path)
    print(f"  Saved: {model_path}")

    # Save feature names for inference
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    # Save metadata JSON
    best_metrics = results_df.iloc[0].to_dict()
    metadata = {
        "best_model": best_name,
        "target_column": target_col,
        "feature_names": feature_names,
        "metrics": best_metrics,
        "all_results": results,
    }
    with open(os.path.join(MODELS_DIR, "model_info.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved: models/model_info.json")

    print("\n[DONE] Training complete.\n")
    return best_pipeline, results_df, feature_names


if __name__ == "__main__":
    train_models()
