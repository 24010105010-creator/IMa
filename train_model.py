"""
Polynomial Regression Model Training Script
Dataset: Advertising (TV, Radio, Newspaper -> Sales)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def load_and_preprocess_data(data_path="advertising.csv"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    df = df.dropna()

    feature_cols = ["TV", "Radio", "Newspaper"]
    target_col = "Sales"

    X = df[feature_cols]
    y = df[target_col]

    return df, X, y


def evaluate_degrees(X_train, X_test, y_train, y_test, max_degree=4):
    """
    Evaluates Polynomial Regression models across degrees 1 to max_degree.
    Returns comparison dataframe and best degree.
    """
    results = []

    for deg in range(1, max_degree + 1):
        pipeline = Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ])

        pipeline.fit(X_train, y_train)

        train_pred = pipeline.predict(X_train)
        test_pred = pipeline.predict(X_test)

        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)

        # 5-Fold Cross Validation on training set
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_r2 = cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="r2").mean()
        cv_rmse = np.sqrt(-cross_val_score(pipeline, X_train, y_train, cv=kf, scoring="neg_mean_squared_error")).mean()

        # Adjusted R2 for test set
        n = len(y_test)
        p = pipeline.named_steps["poly"].n_output_features_
        adj_r2 = 1 - (1 - test_r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else float("nan")

        results.append({
            "degree": deg,
            "num_features": p,
            "train_r2": round(float(train_r2), 4),
            "test_r2": round(float(test_r2), 4),
            "adj_test_r2": round(float(adj_r2), 4) if not np.isnan(adj_r2) else None,
            "train_rmse": round(float(train_rmse), 4),
            "test_rmse": round(float(test_rmse), 4),
            "train_mae": round(float(train_mae), 4),
            "test_mae": round(float(test_mae), 4),
            "cv_r2_mean": round(float(cv_r2), 4),
            "cv_rmse_mean": round(float(cv_rmse), 4)
        })

    results_df = pd.DataFrame(results)
    # Best degree chosen by highest test_r2 and cv_r2 without extreme overfitting
    best_row = results_df.sort_values(by="cv_r2_mean", ascending=False).iloc[0]
    best_degree = int(best_row["degree"])

    return results_df, best_degree


def train_and_save_model(best_degree=2, output_model_path="model.joblib", metrics_path="model_metrics.json"):
    df, X, y = load_and_preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    degree_comparison, auto_best_degree = evaluate_degrees(X_train, X_test, y_train, y_test, max_degree=4)

    # Use specified best_degree or auto detected
    selected_degree = best_degree if best_degree is not None else auto_best_degree

    print(f"--- Degree Comparison Table ---")
    print(degree_comparison.to_string(index=False))
    print(f"\nOptimal Degree Selected: {selected_degree}")

    # Build optimal pipeline with Ridge for numerical stability
    best_pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=selected_degree, include_bias=False)),
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0))
    ])

    best_pipeline.fit(X_train, y_train)

    # Evaluate optimal model
    y_train_pred = best_pipeline.predict(X_train)
    y_test_pred = best_pipeline.predict(X_test)

    poly_feature_names = best_pipeline.named_steps["poly"].get_feature_names_out(X.columns).tolist()
    coefficients = best_pipeline.named_steps["regressor"].coef_.tolist()
    intercept = float(best_pipeline.named_steps["regressor"].intercept_)

    metrics = {
        "dataset_shape": list(df.shape),
        "selected_degree": selected_degree,
        "features": list(X.columns),
        "num_poly_features": len(poly_feature_names),
        "poly_feature_names": poly_feature_names,
        "intercept": round(intercept, 4),
        "coefficients": [round(c, 4) for c in coefficients],
        "train_metrics": {
            "r2": round(float(r2_score(y_train, y_train_pred)), 4),
            "mae": round(float(mean_absolute_error(y_train, y_train_pred)), 4),
            "mse": round(float(mean_squared_error(y_train, y_train_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_train, y_train_pred))), 4),
        },
        "test_metrics": {
            "r2": round(float(r2_score(y_test, y_test_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_test_pred)), 4),
            "mse": round(float(mean_squared_error(y_test, y_test_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_test_pred))), 4),
        },
        "degree_comparison": degree_comparison.to_dict(orient="records")
    }

    # Save model pipeline
    joblib.dump(best_pipeline, output_model_path)
    print(f"\nModel pipeline saved to: {output_model_path}")

    # Save metrics JSON
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")

    print("\n--- Model Performance Summary ---")
    print(f"Train R^2: {metrics['train_metrics']['r2']:.4f} | Train RMSE: {metrics['train_metrics']['rmse']:.4f}")
    print(f"Test R^2:  {metrics['test_metrics']['r2']:.4f} | Test RMSE:  {metrics['test_metrics']['rmse']:.4f}")

    return best_pipeline, metrics


if __name__ == "__main__":
    train_and_save_model(best_degree=2)
