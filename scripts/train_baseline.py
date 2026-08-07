"""Train a deliberately simple demand baseline and register it in MLflow.

The model is intentionally boring - the platform is the point. What matters
here is an honest time-based split and a registered, versioned model.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA = "data/processed/demand_2024-01.parquet"
FEATURES = ["zone_id", "hour", "day_of_week", "is_weekend", "is_holiday"]
TARGET = "demand"
EXPERIMENT = "sentinel-demand-baseline"
MODEL_NAME = "demand-forecaster"


def main():
    df = pd.read_parquet(DATA).sort_values("pickup_hour").reset_index(drop=True)

    # --- Honest time-based split: train on earlier hours, test on later ---
    cutoff = df["pickup_hour"].quantile(0.8)
    train = df[df["pickup_hour"] <= cutoff]
    test = df[df["pickup_hour"] > cutoff]
    print(f"Train: {len(train)} rows (<= {cutoff})")
    print(f"Test:  {len(test)} rows (> {cutoff})")

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run() as run:
        params = {
            "max_iter": 200,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "random_state": 42,
        }
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds) ** 0.5

        # Naive seasonal baseline for context: predict train mean per (zone,hour)
        seasonal = (
            train.groupby(["zone_id", "hour"])[TARGET].mean().rename("naive").reset_index()
        )
        test_naive = test.merge(seasonal, on=["zone_id", "hour"], how="left")
        test_naive["naive"] = test_naive["naive"].fillna(y_train.mean())
        naive_mae = mean_absolute_error(test_naive[TARGET], test_naive["naive"])

        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("naive_mae", naive_mae)

        mlflow.sklearn.log_model(
            model, name="model", registered_model_name=MODEL_NAME
        )

        print(f"\nModel MAE:  {mae:.2f}")
        print(f"Model RMSE: {rmse:.2f}")
        print(f"Naive MAE:  {naive_mae:.2f}  (baseline to beat)")
        print(f"\nRun ID: {run.info.run_id}")
        print(f"Registered as: {MODEL_NAME}")


if __name__ == "__main__":
    main()
