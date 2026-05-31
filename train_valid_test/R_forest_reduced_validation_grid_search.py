import pandas as pd
import numpy as np
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 1. Load engineered dataset
# =========================================================
df = pd.read_csv("Traffic_engineered_fixed.csv")

# =========================================================
# 2. Choose REDUCED input features and target
# =========================================================
feature_cols = [
    "Hour",
    "Minute",
    "QuarterHour",
    "Hour_sin",
    "Hour_cos",
    "CarCount",
    "BusCount",
    "TruckCount",
    "TotalCount",
    "Total_lag_1",
    "Total_lag_2",
    "Total_lag_3",
    "Total_lag_4",
    "Car_lag_1",
    "Bus_lag_1",
    "Truck_lag_1",
    "Total_roll_mean_3",
    "Total_roll_mean_4",
    "Total_roll_std_4",
    "Total_change_1",
    "Total_pct_change_1"
]

target_col = "Target_Next_Total"

X = df[feature_cols]
y = df[target_col]

# =========================================================
# 3. Time-ordered split: 70% train, 10% val, 20% test
# =========================================================
n = len(df)

train_end = int(n * 0.70)
val_end = int(n * 0.80)

X_train = X.iloc[:train_end]
X_val   = X.iloc[train_end:val_end]
X_test  = X.iloc[val_end:]

y_train = y.iloc[:train_end]
y_val   = y.iloc[train_end:val_end]
y_test  = y.iloc[val_end:]

print("Training rows:  ", len(X_train))
print("Validation rows:", len(X_val))
print("Testing rows:   ", len(X_test))

# =========================================================
# 4. Hyperparameter grid
# =========================================================
n_estimators_grid = [100, 200, 300]
max_depth_grid = [8, 10, 12]
min_samples_split_grid = [2, 5, 10]
min_samples_leaf_grid = [1, 2, 4]

param_grid = list(product(
    n_estimators_grid,
    max_depth_grid,
    min_samples_split_grid,
    min_samples_leaf_grid
))

print("\nTotal models to test:", len(param_grid))

# =========================================================
# 5. Validation-based grid search
# =========================================================
results = []

for i, (n_estimators, max_depth, min_samples_split, min_samples_leaf) in enumerate(param_grid, start=1):
    print(f"\nTesting model {i}/{len(param_grid)}")
    print(
        f"n_estimators={n_estimators}, "
        f"max_depth={max_depth}, "
        f"min_samples_split={min_samples_split}, "
        f"min_samples_leaf={min_samples_leaf}"
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    results.append({
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_r2": val_r2
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Sort by best validation MAE (lowest is best)
results_df = results_df.sort_values(by="val_mae", ascending=True).reset_index(drop=True)

print("\n==============================")
print("TOP 10 PARAMETER SETTINGS")
print("==============================")
print(results_df.head(10).to_string(index=False))

# =========================================================
# 6. Select best parameters
# =========================================================
best_params = results_df.iloc[0]

best_n_estimators = int(best_params["n_estimators"])
best_max_depth = int(best_params["max_depth"])
best_min_samples_split = int(best_params["min_samples_split"])
best_min_samples_leaf = int(best_params["min_samples_leaf"])

print("\n==============================")
print("BEST VALIDATION MODEL")
print("==============================")
print(f"n_estimators      = {best_n_estimators}")
print(f"max_depth         = {best_max_depth}")
print(f"min_samples_split = {best_min_samples_split}")
print(f"min_samples_leaf  = {best_min_samples_leaf}")
print(f"Validation MAE    = {best_params['val_mae']:.3f}")
print(f"Validation RMSE   = {best_params['val_rmse']:.3f}")
print(f"Validation R²     = {best_params['val_r2']:.3f}")

# =========================================================
# 7. Refit best model on TRAINING data only
# Then evaluate on TEST data
# =========================================================
best_model = RandomForestRegressor(
    n_estimators=best_n_estimators,
    max_depth=best_max_depth,
    min_samples_split=best_min_samples_split,
    min_samples_leaf=best_min_samples_leaf,
    random_state=42,
    n_jobs=-1
)

best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\n==============================")
print("FINAL TEST PERFORMANCE")
print("==============================")
print(f"Test MAE:  {test_mae:.3f}")
print(f"Test RMSE: {test_rmse:.3f}")
print(f"Test R²:   {test_r2:.3f}")

# =========================================================
# 8. Show first few test predictions
# =========================================================
results_test = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_test_pred
})

print("\nFirst 10 test predictions:")
print(results_test.head(10))

# =========================================================
# 9. Show feature importance for the best model
# =========================================================
feature_importance = pd.Series(best_model.feature_importances_, index=feature_cols)
feature_importance = feature_importance.sort_values(ascending=False)

print("\nFeature Importance (Best Reduced Model):")
print(feature_importance)

# =========================================================
# 10. Save tuning results to CSV
# =========================================================
results_df.to_csv("train_valid_test/rf_reduced_grid_search_results.csv", index=False)
print("\nSaved all reduced-model grid-search results to rf_reduced_grid_search_results.csv")

