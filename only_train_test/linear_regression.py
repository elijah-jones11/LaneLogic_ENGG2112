# FINAL for TESTING --> WITH OPTIMAL PARAMETERS
# FINAL for TESTING --> WITH OPTIMAL PARAMETERS
# FINAL for TESTING --> WITH OPTIMAL PARAMETERS
# FINAL for TESTING --> WITH OPTIMAL PARAMETERS
# FINAL for TESTING --> WITH OPTIMAL PARAMETERS


# AI MODEL --> FIRST LINEAR REGRESSION THEN RANDOM FOREST REGRESSOR:
# AI MODEL --> FIRST LINEAR REGRESSION THEN RANDOM FOREST REGRESSOR:
# AI MODEL --> FIRST LINEAR REGRESSION THEN RANDOM FOREST REGRESSOR:
# AI MODEL --> FIRST LINEAR REGRESSION THEN RANDOM FOREST REGRESSOR:
# AI MODEL --> FIRST LINEAR REGRESSION THEN RANDOM FOREST REGRESSOR:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 1. Load engineered dataset
# =========================================================
df = pd.read_csv("Traffic_engineered_fixed.csv")

# =========================================================
# 2. Choose input features and target
# =========================================================
feature_cols = [
    "Hour",
    "Minute",
    "QuarterHour",
    "DayOfWeekNum",
    "IsWeekend",
    "Hour_sin",
    "Hour_cos",
    "Day_sin",
    "Day_cos",
    "CarCount",
    "BikeCount",
    "BusCount",
    "TruckCount",
    "TotalCount",
    "Total_lag_1",
    "Total_lag_2",
    "Total_lag_3",
    "Total_lag_4",
    "Car_lag_1",
    "Bike_lag_1",
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
# 3. Time-ordered split
# Use first 80% for training, last 20% for testing
# =========================================================
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# =========================================================
# 4. Train Linear Regression model
# =========================================================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================================================
# 5. Make predictions on test set
# =========================================================
y_pred = model.predict(X_test)

# =========================================================
# 6. Evaluate performance
# =========================================================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Results")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

# =========================================================
# 7. Show first few predictions
# =========================================================
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nFirst 10 predictions:")
print(results.head(10))

