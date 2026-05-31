import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 1. Load engineered dataset
# =========================================================
df = pd.read_csv("Traffic_engineered_fixed.csv")

# =========================================================
# 2. Define feature sets
# =========================================================
full_feature_cols = [
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

reduced_feature_cols = [
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

# =========================================================
# 3. Final 80/20 split
# First 80% train, last 20% test
# =========================================================
split_idx = int(len(df) * 0.8)

# Full features
X_full = df[full_feature_cols]
y = df[target_col]

X_full_train = X_full.iloc[:split_idx]
X_full_test  = X_full.iloc[split_idx:]

# Reduced features
X_reduced = df[reduced_feature_cols]
X_reduced_train = X_reduced.iloc[:split_idx]
X_reduced_test  = X_reduced.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

print("Training rows:", len(y_train))
print("Testing rows:", len(y_test))

# =========================================================
# 4. Train models
# =========================================================

# 4A. Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_full_train, y_train)
y_pred_lr = lr_model.predict(X_full_test)

# 4B. Full Random Forest (best parameters)
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_full_train, y_train)
y_pred_rf = rf_model.predict(X_full_test)

# 4C. Reduced Random Forest (best parameters)
rf_reduced_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=2,
    min_samples_leaf=4,
    random_state=42,
    n_jobs=-1
)
rf_reduced_model.fit(X_reduced_train, y_train)
y_pred_rf_reduced = rf_reduced_model.predict(X_reduced_test)

# =========================================================
# 5. Metric function
# =========================================================
def get_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

lr_mae, lr_rmse, lr_r2 = get_metrics(y_test, y_pred_lr)
rf_mae, rf_rmse, rf_r2 = get_metrics(y_test, y_pred_rf)
rf_red_mae, rf_red_rmse, rf_red_r2 = get_metrics(y_test, y_pred_rf_reduced)

# =========================================================
# 6. FIGURE / TABLE 1: Model comparison table
# =========================================================
comparison_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest",
        "Reduced Random Forest"
    ],
    "MAE": [lr_mae, rf_mae, rf_red_mae],
    "RMSE": [lr_rmse, rf_rmse, rf_red_rmse],
    "R²": [lr_r2, rf_r2, rf_red_r2]
})

comparison_df["MAE"] = comparison_df["MAE"].round(3)
comparison_df["RMSE"] = comparison_df["RMSE"].round(3)
comparison_df["R²"] = comparison_df["R²"].round(3)

print("\nModel comparison table:")
print(comparison_df)

# Save table as CSV
comparison_df.to_csv("figures/figure_table_1_model_comparison.csv", index=False)

# Make a visual table figure
fig, ax = plt.subplots(figsize=(8, 2.2))
ax.axis("off")

table = ax.table(
    cellText=comparison_df.values,
    colLabels=comparison_df.columns,
    loc="center",
    cellLoc="center"
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)

plt.title("Figure/Table 1. Model Performance Comparison", pad=12)
plt.tight_layout()
plt.savefig("figures/figure_table_1_model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 7. FIGURE 2: Actual vs Predicted plot for final Random Forest
# =========================================================
# Use the final chosen model = full Random Forest

plot_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred_rf
}).reset_index(drop=True)

# Plot only first 150 points so the figure is readable
n_plot = min(200, len(plot_df))

plt.figure(figsize=(10, 5))
plt.plot(plot_df.index[:n_plot], plot_df["Actual"][:n_plot], label="Actual")
plt.plot(plot_df.index[:n_plot], plot_df["Predicted"][:n_plot], label="Predicted")
plt.xlabel("Test-set sample index")
plt.ylabel("Traffic count")
plt.title("Actual vs Predicted Traffic Count (Best Random Forest Hyperparameters)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/figure_2_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 8. FIGURE 3: Feature importance bar chart
# =========================================================
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=full_feature_cols
).sort_values(ascending=False)

top_n = 10
top_features = feature_importance.head(top_n)

plt.figure(figsize=(9, 5))
top_features.sort_values().plot(kind="barh")
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances (Best Random Forest)")
plt.tight_layout()
plt.savefig("figures/figure_3_feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 9. Optional: save prediction data for later use
# =========================================================
plot_df.to_csv("figures/figure_2_actual_vs_predicted_data.csv", index=False)
feature_importance.to_csv("figures/figure_3_feature_importance_data.csv", header=["Importance"])

print("\nSaved files:")
print("- figure_table_1_model_comparison.csv")
print("- figure_table_1_model_comparison.png")
print("- figure_2_actual_vs_predicted.png")
print("- figure_2_actual_vs_predicted_data.csv")
print("- figure_3_feature_importance.png")
print("- figure_3_feature_importance_data.csv")



