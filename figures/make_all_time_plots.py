import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor

# =========================================================
# 0. Make figures folder
# =========================================================
os.makedirs("figures", exist_ok=True)

# =========================================================
# 1. Load engineered dataset
# =========================================================
df = pd.read_csv("Traffic_engineered_fixed.csv")

# =========================================================
# 2. Define full feature set and target
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
# 3. Final 80/20 split
# =========================================================
split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

test_df = df.iloc[split_idx:].copy().reset_index(drop=True)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# =========================================================
# 4. Train final Random Forest
# =========================================================
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Put predictions into test dataframe
test_df["Actual"] = y_test.values
test_df["Predicted"] = y_pred
test_df["Error"] = test_df["Predicted"] - test_df["Actual"]

# =========================================================
# EXTRA METRICS: Normalised MAE and Low/High Demand MAE
# =========================================================
test_df["AbsoluteError"] = (test_df["Predicted"] - test_df["Actual"]).abs()

# Mean actual traffic count in the test set
mean_actual = test_df["Actual"].mean()

# Normalised MAE as percentage of mean actual traffic
mae = test_df["AbsoluteError"].mean()
normalised_mae_percent = (mae / mean_actual) * 100

# Low-demand / high-demand split using mean actual traffic as boundary
demand_boundary = mean_actual

low_demand_df = test_df[test_df["Actual"] < demand_boundary].copy()
high_demand_df = test_df[test_df["Actual"] >= demand_boundary].copy()

low_demand_mae = low_demand_df["AbsoluteError"].mean()
high_demand_mae = high_demand_df["AbsoluteError"].mean()

low_demand_normalised_mae_percent = (low_demand_mae / low_demand_df["Actual"].mean()) * 100
high_demand_normalised_mae_percent = (high_demand_mae / high_demand_df["Actual"].mean()) * 100

print("\nExtra evaluation metrics:")
print(f"Mean actual traffic count (test set): {mean_actual:.3f}")
print(f"Overall MAE: {mae:.3f}")
print(f"Normalised MAE (% of mean actual): {normalised_mae_percent:.2f}%")
print(f"Demand boundary (mean actual): {demand_boundary:.3f}")

print("\nLow-demand intervals:")
print(f"Rows: {len(low_demand_df)}")
print(f"Mean actual: {low_demand_df['Actual'].mean():.3f}")
print(f"MAE: {low_demand_mae:.3f}")
print(f"Normalised MAE: {low_demand_normalised_mae_percent:.2f}%")

print("\nHigh-demand intervals:")
print(f"Rows: {len(high_demand_df)}")
print(f"Mean actual: {high_demand_df['Actual'].mean():.3f}")
print(f"MAE: {high_demand_mae:.3f}")
print(f"Normalised MAE: {high_demand_normalised_mae_percent:.2f}%")

# Save summary to CSV
extra_metrics_df = pd.DataFrame({
    "Metric": [
        "Mean actual traffic count",
        "Overall MAE",
        "Overall normalised MAE (%)",
        "Demand boundary",
        "Low-demand MAE",
        "Low-demand normalised MAE (%)",
        "High-demand MAE",
        "High-demand normalised MAE (%)"
    ],
    "Value": [
        mean_actual,
        mae,
        normalised_mae_percent,
        demand_boundary,
        low_demand_mae,
        low_demand_normalised_mae_percent,
        high_demand_mae,
        high_demand_normalised_mae_percent
    ]
})

extra_metrics_df.to_csv("figures/extra_error_metrics.csv", index=False)


# Create a readable time-of-day label
test_df["TimeLabel"] = (
    test_df["Hour"].astype(int).astype(str).str.zfill(2)
    + ":"
    + test_df["Minute"].astype(int).astype(str).str.zfill(2)
)

# =========================================================
# 5. Plot 1: Actual vs Predicted over sample index
# =========================================================
n_plot = min(200, len(test_df))

plt.figure(figsize=(11, 5))
plt.plot(test_df.index[:n_plot], test_df["Actual"][:n_plot], label="Actual")
plt.plot(test_df.index[:n_plot], test_df["Predicted"][:n_plot], label="Predicted")
plt.xlabel("Test-set sample index")
plt.ylabel("Traffic count")
plt.title("Actual vs Predicted Traffic Count (Test Set Index)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/plot_1_actual_vs_predicted_index.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 6. Plot 2: Actual vs Predicted with time-of-day labels
# =========================================================
# Use fewer points so axis labels stay readable
n_time_plot = min(96, len(test_df))   # roughly one day of 15-min intervals

subset_time = test_df.iloc[:n_time_plot].copy()

plt.figure(figsize=(12, 5))
plt.plot(subset_time.index, subset_time["Actual"], label="Actual")
plt.plot(subset_time.index, subset_time["Predicted"], label="Predicted")

tick_step = max(1, len(subset_time) // 8)
tick_positions = subset_time.index[::tick_step]
tick_labels = subset_time["TimeLabel"].iloc[::tick_step]

plt.xticks(tick_positions, tick_labels, rotation=45)
plt.xlabel("Time of day")
plt.ylabel("Traffic count")
plt.title("Actual vs Predicted Traffic Count (Time-of-Day View)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/plot_2_actual_vs_predicted_time_labels.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 7. Plot 3: Average daily profile (Actual vs Predicted)
# =========================================================
# Group by 15-minute time slot across the test set
daily_profile = (
    test_df.groupby("TimeLabel")[["Actual", "Predicted"]]
    .mean()
    .reset_index()
)

plt.figure(figsize=(13, 5))
plt.plot(daily_profile.index, daily_profile["Actual"], label="Average Actual")
plt.plot(daily_profile.index, daily_profile["Predicted"], label="Average Predicted")

tick_step = max(1, len(daily_profile) // 8)
tick_positions = daily_profile.index[::tick_step]
tick_labels = daily_profile["TimeLabel"].iloc[::tick_step]

plt.xticks(tick_positions, tick_labels, rotation=45)
plt.xlabel("Time of day")
plt.ylabel("Average traffic count")
plt.title("Average Daily Traffic Profile: Actual vs Predicted")
plt.legend()
plt.tight_layout()
plt.savefig("figures/plot_3_average_daily_profile.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 8. Plot 4: Scatter plot (Actual vs Predicted)
# =========================================================
plt.figure(figsize=(6, 6))
plt.scatter(test_df["Actual"], test_df["Predicted"], alpha=0.6)

# Reference line y = x
min_val = min(test_df["Actual"].min(), test_df["Predicted"].min())
max_val = max(test_df["Actual"].max(), test_df["Predicted"].max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.xlabel("Actual traffic count")
plt.ylabel("Predicted traffic count")
plt.title("Actual vs Predicted Scatter Plot")
plt.tight_layout()
plt.savefig("figures/plot_4_actual_vs_predicted_scatter.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 9. Plot 5: Error histogram
# =========================================================
plt.figure(figsize=(8, 5))
plt.hist(test_df["Error"], bins=30)
plt.xlabel("Prediction error (Predicted - Actual)")
plt.ylabel("Frequency")
plt.title("Prediction Error Distribution")
plt.tight_layout()
plt.savefig("figures/plot_5_error_histogram.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 10. Save plot data too
# =========================================================
test_df.to_csv("figures/test_predictions_full_random_forest.csv", index=False)
daily_profile.to_csv("figures/average_daily_profile.csv", index=False)

# =========================================================
# 11. Plot 6: MAE comparison for low vs high demand
# =========================================================
plt.figure(figsize=(6, 4))
plt.bar(
    ["Low demand", "High demand"],
    [low_demand_mae, high_demand_mae]
)
plt.ylabel("MAE (traffic counts)")
plt.title("Prediction Error for Low vs High Demand Intervals")
plt.tight_layout()
plt.savefig("figures/plot_6_low_vs_high_demand_mae.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 12. Plot 7: Normalised MAE comparison for low vs high demand
# =========================================================
plt.figure(figsize=(6, 4))
plt.bar(
    ["Low demand", "High demand"],
    [low_demand_normalised_mae_percent, high_demand_normalised_mae_percent]
)
plt.ylabel("Normalised MAE (%)")
plt.title("Relative Prediction Error for Low vs High Demand Intervals")
plt.tight_layout()
plt.savefig("figures/plot_7_low_vs_high_demand_normalised_mae.png", dpi=300, bbox_inches="tight")
plt.close()

print("- figures/extra_error_metrics.csv")
print("- figures/plot_6_low_vs_high_demand_mae.png")
print("- figures/plot_7_low_vs_high_demand_normalised_mae.png")

print("\nSaved plots in figures/:")
print("- figures/plot_1_actual_vs_predicted_index.png")
print("- figures/plot_2_actual_vs_predicted_time_labels.png")
print("- figures/plot_3_average_daily_profile.png")
print("- figures/plot_4_actual_vs_predicted_scatter.png")
print("- figures/plot_5_error_histogram.png")
print("- figures/test_predictions_full_random_forest.csv")
print("- figures/average_daily_profile.csv")

