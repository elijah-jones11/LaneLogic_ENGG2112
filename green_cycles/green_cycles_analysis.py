import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# 0. SETTINGS
# =========================================================
INPUT_PATH = "Traffic_engineered_fixed.csv"
OUTPUT_DIR = "green_cycles"

# Final chosen Random Forest hyperparameters
N_ESTIMATORS = 200
MAX_DEPTH = 12
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 1

# Demand thresholds for simple control logic
LOW_DEMAND_THRESHOLD = 75
HIGH_DEMAND_THRESHOLD = 125

# Penalty weights for combined score
WASTED_GREEN_PENALTY = 1
MISSED_HIGH_DEMAND_PENALTY = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_PATH)

# =========================================================
# 2. FEATURES AND TARGET
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
# 3. FINAL 80/20 SPLIT
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
# 4. TRAIN FINAL RANDOM FOREST
# =========================================================
model = RandomForestRegressor(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    min_samples_split=MIN_SAMPLES_SPLIT,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================================================
# 5. EVALUATE MODEL
# =========================================================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nFinal Random Forest Test Results")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²:   {r2:.3f}")

# =========================================================
# 6. BUILD ANALYSIS DATAFRAME
# =========================================================
analysis = pd.DataFrame({
    "Actual_Next_Total": y_test.values,
    "Predicted_Next_Total": y_pred
})

for col in ["Hour", "Minute", "DayOfWeekNum", "IsWeekend"]:
    if col in test_df.columns:
        analysis[col] = test_df[col].values

# =========================================================
# 7. DEFINE FIXED-CYCLE BASELINE
# =========================================================
# Simplified fixed cycle:
# Alternate green/red each interval:
# interval 0 -> green, interval 1 -> red, interval 2 -> green, ...
analysis["IntervalIndex"] = np.arange(len(analysis))
analysis["FixedCycle_GivesGreen"] = (analysis["IntervalIndex"] % 2 == 0).astype(int)

# =========================================================
# 8. DEFINE ADAPTIVE LOGIC
# =========================================================
# Give green if predicted next traffic is above low-demand threshold
analysis["Adaptive_GivesGreen"] = (
    analysis["Predicted_Next_Total"] >= LOW_DEMAND_THRESHOLD
).astype(int)

# =========================================================
# 9. DEFINE ACTUAL LOW/HIGH DEMAND
# =========================================================
analysis["Actual_LowDemand"] = (
    analysis["Actual_Next_Total"] < LOW_DEMAND_THRESHOLD
).astype(int)

analysis["Actual_HighDemand"] = (
    analysis["Actual_Next_Total"] >= HIGH_DEMAND_THRESHOLD
).astype(int)

# =========================================================
# 10. DEFINE WASTED GREEN
# =========================================================
# Wasted green = green given when actual demand is low
analysis["FixedCycle_WastedGreen"] = (
    (analysis["FixedCycle_GivesGreen"] == 1) &
    (analysis["Actual_LowDemand"] == 1)
).astype(int)

analysis["Adaptive_WastedGreen"] = (
    (analysis["Adaptive_GivesGreen"] == 1) &
    (analysis["Actual_LowDemand"] == 1)
).astype(int)

# =========================================================
# 11. DEFINE MISSED HIGH-DEMAND
# =========================================================
# Missed high-demand = no green given when actual demand is high
analysis["FixedCycle_MissedHighDemand"] = (
    (analysis["FixedCycle_GivesGreen"] == 0) &
    (analysis["Actual_HighDemand"] == 1)
).astype(int)

analysis["Adaptive_MissedHighDemand"] = (
    (analysis["Adaptive_GivesGreen"] == 0) &
    (analysis["Actual_HighDemand"] == 1)
).astype(int)

# =========================================================
# 12. SUMMARISE PERFORMANCE
# =========================================================
fixed_green_count = int(analysis["FixedCycle_GivesGreen"].sum())
adaptive_green_count = int(analysis["Adaptive_GivesGreen"].sum())

fixed_wasted = int(analysis["FixedCycle_WastedGreen"].sum())
adaptive_wasted = int(analysis["Adaptive_WastedGreen"].sum())

fixed_missed = int(analysis["FixedCycle_MissedHighDemand"].sum())
adaptive_missed = int(analysis["Adaptive_MissedHighDemand"].sum())

# Reduction percentages
if fixed_wasted > 0:
    wasted_green_reduction_pct = 100 * (fixed_wasted - adaptive_wasted) / fixed_wasted
else:
    wasted_green_reduction_pct = 0.0

if fixed_missed > 0:
    missed_high_demand_reduction_pct = 100 * (fixed_missed - adaptive_missed) / fixed_missed
else:
    missed_high_demand_reduction_pct = 0.0

# Combined penalty score
fixed_total_penalty = (
    WASTED_GREEN_PENALTY * fixed_wasted
    + MISSED_HIGH_DEMAND_PENALTY * fixed_missed
)

adaptive_total_penalty = (
    WASTED_GREEN_PENALTY * adaptive_wasted
    + MISSED_HIGH_DEMAND_PENALTY * adaptive_missed
)

if fixed_total_penalty > 0:
    penalty_reduction_pct = 100 * (fixed_total_penalty - adaptive_total_penalty) / fixed_total_penalty
else:
    penalty_reduction_pct = 0.0

summary = pd.DataFrame({
    "Metric": [
        "Test rows",
        "Fixed-cycle green intervals",
        "Adaptive green intervals",
        "Fixed-cycle wasted green intervals",
        "Adaptive wasted green intervals",
        "Wasted green reduction (%)",
        "Fixed-cycle missed high-demand intervals",
        "Adaptive missed high-demand intervals",
        "Missed high-demand reduction (%)",
        "Fixed-cycle total penalty",
        "Adaptive total penalty",
        "Total penalty reduction (%)",
        "RF Test MAE",
        "RF Test RMSE",
        "RF Test R2"
    ],
    "Value": [
        len(analysis),
        fixed_green_count,
        adaptive_green_count,
        fixed_wasted,
        adaptive_wasted,
        round(wasted_green_reduction_pct, 3),
        fixed_missed,
        adaptive_missed,
        round(missed_high_demand_reduction_pct, 3),
        fixed_total_penalty,
        adaptive_total_penalty,
        round(penalty_reduction_pct, 3),
        round(mae, 3),
        round(rmse, 3),
        round(r2, 3)
    ]
})

print("\nGreen-cycle proxy evaluation summary:")
print(summary.to_string(index=False))

# =========================================================
# 13. SAVE TABLES
# =========================================================
analysis.to_csv(os.path.join(OUTPUT_DIR, "green_cycle_interval_analysis.csv"), index=False)
summary.to_csv(os.path.join(OUTPUT_DIR, "green_cycle_summary.csv"), index=False)

# =========================================================
# 14. FIGURE 1: WASTED GREEN COMPARISON
# =========================================================
plt.figure(figsize=(7, 5))
plt.bar(
    ["Fixed-cycle", "Adaptive"],
    [fixed_wasted, adaptive_wasted]
)
plt.ylabel("Number of wasted green intervals")
plt.title("Wasted Green Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_1_wasted_green_bar.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 15. FIGURE 2: MISSED HIGH-DEMAND COMPARISON
# =========================================================
plt.figure(figsize=(7, 5))
plt.bar(
    ["Fixed-cycle", "Adaptive"],
    [fixed_missed, adaptive_missed]
)
plt.ylabel("Number of missed high-demand intervals")
plt.title("Missed High-Demand Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_2_missed_high_demand_bar.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 16. FIGURE 3: TOTAL PENALTY COMPARISON
# =========================================================
plt.figure(figsize=(7, 5))
plt.bar(
    ["Fixed-cycle", "Adaptive"],
    [fixed_total_penalty, adaptive_total_penalty]
)
plt.ylabel("Total penalty score")
plt.title("Combined Penalty Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_3_total_penalty_bar.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 17. FIGURE 4: ACTUAL VS PREDICTED WITH THRESHOLDS
# =========================================================
n_plot = min(200, len(analysis))

plt.figure(figsize=(12, 5))
plt.plot(analysis.index[:n_plot], analysis["Actual_Next_Total"][:n_plot], label="Actual next traffic")
plt.plot(analysis.index[:n_plot], analysis["Predicted_Next_Total"][:n_plot], label="Predicted next traffic")
plt.axhline(LOW_DEMAND_THRESHOLD, linestyle="--", label=f"Low-demand threshold = {LOW_DEMAND_THRESHOLD}")
plt.axhline(HIGH_DEMAND_THRESHOLD, linestyle="--", label=f"High-demand threshold = {HIGH_DEMAND_THRESHOLD}")
plt.xlabel("Test-set interval index")
plt.ylabel("Traffic count")
plt.title("Actual vs Predicted Next-Interval Traffic")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_4_actual_predicted_with_thresholds.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 18. FIGURE 5: FIXED-CYCLE VS ADAPTIVE GREEN DECISIONS
# =========================================================
n_decision_plot = min(120, len(analysis))
decision_df = analysis.iloc[:n_decision_plot].copy()

plt.figure(figsize=(12, 5))
plt.plot(decision_df.index, decision_df["Actual_Next_Total"], label="Actual next traffic")
plt.plot(decision_df.index, decision_df["Predicted_Next_Total"], label="Predicted next traffic")

fixed_green_idx = decision_df.index[decision_df["FixedCycle_GivesGreen"] == 1]
adaptive_green_idx = decision_df.index[decision_df["Adaptive_GivesGreen"] == 1]

plt.scatter(
    fixed_green_idx,
    decision_df.loc[fixed_green_idx, "Actual_Next_Total"],
    label="Fixed-cycle gives green",
    s=15
)
plt.scatter(
    adaptive_green_idx,
    decision_df.loc[adaptive_green_idx, "Predicted_Next_Total"],
    label="Adaptive gives green",
    s=15
)

plt.axhline(LOW_DEMAND_THRESHOLD, linestyle="--", label=f"Low-demand threshold = {LOW_DEMAND_THRESHOLD}")
plt.xlabel("Test-set interval index")
plt.ylabel("Traffic count")
plt.title("Green Decisions: Fixed-Cycle vs Adaptive")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_5_green_decisions.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved outputs in:", OUTPUT_DIR)
print("- green_cycle_interval_analysis.csv")
print("- green_cycle_summary.csv")
print("- figure_1_wasted_green_bar.png")
print("- figure_2_missed_high_demand_bar.png")
print("- figure_3_total_penalty_bar.png")
print("- figure_4_actual_predicted_with_thresholds.png")
print("- figure_5_green_decisions.png")

