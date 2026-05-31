import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
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

# Fixed chosen adaptive threshold
LOW_DEMAND_THRESHOLD = 90
HIGH_DEMAND_THRESHOLD = 125

# Penalty weights
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
# 5. MODEL PERFORMANCE
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

analysis["IntervalIndex"] = np.arange(len(analysis))

# =========================================================
# 7. DEFINE FIXED-CYCLE BASELINE
# =========================================================
# Alternate green / red by interval
analysis["FixedCycle_GivesGreen"] = (analysis["IntervalIndex"] % 2 == 0).astype(int)

# =========================================================
# 8. DEFINE ADAPTIVE LOGIC
# =========================================================
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

total_low_demand_intervals = int(analysis["Actual_LowDemand"].sum())
total_high_demand_intervals = int(analysis["Actual_HighDemand"].sum())

# =========================================================
# 10. DEFINE WASTED GREEN
# =========================================================
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
analysis["FixedCycle_MissedHighDemand"] = (
    (analysis["FixedCycle_GivesGreen"] == 0) &
    (analysis["Actual_HighDemand"] == 1)
).astype(int)

analysis["Adaptive_MissedHighDemand"] = (
    (analysis["Adaptive_GivesGreen"] == 0) &
    (analysis["Actual_HighDemand"] == 1)
).astype(int)

# =========================================================
# 12. COUNTS
# =========================================================
fixed_green_count = int(analysis["FixedCycle_GivesGreen"].sum())
adaptive_green_count = int(analysis["Adaptive_GivesGreen"].sum())

fixed_wasted = int(analysis["FixedCycle_WastedGreen"].sum())
adaptive_wasted = int(analysis["Adaptive_WastedGreen"].sum())

fixed_missed = int(analysis["FixedCycle_MissedHighDemand"].sum())
adaptive_missed = int(analysis["Adaptive_MissedHighDemand"].sum())

# =========================================================
# 13. RATES
# =========================================================
fixed_wasted_green_rate_pct = (
    fixed_wasted / fixed_green_count * 100 if fixed_green_count > 0 else 0.0
)
adaptive_wasted_green_rate_pct = (
    adaptive_wasted / adaptive_green_count * 100 if adaptive_green_count > 0 else 0.0
)

fixed_missed_high_rate_pct = (
    fixed_missed / total_high_demand_intervals * 100 if total_high_demand_intervals > 0 else 0.0
)
adaptive_missed_high_rate_pct = (
    adaptive_missed / total_high_demand_intervals * 100 if total_high_demand_intervals > 0 else 0.0
)

# =========================================================
# 14. REDUCTION PERCENTAGES
# =========================================================
wasted_green_reduction_pct = (
    100 * (fixed_wasted - adaptive_wasted) / fixed_wasted
    if fixed_wasted > 0 else 0.0
)

missed_high_reduction_pct = (
    100 * (fixed_missed - adaptive_missed) / fixed_missed
    if fixed_missed > 0 else 0.0
)

# =========================================================
# 15. COMBINED PENALTY
# =========================================================
fixed_total_penalty = (
    WASTED_GREEN_PENALTY * fixed_wasted
    + MISSED_HIGH_DEMAND_PENALTY * fixed_missed
)

adaptive_total_penalty = (
    WASTED_GREEN_PENALTY * adaptive_wasted
    + MISSED_HIGH_DEMAND_PENALTY * adaptive_missed
)

penalty_reduction_pct = (
    100 * (fixed_total_penalty - adaptive_total_penalty) / fixed_total_penalty
    if fixed_total_penalty > 0 else 0.0
)

# =========================================================
# 16. DECISION OUTCOME TYPES
# =========================================================
analysis["FixedCycle_Outcome"] = np.select(
    [
        (analysis["FixedCycle_GivesGreen"] == 1) & (analysis["Actual_LowDemand"] == 1),
        (analysis["FixedCycle_GivesGreen"] == 0) & (analysis["Actual_HighDemand"] == 1),
        (analysis["FixedCycle_GivesGreen"] == 1) & (analysis["Actual_LowDemand"] == 0),
        (analysis["FixedCycle_GivesGreen"] == 0) & (analysis["Actual_HighDemand"] == 0)
    ],
    [
        "Wasted green",
        "Missed high-demand",
        "Useful green / acceptable",
        "Acceptable red / low-medium demand"
    ],
    default="Other"
)

analysis["Adaptive_Outcome"] = np.select(
    [
        (analysis["Adaptive_GivesGreen"] == 1) & (analysis["Actual_LowDemand"] == 1),
        (analysis["Adaptive_GivesGreen"] == 0) & (analysis["Actual_HighDemand"] == 1),
        (analysis["Adaptive_GivesGreen"] == 1) & (analysis["Actual_LowDemand"] == 0),
        (analysis["Adaptive_GivesGreen"] == 0) & (analysis["Actual_HighDemand"] == 0)
    ],
    [
        "Wasted green",
        "Missed high-demand",
        "Useful green / acceptable",
        "Acceptable red / low-medium demand"
    ],
    default="Other"
)

# =========================================================
# 17. SUMMARY TABLE
# =========================================================
summary = pd.DataFrame({
    "Metric": [
        "Low-demand threshold",
        "High-demand threshold",
        "Fixed-cycle green intervals",
        "Adaptive green intervals",
        "Fixed-cycle wasted green",
        "Adaptive wasted green",
        "Wasted green reduction (%)",
        "Fixed-cycle wasted green rate (%)",
        "Adaptive wasted green rate (%)",
        "Fixed-cycle missed high-demand",
        "Adaptive missed high-demand",
        "Fixed-cycle missed high-demand rate (%)",
        "Adaptive missed high-demand rate (%)",
        "Fixed-cycle total penalty",
        "Adaptive total penalty",
        "Penalty reduction (%)",
        "RF Test MAE",
        "RF Test RMSE",
        "RF Test R²"
    ],
    "Value": [
        LOW_DEMAND_THRESHOLD,
        HIGH_DEMAND_THRESHOLD,
        fixed_green_count,
        adaptive_green_count,
        fixed_wasted,
        adaptive_wasted,
        round(wasted_green_reduction_pct, 3),
        round(fixed_wasted_green_rate_pct, 3),
        round(adaptive_wasted_green_rate_pct, 3),
        fixed_missed,
        adaptive_missed,
        round(fixed_missed_high_rate_pct, 3),
        round(adaptive_missed_high_rate_pct, 3),
        fixed_total_penalty,
        adaptive_total_penalty,
        round(penalty_reduction_pct, 3),
        round(mae, 3),
        round(rmse, 3),
        round(r2, 3)
    ]
})

print("\nThreshold-90 green-cycle summary:")
print(summary.to_string(index=False))

# Save summary
summary.to_csv(os.path.join(OUTPUT_DIR, "green_cycle_threshold90_summary.csv"), index=False)

# Save interval-level data
analysis.to_csv(os.path.join(OUTPUT_DIR, "green_cycle_threshold90_interval_analysis.csv"), index=False)

# =========================================================
# 18. FIGURE 1: WASTED GREEN COMPARISON
# =========================================================
plt.figure(figsize=(7, 5))
plt.bar(
    ["Fixed-cycle", "Adaptive"],
    [fixed_wasted, adaptive_wasted]
)
plt.ylabel("Wasted green intervals")
plt.title("Threshold 90: Wasted Green Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_threshold90_wasted_green.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 19. FIGURE 2: MISSED HIGH-DEMAND COMPARISON
# =========================================================
plt.figure(figsize=(7, 5))
plt.bar(
    ["Fixed-cycle", "Adaptive"],
    [fixed_missed, adaptive_missed]
)
plt.ylabel("Missed high-demand intervals")
plt.title("Threshold 90: Missed High-Demand Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_threshold90_missed_high.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 20. FIGURE 3: TOTAL PENALTY COMPARISON
# =========================================================
plt.figure(figsize=(7, 5))
plt.bar(
    ["Fixed-cycle", "Adaptive"],
    [fixed_total_penalty, adaptive_total_penalty]
)
plt.ylabel("Total penalty score")
plt.title("Threshold 90: Total Penalty Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_threshold90_total_penalty.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 21. FIGURE 4: WASTED GREEN RATE + MISSED HIGH RATE
# =========================================================
categories = ["Wasted green rate", "Missed high-demand rate"]
fixed_rates = [fixed_wasted_green_rate_pct, fixed_missed_high_rate_pct]
adaptive_rates = [adaptive_wasted_green_rate_pct, adaptive_missed_high_rate_pct]

x = np.arange(len(categories))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width/2, fixed_rates, width, label="Fixed-cycle")
plt.bar(x + width/2, adaptive_rates, width, label="Adaptive")
plt.xticks(x, categories)
plt.ylabel("Rate (%)")
plt.title("Threshold 90: Decision Error Rates")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_threshold90_error_rates.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 22. FIGURE 5: STACKED OUTCOME COMPARISON
# =========================================================
outcome_order = [
    "Wasted green",
    "Missed high-demand",
    "Useful green / acceptable",
    "Acceptable red / low-medium demand"
]

fixed_counts = analysis["FixedCycle_Outcome"].value_counts().reindex(outcome_order, fill_value=0)
adaptive_counts = analysis["Adaptive_Outcome"].value_counts().reindex(outcome_order, fill_value=0)

plt.figure(figsize=(9, 5))

bottom_fixed = 0
bottom_adaptive = 0

for outcome in outcome_order:
    plt.bar("Fixed-cycle", fixed_counts[outcome], bottom=bottom_fixed, label=outcome if bottom_fixed == 0 else "")
    plt.bar("Adaptive", adaptive_counts[outcome], bottom=bottom_adaptive)
    bottom_fixed += fixed_counts[outcome]
    bottom_adaptive += adaptive_counts[outcome]

plt.ylabel("Number of intervals")
plt.title("Threshold 90: Decision Outcome Comparison")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_threshold90_stacked_outcomes.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 23. FIGURE 6: TIMELINE OF GREEN DECISIONS
# =========================================================
n_plot = min(120, len(analysis))
plot_df = analysis.iloc[:n_plot].copy()

plt.figure(figsize=(12, 5))
plt.plot(plot_df.index, plot_df["Actual_Next_Total"], label="Actual next traffic")
plt.plot(plot_df.index, plot_df["Predicted_Next_Total"], label="Predicted next traffic")
plt.axhline(LOW_DEMAND_THRESHOLD, linestyle="--", label=f"Threshold = {LOW_DEMAND_THRESHOLD}")

fixed_green_idx = plot_df.index[plot_df["FixedCycle_GivesGreen"] == 1]
adaptive_green_idx = plot_df.index[plot_df["Adaptive_GivesGreen"] == 1]

plt.scatter(
    fixed_green_idx,
    plot_df.loc[fixed_green_idx, "Actual_Next_Total"],
    s=15,
    label="Fixed-cycle gives green"
)

plt.scatter(
    adaptive_green_idx,
    plot_df.loc[adaptive_green_idx, "Predicted_Next_Total"],
    s=15,
    label="Adaptive gives green"
)

plt.xlabel("Test-set interval index")
plt.ylabel("Traffic count")
plt.title("Threshold 90: Fixed-Cycle vs Adaptive Decisions")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_threshold90_decision_timeline.png"), dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved outputs in:", OUTPUT_DIR)
print("- green_cycle_threshold90_summary.csv")
print("- green_cycle_threshold90_interval_analysis.csv")
print("- figure_threshold90_wasted_green.png")
print("- figure_threshold90_missed_high.png")
print("- figure_threshold90_total_penalty.png")
print("- figure_threshold90_error_rates.png")
print("- figure_threshold90_stacked_outcomes.png")
print("- figure_threshold90_decision_timeline.png")

