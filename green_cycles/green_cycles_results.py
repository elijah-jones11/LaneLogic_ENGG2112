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

# Sweep several low-demand thresholds
LOW_DEMAND_THRESHOLDS = [60, 75, 90, 105, 120]

# High-demand threshold for missed-demand penalty
HIGH_DEMAND_THRESHOLD = 125

# Penalty weights
WASTED_GREEN_PENALTY = 1
MISSED_HIGH_DEMAND_PENALTY = 2

# Which threshold to highlight in final bar chart
CHOSEN_THRESHOLD = 90

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
# 5. PREDICTION METRICS
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
# Alternate green/red by interval
analysis["FixedCycle_GivesGreen"] = (analysis["IntervalIndex"] % 2 == 0).astype(int)

# High-demand definition is fixed
analysis["Actual_HighDemand"] = (
    analysis["Actual_Next_Total"] >= HIGH_DEMAND_THRESHOLD
).astype(int)

# Fixed-cycle missed high-demand does not depend on low threshold
analysis["FixedCycle_MissedHighDemand"] = (
    (analysis["FixedCycle_GivesGreen"] == 0) &
    (analysis["Actual_HighDemand"] == 1)
).astype(int)

fixed_green_count = int(analysis["FixedCycle_GivesGreen"].sum())
fixed_missed = int(analysis["FixedCycle_MissedHighDemand"].sum())
total_high_demand_intervals = int(analysis["Actual_HighDemand"].sum())

# =========================================================
# 8. THRESHOLD SWEEP
# =========================================================
sweep_results = []

for low_threshold in LOW_DEMAND_THRESHOLDS:
    temp = analysis.copy()

    # Adaptive rule
    temp["Adaptive_GivesGreen"] = (
        temp["Predicted_Next_Total"] >= low_threshold
    ).astype(int)

    # Low-demand definition for this threshold
    temp["Actual_LowDemand"] = (
        temp["Actual_Next_Total"] < low_threshold
    ).astype(int)

    total_low_demand_intervals = int(temp["Actual_LowDemand"].sum())

    # Wasted green
    temp["FixedCycle_WastedGreen"] = (
        (temp["FixedCycle_GivesGreen"] == 1) &
        (temp["Actual_LowDemand"] == 1)
    ).astype(int)

    temp["Adaptive_WastedGreen"] = (
        (temp["Adaptive_GivesGreen"] == 1) &
        (temp["Actual_LowDemand"] == 1)
    ).astype(int)

    # Missed high-demand
    temp["Adaptive_MissedHighDemand"] = (
        (temp["Adaptive_GivesGreen"] == 0) &
        (temp["Actual_HighDemand"] == 1)
    ).astype(int)

    adaptive_green_count = int(temp["Adaptive_GivesGreen"].sum())

    fixed_wasted = int(temp["FixedCycle_WastedGreen"].sum())
    adaptive_wasted = int(temp["Adaptive_WastedGreen"].sum())

    adaptive_missed = int(temp["Adaptive_MissedHighDemand"].sum())

    # Rates
    fixed_wasted_green_rate = (
        fixed_wasted / fixed_green_count * 100 if fixed_green_count > 0 else 0.0
    )
    adaptive_wasted_green_rate = (
        adaptive_wasted / adaptive_green_count * 100 if adaptive_green_count > 0 else 0.0
    )

    fixed_missed_high_rate = (
        fixed_missed / total_high_demand_intervals * 100 if total_high_demand_intervals > 0 else 0.0
    )
    adaptive_missed_high_rate = (
        adaptive_missed / total_high_demand_intervals * 100 if total_high_demand_intervals > 0 else 0.0
    )

    # Reductions
    wasted_green_reduction_pct = (
        100 * (fixed_wasted - adaptive_wasted) / fixed_wasted if fixed_wasted > 0 else 0.0
    )
    missed_high_reduction_pct = (
        100 * (fixed_missed - adaptive_missed) / fixed_missed if fixed_missed > 0 else 0.0
    )

    # Combined penalty
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

    sweep_results.append({
        "low_demand_threshold": low_threshold,
        "high_demand_threshold": HIGH_DEMAND_THRESHOLD,
        "fixed_cycle_green_intervals": fixed_green_count,
        "adaptive_green_intervals": adaptive_green_count,
        "total_low_demand_intervals": total_low_demand_intervals,
        "total_high_demand_intervals": total_high_demand_intervals,
        "fixed_cycle_wasted_green": fixed_wasted,
        "adaptive_wasted_green": adaptive_wasted,
        "fixed_cycle_wasted_green_rate_pct": fixed_wasted_green_rate,
        "adaptive_wasted_green_rate_pct": adaptive_wasted_green_rate,
        "wasted_green_reduction_pct": wasted_green_reduction_pct,
        "fixed_cycle_missed_high": fixed_missed,
        "adaptive_missed_high": adaptive_missed,
        "fixed_cycle_missed_high_rate_pct": fixed_missed_high_rate,
        "adaptive_missed_high_rate_pct": adaptive_missed_high_rate,
        "missed_high_reduction_pct": missed_high_reduction_pct,
        "fixed_cycle_total_penalty": fixed_total_penalty,
        "adaptive_total_penalty": adaptive_total_penalty,
        "penalty_reduction_pct": penalty_reduction_pct
    })

sweep_df = pd.DataFrame(sweep_results)

# =========================================================
# 9. SAVE SWEEP TABLE
# =========================================================
sweep_csv_path = os.path.join(OUTPUT_DIR, "green_cycle_threshold_sweep_results.csv")
sweep_df.to_csv(sweep_csv_path, index=False)

print("\nThreshold sweep summary:")
print(sweep_df.to_string(index=False))

# =========================================================
# 10. CHOSEN THRESHOLD SUMMARY
# =========================================================
chosen_df = sweep_df[sweep_df["low_demand_threshold"] == CHOSEN_THRESHOLD].copy()

if len(chosen_df) == 0:
    raise ValueError(f"Chosen threshold {CHOSEN_THRESHOLD} not found in LOW_DEMAND_THRESHOLDS.")

chosen_row = chosen_df.iloc[0]

chosen_summary = pd.DataFrame({
    "Metric": [
        "Low-demand threshold",
        "High-demand threshold",
        "Fixed-cycle green intervals",
        "Adaptive green intervals",
        "Fixed-cycle wasted green",
        "Adaptive wasted green",
        "Fixed-cycle wasted green rate (%)",
        "Adaptive wasted green rate (%)",
        "Wasted green reduction (%)",
        "Fixed-cycle missed high-demand",
        "Adaptive missed high-demand",
        "Fixed-cycle missed high-demand rate (%)",
        "Adaptive missed high-demand rate (%)",
        "Total penalty (fixed-cycle)",
        "Total penalty (adaptive)",
        "Penalty reduction (%)",
        "RF Test MAE",
        "RF Test RMSE",
        "RF Test R²"
    ],
    "Value": [
        int(chosen_row["low_demand_threshold"]),
        int(chosen_row["high_demand_threshold"]),
        int(chosen_row["fixed_cycle_green_intervals"]),
        int(chosen_row["adaptive_green_intervals"]),
        int(chosen_row["fixed_cycle_wasted_green"]),
        int(chosen_row["adaptive_wasted_green"]),
        round(chosen_row["fixed_cycle_wasted_green_rate_pct"], 3),
        round(chosen_row["adaptive_wasted_green_rate_pct"], 3),
        round(chosen_row["wasted_green_reduction_pct"], 3),
        int(chosen_row["fixed_cycle_missed_high"]),
        int(chosen_row["adaptive_missed_high"]),
        round(chosen_row["fixed_cycle_missed_high_rate_pct"], 3),
        round(chosen_row["adaptive_missed_high_rate_pct"], 3),
        int(chosen_row["fixed_cycle_total_penalty"]),
        int(chosen_row["adaptive_total_penalty"]),
        round(chosen_row["penalty_reduction_pct"], 3),
        round(mae, 3),
        round(rmse, 3),
        round(r2, 3)
    ]
})

chosen_summary_csv = os.path.join(OUTPUT_DIR, "green_cycle_chosen_threshold_summary.csv")
chosen_summary.to_csv(chosen_summary_csv, index=False)

print("\nChosen threshold summary:")
print(chosen_summary.to_string(index=False))

# =========================================================
# 11. FIGURE 1: THRESHOLD VS TOTAL PENALTY
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(
    sweep_df["low_demand_threshold"],
    sweep_df["adaptive_total_penalty"],
    marker="o",
    label="Adaptive"
)
plt.axhline(
    y=chosen_row["fixed_cycle_total_penalty"],
    linestyle="--",
    label="Fixed-cycle baseline"
)
plt.xlabel("Low-demand threshold")
plt.ylabel("Total penalty")
plt.title("Adaptive Threshold Sweep: Total Penalty")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_1_threshold_vs_penalty.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 12. FIGURE 2: THRESHOLD VS WASTED/MISSED RATES
# =========================================================
plt.figure(figsize=(9, 5))
plt.plot(
    sweep_df["low_demand_threshold"],
    sweep_df["adaptive_wasted_green_rate_pct"],
    marker="o",
    label="Adaptive wasted green rate"
)
plt.plot(
    sweep_df["low_demand_threshold"],
    sweep_df["adaptive_missed_high_rate_pct"],
    marker="o",
    label="Adaptive missed high-demand rate"
)
plt.axhline(
    y=chosen_row["fixed_cycle_wasted_green_rate_pct"],
    linestyle="--",
    label="Fixed-cycle wasted green rate"
)
plt.axhline(
    y=chosen_row["fixed_cycle_missed_high_rate_pct"],
    linestyle="--",
    label="Fixed-cycle missed high-demand rate"
)
plt.xlabel("Low-demand threshold")
plt.ylabel("Rate (%)")
plt.title("Adaptive Threshold Sweep: Error Trade-Off")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_2_threshold_vs_rates.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 13. FIGURE 3: CHOSEN THRESHOLD BAR CHART
# =========================================================
plt.figure(figsize=(9, 5))

categories = ["Wasted green", "Missed high-demand", "Total penalty"]
fixed_vals = [
    chosen_row["fixed_cycle_wasted_green"],
    chosen_row["fixed_cycle_missed_high"],
    chosen_row["fixed_cycle_total_penalty"]
]
adaptive_vals = [
    chosen_row["adaptive_wasted_green"],
    chosen_row["adaptive_missed_high"],
    chosen_row["adaptive_total_penalty"]
]

x = np.arange(len(categories))
width = 0.35

plt.bar(x - width/2, fixed_vals, width, label="Fixed-cycle")
plt.bar(x + width/2, adaptive_vals, width, label="Adaptive")

plt.xticks(x, categories)
plt.ylabel("Count / Score")
plt.title(f"Chosen Threshold Comparison (threshold = {CHOSEN_THRESHOLD})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_3_chosen_threshold_bar_chart.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 14. FIGURE 4: WASTED GREEN REDUCTION VS THRESHOLD
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(
    sweep_df["low_demand_threshold"],
    sweep_df["wasted_green_reduction_pct"],
    marker="o"
)
plt.xlabel("Low-demand threshold")
plt.ylabel("Wasted green reduction (%)")
plt.title("Adaptive Threshold Sweep: Wasted Green Reduction")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "figure_4_wasted_green_reduction.png"), dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 15. SAVE INTERVAL-LEVEL DATA FOR CHOSEN THRESHOLD
# =========================================================
chosen_analysis = analysis.copy()
chosen_analysis["Adaptive_GivesGreen"] = (
    chosen_analysis["Predicted_Next_Total"] >= CHOSEN_THRESHOLD
).astype(int)
chosen_analysis["Actual_LowDemand"] = (
    chosen_analysis["Actual_Next_Total"] < CHOSEN_THRESHOLD
).astype(int)
chosen_analysis["FixedCycle_WastedGreen"] = (
    (chosen_analysis["FixedCycle_GivesGreen"] == 1) &
    (chosen_analysis["Actual_LowDemand"] == 1)
).astype(int)
chosen_analysis["Adaptive_WastedGreen"] = (
    (chosen_analysis["Adaptive_GivesGreen"] == 1) &
    (chosen_analysis["Actual_LowDemand"] == 1)
).astype(int)
chosen_analysis["Adaptive_MissedHighDemand"] = (
    (chosen_analysis["Adaptive_GivesGreen"] == 0) &
    (chosen_analysis["Actual_HighDemand"] == 1)
).astype(int)

chosen_analysis.to_csv(
    os.path.join(OUTPUT_DIR, "green_cycle_interval_analysis_chosen_threshold.csv"),
    index=False
)

print("\nSaved outputs in:", OUTPUT_DIR)
print("- green_cycle_threshold_sweep_results.csv")
print("- green_cycle_chosen_threshold_summary.csv")
print("- green_cycle_interval_analysis_chosen_threshold.csv")
print("- figure_1_threshold_vs_penalty.png")
print("- figure_2_threshold_vs_rates.png")
print("- figure_3_chosen_threshold_bar_chart.png")
print("- figure_4_wasted_green_reduction.png")