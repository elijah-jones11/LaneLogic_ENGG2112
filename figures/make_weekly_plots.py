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

test_df["Actual"] = y_test.values
test_df["Predicted"] = y_pred
test_df["Error"] = test_df["Predicted"] - test_df["Actual"]

# =========================================================
# 5. Build approximate datetime for plotting
# =========================================================
# Assumes the original dataset month/year used earlier
YEAR = 2023
MONTH = 12

# Rebuild date column into a full date
test_df["DateFull"] = pd.to_datetime(
    f"{YEAR}-{MONTH:02d}-" + test_df["Date"].astype(int).astype(str).str.zfill(2),
    errors="coerce"
)

# Build datetime using DateFull + Hour + Minute
test_df["PlotDateTime"] = pd.to_datetime(
    test_df["DateFull"].dt.strftime("%Y-%m-%d") + " "
    + test_df["Hour"].astype(int).astype(str).str.zfill(2) + ":"
    + test_df["Minute"].astype(int).astype(str).str.zfill(2) + ":00",
    errors="coerce"
)

# If duplicate dates exist because of month ambiguity, keep row order fallback
# and still use the datetime labels mainly for readability
test_df["RowIndex"] = np.arange(len(test_df))

# =========================================================
# 6. Plot A: Entire test period
# =========================================================
plt.figure(figsize=(14, 6))
plt.plot(test_df["RowIndex"], test_df["Actual"], label="Actual")
plt.plot(test_df["RowIndex"], test_df["Predicted"], label="Predicted")

# Mark day boundaries if Date changes
day_change_idx = test_df.index[test_df["Date"].diff().fillna(0) != 0].tolist()
for idx in day_change_idx:
    plt.axvline(idx)

plt.xlabel("Test-set interval index")
plt.ylabel("Traffic count")
plt.title("Traffic Count Across Entire Test Period")
plt.legend()
plt.tight_layout()
plt.savefig("figures/plot_test_all_days.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 7. Plot B: One week of test data
# 7 days * 96 intervals/day = 672 intervals
# =========================================================
intervals_per_day = 96
week_len = min(7 * intervals_per_day, len(test_df))

week_df = test_df.iloc[:week_len].copy().reset_index(drop=True)

plt.figure(figsize=(15, 6))
plt.plot(week_df.index, week_df["Actual"], label="Actual")
plt.plot(week_df.index, week_df["Predicted"], label="Predicted")

# Add day boundary lines and labels
week_day_change_idx = week_df.index[week_df["Date"].diff().fillna(0) != 0].tolist()
for idx in week_day_change_idx:
    plt.axvline(idx)

# Add x tick labels at the start of each day
tick_positions = [0] + week_day_change_idx
tick_labels = []
for pos in tick_positions:
    row = week_df.iloc[pos]
    tick_labels.append(f"{row['Day of the week']}\n{int(row['Date'])}")

plt.xticks(tick_positions, tick_labels)
plt.xlabel("Day in test week")
plt.ylabel("Traffic count")
plt.title("Traffic Count Over One Week of Test Data")
plt.legend()
plt.tight_layout()
plt.savefig("figures/plot_test_one_week.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 8. Plot C: Weekend vs weekday daily profile
# =========================================================
# Average traffic by time-of-day, split by weekend/weekday
test_df["TimeLabel"] = (
    test_df["Hour"].astype(int).astype(str).str.zfill(2)
    + ":"
    + test_df["Minute"].astype(int).astype(str).str.zfill(2)
)

weekday_profile = (
    test_df[test_df["IsWeekend"] == 0]
    .groupby("TimeLabel")["Actual"]
    .mean()
    .reset_index()
)

weekend_profile = (
    test_df[test_df["IsWeekend"] == 1]
    .groupby("TimeLabel")["Actual"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(13, 5))
if len(weekday_profile) > 0:
    plt.plot(weekday_profile.index, weekday_profile["Actual"], label="Weekday average")
if len(weekend_profile) > 0:
    plt.plot(weekend_profile.index, weekend_profile["Actual"], label="Weekend average")

tick_step = max(1, len(weekday_profile) // 8 if len(weekday_profile) > 0 else 12)
tick_positions = weekday_profile.index[::tick_step] if len(weekday_profile) > 0 else []
tick_labels = weekday_profile["TimeLabel"].iloc[::tick_step] if len(weekday_profile) > 0 else []

plt.xticks(tick_positions, tick_labels, rotation=45)
plt.xlabel("Time of day")
plt.ylabel("Average traffic count")
plt.title("Average Daily Traffic Profile: Weekday vs Weekend")
plt.legend()
plt.tight_layout()
plt.savefig("figures/plot_weekday_vs_weekend_profile.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 9. Save useful data
# =========================================================
test_df.to_csv("figures/test_period_predictions.csv", index=False)
week_df.to_csv("figures/test_one_week_data.csv", index=False)

print("\nSaved plots:")
print("- figures/plot_test_all_days.png")
print("- figures/plot_test_one_week.png")
print("- figures/plot_weekday_vs_weekend_profile.png")
print("- figures/test_period_predictions.csv")
print("- figures/test_one_week_data.csv")

