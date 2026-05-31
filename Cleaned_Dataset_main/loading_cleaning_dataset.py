# LOADING AND CLEANING AND SORTING DATASET:
# LOADING AND CLEANING AND SORTING DATASET:
# LOADING AND CLEANING AND SORTING DATASET:
# LOADING AND CLEANING AND SORTING DATASET:
# LOADING AND CLEANING AND SORTING DATASET:



import pandas as pd
import numpy as np

# ==========================================
# 1. Load dataset
# ==========================================
df = pd.read_csv("Datasets_main/2023_December_Traffic.csv")
df.columns = df.columns.str.strip()

print("Columns:")
print(df.columns.tolist())

# ==========================================
# 2. Clean numeric columns
# ==========================================
count_cols = ["CarCount", "BikeCount", "BusCount", "TruckCount", "Total"]

for col in count_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=count_cols).copy()

# ==========================================
# 3. Keep original file order
# IMPORTANT: do NOT sort by fake timestamp
# ==========================================
df = df.reset_index(drop=True)

# Add sequential step index
df["Step"] = np.arange(len(df))

# Since each row is 15 minutes apart
df["ElapsedMinutes"] = df["Step"] * 15
df["ElapsedHours"] = df["ElapsedMinutes"] / 60

# ==========================================
# 4. Rebuild Total safely
# ==========================================
df["TotalCount"] = df["CarCount"] + df["BikeCount"] + df["BusCount"] + df["TruckCount"]
df["TotalMismatch"] = df["Total"] - df["TotalCount"]

print("\nUnique Total mismatch values:")
print(df["TotalMismatch"].value_counts().sort_index())

# ==========================================
# 5. Clean time-of-day features from Time column
# ==========================================
# Parse time only
parsed_time = pd.to_datetime(df["Time"], format="%I:%M:%S %p", errors="coerce")

df["Hour"] = parsed_time.dt.hour
df["Minute"] = parsed_time.dt.minute
df["QuarterHour"] = df["Minute"] // 15

# ==========================================
# 6. Encode day of week
# ==========================================
day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

df["DayOfWeekNum"] = df["Day of the week"].map(day_map)
df["IsWeekend"] = (df["DayOfWeekNum"] >= 5).astype(int)

# ==========================================
# 7. Cyclical time features
# ==========================================
df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

df["Day_sin"] = np.sin(2 * np.pi * df["DayOfWeekNum"] / 7)
df["Day_cos"] = np.cos(2 * np.pi * df["DayOfWeekNum"] / 7)

# ==========================================
# 8. Lag features
# ==========================================
df["Total_lag_1"] = df["TotalCount"].shift(1)
df["Total_lag_2"] = df["TotalCount"].shift(2)
df["Total_lag_3"] = df["TotalCount"].shift(3)
df["Total_lag_4"] = df["TotalCount"].shift(4)

df["Car_lag_1"] = df["CarCount"].shift(1)
df["Bike_lag_1"] = df["BikeCount"].shift(1)
df["Bus_lag_1"] = df["BusCount"].shift(1)
df["Truck_lag_1"] = df["TruckCount"].shift(1)

# ==========================================
# 9. Rolling features
# ==========================================
df["Total_roll_mean_3"] = df["TotalCount"].rolling(window=3).mean()
df["Total_roll_mean_4"] = df["TotalCount"].rolling(window=4).mean()
df["Total_roll_std_4"] = df["TotalCount"].rolling(window=4).std()

# ==========================================
# 10. Trend features
# ==========================================
df["Total_change_1"] = df["TotalCount"] - df["Total_lag_1"]
df["Total_pct_change_1"] = df["TotalCount"].pct_change()

# ==========================================
# 11. Congestion proxy
# ==========================================
df["CongestionScore"] = df["TotalCount"]
df["CongestionScore_norm"] = (
    (df["CongestionScore"] - df["CongestionScore"].min()) /
    (df["CongestionScore"].max() - df["CongestionScore"].min())
)

# ==========================================
# 12. Encode traffic situation for analysis only
# ==========================================
traffic_map = {"low": 0, "normal": 1, "heavy": 2}
df["TrafficSituationEncoded"] = df["Traffic Situation"].map(traffic_map)

# ==========================================
# 13. Target column
# Predict next 15-minute total traffic
# ==========================================
df["Target_Next_Total"] = df["TotalCount"].shift(-1)

# ==========================================
# 14. Drop NaNs caused by rolling/lags/target shift
# ==========================================
df_model = df.dropna().copy()

# ==========================================
# 15. Final model columns
# ==========================================
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

X = df_model[feature_cols]
y = df_model[target_col]

print("\nFinal feature columns:")
print(feature_cols)
print("\nX shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst 5 rows:")
print(df_model[[
    "Step", "Time", "Date", "Day of the week",
    "TotalCount", "Total_lag_1", "Total_roll_mean_4", "Target_Next_Total"
]].head())

df_model.to_csv("Traffic_engineered_fixed.csv", index=False)
print("\nSaved as Traffic_engineered_fixed.csv")



