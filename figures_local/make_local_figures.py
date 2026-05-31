import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 0. SETTINGS
# =========================================================
INPUT_PATH = "Cleaned_Datasets_local/traffic_counts_ALL_DAYS.csv"
OUTPUT_DIR = "figures_local"

# Resampling bin sizes
BIN_1 = "1min"
BIN_5 = "5min"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_PATH)

print("Loaded file:", INPUT_PATH)
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# =========================================================
# 2. CHECK REQUIRED COLUMNS
# =========================================================
required_cols = [
    "video_name",
    "timestamp_seconds",
    "real_timestamp_hhmmss",
    "person_count",
    "bicycle_count",
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    "total_vehicle_count",
    "total_road_user_count"
]

for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# =========================================================
# 3. CLEAN TYPES
# =========================================================
count_cols = [
    "person_count",
    "bicycle_count",
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    "traffic_light_count",
    "total_vehicle_count",
    "total_road_user_count"
]

for col in count_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["timestamp_seconds"] = pd.to_numeric(df["timestamp_seconds"], errors="coerce")

# =========================================================
# 4. BUILD DAY LABELS
# =========================================================
# Use video_name as the day label
df["DayLabel"] = df["video_name"].astype(str)

# Make safe day label for file names
def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text))

# =========================================================
# 5. PARSE REAL CLOCK TIME
# =========================================================
# Use real_timestamp_hhmmss, not timestamp_hhmmss
df["real_time"] = pd.to_datetime(df["real_timestamp_hhmmss"], format="%H:%M:%S", errors="coerce")

if df["real_time"].isna().any():
    bad_rows = df[df["real_time"].isna()]
    print("\nWarning: some real_timestamp_hhmmss values could not be parsed.")
    print(bad_rows[["real_timestamp_hhmmss"]].head())

# Minutes since midnight for plotting
df["time_of_day_minutes"] = df["real_time"].dt.hour * 60 + df["real_time"].dt.minute

# =========================================================
# 6. CREATE A PSEUDO DATETIME FOR RESAMPLING
# =========================================================
unique_days = list(df["DayLabel"].unique())
day_to_base_date = {
    d: pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
    for i, d in enumerate(unique_days)
}

df["base_date"] = df["DayLabel"].map(day_to_base_date)
df["pseudo_datetime"] = (
    df["base_date"]
    + pd.to_timedelta(df["real_time"].dt.hour, unit="h")
    + pd.to_timedelta(df["real_time"].dt.minute, unit="m")
    + pd.to_timedelta(df["real_time"].dt.second, unit="s")
)

# =========================================================
# 7. RESAMPLE TO 1-MIN AND 5-MIN BINS
# =========================================================
def resample_by_day(frame, bin_size):
    pieces = []

    for day, day_df in frame.groupby("DayLabel"):
        day_df = day_df.sort_values("pseudo_datetime").set_index("pseudo_datetime")

        agg_dict = {
            "person_count": "mean",
            "bicycle_count": "mean",
            "car_count": "mean",
            "motorcycle_count": "mean",
            "bus_count": "mean",
            "truck_count": "mean",
            "total_vehicle_count": "mean",
            "total_road_user_count": "mean"
        }

        if "traffic_light_count" in day_df.columns:
            agg_dict["traffic_light_count"] = "mean"

        out = day_df.resample(bin_size).agg(agg_dict).reset_index()
        out["DayLabel"] = day
        out["clock_time"] = out["pseudo_datetime"].dt.strftime("%H:%M:%S")
        out["time_of_day_minutes"] = (
            out["pseudo_datetime"].dt.hour * 60
            + out["pseudo_datetime"].dt.minute
        )
        pieces.append(out)

    return pd.concat(pieces, ignore_index=True)

resampled_1min = resample_by_day(df, BIN_1)
resampled_5min = resample_by_day(df, BIN_5)

resampled_1min.to_csv(os.path.join(OUTPUT_DIR, "local_resampled_1min.csv"), index=False)
resampled_5min.to_csv(os.path.join(OUTPUT_DIR, "local_resampled_5min.csv"), index=False)

# =========================================================
# 8. PLOTTING HELPERS
# =========================================================
def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

def minute_ticks(min_minutes, max_minutes):
    start_hour = int(np.floor(min_minutes / 60))
    end_hour = int(np.ceil(max_minutes / 60))
    ticks = np.arange(start_hour * 60, end_hour * 60 + 1, 60)
    labels = [f"{int(t//60):02d}:00" for t in ticks]
    return ticks, labels

min_mins = int(resampled_5min["time_of_day_minutes"].min())
max_mins = int(resampled_5min["time_of_day_minutes"].max())

# =========================================================
# 9. GRAPH 1: TOTAL TRAFFIC COUNT OVER TIME
# =========================================================
plt.figure(figsize=(12, 6))
for day, day_df in resampled_5min.groupby("DayLabel"):
    plt.plot(day_df["time_of_day_minutes"], day_df["total_road_user_count"], label=day)

ticks, labels = minute_ticks(min_mins, max_mins)
plt.xticks(ticks, labels, rotation=45)
plt.xlabel("Real time of day")
plt.ylabel("Total traffic count")
plt.title("Local Data: Total Traffic Count Over Time (5-minute bins)")
plt.legend()
save_plot(os.path.join(OUTPUT_DIR, "graph_1_total_traffic_over_time.png"))

# =========================================================
# 10. GRAPH 2: VEHICLE COUNTS OVER TIME
# =========================================================
plt.figure(figsize=(12, 6))
for day, day_df in resampled_5min.groupby("DayLabel"):
    plt.plot(day_df["time_of_day_minutes"], day_df["total_vehicle_count"], label=day)

ticks, labels = minute_ticks(min_mins, max_mins)
plt.xticks(ticks, labels, rotation=45)
plt.xlabel("Real time of day")
plt.ylabel("Vehicle count")
plt.title("Local Data: Vehicle Counts Over Time (5-minute bins)")
plt.legend()
save_plot(os.path.join(OUTPUT_DIR, "graph_2_vehicle_counts_over_time.png"))

# =========================================================
# 11. GRAPH 3: PEDESTRIANS OVER TIME
# =========================================================
plt.figure(figsize=(12, 6))
for day, day_df in resampled_5min.groupby("DayLabel"):
    plt.plot(day_df["time_of_day_minutes"], day_df["person_count"], label=day)

ticks, labels = minute_ticks(min_mins, max_mins)
plt.xticks(ticks, labels, rotation=45)
plt.xlabel("Real time of day")
plt.ylabel("Pedestrian count")
plt.title("Local Data: Pedestrian Counts Over Time (5-minute bins)")
plt.legend()
save_plot(os.path.join(OUTPUT_DIR, "graph_3_pedestrians_over_time.png"))

# =========================================================
# 12. GRAPH 4: CARS VS PEDESTRIANS VS BUSES/TRUCKS
# Average across all local days
# =========================================================
avg_5min = resampled_5min.groupby("time_of_day_minutes").mean(numeric_only=True).reset_index()
avg_5min["buses_trucks"] = avg_5min["bus_count"] + avg_5min["truck_count"]

plt.figure(figsize=(12, 6))
plt.plot(avg_5min["time_of_day_minutes"], avg_5min["car_count"], label="Cars")
plt.plot(avg_5min["time_of_day_minutes"], avg_5min["person_count"], label="Pedestrians")
plt.plot(avg_5min["time_of_day_minutes"], avg_5min["buses_trucks"], label="Buses + Trucks")

ticks, labels = minute_ticks(min_mins, max_mins)
plt.xticks(ticks, labels, rotation=45)
plt.xlabel("Real time of day")
plt.ylabel("Average count")
plt.title("Local Data: Cars vs Pedestrians vs Buses/Trucks")
plt.legend()
save_plot(os.path.join(OUTPUT_DIR, "graph_4_cars_vs_pedestrians_vs_busestrucks.png"))

# =========================================================
# 13. GRAPH 5: ONE-DAY PROFILES
# =========================================================
for day, day_df in resampled_5min.groupby("DayLabel"):
    plt.figure(figsize=(12, 6))
    plt.plot(day_df["time_of_day_minutes"], day_df["total_road_user_count"], label="Total traffic")
    plt.plot(day_df["time_of_day_minutes"], day_df["car_count"], label="Cars")
    plt.plot(day_df["time_of_day_minutes"], day_df["person_count"], label="Pedestrians")

    ticks, labels = minute_ticks(int(day_df["time_of_day_minutes"].min()), int(day_df["time_of_day_minutes"].max()))
    plt.xticks(ticks, labels, rotation=45)
    plt.xlabel("Real time of day")
    plt.ylabel("Count")
    plt.title(f"Local Data: One-Day Traffic Profile - {day}")
    plt.legend()

    save_plot(os.path.join(OUTPUT_DIR, f"graph_5_one_day_profile_{safe_name(day)}.png"))

# =========================================================
# 14. GRAPH 6: COMPARISON ACROSS THE 3 DAYS
# =========================================================
plt.figure(figsize=(12, 6))
for day, day_df in resampled_5min.groupby("DayLabel"):
    plt.plot(day_df["time_of_day_minutes"], day_df["total_road_user_count"], linewidth=2, label=day)

ticks, labels = minute_ticks(min_mins, max_mins)
plt.xticks(ticks, labels, rotation=45)
plt.xlabel("Real time of day")
plt.ylabel("Total traffic count")
plt.title("Local Data: Comparison Across Local Days (5-minute bins)")
plt.legend()
save_plot(os.path.join(OUTPUT_DIR, "graph_6_comparison_across_days.png"))

# =========================================================
# 15. GRAPH 7: TIME-OF-DAY AVERAGE
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(avg_5min["time_of_day_minutes"], avg_5min["total_road_user_count"], linewidth=3, label="Average total traffic")

ticks, labels = minute_ticks(min_mins, max_mins)
plt.xticks(ticks, labels, rotation=45)
plt.xlabel("Real time of day")
plt.ylabel("Average traffic count")
plt.title("Local Data: Average Time-of-Day Traffic Profile")
plt.legend()
save_plot(os.path.join(OUTPUT_DIR, "graph_7_time_of_day_average_total.png"))

# =========================================================
# 16. GRAPH 8: SMOOTHED CURVES
# 1-minute bins with rolling mean
# =========================================================
plt.figure(figsize=(12, 6))
for day, day_df in resampled_1min.groupby("DayLabel"):
    smooth = day_df["total_road_user_count"].rolling(window=5, center=True, min_periods=1).mean()
    plt.plot(day_df["time_of_day_minutes"], smooth, linewidth=2, label=day)

ticks, labels = minute_ticks(int(resampled_1min["time_of_day_minutes"].min()), int(resampled_1min["time_of_day_minutes"].max()))
plt.xticks(ticks, labels, rotation=45)
plt.xlabel("Real time of day")
plt.ylabel("Smoothed traffic count")
plt.title("Local Data: Smoothed Total Traffic Curves (1-minute bins)")
plt.legend()
save_plot(os.path.join(OUTPUT_DIR, "graph_8_smoothed_total_curves.png"))

# =========================================================
# 17. GRAPH 9: AVERAGE CLASS COUNTS BY TIME OF DAY
# =========================================================
plt.figure(figsize=(12, 6))
plt.plot(avg_5min["time_of_day_minutes"], avg_5min["car_count"], label="Cars")
plt.plot(avg_5min["time_of_day_minutes"], avg_5min["person_count"], label="Pedestrians")
plt.plot(avg_5min["time_of_day_minutes"], avg_5min["bus_count"], label="Buses")
plt.plot(avg_5min["time_of_day_minutes"], avg_5min["truck_count"], label="Trucks")
plt.plot(avg_5min["time_of_day_minutes"], avg_5min["bicycle_count"], label="Bicycles")
plt.plot(avg_5min["time_of_day_minutes"], avg_5min["motorcycle_count"], label="Motorcycles")

ticks, labels = minute_ticks(min_mins, max_mins)
plt.xticks(ticks, labels, rotation=45)
plt.xlabel("Real time of day")
plt.ylabel("Average count")
plt.title("Local Data: Average Class Counts by Time of Day")
plt.legend()
save_plot(os.path.join(OUTPUT_DIR, "graph_9_average_class_counts_by_time.png"))

# =========================================================
# 18. SAVE SUMMARY TABLES
# =========================================================
summary_by_day = (
    resampled_5min.groupby("DayLabel")[["total_road_user_count", "total_vehicle_count", "person_count"]]
    .mean()
    .reset_index()
    .rename(columns={
        "total_road_user_count": "avg_total_traffic",
        "total_vehicle_count": "avg_vehicle_traffic",
        "person_count": "avg_pedestrian_traffic"
    })
)

summary_by_day.to_csv(os.path.join(OUTPUT_DIR, "summary_by_day.csv"), index=False)
avg_5min.to_csv(os.path.join(OUTPUT_DIR, "average_profile_5min.csv"), index=False)

print("\nSaved outputs to:", OUTPUT_DIR)
print("\nMain figure files created:")
print("graph_1_total_traffic_over_time.png")
print("graph_2_vehicle_counts_over_time.png")
print("graph_3_pedestrians_over_time.png")
print("graph_4_cars_vs_pedestrians_vs_busestrucks.png")
print("graph_6_comparison_across_days.png")
print("graph_7_time_of_day_average_total.png")
print("graph_8_smoothed_total_curves.png")
print("graph_9_average_class_counts_by_time.png")
print("\nAlso created:")
print("- one graph_5_one_day_profile_... file per day")
print("- local_resampled_1min.csv")
print("- local_resampled_5min.csv")
print("- summary_by_day.csv")
print("- average_profile_5min.csv")

