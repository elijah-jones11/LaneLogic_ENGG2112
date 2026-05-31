import os
import pandas as pd
import matplotlib.pyplot as plt


# 0. Make patterns of days folder

os.makedirs("patterns_of_days", exist_ok=True)

# =========================================================
# 1. Load engineered dataset
# =========================================================
df = pd.read_csv("Traffic_engineered_fixed.csv")

# =========================================================
# 2. Build a readable time-of-day label and slot index
# =========================================================
df["TimeLabel"] = (
    df["Hour"].astype(int).astype(str).str.zfill(2)
    + ":"
    + df["Minute"].astype(int).astype(str).str.zfill(2)
)

# Slot index from 0 to 95 for each 15-minute period in a day
df["TimeSlot"] = df["Hour"] * 4 + (df["Minute"] // 15)

# =========================================================
# 3. Create a daily identifier
# Since full month/year may be imperfect, use DayOfWeek + Date
# This is enough for comparing daily shapes
# =========================================================
df["DayID"] = df["Day of the week"].astype(str) + "_" + df["Date"].astype(int).astype(str)

# =========================================================
# 4. PLOT 1: Every day overlayed
# =========================================================
plt.figure(figsize=(14, 7))

for day_id, day_df in df.groupby("DayID"):
    day_df = day_df.sort_values("TimeSlot")
    plt.plot(day_df["TimeSlot"], day_df["TotalCount"], alpha=0.6)

# Nice x-axis labels
tick_positions = [0, 16, 32, 48, 64, 80, 95]
tick_labels = ["00:00", "04:00", "08:00", "12:00", "16:00", "20:00", "23:45"]
plt.xticks(tick_positions, tick_labels)

plt.xlabel("Time of day")
plt.ylabel("Traffic count")
plt.title("All Days Overlayed: Traffic Count by Time of Day")
plt.tight_layout()
plt.savefig("patterns_of_days/pattern_all_days_overlay.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 5. PLOT 2: Separate plot for each weekday
# =========================================================
weekday_order = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday"
]

for weekday in weekday_order:
    weekday_df = df[df["Day of the week"] == weekday].copy()

    if len(weekday_df) == 0:
        continue

    plt.figure(figsize=(14, 7))

    for day_id, day_df in weekday_df.groupby("DayID"):
        day_df = day_df.sort_values("TimeSlot")
        plt.plot(day_df["TimeSlot"], day_df["TotalCount"], alpha=0.7, label=day_id)

    plt.xticks(tick_positions, tick_labels)
    plt.xlabel("Time of day")
    plt.ylabel("Traffic count")
    plt.title(f"Traffic Pattern for All {weekday}s")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"patterns_of_days/pattern_{weekday.lower()}s_overlay.png", dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# 6. PLOT 3: Average daily profile for each weekday
# =========================================================
plt.figure(figsize=(14, 7))

for weekday in weekday_order:
    weekday_df = df[df["Day of the week"] == weekday].copy()

    if len(weekday_df) == 0:
        continue

    avg_profile = (
        weekday_df.groupby("TimeSlot")["TotalCount"]
        .mean()
        .reset_index()
        .sort_values("TimeSlot")
    )

    plt.plot(avg_profile["TimeSlot"], avg_profile["TotalCount"], label=weekday)

plt.xticks(tick_positions, tick_labels)
plt.xlabel("Time of day")
plt.ylabel("Average traffic count")
plt.title("Average Daily Traffic Profile by Weekday")
plt.legend()
plt.tight_layout()
plt.savefig("patterns_of_days/pattern_weekday_average_profiles.png", dpi=300, bbox_inches="tight")
plt.close()

# =========================================================
# 7. PLOT 4: Fridays only, with thicker average line
# =========================================================
friday_df = df[df["Day of the week"] == "Friday"].copy()

if len(friday_df) > 0:
    plt.figure(figsize=(14, 7))

    # Plot each Friday lightly
    for day_id, day_df in friday_df.groupby("DayID"):
        day_df = day_df.sort_values("TimeSlot")
        plt.plot(day_df["TimeSlot"], day_df["TotalCount"], alpha=0.4)

    # Plot average Friday profile
    friday_avg = (
        friday_df.groupby("TimeSlot")["TotalCount"]
        .mean()
        .reset_index()
        .sort_values("TimeSlot")
    )

    plt.plot(friday_avg["TimeSlot"], friday_avg["TotalCount"], linewidth=3, label="Average Friday")

    plt.xticks(tick_positions, tick_labels)
    plt.xlabel("Time of day")
    plt.ylabel("Traffic count")
    plt.title("All Fridays + Average Friday Traffic Profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig("patterns_of_days/pattern_fridays_with_average.png", dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# 8. Save helpful summary tables
# =========================================================
weekday_avg_table = (
    df.groupby(["Day of the week", "TimeSlot"])["TotalCount"]
    .mean()
    .reset_index()
)

weekday_avg_table.to_csv("patterns_of_days/pattern_weekday_average_profiles.csv", index=False)

print("Saved plots:")
print("- patterns_of_days/pattern_all_days_overlay.png")
for weekday in weekday_order:
    print(f"- patterns_of_days/pattern_{weekday.lower()}s_overlay.png")
print("- patterns_of_days/pattern_weekday_average_profiles.png")
print("- patterns_of_days/pattern_fridays_with_average.png")
print("- patterns_of_days/pattern_weekday_average_profiles.csv")

