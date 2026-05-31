import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 1. Load  the grid search results

df = pd.read_csv("rf_grid_search_results.csv")

# Sort so best is at the top
df = df.sort_values(by="val_mae", ascending=True).reset_index(drop=True)

# Create output folder
os.makedirs("train_valid_test", exist_ok=True)


# 2. Get best parameter settings

best_row = df.iloc[0]

best_n_estimators = int(best_row["n_estimators"])
best_max_depth = int(best_row["max_depth"])
best_min_samples_split = int(best_row["min_samples_split"])
best_min_samples_leaf = int(best_row["min_samples_leaf"])
best_val_mae = best_row["val_mae"]

print("Best parameter settings:")
print("n_estimators      =", best_n_estimators)
print("max_depth         =", best_max_depth)
print("min_samples_split =", best_min_samples_split)
print("min_samples_leaf  =", best_min_samples_leaf)
print("Validation MAE    =", round(best_val_mae, 3))

# =========================================================
# 3. TOP-10 BAR CHART
# =========================================================
top10 = df.head(10).copy()

# Make readable labels
top10["label"] = (
    "n=" + top10["n_estimators"].astype(str) +
    ", d=" + top10["max_depth"].astype(str) +
    ", split=" + top10["min_samples_split"].astype(str) +
    ", leaf=" + top10["min_samples_leaf"].astype(str)
)

# Reverse so best appears at top in barh
top10_plot = top10.iloc[::-1]

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(top10_plot["label"], top10_plot["val_mae"])

ax.set_title("Top 10 Random Forest Hyperparameter Combinations")
ax.set_xlabel("Validation MAE (lower is better)")
ax.set_ylabel("Parameter combination")

# Add value labels
for bar, value in zip(bars, top10_plot["val_mae"]):
    ax.text(
        bar.get_width() + 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.3f}",
        va="center",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("train_valid_test/rf_top10_bar_chart.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: train_valid_test/rf_top10_bar_chart.png")

# =========================================================
# 4. 4-PANEL HEATMAP VERSION
# Panels = min_samples_leaf values
# Fix min_samples_split at best value
# =========================================================
leaf_values = sorted(df["min_samples_leaf"].unique())

# Filter to best min_samples_split only
heatmap_source = df[df["min_samples_split"] == best_min_samples_split].copy()

# For common colour scale across all 4 panels
global_vmin = heatmap_source["val_mae"].min()
global_vmax = heatmap_source["val_mae"].max()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, leaf_val in zip(axes, leaf_values):
    panel_df = heatmap_source[heatmap_source["min_samples_leaf"] == leaf_val].copy()

    pivot_table = panel_df.pivot(
        index="max_depth",
        columns="n_estimators",
        values="val_mae"
    )

    pivot_table = pivot_table.sort_index()
    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

    im = ax.imshow(
        pivot_table.values,
        cmap="viridis_r",
        aspect="auto",
        vmin=global_vmin,
        vmax=global_vmax
    )

    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index)

    ax.set_xlabel("n_estimators")
    ax.set_ylabel("max_depth")
    ax.set_title(f"min_samples_leaf = {leaf_val}")

    # Annotate cells
    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            value = pivot_table.iloc[i, j]
            ax.text(
                j, i,
                f"{value:.2f}",
                ha="center", va="center",
                color="white",
                fontsize=8
            )

    # Highlight best overall model if it lies in this panel
    if leaf_val == best_min_samples_leaf:
        best_i = list(pivot_table.index).index(best_max_depth)
        best_j = list(pivot_table.columns).index(best_n_estimators)

        rect = Rectangle(
            (best_j - 0.5, best_i - 0.5),
            1, 1,
            fill=False,
            edgecolor="red",
            linewidth=3
        )
        ax.add_patch(rect)

fig.suptitle(
    f"Random Forest Validation MAE Heatmaps\n"
    f"(Fixed min_samples_split = {best_min_samples_split})",
    fontsize=16
)

cbar = fig.colorbar(im, ax=axes, shrink=0.9)
cbar.set_label("Validation MAE (lower is better)")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("train_valid_test/rf_4panel_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: train_valid_test/rf_4panel_heatmap.png")