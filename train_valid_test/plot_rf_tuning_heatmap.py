import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =========================================================
# 1. Load grid-search results
# =========================================================
df = pd.read_csv("rf_grid_search_results.csv")

# Make sure results are sorted best-to-worst by validation MAE
df = df.sort_values("val_mae", ascending=True).reset_index(drop=True)

# =========================================================
# 2. Get best hyperparameters
# =========================================================
best_row = df.iloc[0]

best_n_estimators = int(best_row["n_estimators"])
best_max_depth = int(best_row["max_depth"])
best_min_samples_split = int(best_row["min_samples_split"])
best_min_samples_leaf = int(best_row["min_samples_leaf"])
best_val_mae = best_row["val_mae"]

print("Best hyperparameters found from grid search:")
print("n_estimators      =", best_n_estimators)
print("max_depth         =", best_max_depth)
print("min_samples_split =", best_min_samples_split)
print("min_samples_leaf  =", best_min_samples_leaf)
print("best val_mae      =", round(best_val_mae, 3))

# =========================================================
# 3. Filter to the best split/leaf combination
# =========================================================
heatmap_df = df[
    (df["min_samples_split"] == best_min_samples_split) &
    (df["min_samples_leaf"] == best_min_samples_leaf)
].copy()

# =========================================================
# 4. Create pivot table
# rows = max_depth
# cols = n_estimators
# values = validation MAE
# =========================================================
pivot_table = heatmap_df.pivot(
    index="max_depth",
    columns="n_estimators",
    values="val_mae"
)

# Sort axes nicely
pivot_table = pivot_table.sort_index()
pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

print("\nHeatmap table:")
print(pivot_table)

# =========================================================
# 5. Plot heatmap
# =========================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

# Lower MAE is better, so reversed colormap is nice
im = ax.imshow(pivot_table.values, cmap="viridis_r", aspect="auto")

# Axis ticks
ax.set_xticks(np.arange(len(pivot_table.columns)))
ax.set_yticks(np.arange(len(pivot_table.index)))

ax.set_xticklabels(pivot_table.columns)
ax.set_yticklabels(pivot_table.index)

ax.set_xlabel("n_estimators")
ax.set_ylabel("max_depth")

ax.set_title(
    "Random Forest Hyperparameter Tuning Heatmap\n"
    f"(Validation MAE, min_samples_split={best_min_samples_split}, "
    f"min_samples_leaf={best_min_samples_leaf})"
)

# =========================================================
# 6. Annotate each cell with MAE value
# =========================================================
for i in range(pivot_table.shape[0]):
    for j in range(pivot_table.shape[1]):
        value = pivot_table.iloc[i, j]
        ax.text(
            j, i,
            f"{value:.3f}",
            ha="center", va="center",
            color="white",
            fontsize=9
        )

# =========================================================
# 7. Highlight best cell
# =========================================================
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

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Validation MAE (lower is better)")

plt.tight_layout()

# =========================================================
# 8. Save figure
# =========================================================
os.makedirs("train_valid_test", exist_ok=True)
save_path = "train_valid_test/rf_tuning_heatmap.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure to:", save_path)


