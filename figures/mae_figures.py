import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================
# 1. Create the results table
# =========================================================
results_df = pd.DataFrame({
    "Demand level": ["Overall", "Low demand", "High demand"],
    "MAE": [18.713, 15.565, 22.156],
    "Normalised MAE (%)": ["16.42%", "24.24%", "13.15%"]
})

# =========================================================
# 2. Make matplotlib table
# =========================================================
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis("off")

table = ax.table(
    cellText=results_df.values,
    colLabels=results_df.columns,
    loc="center",
    cellLoc="center",
    colWidths=[0.33, 0.33, 0.34]   # manually widen the last column
)

# =========================================================
# 3. Style the table
# =========================================================
table.auto_set_font_size(False)
table.set_fontsize(20)
table.scale(1.4, 2.6)

for (row, col), cell in table.get_celld().items():
    cell.set_linewidth(2.0)
    if row == 0:
        cell.set_text_props(weight="bold", fontsize=18)

# =========================================================
# 4. Save
# =========================================================
os.makedirs("figures", exist_ok=True)
plt.tight_layout()
plt.savefig("figures/random_forest_error_table_fixed.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: figures/random_forest_error_table_fixed.png")