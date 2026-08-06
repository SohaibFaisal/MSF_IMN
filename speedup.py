x_values = [1.0, 1.5, 2.0, 3.0]          # Mesh refinement
y_values = [3/9, 20/9, 141/9, 2670/9]

import matplotlib.pyplot as plt

# Keep text editable in the exported SVG
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "svg.fonttype": "none",
})

# ==========================
# Input data
# ==========================


# ==========================
# Plot
# ==========================
fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(
    x_values,
    y_values,
    color="#4C72B0",
    linewidth=1.5,
    marker="o",
    markersize=6,
    markerfacecolor="white",
    markeredgecolor="#4C72B0",
    markeredgewidth=1.5,
)

# Display only the supplied x-values
ax.set_xticks(x_values)
ax.set_xticklabels([f"{x:g}" for x in x_values])

# Write each y-value above its corresponding point
for x, y in zip(x_values, y_values):
    ax.annotate(
        f"{y:.1f}",
        xy=(x, y),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=10,
    )

ax.set_xlabel("Mesh refinement")
ax.set_ylabel("Speed-up factor")

ax.grid(
    True,
    linestyle="--",
    linewidth=0.6,
    alpha=0.5,
)

ax.tick_params(
    axis="both",
    direction="in",
    top=True,
    right=True,
)

# Add space above the highest point for the value labels
y_range = max(y_values) - min(y_values)
padding = 0.12 * y_range if y_range > 0 else 1.0
ax.set_ylim(
    min(y_values) - 0.08 * y_range,
    max(y_values) + padding,
)

# Uncomment when smaller mesh size means greater refinement
# ax.invert_xaxis()

fig.tight_layout()

# Export for editing in Inkscape
fig.savefig(
    "speedup_mesh_refinement.svg",
    format="svg",
    bbox_inches="tight",
    transparent=True,
)

plt.show()