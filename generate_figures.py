import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV
df = pd.read_csv("results_complete.csv")

# === Define decomposition names and phases
decomp_names = {1: "Row-Slab", 2: "Column-Slab", 3: "Tiled"}
phases = ["Scatter_ms", "Sobel_ms", "Gather_ms"]
colors = ["blue", "orange", "green"]
markers = ["o", "s", "^"]

# Create a figure with 3 subplots
plt.figure(figsize=(15, 4))

for i, (decomp, name) in enumerate(decomp_names.items(), start=1):
    plt.subplot(1, 3, i)
    sub = df[df["Decomp"] == decomp].copy().sort_values("Nprocs")

    # Select the baseline as the smallest number of processes (usually 4)
    baseline = sub[sub["Nprocs"] == sub["Nprocs"].min()].iloc[0]

    # Compute speedup for each phase: Speedup = baseline_time / current_time
    for phase, color, marker in zip(phases, colors, markers):
        sub[f"{phase}_speedup"] = baseline[phase] / sub[phase]
        plt.plot(sub["Nprocs"], sub[f"{phase}_speedup"],
                 label=phase.replace("_ms", ""),
                 marker=marker, color=color)

    # Configure plot aesthetics
    plt.title(f"{name} Decomposition")
    plt.xlabel("Concurrency (Number of Processes)")
    plt.ylabel("Speedup")
    plt.legend()
    plt.grid(True)

# Adjust layout and save figure
plt.tight_layout()
plt.savefig("speedup_plot.png", dpi=300)
plt.show()

print("✅ Speedup plot saved as speedup_plot.png")


