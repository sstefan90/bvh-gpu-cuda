"""Regenerate roofline.png with higher DPI and larger fonts.

Data hardcoded from Table 1 of the final report (N=1M, atomic refit).
Byte/FLOP estimates match those in m3_performance.ipynb.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPORTS_DIR = Path(__file__).parent.parent / "reports"

plt.rcParams.update({"font.size": 13, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})

# Hardware: Quadro RTX 6000, sm_75
PEAK_BW_GBs   = 672.0        # GB/s
PEAK_FP32_GFs = 16310.0      # GFLOPS (72 SMs × 64 cores × 2 × 2.1 GHz × 1e-3)
RIDGE_FLOPS_BYTE = PEAK_FP32_GFs / PEAK_BW_GBs  # ~24.3

N = 1_000_000  # reference scene size from Table 1

stages = ["morton", "sort", "karras", "refit"]
labels = ["Morton encode", "CUB radix sort", "Karras construction", "AABB refit"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
markers = ["o", "s", "^", "D"]

# From Table 1 (median of 5 runs, N=1M, atomic refit)
t_ms = {"morton": 0.28, "sort": 0.45, "karras": 3.50, "refit": 0.46}

# Byte and FLOP estimates from m3_performance.ipynb
bytes_per_tri = {"morton": 44, "sort": 32, "karras": 84, "refit": 128}
flops_per_tri = {"morton": 35, "sort": 1,  "karras": 35, "refit": 12}

ai    = {s: flops_per_tri[s] / bytes_per_tri[s] for s in stages}
gflops = {s: flops_per_tri[s] * N / (t_ms[s] * 1e-3) / 1e9 for s in stages}

fig, ax = plt.subplots(figsize=(8, 5))

ai_range = np.logspace(-2, 3, 500)
roofline = np.minimum(PEAK_FP32_GFs, PEAK_BW_GBs * ai_range)
ax.loglog(ai_range, roofline, "k-", linewidth=2, label="Roofline (Quadro RTX 6000)")
ax.axvline(RIDGE_FLOPS_BYTE, color="gray", linestyle=":", linewidth=1.5,
           label=f"Ridge point ({RIDGE_FLOPS_BYTE:.1f} FLOP/B)")

for s, label, color, mk in zip(stages, labels, colors, markers):
    ax.scatter(ai[s], gflops[s], s=120, color=color, marker=mk, zorder=5)
    ax.annotate(label, (ai[s], gflops[s]),
                textcoords="offset points", xytext=(10, 4),
                fontsize=11, color=color)

ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
ax.set_ylabel("Achieved throughput (GFLOPS)")
ax.set_title(f"Roofline — Quadro RTX 6000  (N=1M triangles)")
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, which="both", alpha=0.25)
ax.set_xlim(1e-2, 1e2)
ax.set_ylim(1e0, PEAK_FP32_GFs * 1.5)

plt.tight_layout()
out = REPORTS_DIR / "roofline.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"Saved {out}")
