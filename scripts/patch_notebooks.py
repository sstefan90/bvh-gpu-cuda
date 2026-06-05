"""Patch m4_scaling.ipynb cells for higher DPI, larger fonts, and label fix."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "m4_scaling.ipynb"

nb = json.loads(NB_PATH.read_text())

patches = {
    # ── rcParams cell: dpi 110→200, font 11→13 ──────────────────────────
    "f6d7ec12": (
        '    "figure.dpi": 110,\n    "axes.spines.top": False,\n    "axes.spines.right": False,\n    "axes.grid": True,\n    "grid.alpha": 0.3,\n    "font.size": 11,',
        '    "figure.dpi": 200,\n    "axes.spines.top": False,\n    "axes.spines.right": False,\n    "axes.grid": True,\n    "grid.alpha": 0.3,\n    "font.size": 13,',
    ),
    # ── strong scaling: bump legend fontsize ────────────────────────────
    "0cb79af5": (
        'ax.legend(loc="upper left", fontsize=9)',
        'ax.legend(loc="upper left", fontsize=11)',
    ),
    # ── weak scaling: bump legend fontsize ──────────────────────────────
    "a53fb351": (
        'ax.legend(loc="upper left", fontsize=9)',
        'ax.legend(loc="upper left", fontsize=11)',
    ),
    # ── compute vs comm: move label inside bar, add ylim headroom ───────
    "b2c74b25": (
        '    top = max(g + m for g, m in zip(gpu_compute, mpi_comm))\n    for i, (g, m) in enumerate(zip(gpu_compute, mpi_comm)):\n        pct = m / (g + m) * 100 if (g + m) > 0 else 0\n        ax.text(i, g + m + top * 0.02, f"{pct:.0f}%\\ncomm",\n                ha="center", fontsize=8, color="#e15759")',
        '    top = max(g + m for g, m in zip(gpu_compute, mpi_comm))\n    ax.set_ylim(0, top * 1.20)\n    for i, (g, m) in enumerate(zip(gpu_compute, mpi_comm)):\n        pct = m / (g + m) * 100 if (g + m) > 0 else 0\n        if m > top * 0.03:\n            ax.text(i, g + m * 0.55, f"{pct:.0f}%\\ncomm",\n                    ha="center", va="center", fontsize=10, color="white", fontweight="bold")\n        else:\n            ax.text(i, g + m + top * 0.02, f"{pct:.0f}%\\ncomm",\n                    ha="center", va="bottom", fontsize=10, color="#e15759")',
    ),
}

for cell in nb["cells"]:
    cell_id = cell.get("id", "")
    if cell_id in patches and cell["cell_type"] == "code":
        old, new = patches[cell_id]
        src = "".join(cell["source"])
        if old in src:
            cell["source"] = list(src.replace(old, new, 1))
            print(f"  Patched cell {cell_id}")
        else:
            print(f"  WARNING: pattern not found in cell {cell_id}")

NB_PATH.write_text(json.dumps(nb, indent=1))
print("Done.")
