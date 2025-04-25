"""
Plot helpers for README figures.
"""
from pathlib import Path
import matplotlib.pyplot as plt


def mae_bar(metrics: dict[str, float], out: str | Path = "docs/images/mae_bar.png"):
    names = list(metrics.keys())
    values = [metrics[n]["mae"] for n in names]

    plt.figure(figsize=(5, 3))
    plt.bar(names, values)
    plt.ylabel("MAE")
    plt.title("Model comparison (lower is better)")
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
