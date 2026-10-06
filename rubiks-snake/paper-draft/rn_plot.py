"""Generate the paper's consecutive-ratio plot from the stored exact counts."""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rubiks_snake import RubiksSnakeCounter


def main() -> None:
    counts = np.asarray(RubiksSnakeCounter.S[1:], dtype=np.float64)
    n = np.arange(1, len(counts))
    ratios = counts[1:] / counts[:-1]
    output = Path(__file__).resolve().with_suffix(".png")

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 0.6,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    ):
        fig, ax = plt.subplots(figsize=(3.5, 2.35), layout="constrained")
        ax.plot(
            n,
            ratios,
            color="black",
            linewidth=0.9,
            marker="o",
            markersize=2.5,
            markerfacecolor="white",
            markeredgewidth=0.6,
        )
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(r"$r_n = S_{n+1}/S_n$")
        ax.set_xlim(0, n[-1] + 1)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.tick_params(direction="out", length=3, width=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.85", linewidth=0.4)
        ax.set_axisbelow(True)
        fig.savefig(output, dpi=600, facecolor="white")
        plt.close(fig)

    print(f"Saved {output}")


if __name__ == "__main__":
    main()
