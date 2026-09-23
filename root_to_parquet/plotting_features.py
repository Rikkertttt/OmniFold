import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Optional

import awkward as ak
import numpy as np

import awkward as ak
import matplotlib.pyplot as plt
import numpy as np


def plot_hist(
    data: ak.Array,
    filename: Optional[str] = None,
    *,
    show: bool = False,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Entries",
    show_mean_std: bool = False,
    bins: int = 50,
    density: bool = False,
):
    """Plot a clean histogram of nested or flat data."""

    values = ak.to_numpy(
        ak.flatten(ak.drop_none(data, axis=None))
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        values,
        bins=bins,
        histtype="step",
        linewidth=1.8,
        color="#0072B2",
        density=density,
    )

    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel="Normalized entries" if density else ylabel,
    )

    ax.grid(axis="y", alpha=0.25, linestyle="--")

    # Cleaner analysis-style axes.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", top=False, right=False)

    if show_mean_std:
        stats = (
            f"Entries: {len(values)}\n"
            f"Mean: {np.mean(values):.4f}\n"
            f"Std: {np.std(values):.3f}"
        )

        ax.text(
            0.98,
            0.98,
            stats,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "edgecolor": "lightgray",
                "alpha": 0.9,
            },
        )

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename + ".pdf", bbox_inches="tight")
        fig.savefig(filename + ".png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax

def plot_comparison_hist(
    data: list[ak.Array],
    datanames: Optional[list[str]] = None,
    filename: Optional[str] = None,
    *,
    show: bool = False,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Entries",
    bins: int = 50,
) -> tuple[Figure, Axes]:
    """Plot a comparison histogram of two datasets, optionally saving or showing it."""

    fig, ax = plt.subplots()

    for i in range(len(data)):
        values = ak.to_numpy(ak.flatten(data[i]))
        ax.hist(values, bins=bins, alpha=0.5, label=datanames[i] if datanames is not None else f"Dataset {i+1}")
    
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()

    if filename is not None:
        fig.savefig(filename + ".pdf", bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_comparison_hist_y(
    data: list[ak.Array],
    datanames: Optional[list[str]] = None,
    filename: Optional[str] = None,
    *,
    show: bool = False,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Entries",
    bins: int = 50,
    density: bool = False,
) -> tuple[Figure, Axes]:
    """Plot a visually clean comparison histogram."""

    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

    # Flatten once and use shared bin edges for every dataset.
    values_list = [ak.to_numpy(ak.flatten(dataset)) for dataset in data]

    global_min = min(values.min() for values in values_list)
    global_max = max(values.max() for values in values_list)
    bin_edges = np.linspace(global_min, global_max, bins + 1)

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, values in enumerate(values_list):
        label = datanames[i] if datanames is not None else f"Dataset {i + 1}"

        ax.hist(
            values,
            bins=bin_edges,
            histtype="step",
            linewidth=1.8,
            color=colors[i % len(colors)],
            label=label,
            density=density,
        )

    ax.set(
        title=title,
        xlabel=xlabel,
        ylabel="Normalized entries" if density else ylabel,
    )

    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)

    # Cleaner analysis-style axes.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", top=False, right=False)

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename + ".pdf", bbox_inches="tight")
        fig.savefig(filename + ".png", dpi=200, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax