"""Small matplotlib helpers shared by the notebooks (consistent style, saved figures)."""
import matplotlib.pyplot as plt

from src import config

COLORS = {
    "actual": "#1f2a44",
    "forecast": "#d1495b",
    "interval": "#f4b6bd",
    "accent": "#00798c",
    "muted": "#8d99ae",
    "highlight": "#edae49",
}


def apply_style() -> None:
    """Apply one consistent look to every figure."""
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 130, "figure.figsize": (11, 4.2),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10, "legend.frameon": False, "font.size": 10,
    })


def save_figure(fig, name: str) -> str:
    """Save a figure into reports/figures/ and return the file name."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight")
    return path.name


def gbp(x, _pos=None) -> str:
    """Axis formatter: 12,345 -> '£12k'."""
    return f"£{x / 1000:,.0f}k"
