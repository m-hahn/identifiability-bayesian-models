import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

LABEL_FONT_SIZE = 14
TICK_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
TITLE_FONT_SIZE = 15


SIGMA2_VALUES = [0.2, 0.4, 0.5, 1.0, 2.0]
SIGMA2_VALUES_INTERVAL = [0.1, 0.5, 1.0]
COSINE_P_VALUES = [0, 1, 2, 4, 6]
INTERVAL_P_VALUES = [0, 1, 2, 4, 6, 8]


def vonMises_function(x, mean, sigma2):
    return np.exp(np.cos(x - mean) / sigma2 - 1)


def make_loss_function(MIXTURE_OF_GAUSSIANS_SETUP):
    pi_l = MIXTURE_OF_GAUSSIANS_SETUP["pi_l"]
    mean_1 = MIXTURE_OF_GAUSSIANS_SETUP["mean_1"]
    mean_2 = MIXTURE_OF_GAUSSIANS_SETUP["mean_2"]
    sigma2_1 = MIXTURE_OF_GAUSSIANS_SETUP["sigma2_1"]
    sigma2_2 = MIXTURE_OF_GAUSSIANS_SETUP["sigma2_2"]
    assert pi_l == 1.0

    def LOSS_FUNCTION(difference):
        return (
            (pi_l / math.sqrt(2 * math.pi * sigma2_1) + (1 - pi_l) / math.sqrt(2 * math.pi * sigma2_2))
            - pi_l * vonMises_function(difference, mean_1, sigma2_1)
            - (1 - pi_l) * vonMises_function(difference, mean_2, sigma2_2)
        )

    return LOSS_FUNCTION



def normalize_curve(y):
    y = y - y.min()
    if y.max() <= 1e-12:
        return np.zeros_like(y)
    return y / y.max()


def gaussian_function(x, mean, sigma2):
    return np.exp(-((x - mean) ** 2) / (2 * sigma2)) / math.sqrt(2 * math.pi * sigma2)


def acerbi_interval_loss(difference, sigma2_1, sigma2_2=0.1, pi_l=1.0, mean_1=0.0, mean_2=0.0):
    assert pi_l == 1.0
    return (
        (pi_l / math.sqrt(2 * math.pi * sigma2_1) + (1 - pi_l) / math.sqrt(2 * math.pi * sigma2_2))
        - pi_l * gaussian_function(difference, mean_1, sigma2_1)
        - (1 - pi_l) * gaussian_function(difference, mean_2, sigma2_2)
    )



def make_interval_loss_plot():
    differences = np.linspace(0.0, 3.0, 1000)

    figure, axis = plt.subplots(1, 1, figsize=(7.0, 4.4))
    acerbi_cmap = plt.get_cmap("viridis")

    for i, sigma2_1 in enumerate(SIGMA2_VALUES_INTERVAL):
        values = acerbi_interval_loss(differences, sigma2_1=sigma2_1, sigma2_2=0.1)
        color = acerbi_cmap(i / (len(SIGMA2_VALUES_INTERVAL) - 1))
        axis.plot(
            differences,
            normalize_curve(values),
            color=color,
            linewidth=2.0,
            label=rf"$\sigma={sigma2_1}$",
        )

    axis.set_xlabel(r"Absolute estimation error $|\Delta|$")
    axis.set_ylabel("Loss")
    axis.set_xlim(0.0, 3.0)
    axis.set_ylim(-0.02, 1.02)
    axis.legend(fontsize=LEGEND_FONT_SIZE, ncol=2, frameon=False)
    axis.xaxis.label.set_size(LABEL_FONT_SIZE)
    axis.yaxis.label.set_size(LABEL_FONT_SIZE)
    axis.tick_params(labelsize=TICK_FONT_SIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "acerbi_loss_shapes_interval.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    make_interval_loss_plot()
