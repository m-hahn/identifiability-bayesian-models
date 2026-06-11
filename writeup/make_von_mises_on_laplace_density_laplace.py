import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch


FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

GRID = 180
MIN_GRID = 0
MAX_GRID = 360
NOISE_CONDITIONS = [2, 3, 4, 5]
SIGMA_LOGITS = {
    2: -1.4008985757827759,
    3: -3.219881057739258,
    4: -3.7730233669281006,
    5: -4.337666988372803,
}
NOISE_COLORS = {
    2: "#1f77b4",
    3: "#ff7f0e",
    4: "#2ca02c",
    5: "#d62728",
}
PANEL_LABEL_FONTSIZE = 24
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 15


def MakeZeros(size):
    return torch.zeros(size)


def wrapped_circular_unsigned_distance(x):
    y = torch.remainder(x + math.pi, 2 * math.pi) - math.pi
    return y.abs()


def centered_degrees(grid, reference_index):
    reference = float(grid[reference_index])
    return torch.remainder(grid - reference + 180.0, 360.0) - 180.0


def orientation_axis_and_density(grid, reference_index, density):
    internal_axis = centered_degrees(grid, reference_index)
    return internal_axis / 2.0, 2.0 * density


def plot_density_panels():
    grid = torch.tensor([x / GRID * (MAX_GRID - MIN_GRID) for x in range(GRID)]) + MIN_GRID
    volumeElement = MakeZeros(GRID) + (2.0 * math.pi / GRID)
    reference_index = GRID // 2
    figure, axis = plt.subplots(1, 1, figsize=(7.0, 4.0))

    for condition in NOISE_CONDITIONS:
        sigma_logit = torch.tensor(SIGMA_LOGITS[condition])
        sigma2 = 4 * torch.sigmoid(sigma_logit)
        F = torch.cat([MakeZeros(1), torch.cumsum(volumeElement, dim=0)], dim=0)

        # Keep this density construction visually close to
        # `SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LaplaceNoise.py`
        sensory_likelihoods_laplace = torch.softmax(
            ((-wrapped_circular_unsigned_distance(F[:-1].unsqueeze(0) - F[:-1].unsqueeze(1))) / (sigma2).sqrt())
            + volumeElement.unsqueeze(1).log(),
            dim=0,
        )

        x_axis, density = orientation_axis_and_density(grid, reference_index, sensory_likelihoods_laplace[:, reference_index])
        x_order = torch.argsort(x_axis)
        axis.plot(
            x_axis[x_order].numpy(),
            density[x_order].numpy(),
            color=NOISE_COLORS[condition],
            linewidth=2.8,
        )

    axis.set_xlim(-90, 90)
    axis.set_xticks([-90, -45, 0, 45, 90])
    axis.set_xlabel("Orientation error (degrees)", fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel("Laplace density", fontsize=AXIS_LABEL_FONTSIZE)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_FONTSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.0,
        1.02,
        "B",
        transform=axis.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "von_mises_on_laplace_density_laplace.pdf", bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    plot_density_panels()
