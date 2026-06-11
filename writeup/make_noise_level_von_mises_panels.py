import ast
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
MPLCONFIGDIR = HERE / ".mplconfig"
MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy.stats import vonmises

REPO_ROOT = HERE.parent
OTHER_REPO_ROOT = REPO_ROOT.parent / "unifying-theory-biases"

OUTPUT_PATH = HERE / "noise_level_von_mises_panels.pdf"

SYNTHETIC_LOG = (
    REPO_ROOT
    / "code"
    / "Synthetic"
    / "logs"
    / "SIMULATED_REPLICATE"
    / "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_2345_N10000_STEEPPERIODIC_STEEPPERIODIC.txt"
)
LEVEL_COLORS = {
    2: "#1f77b4",
    3: "#ff7f0e",
    4: "#2ca02c",
    5: "#d62728",
}
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12


def extract_sigma_logit(path: Path):
    for line in path.read_text().splitlines():
        if line.startswith("sigma_logit"):
            _, value = line.split("\t", 1)
            return ast.literal_eval(value.strip())
    raise ValueError(f"No sigma_logit line found in {path}")


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def sigma2_from_logit(logit, scale):
    return scale * sigmoid(float(logit))


def circular_density_from_sigma2(theta_deg, sigma2):
    theta_rad = np.deg2rad(theta_deg)
    kappa = 1.0 / sigma2
    return vonmises.pdf(theta_rad, kappa)


def orientation_density_from_sigma2(theta_deg, sigma2):
    theta_deg = np.asarray(theta_deg)
    internal_theta_deg = 2.0 * theta_deg
    return 2.0 * circular_density_from_sigma2(internal_theta_deg, sigma2)


def circular_sd_degrees_from_sigma2(sigma2):
    kappa = 1.0 / sigma2
    ratio = float(special.i1(kappa) / special.i0(kappa))
    variance = max(0.0, -2.0 * math.log(max(ratio, 1e-12)))
    return math.degrees(math.sqrt(variance))


def plot_series(axis, theta_deg, labels_and_sigma2, colors):
    ymax = 0.0
    for (_, sigma2), color in zip(labels_and_sigma2, colors):
        density = orientation_density_from_sigma2(theta_deg, sigma2)
        ymax = max(ymax, float(density.max()))
        axis.plot(
            theta_deg,
            density,
            color=color,
            linewidth=2.3,
        )

    axis.set_xlim(-60, 60)
    axis.set_xticks([-60, -30, 0, 30, 60])
    axis.set_xticklabels(["-60°", "-30°", "0°", "30°", "60°"], fontsize=TICK_LABEL_FONTSIZE)
    axis.set_xlabel("Orientation error (degrees)", fontsize=AXIS_LABEL_FONTSIZE)
    axis.set_ylabel("Density", fontsize=AXIS_LABEL_FONTSIZE)
    axis.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    axis.set_ylim(0, ymax * 1.08)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def synthetic_levels():
    sigma_logit = extract_sigma_logit(SYNTHETIC_LOG)
    return [
        (str(level), sigma2_from_logit(sigma_logit[level], scale=4.0))
        for level in [2, 3, 4, 5]
    ]


def main():
    theta_deg = np.linspace(-60, 60, 4001)
    figure, axis = plt.subplots(1, 1, figsize=(5.8, 3.6), constrained_layout=True)

    synthetic = synthetic_levels()
    plot_series(
        axis,
        theta_deg,
        synthetic,
        [LEVEL_COLORS[level] for level in [2, 3, 4, 5]],
    )

    figure.savefig(OUTPUT_PATH, bbox_inches="tight")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
