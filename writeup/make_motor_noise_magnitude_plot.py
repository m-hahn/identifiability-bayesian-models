import ast
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OTHER_REPO_ROOT = REPO_ROOT.parent / "unifying-theory-biases"
OUTPUT_PATH = HERE / "motor_noise_magnitudes.pdf"
FIGURES_OUTPUT_PATH = HERE / "figures" / "motor_noise_magnitudes.pdf"

ORIENTATION_USUAL_LOG = (
    REPO_ROOT
    / "code"
    / "Synthetic"
    / "logs"
    / "SIMULATED_REPLICATE"
    / "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize.py_180_2_2345_N10000_STEEPPERIODIC_STEEPPERIODIC.txt"
)
ORIENTATION_MEDIUM_LARGE_SCRIPT = (
    REPO_ROOT
    / "code"
    / "Synthetic"
    / "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_MediumLargeMotorNoise.py"
)
ORIENTATION_LARGE_SCRIPT = (
    REPO_ROOT
    / "code"
    / "Synthetic"
    / "SimulateSynthetic_Parameterized_OtherNoiseLevels_Grid_VarySize_LargeMotorNoise.py"
)
COLORS = {
    "Orientation (de Gardelle et al)": "#1b9e77",
    "Medium large motor noise": "#d95f02",
    "Large motor noise": "#7570b3"
}

XTICKS = [-30, -15, 0, 15, 30]
XTICKLABELS = ["-30°", "-15°", "0°", "15°", "30°"]
AXIS_LABEL_FONTSIZE = 18
TICK_LABEL_FONTSIZE = 15
LEGEND_FONTSIZE = 14
LINEWIDTH = 2.8


def read_text(path):
    return path.read_text()


def extract_singleton_log_motor_var(path):
    text = read_text(path)
    match = re.search(r"log_motor_var\s*\t\s*(\[[^\]]+\])", text)
    assert match is not None, path
    values = ast.literal_eval(match.group(1))
    assert len(values) == 1, (path, values)
    return float(values[0])


def extract_script_log_motor_var(path):
    text = read_text(path)
    match = re.search(
        r'parameters\["log_motor_var"\]\s*=\s*MakeFloatTensor\(\[([^\]]+)\]\)',
        text,
    )
    assert match is not None, path
    return float(match.group(1).strip())


def extract_vector_log_motor_var(path):
    text = read_text(path)
    match = re.search(r"log_motor_var\s*\t\s*(\[[^\]]+\])", text)
    assert match is not None, path
    values = ast.literal_eval(match.group(1))
    return [float(x) for x in values]


def circular_density(theta_deg, log_motor_var):
    kappa = math.exp(log_motor_var)
    theta_rad = np.deg2rad(theta_deg)
    return scipy.stats.vonmises.pdf(theta_rad, kappa)


def rescaled_orientation_density(theta_deg, log_motor_var):
    theta_deg = np.asarray(theta_deg)
    internal_theta_deg = 2.0 * theta_deg
    return 2.0 * circular_density(internal_theta_deg, log_motor_var)


def average_direction_density(theta_deg, log_motor_vars):
    densities = [circular_density(theta_deg, value) for value in log_motor_vars]
    return np.mean(densities, axis=0)


def style_axis(axis):
    axis.set_xlim(-60, 60)
    axis.set_xlim(-30, 30)
    axis.set_xticks(XTICKS)
    axis.set_xticklabels(XTICKLABELS, fontsize=TICK_LABEL_FONTSIZE)
    axis.tick_params(axis="y", labelsize=TICK_LABEL_FONTSIZE)
    axis.set_ylabel("Density", fontsize=AXIS_LABEL_FONTSIZE)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


if __name__ == "__main__":
    modeling_log_motor_vars = {
        "Orientation (de Gardelle et al)": extract_singleton_log_motor_var(ORIENTATION_USUAL_LOG),
        "Medium large motor noise": extract_script_log_motor_var(ORIENTATION_MEDIUM_LARGE_SCRIPT),
        "Large motor noise": extract_script_log_motor_var(ORIENTATION_LARGE_SCRIPT),
    }

    theta_deg = np.linspace(-180, 180, 4001)

    figure, axis = plt.subplots(1, 1, figsize=(11.5, 4.2))

    for label, log_motor_var in modeling_log_motor_vars.items():
        axis.plot(
            theta_deg,
            rescaled_orientation_density(theta_deg, log_motor_var),
            linewidth=LINEWIDTH,
            color=COLORS[label],
            label=label,
        )
    style_axis(axis)
    axis.legend(
        frameon=False,
        ncol=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=LEGEND_FONTSIZE,
    )
    axis.set_xlabel("Orientation error (degrees)", fontsize=AXIS_LABEL_FONTSIZE)



    figure.tight_layout(rect=(0.0, 0.0, 0.86, 1.0))
    figure.savefig(OUTPUT_PATH)
    figure.savefig(FIGURES_OUTPUT_PATH)
    print(OUTPUT_PATH)
