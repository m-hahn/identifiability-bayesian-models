


__file__ = __file__.split("/")[-1]

import math
import json
import re
import sys
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from util import MakeFloatTensor
from util import MakeLongTensor

def symmetric_pair_prob(responses, stimulus_indices, theta_grid, h_idx,
                         circular):
    """Estimate P(\thetahat(\theta + h/2) > \thetahat(\theta − h/2)) for every \theta on the grid.

    Args
    ----
    responses : torch.Tensor  (N,)
        Observed \hatθ values (in *degrees*, not yet wrapped).
    stimulus_indices : torch.LongTensor  (N,)
        Integer indices mapping each trial's θ to the nearest grid point.
    theta_grid : torch.Tensor  (G,)
        The discretised stimulus grid in degrees.
    h_idx : int
        Half‑width in *grid steps* to use for the finite difference.
    circular : bool
        Whether the stimulus space is circular.

    Returns
    -------
    torch.Tensor  (G,)
        Empirical pairwise probabilities for every θ.
    """
    G = theta_grid.numel()
    probs = torch.full((G,), float('nan'), dtype=torch.float32, device=responses.device)

    # Pre‑compute an index list for each grid position
    bins = [[] for _ in range(G)]
    for idx, g in enumerate(stimulus_indices.tolist()):
        bins[g].append(idx)

    for g in range(G):
        g_plus = (g + h_idx) % G if circular else g + h_idx
        g_minus = (g - h_idx) % G if circular else g - h_idx
        if g_minus < 0 or g_plus >= G:
            continue  # skip out‑of‑bounds in interval case
        plus_trials = bins[g_plus]
        minus_trials = bins[g_minus]
        if len(plus_trials) == 0 or len(minus_trials) == 0:
            continue
        plus_resp = responses[plus_trials]
        minus_resp = responses[minus_trials]
        
        # Broadcast comparison: shape (A, B)
        diff_rad = (plus_resp[:, None] - minus_resp[None, :]) * math.pi/180.0
        comp = (torch.sin(diff_rad) > 0).float()
        probs[g] = comp.mean()
    
    return probs

def estimate_sqrtJ(responses, stimulus_indices, theta_grid, h_deg=4.0,
                   circular=True):
    """
    Compute an estimate of sqrt(J) for each \theta on the grid.
    """
    dtheta = theta_grid[1] - theta_grid[0]
    h_idx = max(1, int(round(h_deg / dtheta.item() / 2)))  # h/2 in grid steps

    P_hat = symmetric_pair_prob(responses, stimulus_indices, theta_grid,
                                h_idx, circular)
    valid = ~torch.isnan(P_hat)
    P_hat[~valid] = 0.5  # dummy fill (will mask later)

    sqrtJ = (P_hat - 0.5) * math.sqrt(4 * math.pi) / (h_idx * 2 * dtheta)
    return sqrtJ, valid

if __name__ == "__main__":
    GRID = int(sys.argv[1])
    h_deg = float(sys.argv[2])
    smooth_sigma = float(sys.argv[3])
    FIT = sys.argv[4]
    
    circular = True
    
    parameters = {}

    with open(f"logs/SIMULATED_REPLICATE/{FIT}", "r") as inFile:
        content = inFile.read()

        match = re.search(r"volume\s+\[([^\]]+)\]", content)
        if match:
            volume_str = match.group(1)
            volume = [float(v.strip()) for v in volume_str.split(',')]
        
        match = re.search(r"sigma_logit\s+\[([^\]]+)\]", content)
        if match:
            noise_str = match.group(1)
            noise_sigma = [float(v.strip()) for v in noise_str.split(',')]

        inFile.seek(0)
        data = [z.split(" ") for z in inFile.read().strip().split("=======\n")[1].split("\n")]

    duration__, sample__, responses__ = zip(*data)
    DURATION = int(duration__[0])
    sample = MakeFloatTensor([float(x) for x in sample__])
    response = MakeFloatTensor([float(x) for x in responses__])

    # Grid settings
    theta_grid = torch.linspace(0, 360 - 360 / GRID, GRID)
    stimulus_idx = ((sample / 360.0) * GRID).round().long() % GRID
    
    # Estimate sqrtJ
    sqrtJ, valid = estimate_sqrtJ(response, stimulus_idx, theta_grid,
                                      h_deg=h_deg, circular=circular)
    sqrtJ[~valid] = 0
    
    # Smoothing if requested
    if smooth_sigma:
        print("smoothing")
        sqrtJ_smooth = sqrtJ.cpu().numpy()
        mode = 'wrap' if circular else 'nearest'
        sqrt_smooth = gaussian_filter1d(sqrtJ_smooth, smooth_sigma, mode=mode)
        sqrtJ = torch.tensor(sqrt_smooth, device=response.device)
    
    # Recover F' using noise
    
    noise_sigma2 = 4 * torch.sigmoid(torch.tensor(noise_sigma[DURATION], device=response.device))
    F_recover = sqrtJ / sqrtJ.sum()

    volume_tensor = torch.tensor(volume)
    F_ = 2*math.pi*torch.softmax(volume_tensor, dim=0)
    sqrtJ_ = F_ / noise_sigma2.sqrt() * 180/360 # this is the \sqrt{J} for the [0,360] parameterization of the stimulus space, per the formula used in Hahn&Wei 2024.
    
    # Plot sqrtJ
    plt.figure(figsize=(8, 4))

    # Note: For plotting, there is a factor of 2 because we internally represent the space as [0,360] but plot the conventional [0,180] for orientation perception
    plt.plot(theta_grid.cpu().numpy(), 2*sqrtJ.cpu().numpy(), label="sqrt(J)") 
    plt.plot(theta_grid.cpu().numpy(), 2*sqrtJ_.cpu().numpy(), label="ground truth sqrt(J)", linestyle='--')
    plt.xlabel("theta (degrees)")
    plt.ylabel("Value")
    plt.title("Estimated and ground truth sqrt(J)")
    plt.ylim(bottom=0)
    plt.legend(loc="lower left")
    ax = plt.gca()
    ax.set_xticks(ticks=[0, 180, 360], labels=[0, 90, 180])
    ax.set_ylim(0, 0.4)
    plt.grid(True)
    plt.tight_layout()
#    plt.show()

    fname_j = f"figures/{__file__}_{FIT}_{GRID}_{h_deg}_{smooth_sigma}_sqrtJ.pdf"
    plt.savefig(fname_j)


    F_ground = F_ / F_.sum()
   
    # Plot recovered vs ground truth F'
    plt.figure(figsize=(8, 4))
    plt.plot(theta_grid.cpu().numpy(), F_recover.cpu().numpy(), label="recovered F'")
    plt.plot(theta_grid.cpu().numpy(), F_ground.cpu().numpy(), label="ground truth F'", linestyle='--')
    plt.xlabel("theta (degrees)")
    plt.ylabel("Value")
    plt.title("Estimated and ground truth F'")
    plt.ylim(bottom=0)
    plt.legend(loc="lower left")
    ax = plt.gca()
    ax.set_xticks(ticks=[0, 180, 360], labels=[0, 90, 180])
    plt.grid(True)
    plt.tight_layout()
    fname_f = f"figures/{__file__}_{FIT}_{GRID}_{h_deg}_{smooth_sigma}_Fprime.pdf"
    plt.savefig(fname_f)
    
    with open(f"logs/CROSSVALID/{__file__}_{FIT}_{GRID}_{h_deg}_{smooth_sigma}.txt", "w") as outFile:
        print(F_recover)
        print("sqrtJ", file=outFile)
        print(" ".join(f"{v:.6f}" for v in sqrtJ.tolist()), file=outFile)
        print("F_recover", file=outFile)
        print(" ".join(f"{v:.6f}" for v in F_recover.tolist()), file=outFile)
