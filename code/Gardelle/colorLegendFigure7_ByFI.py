from matplotlib import rc

__file__ = __file__.split("/")[-1]
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.rcParams.update({'font.size': 60})

# Get the reversed plasma colormap
cmap = plt.get_cmap('plasma_r')

# Normalize over 0–17
norm = Normalize(vmin=0, vmax=17)

# Make a wide, short figure
fig, ax = plt.subplots(figsize=(6, 1))

# Draw the horizontal colorbar
cb = plt.colorbar(
    ScalarMappable(norm=norm, cmap=cmap),
    cax=ax,
    orientation='horizontal'
)

cb.set_ticks([0, 5, 10, 15])

# Increase tick‐label font size and label font size
cb.ax.tick_params(labelsize=30)           # tick labels
cb.set_label('Δ Total FI', fontsize=35, labelpad=10)

plt.savefig(f"figures/{__file__.split('/')[-1]}.pdf",
            format="pdf",
            bbox_inches="tight")

