import numpy as np
import ot
import matplotlib.pyplot as plt

import seaborn as sns

np.random.seed(3450)
n = 1000
n_bins = 100
d = n_bins


m_1 = 0.2
m_2 = 0.7
s_1 = 0.3
s_2 = 0.1

ot1 = ot.datasets.make_1D_gauss(n_bins, m_1*n_bins, s_1*n_bins)
ot2 = ot.datasets.make_1D_gauss(n_bins, m_2*n_bins, s_2*n_bins)

# combine them as a single dist by just averaging
A = (ot1 + ot2)/2 + (np.random.rand(n_bins)-1)/200
A = A - min(A)
A = A/sum(A)
S = np.zeros(n_bins)
S[86] = max(A)

myBlue = '#005baa'
myYellow = '#f0e442'

bin_boundaries = range(d+1)

plt.rcParams["axes.grid"] = False
plt.rcParams["axes.linewidth"]  = 0.0

fig = plt.figure(figsize=(9,3), dpi=300, frameon=False)
plt.tight_layout(pad=0.0)
ax = fig.add_axes([0, 0, 1, 1])

# clear everything except plot area
ax.axis('off')
for item in [fig, ax]:
    item.patch.set_visible(False)
    item.patch.set_linewidth(0.0)
ax.set_axis_off()
ax.set_frame_on(False)
ax.minorticks_off()
plt.tick_params(
    which='both',      # both major and minor ticks are affected
    right=False,
    left=False,      # ticks along the bottom edge are off
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

sns.histplot(x=range(d),
                    weights=A, 
                    stat='count',
                    bins=bin_boundaries,
                    color=myBlue,
                    alpha=1,
                    ax=ax,
                    legend=False
                    )

sns.histplot(x=range(d),
                    weights=S, 
                    stat='count',
                    bins=bin_boundaries,
                    color=myYellow,
                    alpha=1,
                    ax=ax,
                    legend=False
                    )
ax.set_xlim(0, d)

plt.savefig('figs/null_dist.png', bbox_inches='tight', pad_inches=0.0)