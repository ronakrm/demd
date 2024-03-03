
# For this we probably need the local rule gradient update
# that way we know exactly how much mass moves,
# and for the dirac (sample) case, we can move a single
# sample "one bin over"

import numpy as np
import ot

from dmmot import discrete_mmot_converge
from utils import univariateGiffer

n = 1000
batch_size = 1000
n_bins = 50

n_epochs = 100

seed = 345890
np.random.seed(seed)

m_1 = 0.2
m_2 = 0.7
s_1 = 0.3
s_2 = 0.1

# symmetric
# m_1 = 0.5
# m_2 = 0.5
# s_1 = 0.1
# s_2 = 0.1

ot1 = ot.datasets.make_1D_gauss(n_bins, m_1*n_bins, s_1*n_bins)
ot2 = ot.datasets.make_1D_gauss(n_bins, m_2*n_bins, s_2*n_bins)

plim = 1.2*np.max(ot1)

g1 = np.random.choice(n_bins, size=n, p=ot1)
g2 = np.random.choice(n_bins, size=n, p=ot2)

logs = []

for i in range(n_epochs):

    # get a batch from each group
    # we have a distribution over the set of bins, and we want to sample from this distribution
    idx_1 = np.random.choice(n, size=batch_size, replace=False)
    idx_2 = np.random.choice(n, size=batch_size, replace=False)
    g1_batch = g1[idx_1]
    g2_batch = g2[idx_2]

    A1_batch = np.histogram(g1_batch, bins=np.arange(n_bins+1)-0.5)[0]/batch_size
    A2_batch = np.histogram(g2_batch, bins=np.arange(n_bins+1)-0.5)[0]/batch_size

    A = np.array([A1_batch, A2_batch]) + 1/(n_bins*100)
    # normalize A along the groups
    A = discrete_mmot_converge(A, niters=1, print_rate=1, verbose=False, log=False)

    # after the demd compute and update, we need to update the samples in our full set 
    g1[idx_1] = np.random.choice(n_bins, size=batch_size, p=A[0])
    g2[idx_2] = np.random.choice(n_bins, size=batch_size, p=A[1])

    logs.append(
        {'A': np.array(
            [
                np.histogram(g1, bins=np.arange(n_bins+1)-0.5)[0]/n,
                np.histogram(g2, bins=np.arange(n_bins+1)-0.5)[0]/n,
            ]
        )
        }
    )

    print(f'Epoch {i}:\t m1: {np.mean(g1):.3f} s1: {np.std(g1):.3f} m2: {np.mean(g2):.3f} s2: {np.std(g2):.3f}')


outFolder = './figs/'
outName = outFolder + 'sampler.gif'
print('Generating ' + outName + '...')
univariateGiffer(logs, outName, plim=plim, show_vals=False)