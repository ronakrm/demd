
# For this we probably need the local rule gradient update
# that way we know exactly how much mass moves,
# and for the dirac (sample) case, we can move a single
# sample "one bin over"

import numpy as np
import ot

from dmmot import discrete_mmot_converge
from utils import univariateGiffer

import torch
from torch.utils.data import DataLoader, TensorDataset


from demdLayer import DEMDLayer

n = 100
batch_size = n
n_bins = 50

n_epochs = 200
lr = 1e-2

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

plim = 1.2*max(np.max(ot1), np.max(ot2))

g1 = np.random.choice(n_bins, size=n, p=ot1)
g2 = np.random.choice(n_bins, size=n, p=ot2)

labels_1 = np.zeros(n)
labels_2 = np.ones(n)

# collect all data for creating single torch dataset
data = np.concatenate([g1, g2])
labels = np.concatenate([labels_1, labels_2])

assumed_full_dist_size = 2*n

dataset = TensorDataset(torch.tensor(data), torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = DEMDLayer(discretization=n_bins, order='fixed', verbose=False)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

logs = []

for i in range(n_epochs):

    for batch in dataloader:
        acts, group_labels = batch
        obj = model(acts, group_labels)
        obj.backward()



    # get a batch from each group
    # we have a distribution over the set of bins, and we want to sample from this distribution
    # idx_1 = np.random.choice(n, size=batch_size, replace=False)
    # idx_2 = np.random.choice(n, size=batch_size, replace=False)
    # g1_batch = g1[idx_1]
    # g1_else = g1[np.setdiff1d(np.arange(n), idx_1)]
    # g2_batch = g2[idx_2]
    # g2_else = g2[np.setdiff1d(np.arange(n), idx_2)]

    g1_batch = np.random.choice(n_bins, size=batch_size, p=ot1)
    g2_batch = np.random.choice(n_bins, size=batch_size, p=ot2)

    A1_batch = np.histogram(g1_batch, bins=np.arange(n_bins+1)-0.5)[0]/batch_size
    A2_batch = np.histogram(g2_batch, bins=np.arange(n_bins+1)-0.5)[0]/batch_size
    
    A = np.array([A1_batch, A2_batch])# + 1/(n_bins*100) # add a small amount of mass to each bin to avoid numerical issues
    # normalize A along the groups
    # A = A / np.sum(A, axis=1)[:, np.newaxis]
    if i > 100 and i < 120:
        lr = lr*0.9
    if i > 120:
        lr = lr*0.95
    A = discrete_mmot_converge(A, niters=1, print_rate=1, verbose=False, log=False, lr=lr)

    # after the demd compute and update, we need to update the samples in our full set 
    ot1 = (batch_size/assumed_full_dist_size)*A[0] + (1 - batch_size/assumed_full_dist_size)*ot1
    ot2 = (batch_size/assumed_full_dist_size)*A[1] + (1 - batch_size/assumed_full_dist_size)*ot2

    logs.append(
        {'A': np.array(
            [
                ot1,
                ot2,
            ]
        )
        }
    )

    # print(f"Epoch {i}:\t m1: {np.mean(g1):.3f} s1: {np.std(g1):.3f} m2: {np.mean(g2):.3f} s2: {np.std(g2):.3f}")
    print(f"Epoch {i}:\t m1: {np.sum(ot1*np.arange(1,n_bins+1)):.3f} m2: {np.sum(ot2*np.arange(1,n_bins+1)):.3f}")


outFolder = './figs/'
outName = outFolder + 'sampler.gif'
print('Generating ' + outName + '...')
univariateGiffer(logs, outName, plim=plim, show_vals=False, total_time=10000)