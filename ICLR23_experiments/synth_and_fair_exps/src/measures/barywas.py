# wasserstein barycenter via POT + Torch Backend
import numpy as np
import torch
import torch.nn as nn

import ot

class OTWBLoss(nn.Module):
	def __init__(self, n, device='cpu'):
		super().__init__()

		self.n = n # problem dimension, d in sinkbary notation
		x = np.arange(n, dtype=np.float64).reshape((n, 1))
		self.M = torch.Tensor(ot.utils.dist(x, x, metric='minkowski')).to(device)
		
		tmp = np.ones([n,1])/n
		self.bary_est = torch.from_numpy(tmp).to(device=device).requires_grad_(True)

		# import pdb; pdb.set_trace()

	def forward(self, x):
		d = len(x)
		loss = torch.Tensor([0]).to(x[0].device)
		for i in range(d):
			for j in range(1):
				group_loss = ot.emd2(x[i], self.bary_est, self.M)[0]
				# group_loss = ot.sinkhorn2(x[i], self.bary_est, self.M, reg=torch.Tensor([0.01]))
				loss += group_loss

		return loss, self.bary_est

# class DebiasedSinkhornBarycenter(nn.Module):
# 	def __init__(self, n, device='cpu', eps=1e-6):

# 		self.n = n # problem dimension, d in sinkbary notation
# 		self.eps = eps
# 		x = np.arange(n, dtype=np.float64).reshape((n, 1))
# 		self.C = torch.Tensor(ot.utils.dist(x, x, metric='minkowski')).to(device)
# 		self.K = torch.exp(-1*self.C/self.eps)

# 	def forward(self, x):
		
# 		a = torch.zeros_like(x)
# 		for k in range(d):
# 			a[k,:] = x[k,:]/(self.K*b[k,:])


class POTWassersteinBary(nn.Module):
	def __init__(self, n, device='cpu'):
		super().__init__()

		self.n = n # problem dimension, d in sinkbary notation
		x = np.arange(n, dtype=np.float64).reshape((n, 1))
		self.bins = torch.tensor(x).to(device=device)
		self.C = torch.Tensor(ot.utils.dist(x, x, metric='minkowski')).to(device)

		
		tmp = np.ones([n,1])/n

		self.bary_est = torch.from_numpy(tmp).to(device=device).requires_grad_(True)

		# import pdb; pdb.set_trace()

	def forward(self, x):
		d = len(x)
		loss = torch.Tensor([0]).to(x[0].device)
		for i in range(d):
			group_loss = ot.wasserstein_1d(self.bins, self.bins, x[i], self.bary_est, p=2)
			loss += group_loss

		return loss, self.bary_est


class WassersteinBarycenter(nn.Module):
	def __init__(self, discretization=10, verbose=False, device='cpu'):
		super().__init__()
		self.verbose = verbose
		self.discretization = discretization
		
		self.cdf = nn.Sigmoid()
		self.Hist = HistoBin(nbins=discretization)

		# self.fairMeasure = OTWBLoss(n=self.discretization, device='cuda:0')
		self.fairMeasure = POTWassersteinBary(n=self.discretization, device=device)

	def forward(self, acts, group_labels):
		groups = torch.unique(group_labels)
		d = len(groups)
		# first organize output into distributions.
		grouped_dists = []
		for i in range(d):
			idxs = group_labels==groups[i]
			g_dist = self.genHists(acts[idxs], nbins=self.discretization)
			grouped_dists.append(g_dist)

		# torch_dists = torch.stack(grouped_dists).requires_grad_(requires_grad=True)

		fairObj = self.fairMeasure(grouped_dists)
		return fairObj


	def genHists(self, samples, nbins=10):
		# convert to [0,1] via sigmoid
		cdfs = self.cdf(samples) - 0.0001 # for boundary case at end
		dist = self.Hist(cdfs)
		# dist = torch.histc(cdfs, bins=nbins, min=0, max=1)
		return dist/dist.sum()

class HistoBin(nn.Module):
	def __init__(self, nbins, norm=True):
		super(HistoBin, self).__init__()
		self.locs = torch.arange(0,1,1.0/nbins)
		self.r = 1.0/nbins
		self.norm = norm
	
	def forward(self, x):
		
		counts = []
		
		for loc in self.locs:
			dist = torch.abs(x - loc)
			#print dist
			ct = torch.relu(self.r - dist).sum() 
			counts.append(ct)
		
		# out = torch.stack(counts, 1)
		out = torch.stack(counts)
		
		if self.norm:
			out = out + 0.0001
			out = out / out.sum()
		return out
