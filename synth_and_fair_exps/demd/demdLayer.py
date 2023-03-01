import torch
import torch.nn as nn

import random

from .demdFunc import dEMD, OBJ
from .demdLoss import dEMDLossFunc


class DEMDLayer(nn.Module):
	def __init__(self, cost=OBJ, discretization=10, order='fixed', verbose=False):
		super().__init__()

		self.cost = cost
		self.verbose = verbose
		self.discretization = discretization
		
		self.cdf = nn.Sigmoid()
		self.Hist = HistoBin(nbins=discretization)

		self.fairMeasure = dEMDLossFunc

		self.order = order

	def forward(self, acts, group_labels):
		groups = torch.unique(group_labels)
		d = len(groups)
		# first organize output into distributions.
		grouped_dists = []
		for i in range(d):
			idxs = group_labels==groups[i]
			g_dist = self.genHists(acts[idxs], nbins=self.discretization)
			grouped_dists.append(g_dist)

		if self.order == 'randomized':
			random.shuffle(grouped_dists)
		elif self.order == 'fixed':
			pass

		torch_dists = torch.stack(grouped_dists).requires_grad_(requires_grad=True)

		fairObj = self.fairMeasure(torch_dists)
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
