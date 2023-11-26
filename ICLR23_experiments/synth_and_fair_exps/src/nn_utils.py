import torch
import numpy as np
import random
from tqdm import tqdm

from src.utils import genClassificationReport, genRegressionReport

import ot

def do_reg_epoch(model, dataloader, criterion, reg, dist, 
					epoch, nepochs, lambda_reg, nbins, regType,
					threshold=0.5, problemType='class',
					optim=None, device='cpu', outString=''):
	# saves last two epochs gradients for computing finite difference Hessian
	total_loss = 0
	total_accuracy = 0
	nsamps = 0
	if optim is not None:
		model.train()
		reg.train()
	else:
		model.eval()
		reg.eval()

	acts = []
	targets = []
	attrs = []

	for x, target in tqdm(dataloader):

		(y_true, attr) = target[:, 0], target[:, 1].int().to(device)
		x, y_true = x.to(device), y_true.to(device).float()#.unsqueeze(1)

		act = model(x).squeeze()
		if problemType=='class':
			y_sig = torch.sigmoid(act)
		elif problemType=='regress':
			y_sig = act
		recon_loss = criterion(y_sig, y_true)

		# import pdb; pdb.set_trace()
		if regType == 'none':
			reg_loss = 0
		elif regType == 'demd':
			reg_loss = reg(act, attr)
		elif regType == 'wasbary':
			reg_loss, bary_est = reg(act, attr)
		else:
			# sens = attr.cpu().detach().numpy().astype(int)
			reg_loss = reg(X=None, y=y_true.int(), out=act, sensitive=attr)

		loss = recon_loss + lambda_reg*reg_loss

		# for training
		if optim is not None:
			optim.zero_grad()
			loss.backward()
			optim.step()

		if regType == 'wasbary:':
		# performs a step of projected gradient descent
		    with torch.no_grad():
		        grad = bary_est.grad
		        bary_est -= bary_est.grad * 0.001  # step
		        bary_est.grad.zero_()
		        bary_est.data = ot.proj_simplex(bary_est)  # projection onto the simplex

		nsamps += len(y_true)
		total_loss += loss.item()

		total_accuracy += ((y_sig>threshold) == y_true).float().mean().item()

		acts.extend(act)
		targets.extend(y_true)
		attrs.extend(attr)

	mean_loss = total_loss / len(dataloader)
	mean_accuracy = total_accuracy / len(dataloader)

	tacts = torch.stack(acts)
	ttargets = torch.stack(targets)
	tattrs = torch.stack(attrs)

	if optim is None:
		verbose = True
	else:
		verbose = False

	vals = {}
	if problemType == 'class':
		gtdists, accs, dp, eo, valid_dist = genClassificationReport(tacts.cpu(), ttargets.cpu(), tattrs.cpu(),
							dist=dist, nbins=nbins, threshold=threshold,verbose=verbose)
		
		vals['gt_0'] = gtdists[0]
		vals['gt_1'] = gtdists[1]
		vals['maxacc'] = max(accs.values()).item()
		vals['minacc'] = min(accs.values()).item()
		vals['dp_gap'] = (max(dp.values()) - min(dp.values())).item()
		vals['eo_gap'] = (max(eo.values()) - min(eo.values())).item()

	elif problemType == 'regress':
		gtdists, mse, ks, mses, valid_dist = genRegressionReport(tacts.cpu(), ttargets.cpu(), tattrs.cpu(),
		dist=dist, nbins=nbins, verbose=verbose)

		vals['gt_0'] = gtdists[0]
		vals['gt_1'] = gtdists[1]
		vals['maxmse'] = max(mses.values()).item()
		vals['minmse'] = min(mses.values()).item()
		vals['mse'] = (mse).item()
		vals['ks'] = ks

	return mean_loss, mean_accuracy, valid_dist, vals

