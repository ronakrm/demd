# emd_vanilla.py

import numpy as np
from scipy.optimize import approx_fprime

from .emd import greedy_primal_dual

def matricize(x, d, n):
	return np.reshape(x, (d, n))

def listify(x):
	tmp = []
	for i in range(x.shape[0]):
		tmp.append(x[i,:])

	return tmp

def vectorize(x, vecsize):
	return np.reshape(x, vecsize)

def approxGrad(f, x, d, n):
	grad = approx_fprime(x, f, 1e-8, d, n)
	return grad

def demd_func(x, d, n, return_dual_vars=False):
	x = matricize(x, d, n)
	x = listify(x)
	log = greedy_primal_dual(x)

	if return_dual_vars:
		dual = log['dual']
		# dualshift = []
		# for d in dual:
			# dualshift.append(d - d[-1])
		# return_dual = np.array(dualshift)
		return_dual = np.array(dual)
		# import pdb; pdb.set_trace()
		dualobj = log['dual objective']
		return log['primal objective'], return_dual, dualobj
	else:
		return log['primal objective']

def takeStep(x, grad, lr=0.1):
	xnew = x - lr*grad
	return xnew

def renormalize(x, d, n, vecsize):
	x = matricize(x, d, n)
	for i in range(x.shape[0]):
		if min(x[i,:]) < 0:
			x[i,:] -= min(x[i,:])
		x[i,:] /= np.sum(x[i,:])
	return x

def dualIter(f, x, d, n, vecsize, lr):
	funcval, grad, _ = f(x, d, n, return_dual_vars=True)
	xnew = takeStep(matricize(x, d, n), grad, lr)
	return funcval, xnew, grad

def autoIter(f, x, d, n, vecsize, lr):
	x = vectorize(x, vecsize)
	funcval = f(x, d, n, return_dual_vars=False)
	grad = approxGrad(f, x, d, n)
	xnew = takeStep(x, grad, lr)
	return funcval, xnew, grad

def minimize(f, x_0, d, n, vecsize, niters=100, lr=0.1, print_rate=100):

	x = x_0
	funcval, _, grad = dualIter(f, x, d, n, vecsize, lr)
	gn = np.linalg.norm(grad)

	print(f'Inital:\t\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

	for i in range(niters):

		x = renormalize(x, d, n, vecsize)
		# import pdb; pdb.set_trace()
		funcval, x, grad = dualIter(f, x, d, n, vecsize, lr)
		gn = np.linalg.norm(grad)
				
		if i % print_rate == 0:
			# print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm Auto:\t{gn:.4f}\tGradNorm Dual:\t{gnd:.4f}')
			print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

	x = renormalize(x, d, n, vecsize)
	return listify(matricize(x, d, n))






