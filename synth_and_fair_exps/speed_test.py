# grad_test.py
import argparse

import pandas as pd
import time
import os

import numpy as np
import ot

from scipy.optimize import approx_fprime

import torch

from demd.emd import greedy_primal_dual
from demd.emd_vanilla import demd_func, approxGrad, matricize
from demd.demdFunc import dEMD
from demd.demdLoss import dEMDLossFunc

from utils import manual_seed

def genNumpyData(n, d):

	data = []

	# Gaussianlike distributions
	data = []
	for i in range(d):
		m = 100*np.random.rand(1)
		# a = ot.datasets.make_1D_gauss(n, m=m, s=5)
		# print(a)
		# data.append(a)
		a = np.random.rand(n)
		a = a/sum(a)
		data.append(a)

	return data

def test(n, d, seed, gradType, outfile):

	tmp = {}
	tmp['n'] = [n]
	tmp['d'] = [d]
	tmp['seed'] = [seed]
	tmp['gradType'] = [gradType]

	manual_seed(seed)

	if args.dataType == 'N52D':
		np_data = np.array([[0.1, 0.1, 0.2, 0.5, 0.1], 
							[0.2, 0.1, 0.3, 0.1, 0.3]])
	else:
		np_data = np.array(genNumpyData(n, d))


	if gradType == 'scipy' or gradType == 'npdual':
		d,n = np_data.shape
		x = np.reshape(np_data, d*n)

	elif gradType == 'torchdual' or gradType == 'autograd':
		# torch stuff
		x = torch.from_numpy(np_data).clone().requires_grad_(requires_grad=True)

	else:
		print(f'Unknown GradType: {gradType}')
		exit(1)

	t1 = time.time()

	if gradType == 'scipy':
		funcval, _, dualobj = demd_func(x, d, n, return_dual_vars=True)

		t2 = time.time()

		grad = matricize(approxGrad(demd_func, x, d, n), d, n)

	elif gradType == 'npdual':
		funcval, grad, dualobj = demd_func(x, d, n, return_dual_vars=True)

		t2 = time.time() # gradient already computed from dual

	elif gradType == 'torchdual':
		funcval = dEMDLossFunc(x)
		t2 = time.time()

		funcval.backward()

		dualobj = sum([_.dot(_d) for _, _d in zip(x, x.grad)])

		grad = x.grad

	elif gradType == 'autograd':
		func = dEMD(computeDual=True)
		funcval, dualobj = func(x)
		t2 = time.time()

		funcval.backward()

		grad = x.grad

	else:
		print(f'Unknown GradType: {gradType}')
		exit(1)

	print(grad)
	print(funcval)
	print(dualobj)

	t3 = time.time()
	fp_time = t2 - t1
	bk_time = t3 - t2
	total_time = t3 - t1

	tmp['forward_time'] = [fp_time]
	tmp['backward_time'] = [bk_time]
	tmp['total_time'] = [total_time]

	df = pd.DataFrame(tmp)
	if os.path.isfile(outfile):
		df.to_csv(outfile, mode='a', header=False, index=False)
	else:
		df.to_csv(outfile, mode='a', header=True, index=False)

	return


def main(args):
	test(args.n, args.d, args.random_seed, args.gradType, args.outfile)


if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser(description='Speed Test')

	arg_parser.add_argument('--random_seed', type=int, default=0)
	arg_parser.add_argument('--n', type=int, default=5)
	arg_parser.add_argument('--d', type=int, default=3)

	arg_parser.add_argument('--gradType', type=str, default='scipy',
						choices=['scipy', 'npdual', 'torchdual', 'autograd'])

	arg_parser.add_argument('--outfile', type=str, default='results/speed_test_results.csv')

	arg_parser.add_argument('--dataType', type=str, default='gausspseudo',
						choices=['gausspseudo', 'N52D'])

	args = arg_parser.parse_args()
	main(args)