import argparse
import numpy as np
import os
import pandas as pd

from demd.emd_vanilla import demd_func, minimize
from demd.datagen import getData

from demd.emd_utils import lp_1d_bary#, sink_1d_bary

def main(args):

	run_dict = vars(args)

	n = args.n  # nb bins
	d = args.d

	vecsize = n*d

	# data, M = getData(n, d, 'uniform')
	data, M = getData(n, d, 'skewedGauss')

	if args.baryType == 'sink':
		obj, bary = sink_1d_bary(data, M, n, d)
	elif args.baryType == 'lp':
		obj, bary = lp_1d_bary(data, M, n, d)
	elif args.baryType == 'demd':
		x = minimize(demd_func, data, d, n, vecsize,
	                 niters=args.iters, lr=args.learning_rate)
		bary = x[0]


	for j, v in enumerate(bary):
		run_dict['bin'] = j
		run_dict['val'] = v

		df = pd.DataFrame.from_records([run_dict])	
		if os.path.isfile(args.outfile):
			df.to_csv(args.outfile, mode='a', header=False, index=False)
		else:
			df.to_csv(args.outfile, mode='a', header=True, index=False)


if __name__ == "__main__":
	arg_parser = argparse.ArgumentParser(description='Distance Compute')
	arg_parser.add_argument('--seed', type=int, default=0)
	arg_parser.add_argument('-n', type=int, default=50, help='Number of bins')
	arg_parser.add_argument('-d', type=int, default=7, help='Number of dists')
	arg_parser.add_argument('--iters', type=int, default=0)
	arg_parser.add_argument('--learning_rate', type=float, default=1e-6)
	arg_parser.add_argument('--baryType', type=str, default='demd')
	arg_parser.add_argument('--outfile', type=str, default='results/hist_comp_results.csv', help='results file to print to')
	args = arg_parser.parse_args()

	np.random.seed(args.seed)
	main(args)

