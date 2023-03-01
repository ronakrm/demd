import argparse
import numpy as np
import pandas as pd
import os

import ot

from demd.datagen import getData


def main(args):

	run_dict = vars(args)

	data, M = getData(args.n, args.d)

	exec("from demd.emd_utils import %s" % args.model)
	model = eval(args.model)

	ot.tic()
	obj, _ = model(data, M, args.n, args.d)
	time = ot.toc('')
  
	# run_dict['bary'] = bary
	run_dict['time'] = time
	run_dict['obj'] = obj
  
	df = pd.DataFrame.from_records([run_dict])
	if os.path.isfile(args.outfile):
		df.to_csv(args.outfile, mode='a', header=False, index=False)
	else:
		df.to_csv(args.outfile, mode='a', header=True, index=False)


if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser(description='Distance Compute')
	arg_parser.add_argument('--seed', type=int, default=0)
	# arg_parser.add_argument('--dataset', type=str, default='1D-dists')
	arg_parser.add_argument('-n', type=int, default=10, help='Number of bins')
	arg_parser.add_argument('-d', type=int, default=4, help='Number of dists')
	arg_parser.add_argument('--model', type=str, default='demd')
	# arg_parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
	# arg_parser.add_argument('--epochs', type=int, default=5)
	# arg_parser.add_argument('--learning_rate', type=float, default=0.1)
	arg_parser.add_argument('--outfile', type=str, default='results/tmp_1d_results.csv', help='results file to print to')
	args = arg_parser.parse_args()
	# arg_parser.print_help()

	np.random.seed(args.seed)
	main(args)


