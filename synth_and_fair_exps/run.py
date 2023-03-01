import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

import torch
import torchvision

from src.nn_utils import do_reg_epoch

from src.dataset_helper import getDatasets
from utils import manual_seed

from src.measures.fairtorch_constraints import DemographicParityLoss, EqualiedOddsLoss
from src.measures.barywas import WassersteinBarycenter
from demd import DEMDLayer, dEMD

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):

	run_dict = vars(args)
	run_dict['start_time'] = time.time()

	manual_seed(args.train_seed)
	outString = 'trained_models/'+args.dataset+"_"+args.model+'_epochs_' + str(args.epochs)+'_lr_' + str(args.learning_rate)+'_' + args.regType +'_lamb_' + str(args.lambda_reg)+'_optim_' + str(args.optim)
	print(outString)
	
	exec("from src.models import %s" % args.model)
	model = eval(args.model)(input_size=args.input_size, num_classes=args.n_classes).to(device)

	if args.problemType == 'class':
		criterion = torch.nn.BCELoss().to(device)
	elif args.problemType == 'regress':
		criterion = torch.nn.MSELoss().to(device)
	sens_classes = [*range(0,args.nSens)]

	if args.regType == 'demd':
		reg = DEMDLayer(discretization=args.nbins).to(device)
	elif args.regType == 'wasbary':
		reg = WassersteinBarycenter(discretization=args.nbins,device=device).to(device)
	elif args.regType == 'dp':
		reg = DemographicParityLoss(sensitive_classes=sens_classes, alpha=1.0).to(device)
	elif args.regType == 'eo':
		reg = EqualiedOddsLoss(sensitive_classes=sens_classes, alpha=1.0).to(device)
	elif args.regType == 'none':
		reg = torch.nn.Identity()

	dist = dEMD()

	print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

	optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

	train_dataset, valid_dataset = getDatasets(args.dataset,
												download=True, seed=args.train_seed)

	print(f'Train Dataset Size: {len(train_dataset)}')
	print(f'Valid Dataset Size: {len(valid_dataset)}')

	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
											  shuffle=True, num_workers=1, drop_last=args.droplast)

	valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
											 shuffle=False, num_workers=1)

	# import pdb; pdb.set_trace()

	tottraintime = 0
	for epoch in range(args.epochs):
		print(f'*** EPOCH {epoch} ***')
		tic = time.time()
		train_loss, train_accuracy, train_dist, _ = do_reg_epoch(model, train_loader, criterion, reg, dist, epoch, args.epochs, args.lambda_reg, args.nbins, problemType=args.problemType, regType=args.regType, optim=optim, device=device, outString=outString)

		tottraintime += (time.time() - tic)
		with torch.no_grad():
			valid_loss, valid_accuracy, valid_dist, valstats = do_reg_epoch(model, valid_loader, criterion, reg, dist, epoch, args.epochs, args.lambda_reg, args.nbins, problemType=args.problemType, regType=args.regType, optim=None, device=device, outString=outString)

		tqdm.write(f'{args.model} EPOCH {epoch:03d}: train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.4f} '
				   f'valid_loss={valid_loss:.4f}, valid_accuracy={valid_accuracy:.4f}, train_demd_dist={train_dist:f}')

		print('Saving model...')
		torch.save(model.state_dict(), outString + '.pt')

	run_dict.update(valstats)
	run_dict['avg_train_epoch_time'] = tottraintime / args.epochs
	run_dict['final_train_acc'] = train_accuracy
	run_dict['final_train_loss'] = train_loss
	run_dict['final_train_dist'] = train_dist
	run_dict['final_valid_acc'] = valid_accuracy
	run_dict['final_valid_loss'] = valid_loss
	run_dict['final_valid_dist'] = valid_dist

	df = pd.DataFrame.from_records([run_dict])
	if os.path.isfile(args.outfile):
		df.to_csv(args.outfile, mode='a', header=False, index=False)
	else:
		df.to_csv(args.outfile, mode='a', header=True, index=False)
	

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser(description='D-EMD NNs')
	arg_parser.add_argument('--train_seed', type=int, default=0)
	arg_parser.add_argument('--dataset', type=str, default='acs-employ')
	arg_parser.add_argument('--model', type=str, default='ACSNet')
	arg_parser.add_argument('--problemType', type=str, default='class', choices=['class', 'regress'])
	arg_parser.add_argument('--batch_size', type=int, default=128)
	arg_parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'adam'])
	arg_parser.add_argument('--epochs', type=int, default=1)
	arg_parser.add_argument('--n_classes', type=int, default=1)
	arg_parser.add_argument('--input_size', type=int, default=1)
	arg_parser.add_argument('--learning_rate', type=float, default=0.001)
	arg_parser.add_argument('--momentum', type=float, default=0.9)
	arg_parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay, or l2_regularization for SGD')
	arg_parser.add_argument('--regType', type=str, default='demd', choices=['none', 'demd', 'dp', 'eo', 'wasbary'], help='none, demd, wasbary, dp, or eo')
	arg_parser.add_argument('--lambda_reg', type=float, default=1e-5, help='dEMD reg weight')
	arg_parser.add_argument('--nbins', type=int, default=10, help='number of bins for histogram')
	arg_parser.add_argument('--nSens', type=int, default=10, help='number of sensitive classes')
	arg_parser.add_argument('--droplast', type=bool, default=False, help='drop last batch in training dloader')
	arg_parser.add_argument('--outfile', type=str, default='results/tmp_resnew.csv', help='results file to print to')
	args = arg_parser.parse_args()
	# arg_parser.print_help()
	main(args)
