'''
This is the first file where the program control lands. 
The necessary parameters and hyper-parameters are defined. 

Training our proposed model requires one flag
    -flag_train_equivar

For evaluation, two flags are provided, one for each metric
    -flag_test_adv #For $\mathcal{adv}$ metric
    -flag_test_mmd #For $\mathcal{M}$ metric

'''
import argparse
from src import dataloader as mydatasets
import os, sys, time, shutil
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
import datetime as dt
import random
from acs import ACSEmployment, ACSIncome
from crimes import CommunitiesCrime

from src.train_equivariance import run_equivariance
from src.test_adversaries import run_adversaries
from src.test_mmd_measure import run_mmd_measure

from src.average_meter import AverageMeter
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
sys.path.append('../..')

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', type=str, default='BaselineComparison')
parser.add_argument('--dataset_name', type=str, default='German')
parser.add_argument('--model_name', type=str, default='FC')
parser.add_argument('--result_path', type=str, default='result', help='output path')
parser.add_argument('--data_path', type=str, default='data/', help='path for data')
parser.add_argument("--fold", default=0, type=int)
parser.add_argument('--flag_debug', default=False, action='store_true', help='debug flag')
parser.add_argument('--user_output_path', type=str, default='/path/to/trained/model', help='path for data')

# Parameters for generic encoder-decoder training 
parser.add_argument('--num_epochs', type=int, default=201)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed to use')
parser.add_argument('--log_step', type=int, default=100, help='step size for prining log info')
parser.add_argument('--save_step', type=int, default=100, help='step size for saving trained models')
parser.add_argument('--flag_retrain', default=False, action='store_true', help='Re train')
parser.add_argument('--flag_reg', default=False, action='store_true', help='Regularizer')
parser.add_argument('--flag_plot', default=False, action='store_true', help='Plot')

# Parameters for ResNet architectures if used 
parser.add_argument("--blocks", default=[2, 2, 2, 2], type=int, nargs='+')
parser.add_argument("--channels", default=[8, 8, 16, 32], type=int, nargs='+')


# Parameters for the adversarial testing metric
parser.add_argument('--num_adv', type=int, default=2)
parser.add_argument('--adv_lr', type=float, default=0.1, help='lr for the adversaries')

parser.add_argument('--adv_hidden_dim', type=int, default=64, help='hidden layers dim in adversaries')
parser.add_argument('--adv_batch_size', type=int, default=128)
parser.add_argument('--adv_num_epochs', type=int, default=250)
parser.add_argument('--adv_log_step', type=int, default=100)
parser.add_argument('--adv_use_weighted_loss', default=False, action='store_true')

# Other params
parser.add_argument('--latent_dim', type=int, default=30)
parser.add_argument('--comp_lambda', type=float, default=1e-2)
parser.add_argument('--beta', type=float, default=1e-2)
parser.add_argument('--use_bottleneck_layers', default=False, action="store_true")

# Parameters for the loss functions in baselines
parser.add_argument('--alpha', type=float, default=0.1, help='Additional weight on the compression')
parser.add_argument('--alpha_max', type=float, default=10.0, help='Max value of the regularizer') 
parser.add_argument('--alpha_gamma', type=float, default=1.5, help='Multiplicative factor for alpha')
parser.add_argument('--recon_lambda', type=float, default=1e-2, help='Additional weight on the compression')
parser.add_argument('--equiv_lambda', type=float, default=1.0, help='Max value of the regularizer') 
parser.add_argument('--delta_lambda', type=float, default=1.0, help='Max value of the regularizer') 
parser.add_argument('--demd_nbins', type=int, default=2)
parser.add_argument('--demd_type', type=str, default='latent')

parser.add_argument('--equiv_type', type=str, default='none', help='Choose from mmd_lap, cai, none, demd')


parser.add_argument('--run_type', type=str, default=None,
                    help='To create multiple runs')
parser.add_argument('--add_prior', default=False, action='store_true', 
                    help='Add the gaussian prior term like VAE')

parser.add_argument('--use_weighted_loss', default=False, action='store_true')
parser.add_argument('--run_multi_adv', default=False, action='store_true', 
                    help='Runs multiple adversaries for this experiment.')

parser.add_argument('--mmd_lap_p', type=float, default=1.0, 
                    help='Argument for mmd laplacian')
parser.add_argument('--disc_lr', type=float, default=0.1, help='lr for the discriminator')
parser.add_argument('--adv_hidden_layers', type=int, default=3)
parser.add_argument('--gpu_ids', type=str, default=str(random.randrange(1)))

# Flags for which setting to run
parser.add_argument('--flag_train_equivar', default=False, action='store_true')
parser.add_argument('--flag_test_adv', default=False, action='store_true')
parser.add_argument('--flag_test_mmd', default=False, action='store_true')

# All model parametrs in args
args = parser.parse_args()

torch.manual_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu_ids
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


'''
Prepare the results directory to save tensorboard events file, log files and
checkpoints
'''

params = ['equiv_type', args.equiv_type,
          'lr', args.lr,
          'latent', args.latent_dim,
          'beta', args.beta,
          'comp_lambda', args.comp_lambda,
          'recon_lambda', args.recon_lambda,
          'equiv_lambda', args.equiv_lambda,
          'seed', args.seed]

params_str = '_'.join([str(x) for x in params])

if args.run_type is not None:
    params_str = args.run_type + '_' + params_str
if args.adv_use_weighted_loss:
    params_str += '_advw'

# All paths
dataset_path=os.path.join(args.data_path, args.dataset_name)
runTime = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
if args.flag_debug:
    output_path=os.path.join(args.result_path, args.experiment_name + '_' + 'debug')
elif args.equiv_type == 'demd':
    output_path=os.path.join(args.result_path, args.experiment_name + '_' + args.demd_type + '_' + str(args.equiv_lambda) + '_' + str(args.demd_nbins) )
else:
    output_path=os.path.join(args.result_path, args.experiment_name) 

if args.dataset_name == 'Adult' or args.dataset_name == 'German' or args.dataset_name == 'ACSEmploy' or args.dataset_name == 'ACSIncome' or args.dataset_name == 'Crimes':
    output_path = os.path.join(output_path, 'run_' + str(args.seed))
else:
    raise NotImplementedError

    
log_path=os.path.join(output_path, "log.txt")
model_path=os.path.join(output_path, 'snapshots')


# makedir
def make_dir(dirname, rm=False):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif rm:
        print('rm and mkdir ', dirname)
        shutil.rmtree(dirname)
        os.makedirs(dirname)

if args.flag_train_equivar:
    make_dir(args.result_path)
    make_dir(output_path)
    logf=open(log_path, 'w')
    logf.write(params_str + '\n')
    make_dir(model_path)
    writer=SummaryWriter(comment=args.experiment_name, log_dir=output_path)

'''
Create the dataloaders for the individual datasets.
Four datasets are used in the paper -- Adult, German,
ACSEmploy, ACSIncome, Crimes
'''    
#### Dataloaders
if args.dataset_name == 'Adult':
    args.data_path = os.path.join(args.data_path, 'adult_proc_gattr.z')

    trainset = mydatasets.GattrDataset(args.data_path, split='train')
    valset = mydatasets.GattrDataset(args.data_path, split='val')
    testset = mydatasets.GattrDataset(args.data_path, split='test')
elif args.dataset_name == 'German':
    args.data_path = os.path.join(args.data_path, 'german_proc_gattr.z')

    trainset = mydatasets.GattrDataset(args.data_path, split='train')
    valset = mydatasets.GattrDataset(args.data_path, split='val')
    testset = mydatasets.GattrDataset(args.data_path, split='test')
elif args.dataset_name == 'Crimes':
    args.data_path = os.path.join(args.data_path, 'Crime.mat')

    trainset = CommunitiesCrime(path=args.data_path, train=True)
    valset = CommunitiesCrime(path=args.data_path, train=False)
    testset = None
elif args.dataset_name == 'ACSIncome':
    trainset = ACSIncome(train=True)
    valset = ACSIncome(train=False)
    testset = None
else:
    raise NotImplementedError


'''
Stage one: Train for harmonized representations
'''
if args.flag_train_equivar:
    run_equivariance(args, device, model_path, logf, trainset, valset, testset, writer)

'''
Test Metric: Evaluate the MMD measure on the test set
'''
if args.flag_test_mmd:
    if args.flag_train_equivar:
        equivar_model_name = 'Equivar_best_val_acc'
        invar_model_name = ''
    else:
        equivar_model_name = 'Equivar_best_val_acc'
        invar_model_name = ''
        output_path = args.user_output_path

    log_path=os.path.join(output_path, "log.txt") 
    logf=open(log_path, 'a+')
    model_path=os.path.join(output_path, 'snapshots')
    print('Running MMD measure now')

    if testset is None:
        run_mmd_measure(args, device, valset, equivar_model_name, invar_model_name, model_path, output_path, logf)
    else:
        run_mmd_measure(args, device, testset, equivar_model_name, invar_model_name, model_path, output_path, logf)
        
    print('MMD measure done')
    
    
'''
Test Metric: Evaluate the adversarial test accuracy of predicting
the site information from the latent representations
'''
if args.flag_test_adv:
    if args.flag_train_equivar:
        equivar_model_name = 'Equivar_best_val_acc'
        invar_model_name = ''
    else:
        equivar_model_name = 'Equivar_best_val_acc'
        invar_model_name = ''
        output_path = args.user_output_path


    writer=SummaryWriter(comment='adv_'+args.experiment_name, log_dir=output_path)
    log_path=os.path.join(output_path, "log.txt") 
    logf=open(log_path, 'a+')
    model_path=os.path.join(output_path, 'snapshots')
    print('Running Adversarial training now')
    run_adversaries(args, device, model_path, logf, trainset, valset, testset, writer, equivar_model_name, invar_model_name)
    print('Adversarial training done')

