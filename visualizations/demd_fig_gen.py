import numpy as np
import ot
from dmmot import discrete_mmot_converge

from utils import univariateGiffer

import argparse

parser = argparse.ArgumentParser(description='Generate DEMD figs.')
parser.add_argument('--seed', type=int, default=8950)
parser.add_argument('-n', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=1e-1)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--print_rate', type=int, default=10)
parser.add_argument('--outFolder', type=str, default='./figs/')
parser.add_argument('--example', type=str, default='all')
parser.add_argument('--first_fixed', action='store_true', default=False)
parser.add_argument('--alpha', action='store_true', default=False)

args = parser.parse_args()
np.random.seed(args.seed)

def theEndstoMid(n, first_fixed=False):

    a1 = [1e-4 for i in range(n)]
    a1[0] = 1.0
    a1 = np.array(a1)/sum(a1)
    a2 = [1e-4 for i in range(n)]
    a2[-1] = 1.0
    a2 = np.array(a2)/sum(a2)

    a3 = [1e-4 for i in range(n)] + ot.datasets.make_1D_gauss(n, 0.5*n, 0.01*n)
    a3 = np.array(a3)/sum(a3)

    A = np.array([a1, a2, a3])

    return A, 'dirac_to_mid', first_fixed
 
def theEndsWaveNoise(n, first_fixed=False):

    a1 = [1e-4*np.random.rand() for i in range(n)]
    a1[0] = 1.0
    a1 = np.array(a1)/sum(a1)
    a2 = [1e-4*np.random.rand() for i in range(n)]
    a2[-1] = 1.0
    a2 = np.array(a2)/sum(a2)

    A = np.array([a1, a2])

    return A, 'delta_ends_wave', first_fixed

def DiracDirac(n, first_fixed=True):

    a1 = [0 for i in range(n)]
    a1[0] = 1.0
    a2 = [0 for i in range(n)]
    a2[-1] = 1.0

    A = np.array([a1, a2])

    return A, 'dirac_dirac', first_fixed

def theEnds(n, first_fixed=False):

    a1 = 0.1*ot.datasets.make_1D_gauss(n, 0.5*n, n)
    #a1 = [1e-4 for i in range(n)]
    a1[0] = 1.0
    a1 = np.array(a1)/sum(a1)
    a2 = 0.1*ot.datasets.make_1D_gauss(n, 0.5*n, n)
    #a2 = [1e-4 for i in range(n)]
    a2[-1] = 1.0
    a2 = np.array(a2)/sum(a2)

    A = np.array([a1, a2])

    return A, 'delta_ends', first_fixed

def four_dists(n, first_fixed=True):

    a1 = ot.datasets.make_1D_gauss(n, 0.2*n, 0.3*n)
    a2 = ot.datasets.make_1D_gauss(n, 0.7*n, 0.2*n)
    a3 = ot.datasets.make_1D_gauss(n, 0.5*n, 0.1*n) \
        + ot.datasets.make_1D_gauss(n, 0.1*n, 0.1*n) 
    a3 = a3/np.sum(a3)
    a4 = ot.datasets.make_1D_gauss(n, 0.9*n, 0.4*n)

    A = np.array([a3, a2, a1, a4])

    return A, 'four_dists', first_fixed


def dirac_to_multimodal(n, first_fixed=True):

    a1 = ot.datasets.make_1D_gauss(n, 0.5*n, 0.1*n) \
        + ot.datasets.make_1D_gauss(n, 0.1*n, 0.1*n) 

    a2 = [1e-4*np.random.rand() for i in range(n)]
    a2[-1] = 1.0
    a2 = np.array(a2)/sum(a2)

    A = np.array([a1, a2])
    return A, 'dirac_to_multimodal', first_fixed


def simple_calibration_centered(n, first_fixed=False):

    a1 = ot.datasets.make_1D_gauss(n, 0.8*n, 0.1*n)
    a2 = ot.datasets.make_1D_gauss(n, 0.2*n, 0.1*n) 

    A = np.array([a1, a2])
    return A, 'simple_calibrate_symmetric', first_fixed


def simple_calibration(n, first_fixed=True):

    a1 = ot.datasets.make_1D_gauss(n, 0.5*n, 0.1*n) \
        + ot.datasets.make_1D_gauss(n, 0.1*n, 0.1*n)
    return A, 'simple_calibrate', first_fixed

def runExample(example, n, first_fixed):
    A, name, first_fixed = example(n, first_fixed=first_fixed)
    res, logs = discrete_mmot_converge(A, first_fixed=first_fixed, niters=args.niters, lr=args.learning_rate, print_rate=args.print_rate, verbose=False, log=True)
    plim = 1.2*np.max(A)
    outName = args.outFolder + name + '_ff_' + str(first_fixed) + '_n_' + str(args.n) + '_lr_' + str(args.learning_rate) + '_niters_' + str(args.niters) + '.gif'
    print('Generating ' + outName + '...')
    univariateGiffer(logs, outName, plim=plim, show_vals=False, alpha=args.alpha)

runExample(eval(args.example), n=args.n, first_fixed=args.first_fixed)