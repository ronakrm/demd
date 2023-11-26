import numpy as np
import ot
from dmmot import discrete_mmot_converge

from simple import twoddpemd
from utils import univariateGiffer

np.random.seed(8950)
outFolder = './figs/'

def theEndstoMid():
    n = 50

    a1 = [1e-4 for i in range(n)]
    a1[0] = 1.0
    a1 = np.array(a1)/sum(a1)
    a2 = [1e-4 for i in range(n)]
    a2[-1] = 1.0
    a2 = np.array(a2)/sum(a2)

    a3 = [1e-4 for i in range(n)] + ot.datasets.make_1D_gauss(n, 0.5*n, 0.01*n)
    a3 = np.array(a3)/sum(a3)

    A = np.array([a1, a2, a3])

    return A, f'dirac_to_mid_{n}.gif'
 
def theEndsWaveNoise():
    n = 50

    a1 = [1e-4*np.random.rand() for i in range(n)]
    a1[0] = 1.0
    a1 = np.array(a1)/sum(a1)
    a2 = [1e-4*np.random.rand() for i in range(n)]
    a2[-1] = 1.0
    a2 = np.array(a2)/sum(a2)

    A = np.array([a1, a2])

    return A, f'delta_ends_wave{n}.gif'

def theEnds():
    n = 50

    a1 = 0.1*ot.datasets.make_1D_gauss(n, 0.5*n, n)
    #a1 = [1e-4 for i in range(n)]
    a1[0] = 1.0
    a1 = np.array(a1)/sum(a1)
    a2 = 0.1*ot.datasets.make_1D_gauss(n, 0.5*n, n)
    #a2 = [1e-4 for i in range(n)]
    a2[-1] = 1.0
    a2 = np.array(a2)/sum(a2)

    A = np.array([a1, a2])

    return A, f'delta_ends_{n}.gif'

def first_fixed():
    n = 50

    a1 = ot.datasets.make_1D_gauss(n, 0.2*n, 0.3*n)
    a2 = ot.datasets.make_1D_gauss(n, 0.7*n, 0.2*n)
    a3 = ot.datasets.make_1D_gauss(n, 0.5*n, 0.1*n) \
        + ot.datasets.make_1D_gauss(n, 0.1*n, 0.1*n) 
    a3 = a3/np.sum(a3)
    a4 = ot.datasets.make_1D_gauss(n, 0.9*n, 0.4*n)

    A = np.array([a3, a2, a1, a4])

    return A, f'first_fixed_{n}.gif'


def simple_calibration():
    n = 50

    a1 = ot.datasets.make_1D_gauss(n, 0.5*n, 0.1*n) \
        + ot.datasets.make_1D_gauss(n, 0.1*n, 0.1*n) 
    cuma2 = [(i+1)/n for i in range(n)]
    a2 = [i/sum(cuma2) for i in cuma2]

    A = np.array([a1, a2])
    return A, f'simple_calibrate_{n}.gif'

#A, name = theEnds()
#A, name = theEndsWaveNoise()
#A, name = theEndstoMid()
A, name = first_fixed()
plim = 1.2*np.max(A)
res, logs = discrete_mmot_converge(A, first_fixed=True, niters=10000, lr=1e-02, print_rate=100, verbose=False, log=True)

#univariateGiffer(logs, name, plim=plim, show_vals=False)
univariateGiffer(logs, outFolder+name, plim=plim, show_vals=False)