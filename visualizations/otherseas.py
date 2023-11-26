import numpy as np
import ot

from simple import twoddpemd
from utils import giffer

np.random.seed(8950)

n = 10
#n = 50
#n = 100
show_vals = False

a1 = ot.datasets.make_1D_gauss(n, 0.2*n, 0.3*n)
#a2 = ot.datasets.make_1D_gauss(n, 0.7*n, 0.4*n)
a2 = [0]*n
a2[-1] = 1.0
#a1 = [0.50, 0.30, 0.10, 0.05, 0.05]
#a2 = [0.10, 0.15, 0.25, 0.30, 0.20]
#print(a1, a2)
plim = max(max(a1), max(a2))

emdval, resdp, _, _, logs = twoddpemd(a1, a2, verbose=False, full_log=True)

giffer(logs, f'gauss_to_dirac_{n}.gif', plim=plim, show_vals=show_vals)

# sns.displot(data=tips, kind="ecdf", x="total_bill", col="time", hue="smoker", rug=True)

