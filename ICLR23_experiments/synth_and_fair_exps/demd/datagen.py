import numpy as np

import ot

def getData(n, d, dist='skewedGauss'):
    print(f'Data: {d} Random Dists with {n} Bins ***')

    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')

    data = []
    for i in range(d):
        # m = 100*np.random.rand(1)
        m = n*(0.5*np.random.rand(1))*float(np.random.randint(2)+1)
        if dist == 'skewedGauss':
        	a = ot.datasets.make_1D_gauss(n, m=m, s=5)
        elif dist == 'uniform':
        	a = np.random.rand(n)
        	a = a / sum(a)
        else:
        	print('unknown dist')
        data.append(a)

    return data, M

