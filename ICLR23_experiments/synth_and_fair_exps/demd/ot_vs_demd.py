import numpy as np

import ot

from emd_utils import compare_all

def yetanother():

    print('\n')
    print('*'*10)
    print('*** 5 Fixed Dists with 4 Bins ***')
    #######
    n = 4  # nb bins
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')

    # "interior"
    #a1 = np.array([0.1, 0.6, 0.2, 0.1])
    #a2 = np.array([0.1, 0.6, 0.2, 0.1])
    a1 = np.array([0.15, 0.55, 0.2, 0.1]) 
    a2 = np.array([0.05, 0.6, 0.25, 0.1])
    a3 = np.array([0.1, 0.6, 0.2, 0.1])

    # "hull"
    a4 = np.array([0.4, 0.3, 0.2, 0.1])
    a5 = np.array([0.1, 0.1, 0.2, 0.6])

    data = [a1, a2, a3, a4, a5]
    #data = [a3, a4, a5]
    d = len(data)
    print(data)

    compare_all(data, M, n, d)
    print('*'*10)

    # "interior"
    a1 = np.array([0.1, 0.2, 0.55, 0.15])
    a2 = np.array([0.1, 0.25, 0.6, 0.05])
    a3 = np.array([0.1, 0.2, 0.6, 0.1])

    a4 = np.array([0.1, 0.2, 0.3, 0.4])
    a5 = np.array([0.6, 0.2, 0.1, 0.1])

    data = [a1, a2, a3, a4, a5]
    #data = [a3, a4, a5]
    d = len(data)
    print(data)

    compare_all(data, M, n, d)
    print('*'*10)

    print('\n')
    #######



def known2d_simple():

    ####### Two known dists
    print('\n')
    print('*'*10)
    print('*** 2 Fixed Dists with 4 Bins ***')
    #######
    n = 2  # nb bins
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')

    a1 = np.array([0.2, 0.8])
    a2 = np.array([0.4, 0.6])
    a3 = np.array([0.2, 0.8])
    a4 = np.array([0.4, 0.6])
    a5 = np.array([0.2, 0.8])
    a6 = np.array([0.4, 0.6])
    a7 = np.array([0.3, 0.7])

    data = [a1, a2, a3, a4, a5, a6, a7]
    d = len(data)
    print(data)

    compare_all(data, M, n, d)
    print('*'*10)
    print('\n')
    #######

    return

def known3d_simple():

    ####### Three known dists
    print('\n')
    print('*'*10)
    print('*** 3-4 Fixed Dists with 3 Bins ***')
    #######

    a1 = np.array([1.0, 0, 0])
    a2 = np.array([0, 1.0, 0])
    a3 = np.array([0, 0, 1.0])

    data = [a1, a2, a3]
    d = len(data) # n samples/dimensions

    print(data)

    n = 3  # nb bins
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')

    compare_all(data, M, n, d)
    print('*'*10)
 
    a4 = np.array([0.25, 0.5, 0.25])

    data = [a1, a2, a3, a4]
    d = len(data) # n samples/dimensions

    print(data)

    n = 3  # nb bins
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')

    compare_all(data, M, n, d)
    print('*'*10)
    print('\n')
    #######

    return

def known2d():

    ####### Two known dists
    print('\n')
    print('*'*10)
    print('*** 2 Fixed Dists with 2 Bins ***')
    #######
    d = 2 # n samples/dimensions
    n = 4  # nb bins
    a1 = np.array([0.25, 0.25, 0.25, 0.25])
    a2 = np.array([0.05, 0.25, 0.25, 0.45])
    print(a1)
    print(a2)
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')
    #M = np.array([[0,1,1,1.0],[1,0,1,1],[1,1,0,1],[1,1,1,0]])
    compare_all([a1, a2], M, n, d)
    print('*'*10)
    print('\n')
    #######

    return

def increasing_bins():

    ns = [5, 10, 20, 50, 100]
    for n in ns:
        random2d(n=n)

    return

def random2d(n=4):

    ####### Two random dists
    print('\n')
    print('*'*10)
    print('*** 2 Random Dists with 4 Bins ***')
    #######
    d = 2 # n samples/dimensions
    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
    print(a1)
    print(a2)
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')
    print(M)
    compare_all([a1, a2], M, n, d)
    print('*'*10)
    print('\n')
    #######

    return

def random4d():

    ####### Four random dists, 10 bins
    print('\n')
    print('*'*10)
    print('*** 4 Random Dists with 10 Bins ***')
    #######

    n = 10 # nb bins
    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')

    # Gaussian distributions
    a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
    a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)
    a3 = ot.datasets.make_1D_gauss(n, m=40, s=3)
    a4 = ot.datasets.make_1D_gauss(n, m=20, s=25)

    data = [a1, a2, a3, a4]
    d = len(data)

    print(data)

    compare_all(data, M, n, d)
    print('*'*10)
    print('\n')
    #######

    return

def random(n, d):

    ####### Four random dists, 10 bins
    print('\n')
    print('*'*10)
    print(f'*** {d} Random Dists with {n} Bins ***')
    #######

    x = np.arange(n, dtype=np.float64).reshape((n, 1))
    M = ot.utils.dist(x, metric='minkowski')

    # Gaussian distributions
    data = []
    for i in range(d):
        m = 100*np.random.rand(1)
        a = ot.datasets.make_1D_gauss(n, m=m, s=5)
        data.append(a)

    d = len(data)

    print(data)

    compare_all(data, M, n, d)
    print('*'*10)
    print('\n')
    #######

    return



if __name__ == "__main__":

    np.random.seed(0)

    #yetanother()
    #known2d_simple()
    #known3d_simple()
    #known2d()
    #random2d()
    #random4d()
    random(4, 5)


    #increasing_bins()



