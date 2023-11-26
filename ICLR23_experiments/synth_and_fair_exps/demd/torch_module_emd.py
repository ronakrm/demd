
import torch
import numpy as np
import time

from emd_torch import dEMD, dEMDLossFunc

def minimize(func, x_0, niters=100, lr=0.1, verbose=False):

    x = x_0
    
    opt = torch.optim.SGD([x], lr=lr)

    if verbose:
        tic = time.time()
        funcval = func(x)
        print(time.time()-tic, ' seconds for forward.')

        tic = time.time()
        funcval.backward()
        print(time.time()-tic, ' seconds for gradient.')


    for i in range(niters):

        with torch.no_grad():
            denom = (torch.sum(x, 1).unsqueeze(-1))
            # print(denom)
        
        nx = x / denom
        # print(nx)
        funcval = func(nx)

        opt.zero_grad()
        funcval.backward()
        # print(x.grad)
        opt.step()

        gn = np.linalg.norm(x.grad)
        
        if i % 100 == 0:
            print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')


    with torch.no_grad():
        denom = (torch.sum(x, 1).unsqueeze(-1))
        # print(denom)
    
    nx = x / denom
    print(nx)
    return



if __name__ == "__main__":
    
    np.random.seed(0)

    print('*'*10)
    print('*** 2 Fixed Dists with 6 Bins ***')
    #######
    n = 5  # nb bins
    x = np.arange(n, dtype=np.float64).reshape((n, 1))

    a1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    a2 = np.array([0.2, 0.1, 0.2, 0.3, 0.2])
    a3 = np.array([0.1, 0.1, 0.5, 0.2, 0.1])
    data = [a1, a2, a3]
    d = len(data)
    print(data)

    np_data = np.array(data)

    torch_data = torch.from_numpy(np.array(np_data)).clone().requires_grad_(requires_grad=True)

    # func = dEMD()
    func = dEMDLossFunc
    minimize(func, torch_data, niters=1000, lr=0.001, verbose=True)

   