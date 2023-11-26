# -*- coding: utf-8 -*-
"""
d-MMOT solvers for optimal transport
"""
import numpy as np

def discrete_mmot(A, verbose=False, log=False):
    def OBJ(i):
        return max(i) - min(i)

    AA = [np.copy(_) for _ in A]

    dims = tuple([len(_) for _ in AA])
    xx = {}
    dual = [np.zeros(d) for d in dims]

    idx = [0, ] * len(AA)
    obj = 0
    
    while all([i < _ for _, i in zip(dims, idx)]):
        vals = [v[i] for v, i in zip(AA, idx)]
        minval = min(vals)
        i = vals.index(minval)
        xx[tuple(idx)] = minval
        obj += (OBJ(idx)) * minval
        for v, j in zip(AA, idx):
            v[j] -= minval
        oldidx = np.copy(idx)
        idx[i] += 1
        if idx[i] < dims[i]:
            dual[i][idx[i]] += OBJ(idx) - OBJ(oldidx) + dual[i][idx[i]-1]
        if verbose:
            print('oldidx\tminval\ti\tobj\tvals')
            print(oldidx,'\t', f'{round(minval,2)}', '\t', i, '\t', f'{round(obj,2)}', '\t', vals)
            print(dual)
            print(xx)
            print(np.round(AA[0],2))
            print(np.round(AA[1],2))

    # the above terminates when any entry in idx equals the corresponding
    # value in dims this leaves other dimensions incomplete; the remaining
    # terms of the dual solution must be filled-in
    for _, i in enumerate(idx):
        try:
            dual[_][i:] = dual[_][i]
        except Exception:
            pass

    dualobj = np.sum([np.dot(arr, dual_arr) for arr, dual_arr in zip(A, dual)])

    log_dict = {'A': xx, 
           'primal objective': obj,
           'dual': dual, 
           'dual objective': dualobj}
    
    if log:
        return obj, log_dict
    else:
        return obj


def discrete_mmot_converge(A, first_fixed=False, niters=100, lr=0.1, print_rate=100, verbose=False, log=False):

    d, n = A.shape
    def dualIter(A, lr):
        funcval, log_dict = discrete_mmot(A, verbose=False, log=True)
        grad = np.array(log_dict['dual'])
        if first_fixed:
            A_new = A
            A_new[1:,:] = A[1:,:] - A[1:,:]*(grad[1:,:]/n) * lr
        else:
            A_new = A - np.multiply(A, grad)/n * lr
        return funcval, A_new, grad, log_dict

    def renormalize(A):
        for i in range(A.shape[0]):
            if min(A[i, :]) < 0:
                A[i, :] -= min(A[i, :])
            A[i, :] /= np.sum(A[i, :])
        return A

    funcval, _, grad, _ = dualIter(A, lr)
    gn = np.linalg.norm(grad)

    if log:
        logs = []
        log_dict = {}
        log_dict['A'] = renormalize(A)
        log_dict['func_val'] = funcval
        log_dict['grad_norm'] = gn
        logs.append(log_dict)

    if verbose:
        print(f'Inital:\t\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    for i in range(niters):

        funcval, A, grad, _ = dualIter(A, lr)

        gn = np.linalg.norm(grad)
        A = renormalize(A)

        if i % print_rate == 0:
            if verbose:
                print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')
            if log:
                log_dict = {}
                log_dict['A'] = renormalize(A.copy())
                log_dict['func_val'] = funcval
                log_dict['grad_norm'] = gn
                logs.append(log_dict)

    #import pdb; pdb.set_trace()
    #_, _, _, _ = dualIter(A, lr)
    log_dict = {}
    log_dict['A'] = renormalize(A)
    log_dict['func_val'] = funcval
    log_dict['grad_norm'] = gn
    logs.append(log_dict)


    if log:
        return A, logs
    else:
        return A 


if __name__ == "__main__":

    a1 = [0.25,0.3,0.3,0.1,0.05]
    a2 = [0.1,0.1,0.2,0.4,0.2]

    A = np.vstack((a1,a2))

    funcval, log_dict = discrete_mmot(A, verbose=True, log=True)
    print(a1)
    print(a2)

    print(log_dict)