#!/usr/bin/env python3.7
import scipy as sp
import scipy.spatial as spt
import scipy.stats as sts
import numpy as np
import numpy.linalg as nl
import numpy.random as nr
import sys

from .emd_vanilla import matricize, listify, takeStep, renormalize

def img_demd_func(x, d, n, imsize=1, return_dual_vars=False):
    x = matricize(x, d, n)
    x = listify(x)
    log = greedy_primal_dual(x, imsize)

    if return_dual_vars:
        dual = log['dual']
        # dualshift = []
        # for d in dual:
            # dualshift.append(d - d[-1])
        # return_dual = np.array(dualshift)
        return_dual = np.array(dual)
        # import pdb; pdb.set_trace()
        dualobj = log['dual objective']
        return log['primal objective'], return_dual, dualobj
    else:
        return log['primal objective']


def IMGOBJ(i, imsize=1):
    # backcompute cost for 2D for all d dims
    costs = []
    for j in i:
        row = j // imsize
        col = j % imsize
        costs.append((row + col))
    return max(costs) - min(costs)


def greedy_primal_dual(aa, imsize=1, verbose=False):
    sum_aa = [sum(_) for _ in aa]
    #assert abs(max(sum_aa)-min(sum_aa)) < 1e-10
    AA = [np.copy(_) for _ in aa]

    dims = tuple([len(_) for _ in AA])
    xx = {}
    dual = [np.zeros(d) for d in dims]

    idx = [0,]*len(AA)
    obj = 0
    if verbose:
        print('i minval oldidx\t\tobj\t\tvals')
    while all([i < _ for _, i in zip(dims, idx)]):
        vals = [v[i] for v, i in zip(AA, idx)]
        minval = min(vals)
        i = vals.index(minval)
        xx[tuple(idx)] = minval
        obj += (IMGOBJ(idx, imsize)) * minval
        for v, j in zip(AA, idx): v[j] -= minval
        oldidx = np.copy(idx)
        idx[i] += 1
        if idx[i]<dims[i]:
            dual[i][idx[i]] += IMGOBJ(idx, imsize) - IMGOBJ(oldidx, imsize) + dual[i][idx[i]-1]
        if verbose:
            print(i, minval, oldidx, obj, '\t', vals)

    # the above terminates when any entry in idx equals the corresponding value in dims
    # this leaves other dimensions incomplete; the remaining terms of the dual solution 
    # must be filled-in
    for _, i in enumerate(idx):
        try: dual[_][i:] = dual[_][i]
        except: pass

    dualobj =  sum([_.dot(_d) for _, _d in zip(aa, dual)])
    
    return {'x': xx, 'primal objective': obj,
            'dual': dual, 'dual objective': dualobj}


def dualIter(f, x, d, n, vecsize, imsize, lr):
    funcval, grad, _ = f(x, d, n, imsize, return_dual_vars=True)
    xnew = takeStep(matricize(x, d, n), grad, lr)
    return funcval, xnew, grad

def img_minimize(f, x_0, d, n, vecsize, imsize, niters=100, lr=0.1, print_rate=100):

    x = x_0
    funcval, _, grad = dualIter(f, x, d, n, vecsize, imsize, lr)
    gn = np.linalg.norm(grad)

    print(f'Inital:\t\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    for i in range(niters):

        x = renormalize(x, d, n, vecsize)
        # import pdb; pdb.set_trace()
        funcval, x, grad = dualIter(f, x, d, n, vecsize, imsize, lr)
        gn = np.linalg.norm(grad)
                
        if i % print_rate == 0:
            # print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm Auto:\t{gn:.4f}\tGradNorm Dual:\t{gnd:.4f}')
            print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}\tGradNorm:\t{gn:.4f}')

    x = renormalize(x, d, n, vecsize)
    return listify(matricize(x, d, n))
