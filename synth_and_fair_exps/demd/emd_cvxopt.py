#!/usr/bin/env python3.7
import numpy as np
import cvxopt 
import cvxopt.solvers as cvx_solvers
import sys
import random
from .emd import OBJ, greedy_primal_dual

cvxopt.solvers = {'feastol': 1e-7, 'reltol': 1e-6, 'abstol': 1e-7, 'refinement': 2}

def cvxdual(aa):
    sum_aa = [sum(_) for _ in aa]
    assert abs(max(sum_aa)-min(sum_aa)) < 1e-10
    AA = [np.copy(_) for _ in aa]
    dims = [len(_) for _ in AA]

    # maximize sum dot(a_{i}, y_{i})
    # s.t. y_{0}[i_0] + y_{1}[i_1] + ... + y_{n-1}[i_{n-1}] \leq c_{i_0, ... i_n}
    cc = []
    for _ in AA: cc.extend(-_)
    GG = []
    hh = []
    for i in np.ndindex(*dims): 
        tmp = []
        for _i, d in zip(i, dims):
            _ = np.zeros(d)
            _[_i] = 1.
            tmp.extend(_)
        GG.append(tmp)
        # hh.append( max(i) - min(i) +0.)
        hh.append( OBJ(i) + 0.)
    # ensure the polytope is bounded
    # it only serves to get the algorithm to run.
    # Without this constraint, the program
    # raises a ValueError, deficient rank problem 
    for i in range(len(cc)):
        _ = np.zeros(len(cc))
        _[i] = -1.
        GG.append(_)
        hh.append(10)
    GG = np.array(GG)
    hh = np.array(hh)
    cc = np.array(cc)

    GG = cvxopt.matrix(GG)
    hh = cvxopt.matrix(hh)
    GG = cvxopt.sparse(GG)
    cc = cvxopt.matrix(cc)

    primalstart = {'x': cvxopt.matrix(0., (len(cc), 1)), 's': cvxopt.matrix(1e-3, (GG.size[0], 1))}
    
    ret_cvx= cvx_solvers.lp(cc, GG, hh, primalstart=primalstart )
    ret_cvx['c'] = cc
    return ret_cvx

def cvxprimal(aa, assertsum=True):
    sum_aa = [sum(_) for _ in aa]
    assert abs(max(sum_aa)-min(sum_aa)) < 1e-10
    AA = [np.copy(_) for _ in aa]
    dims = [len(_) for _ in AA]

    cc = np.zeros(dims)

    # minimize sum c_{ij} f_{ij}
    # s.t. sum_{j} f_{ij} = p_i, 0<= i < n
    #      sum_{i} f_{ij} = q_j, 0<= j < m
    #      f_{ij} >= 0
    # this is the cost matrix
    indices = np.array(range(np.prod(dims)), dtype=int)
    indices = np.reshape( indices, dims)
    for i in np.ndindex(*dims): 
        # cc[i] = (  max(i) - min(i)) ** 1.0001
        cc[i] = (  OBJ(i) ) ** 1.0000
    
    aeq = []
    beq = []
    for dim, v in enumerate(AA):
        beq.extend(v)
        _aeq = np.zeros(cc.size)
        _idx = np.swapaxes(indices, 0, dim)
        _indices = _idx[0,:].flatten()
        _aeq[_indices] = 1
        _aeq = _aeq.reshape(dims)
        for idx, v in enumerate(AA[dim]): 
            aeq.append(np.roll(_aeq, idx, axis=dim).flatten())

    aeq = np.array(aeq)
    beq = np.array(beq)
    cc = cc.flatten()
 
    GG = np.zeros(cc.size*cc.size)
    GG[::cc.size+1] = -1
    GG = cvxopt.matrix(np.reshape(GG, (cc.size, cc.size)))
    _aeq = cvxopt.matrix(aeq)
    GG = cvxopt.matrix([GG, _aeq])
    hh = cvxopt.matrix([cvxopt.matrix(np.zeros(cc.size)), cvxopt.matrix(beq)])
    bb = cvxopt.matrix(sum(sum_aa)/len(aa)*1.)
    AA_eq = cvxopt.matrix(1., (1,cc.size))

    GG = cvxopt.sparse(GG)
   
    ret_cvx= cvx_solvers.lp(cvxopt.matrix(cc), GG, hh, A=AA_eq, b=bb)
    return ret_cvx


if __name__ == '__main__':
    dims = [4,]*6
    aa = [np.random.random(d) for d in dims]
    aa = [_/sum(_) for _ in aa]
    
    # solve the primal and dual problems explicity
    emd_aa= greedy_primal_dual(aa)
    cvx_aa= cvxprimal(aa)
    dul_aa= cvxdual(aa)
    
    print('='*50)
    print('Objectives')
    print('  greedy primal obj      : %6.4f' % emd_aa['primal objective'])
    print('  greedy dual   obj      : %6.4f' % emd_aa['dual objective'])
    print('  cvxopt primal obj      : %6.4f' % cvx_aa['primal objective'])
    print('  cvxopt dual   obj      : %6.4f' % cvx_aa['dual objective'])
    print('  cvxopt dual primal obj : %6.4f' % -dul_aa['primal objective'])
    print('  cvxopt dual dual obj   : %6.4f' % -dul_aa['dual objective'])
