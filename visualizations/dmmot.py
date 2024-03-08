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


def discrete_mmot_converge(A, first_fixed=False, niters=100, lr=0.1, print_rate=100, verbose=False, log=False, sched_drop_time=250):

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

        if i > sched_drop_time:
            lr = lr*0.995
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
  
    if log:
        log_dict = {}
        log_dict['A'] = renormalize(A)
        log_dict['func_val'] = funcval
        log_dict['grad_norm'] = gn
        logs.append(log_dict)

        return A, logs
    else:
        return A 

def list_discrete_mmot(aa):

    def OBJ(i):
        return max(i) - min(i)
        # return 0 if max(i) == min(i) else 1

    sum_aa = [sum(_) for _ in aa]
    #assert abs(max(sum_aa)-min(sum_aa)) < 1e-10
    AA = [np.copy(_) for _ in aa]

    dims = tuple([len(_) for _ in AA])
    xx = {}
    dual = [np.zeros(d) for d in dims]

    idx = [0,]*len(AA)
    obj = 0

    while all([i < _ for _, i in zip(dims, idx)]):
        vals = [v[i] for v, i in zip(AA, idx)]
        minval = min(vals)
        i = vals.index(minval)
        xx[tuple(idx)] = minval
        obj += (OBJ(idx)) * minval
        for v, j in zip(AA, idx): v[j] -= minval
        oldidx = np.copy(idx)
        idx[i] += 1
        if idx[i]<dims[i]:
            dual[i][idx[i]] += OBJ(idx) - OBJ(oldidx) + dual[i][idx[i]-1]

    # the above terminates when any entry in idx equals the corresponding value in dims
    # this leaves other dimensions incomplete; the remaining terms of the dual solution 
    # must be filled-in
    for _, i in enumerate(idx):
        try: dual[_][i:] = dual[_][i]
        except: pass

    dualobj =  sum([np.array(_).dot(np.array(_d)) for _, _d in zip(aa, dual)])
    
    return {'x': xx, 'primal objective': obj,
            'dual': dual, 'dual objective': dualobj}

# For differently sized inputs we can't use a single np array
# this function is a list-based version of the above, which 
# calls list_discrete_mmot so varying length inputs can be used
def list_discrete_mmot_converge(aa, niters=100, lr=0.1, print_rate=100, verbose=False, log=False, sched_drop_time=250):

    def dualIter(aa, lr):
        log_dict = list_discrete_mmot(aa)
        aa_new = [np.copy(_) for _ in aa]
        grad = log_dict['dual']
        for i in range(len(aa)):
            aa_new[i] -= aa[i] * (grad[i]/len(aa[i])) * lr
        return log_dict['primal objective'], aa_new, grad, log_dict

    def renormalize(aa):
        for i in range(len(aa)):
            if min(aa[i]) < 0:
                aa[i] -= min(aa[i])
            s = sum(aa[i])
            aa[i] = [aa[i][j]/s for j in range(len(aa[i]))]
        return aa

    funcval, _, grad, _ = dualIter(aa, lr)

    if log:
        logs = []
        log_dict = {}
        log_dict['A'] = renormalize(aa)
        log_dict['func_val'] = funcval
        logs.append(log_dict)

    if verbose:
        print(f'Inital:\t\tObj:\t{funcval:.4f}')

    for i in range(niters):

        if i > sched_drop_time:
            lr = lr*0.995
        funcval, aa, grad, _ = dualIter(aa, lr)

        aa = renormalize(aa)

        if i % print_rate == 0:
            if verbose:
                print(f'Iter {i:2.0f}:\tObj:\t{funcval:.4f}')
            if log:
                log_dict = {}
                log_dict['A'] = renormalize(aa.copy())
                log_dict['func_val'] = funcval
                logs.append(log_dict)

    if log:
        log_dict = {}
        log_dict['A'] = renormalize(aa)
        log_dict['func_val'] = funcval
        logs.append(log_dict)

        return aa, logs
    else:
        return aa


if __name__ == "__main__":

    a1 = [0.25,0.3,0.3,0.1,0.05]
    a2 = [0.1,0.1,0.2,0.4,0.2]

    A = np.vstack((a1,a2))

    funcval, log_dict = discrete_mmot(A, verbose=True, log=True)


    a1 = [0.25,0.3,0.3,0.1,0.05]
    a2 = [0.1,0.2,0.5,0.2]

    A = [a1,a2]

    # log_dict = list_discrete_mmot(A)
    # print(log_dict)

    A, logs = list_discrete_mmot_converge(A, niters=1000, verbose=True, log=True)
    for a in A:
        print(np.round(np.array(a),2))