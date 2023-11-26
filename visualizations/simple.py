import numpy as np

np.set_printoptions(precision=2)

def twoddpemd(a, b, verbose=True, full_log=False):

    dpmat = np.zeros([len(a), len(b)])
    
    da = np.zeros(len(a))
    db = np.zeros(len(b))
    emd = 0

    logs = []
    iterlog = {}
    if full_log:
        iterlog['a'] = a.copy()
        iterlog['b'] = b.copy()
        iterlog['dpmat'] = dpmat.copy()
        iterlog['emd'] = emd
    logs.append(iterlog)

    i,j = 0,0

    while (i < len(a)) and (j < len(b)):
        if verbose: print(i,j,emd)
        if verbose: print(dpmat)

        v_a, v_b = a[i], b[j]
        if v_a < v_b:
            dpmat[i,j] = v_a
        else:
            dpmat[i,j] = v_b

        emd += np.abs(i - j)*dpmat[i,j]
        a[i] -= dpmat[i,j]
        b[j] -= dpmat[i,j]
        if v_a < v_b:
            i += 1
            if i < len(a):
                da[i] += np.abs(i-j)-np.abs((i-1)-j) + da[i-1]
        else:
            j += 1
            if j < len(b):
                db[j] += np.abs(i-j)-np.abs(i-(j-1)) + db[j-1]
        if full_log:
            iterlog = {}
            iterlog['a'] = a.copy()
            iterlog['b'] = b.copy()
            iterlog['dpmat'] = dpmat.copy()
            iterlog['emd'] = emd
            logs.append(iterlog)

    if i < len(a): da[i:] = da[i]
    if j < len(b): db[j:] = da[j]

    if verbose: print(dpmat)
    if full_log:
        return emd, dpmat, da, db, logs
    else:
        return emd, dpmat, da, db