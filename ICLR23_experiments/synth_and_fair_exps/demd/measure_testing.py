import numpy as np

from emd import greedy_primal_dual

def primalObjVal(aa):
    return greedy_primal_dual(aa)['primal objective']


print('moving single point')
vals = np.linspace(0, 0.5, 100)
for p in vals:
    corner = [np.array([1,0,0]),np.array([0,1,0]),np.array([p,0.5-p,0.5])]
    print(primalObjVal(corner))


print('added center')
vals = np.linspace(0, 0.66666667, 100)
for p in vals:
    corner = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1]), np.array([0.3333333,p, 0.6666667-p])]
    print(primalObjVal(corner))


print('other full')
corner = [np.array([1,0,0]),np.array([0,1,0]),np.array([0,0,1])]
print(primalObjVal(corner))

print('other tenth')
corner = [np.array([0.9,0.1,0]),np.array([0,0.9,0.1]),np.array([0.1,0,0.9])]
print(primalObjVal(corner))

print('other third')
corner = [np.array([0.6667,0.3333,0]),np.array([0,0.6667,0.3333]),np.array([0.3333,0,0.6667])]
print(primalObjVal(corner))

print('other half')
corner = [np.array([0.5,0.5,0]),np.array([0,0.5,0.5]),np.array([0.5,0,0.5])]
print(primalObjVal(corner))


print('other equal')
corner = [np.array([0.333,0.333,0.334]),np.array([0.333,0.334,0.333]),np.array([0.334,0.333,0.333])]
print(primalObjVal(corner))

print('moving single point equality')
vals = np.linspace(0, 0.666, 10)
for p in vals:
    corner = [np.array([0.334,0.333,0.333]),np.array([0.333,0.334,0.333]),np.array([p,0.666-p,0.334])]
    print(primalObjVal(corner))
