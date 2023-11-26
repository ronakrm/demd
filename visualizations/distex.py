import numpy as np
import ot

from simple import twoddpemd

a1 = [0.10, 0.10, 0.25, 0.35, 0.20]
a2 = [0.50, 0.30, 0.10, 0.05, 0.05]

a1 = [0.25,0.30,0.30,0.10,0.05]
a2 = [0.10,0.10,0.20,0.40,0.20]

print(a1)
print(a2)

simp = twoddpemd(a1, a2)
print(simp)

