import numpy as np

from simple import twoddpemd

st1 = 'illicit'
st2 = 'explicit'

a1 = [ord(c)-ord('a') for c in st1]
a2 = [ord(c)-ord('a') for c in st2]

print(a1)
print(a2)

simp = twoddpemd(a1, a2)
print(simp)

a1 = [ord(c)-ord('a') for c in st1]
a2 = [ord(c)-ord('a') for c in st2]
a1.reverse()
a2.reverse()

print(a1)
print(a2)

simp = twoddpemd(a1, a2)
print(simp)


