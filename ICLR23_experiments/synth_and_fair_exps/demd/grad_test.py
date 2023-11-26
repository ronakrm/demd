# grad_test.py
import numpy as np
from scipy.optimize import approx_fprime

from emd import greedy_primal_dual

def matricize(x, n, d):
	return np.reshape(x, (n, d))

def listify(x):
	tmp = []
	for i in range(x.shape[0]):
		tmp.append(x[i,:])

	return tmp

def demd_func(x, d, n):
	x = matricize(x, d, n)
	x = listify(x)
	log = greedy_primal_dual(x)
	return log['primal objective']

def approxGrad(f, x, d, n):
	grad = approx_fprime(x, f, 1e-8, d, n)
	return grad

def main():

	a1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
	a2 = np.array([0.2, 0.1, 0.2, 0.3, 0.2])
	# a3 = np.array([0.1, 0.1, 0.5, 0.2, 0.1])
	# data = [a1, a2, a3]
	data = [a1, a2]
	d = len(data)
	print(data)

	npdata = np.array(data)
	d,n = npdata.shape
	x = np.reshape(npdata, d*n)

	funcval = demd_func(x, d, n)
	grad = approxGrad(demd_func, x, d, n)
	grad = matricize(grad, d, n)
	grad = listify(grad)
	
	print('scipy approx grad:')
	print(np.round(grad))

	print('dual variables:')
	print(greedy_primal_dual(data)['dual'])

	print('obj:')
	print(greedy_primal_dual(data)['primal objective'])

	return


if __name__ == "__main__":
	main()