import numpy as np
import time
import torch
import ot

from diff_test import demd_func
from diff_test import minimize as np_demd_min
from torch_diff import torch_demd_func
from torch_diff import minimize as torch_demd_min

lr = 0.0001
niters = 2000

def run(data, torch_data):
	np_demd_min(demd_func, data, niters=niters, lr=lr)
	torch_demd_min(torch_demd_func, torch_data, niters=niters, lr=lr, verbose=False)


def random_data(n, d):
	
	# Gaussian distributions
	data = []
	torch_list = []
	for i in range(d):
		m = 100*np.random.rand(1)
		a = ot.datasets.make_1D_gauss(n, m=m, s=5)
		data.append(a)
		torch_list.append(torch.Tensor(a))

	#print(data)
	torch_data = torch.stack(torch_list).clone().requires_grad_(requires_grad=True)

	return data, torch_data


def simple_data():
	n = 5  # nb bins

	a1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
	a2 = np.array([0.2, 0.1, 0.2, 0.3, 0.2])
	a3 = np.array([0.1, 0.1, 0.5, 0.2, 0.1])
	data = [a1, a2, a3]
	d = len(data)
	print(data)

	# data = np.array(data)
	# data = vectorize(data, vecsize)

	ta1 = torch.Tensor(a1)
	ta2 = torch.Tensor(a2)
	ta3 = torch.Tensor(a3)
	torch_data = [ta1, ta2, ta3]
	torch_data = torch.stack(torch_data).clone().requires_grad_(requires_grad=True)

	return data, torch_data



if __name__ == "__main__":
	
	np.random.seed(0)
	print('*'*10)

	data, torch_data = random_data(10, 10)
	torch_demd_min(torch_demd_func, torch_data, niters=niters, lr=lr, verbose=False)
	# run(data, torch_data)

	# data, torch_data = random_data(100, 10)
	# torch_demd_min(torch_demd_func, torch_data, niters=niters, lr=lr, verbose=False)

	# data, torch_data = random_data(10, 100)
	# torch_demd_min(torch_demd_func, torch_data, niters=niters, lr=lr, verbose=False)

	# data, torch_data = random_data(100, 100)
	# torch_demd_min(torch_demd_func, torch_data, niters=niters, lr=lr, verbose=False)

 
