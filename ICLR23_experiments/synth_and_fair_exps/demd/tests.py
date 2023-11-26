# tests
import numpy as np
import torch
from torch_diff import minimize, torch_demd_func

def reverse_test():
	func = torch_demd_func

	a1 = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
	a2 = np.array([0.2, 0.1, 0.2, 0.3, 0.2])
	a3 = np.array([0.1, 0.1, 0.5, 0.2, 0.1])
	ta1 = torch.Tensor(a1)
	ta2 = torch.Tensor(a2)
	ta3 = torch.Tensor(a3)
	torch_data = [ta1, ta2, ta3]
	torch_data = torch.stack(torch_data).clone().requires_grad_(requires_grad=True)

	print(func(torch_data))
	# minimize(func, torch_data, niters=500, lr=0.001, verbose=True)

	a1q = np.array(np.flip(a1))
	a2q = np.array(np.flip(a2))
	a3q = np.array(np.flip(a3))
	qta1 = torch.Tensor(a1q)
	qta2 = torch.Tensor(a2q)
	qta3 = torch.Tensor(a3q)
	torch_data = [qta1, qta2, qta3]
	torch_data = torch.stack(torch_data).clone().requires_grad_(requires_grad=True)

	print(func(torch_data))
	# minimize(func, torch_data, niters=500, lr=0.001, verbose=True)


def shuffle_test(nshuffles=10):

	func = torch_demd_func

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

	for i in range(nshuffles):
		torch_data = torch_data[:, np.random.permutation(torch_data.shape[1])]
		print(func(torch_data))
		# minimize(func, torch_data, niters=500, lr=0.001, verbose=True)

if __name__ == '__main__':
	np.random.seed(0)
	reverse_test()
	# shuffle_test(10)