import os
import numpy as np
import torch.utils.data
# from torch.utils.data import Subset
from torchvision.transforms import Compose, ToTensor, Normalize
# from torchvision.transforms import Resize, RandomCrop, RandomHorizontalFlip

def getDatasets(name='adult',
					target=None,
					attr_cols='race',
					data_augment=False,
					download=True,
					seed=0):

	if name == 'adult':
	# 	from src.datasets import Adult
	# 	train_dataset = Adult(attr_col=attr_cols, train=True)
	# 	valid_dataset = Adult(attr_col=attr_cols, train=False)

		from src.datasets import localds as mydatasets
		data_path = os.path.join('./data/', 'adult_proc_gattr.z')

		train_dataset = mydatasets.GattrDataset(data_path, split='train')
		valid_dataset = mydatasets.GattrDataset(data_path, split='val')
		test_dataset = mydatasets.GattrDataset(data_path, split='test')

	elif name == 'acs-employ':
		from src.datasets import ACSEmployment
		train_dataset = ACSEmployment(train=True)
		valid_dataset = ACSEmployment(train=False)
	elif name == 'acs-income':
		from src.datasets import ACSIncome
		train_dataset = ACSIncome(train=True)
		valid_dataset = ACSIncome(train=False)
	elif name == 'crime':
		from src.datasets import CommunitiesCrime
		train_dataset = CommunitiesCrime(seed=seed, train=True)
		valid_dataset = CommunitiesCrime(seed=seed, train=False)

	elif name == 'German':
		from src.datasets import localds as mydatasets
		data_path = os.path.join('./data/', 'german_proc_gattr.z')

		train_dataset = mydatasets.GattrDataset(data_path, split='train')
		valid_dataset = mydatasets.GattrDataset(data_path, split='val')
		test_dataset = mydatasets.GattrDataset(data_path, split='test')

	# elif name.count('gerry') > 0:
	# 	from src.datasets import GerryDataset
	# 	train_dataset = GerryDataset(seed=seed, dataset=name.split('-')[1], train=True)
	# 	valid_dataset = GerryDataset(seed=seed, dataset=name.split('-')[1], train=False)

	# elif name == 'binary-mnist':
	# 	from src.datasets import BinarySizeMNIST

	# 	train_transformation = Compose([
	# 		ToTensor(),
	# 		Normalize((0.1307,), (0.3081,)),
	# 	])

	# 	train_dataset = BinarySizeMNIST(root='./data', train=True, download=download, transform=train_transformation)
	# 	# train_dataset = LabelSubsetWrapper(train_dataset, which_labels=(0,1))
		
	# 	valid_dataset = BinarySizeMNIST(root='./data', train=False, download=download, transform=train_transformation)
	# 	# val_dataset = LabelSubsetWrapper(val_dataset, which_labels=(0,1))

	# elif name == 'celeba-test':
	# 	target = 'Smiling'
	# 	attr_cols = ['Young']
	# 	# attr_cols = ['Young', 'Brown_Hair']
	# 	from src.datasets import CelebA

	# 	train_dataset = CelebA('./data/celeba/', train=True, target=target, 
	# 									spurious=attr_cols,
	# 									n_samples=1000)
	# 	valid_dataset = CelebA('./data/celeba/', train=False, target=target, 
	# 									spurious=attr_cols,
	# 									n_samples=1000)
	
	# elif name == 'celeba':
	# 	target = 'Smiling'
	# 	attr_cols = ['Young']
	# 	from src.datasets import CelebA

	# 	train_dataset = CelebA(root='./data/celeba/', train=True, target=target, 
	# 									spurious=attr_cols)
	# 	valid_dataset = CelebA('./data/celeba/', train=False, target=target, 
	# 									spurious=attr_cols)

	else:
		error('unknown dataset')

	return train_dataset, valid_dataset
