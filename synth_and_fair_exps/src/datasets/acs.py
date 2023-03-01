import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import folktables

class myDataset(Dataset):
	def __init__(self, train=True):
			super().__init__()

			X_train, X_test, y_train, y_test, group_train, group_test = train_test_split(
	    self.features, self.label, self.group, test_size=0.2, random_state=0)
		
			if train:
				X = X_train
				y = y_train
				group = group_train
			else:
				X = X_test
				y = y_test
				group = group_test

			scalar = StandardScaler()
			self.X = scalar.fit_transform(X)
			self.attrs = group
			self.y = y

	def __getitem__(self, index):
		X = torch.from_numpy(self.X[index, :]).float()
		y = self.y[index]
		attr = self.attrs[index]
		return X, torch.Tensor([int(y), int(attr)])

	def __len__(self):
		return len(self.y)


class ACSEmployment(myDataset):
	def __init__(self, **kwargs):
		data_source = folktables.ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
		acs_data = data_source.get_data(states=["LA"], download=True)
		self.features, self.label, self.group = folktables.ACSEmployment.df_to_numpy(acs_data)
		super().__init__(**kwargs)


class ACSIncome(myDataset):
	def __init__(self, **kwargs):
		data_source = folktables.ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
		acs_data = data_source.get_data(states=["LA"], download=True)
		self.features, self.label, self.group = folktables.ACSIncome.df_to_numpy(acs_data)
		super().__init__(**kwargs)
