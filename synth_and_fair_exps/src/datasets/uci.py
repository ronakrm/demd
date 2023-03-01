from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd

from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import fetch_openml

class UCI(Dataset):
	def __init__(self, attr_col, train=True, threshold=True):
		super().__init__()

		X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, 
					test_size=0.1, random_state=0)
	
		if train:
			X = X_train
			y = y_train
		else:
			X = X_test
			y = y_test

		print('treating first col as attribute!')
		self.features = X[:, 1:]
		self.attrs = X[:, 0]
		self.labels = y

		if threshold:
			self.labels = self.labels > np.mean(y_train)
			self.attrs = self.attrs > np.mean(X_train[attr_col])

		# import pdb; pdb.set_trace()

	def __getitem__(self, index):
		X = torch.from_numpy(self.features[index, :]).float()
		y = self.labels[index]
		attr = self.attrs[index]
		return X, torch.Tensor([int(y), int(attr)])

	def __len__(self):
		return len(self.labels)


class Boston(UCI):
	def __init__(self, **kwargs):

		boston = load_boston()
		self.data, self.target   = (boston.data, boston.target)
		
		super().__init__(**kwargs)

class CaliHousing(UCI):
	def __init__(self, **kwargs):

		calihouse = fetch_california_housing()
		self.data, self.target   = (calihouse.data, calihouse.target)
		
		super().__init__(**kwargs)

# class AmesHousing(UCI):
# 	def __init__(self, **kwargs):

# 		ameshouse = fetch_openml(name="house_prices", as_frame=True)
# 		self.data, self.target   = (ameshouse.data, ameshouse.target)
		
# 		super().__init__(**kwargs)

class Adult(Dataset):
	def __init__(self, attr_col, train=True):
		super().__init__()

		adult = fetch_openml(data_id="1590", as_frame=True)

		X, y  = (adult.data, adult.target)

		X_train, X_test, y_train, y_test = train_test_split(X, y, 
					test_size=0.1, random_state=0)
	
		if train:
			X = X_train
			y = y_train
		else:
			X = X_test
			y = y_test

		# cleanup NaNs
		for col in ['workclass', 'occupation', 'native-country']:
		    X[col].fillna(X[col].mode()[0], inplace=True)

		# encode labels to categoricals
		categorical = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
		for feature in categorical:
			le = preprocessing.LabelEncoder()
			X[feature] = le.fit_transform(X[feature])

		le = preprocessing.LabelEncoder()	
		y = le.fit_transform(y)		

		# normalize features
		scaler = StandardScaler()
		X = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)	

		self.features = X.drop(columns=attr_col)
		self.attrs = X[attr_col]
		self.labels = y

		# import pdb; pdb.set_trace()

	def __getitem__(self, index):
		X = torch.from_numpy(self.features.loc[index].values).float()
		y = self.labels[index]
		attr = self.attrs.loc[index]
		return X, torch.Tensor([int(y), int(attr)])

	def __len__(self):
		return len(self.labels)



