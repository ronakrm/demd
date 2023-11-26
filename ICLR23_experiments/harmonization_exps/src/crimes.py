from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import scipy.io

import numpy as np
import torch

class CommunitiesCrime(Dataset):
        def __init__(self, seed=0, path='data/Crime.mat', train=True):
                super().__init__()

                crime = scipy.io.loadmat(path)

                X, group, y  = (crime['X'], crime['Scat'], crime['Y'])

                # reverse 0/1 group label
                # for general consistency and comparison to other results
                group = (~(group.astype(bool))).astype(group.dtype)

                X_train, X_test, g_train, g_test, y_train, y_test = train_test_split(X, group, y, test_size=0.3, random_state=seed)
        
                if train:
                        self.features = X_train
                        self.attrs = g_train
                        self.labels = 1.0*(y_train > np.quantile(y_train, .3)) # as in other work
                else:
                        self.features = X_test
                        self.attrs = g_test
                        self.labels = 1.0*(y_test > np.quantile(y_train, .3)) # as in other work

        def __getitem__(self, index):
                X = torch.from_numpy(self.features[index,:]).float()
                y = self.labels[index]
                attr = self.attrs[index]
                #return X, torch.Tensor([y[0], int(attr)])
                return torch.tensor(X).float(), torch.tensor(y[0]).long(), torch.tensor(attr[0]).long(), torch.tensor(0).long()                

        def __len__(self):
                return len(self.labels)



