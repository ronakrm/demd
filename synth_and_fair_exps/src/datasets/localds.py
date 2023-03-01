# vish_ds.py

from __future__ import print_function

import os
import os.path
import numpy as np
import torch
import joblib
import pdb
import torchvision.transforms.functional as tvf

from torch.utils.data import Dataset
from PIL import Image

class BaselineDataset(Dataset):
    """
        Adult dataset, relies on the processed data generated from uci_data.py
        data_path : path to *_proc.z
        split : 'train'/'test'/'val'
        transform : None at the moment
    """

    def __init__(self, data_path, split='train', continuous_confound=False):
        super(BaselineDataset, self).__init__()
        self.data_path = data_path
        self.continuous_confound = continuous_confound
        self.confound_max = 100.0
        train_total_data, split_numbers, train_size, \
        validation_data, validation_labels, validation_confounds, \
        test_data, test_labels, test_confounds = \
            joblib.load(self.data_path)
        
        self.split = split
        if self.split == 'train':
            self.data = train_total_data[:, :split_numbers[0]]
            self.labels = train_total_data[:, split_numbers[0]]
            self.confounds = train_total_data[:, split_numbers[1]]
        elif self.split == 'test':
            self.data = test_data
            self.labels = test_labels
            self.confounds = test_confounds
        elif self.split == 'val':
            self.data = validation_data
            self.labels = validation_labels
            self.confounds = validation_confounds
        else:
            raise ('Not Implemented Error')
        if self.continuous_confound:
            self.confounds = self.confounds / self.confound_max
        # Compute the weights for the labels
        self.label_w = torch.zeros(2)
        for c in range(0, 2):
            self.label_w[c] = np.sum(self.labels != c) * 1.0
            self.label_w[c] /= self.labels.shape[0] * 1.0
        # Compute the weights for the confounds
        if not self.continuous_confound:
            self.confound_w = torch.zeros(2)
            for c in range(0, 2):
                self.confound_w[c] = np.sum(self.confounds != c) * 1.0
                self.confound_w[c] /= self.confounds.shape[0] * 1.0

    def __len__(self):
        return self.data.shape[0]

    def get_label_weights(self):
        # Returns the weights for different class labels based on self.data
        return self.label_w

    def get_confound_weights(self):
        # Returns the weights for different confounds based on self.data
        if self.continuouos_confound:
            return 1.0
        return self.confound_w

    def __getitem__(self, index):
        x = self.data[index, :].astype(float)
        # pdb.set_trace()
        y = int(self.labels[index])
        if self.continuous_confound:
            c = float(self.confounds[index])
            return torch.tensor(x).float(), torch.tensor(y).long(), torch.tensor(c).float()
        else:
            c = int(self.confounds[index])
            return torch.tensor(x).float(), torch.tensor(y).long(), torch.tensor(c).long()

class GattrDataset(Dataset):
    """
        Adult dataset, relies on the processed data generated from uci_data.py
        data_path : path to *_proc_gattr.z
        split : 'train'/'test'/'val'
        transform : None at the moment
    """

    def __init__(self, data_path, split='train'):
        super(GattrDataset, self).__init__()
        self.data_path = data_path
        self.confound_max = 100.0

        train_total_data, split_numbers, train_size, \
        validation_data, validation_labels, validation_confounds, validation_gattr, \
        test_data, test_labels, test_confounds, test_gattr = \
            joblib.load(self.data_path)

        self.split = split
        if self.split == 'train':
            self.data = train_total_data[:, :split_numbers[0]]
            self.labels = train_total_data[:, split_numbers[0]]
            self.confounds = train_total_data[:, split_numbers[1]]
            self.gattr = train_total_data[:, split_numbers[2]]
        elif self.split == 'test':
            self.data = test_data
            self.labels = test_labels
            self.confounds = test_confounds
            self.gattr = test_gattr
        elif self.split == 'val':
            self.data = validation_data
            self.labels = validation_labels
            self.confounds = validation_confounds
            self.gattr = validation_gattr            
        else:
            raise ('Not Implemented Error')

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index, :].astype(float)
        y = int(self.labels[index])
        c = int(self.confounds[index])
        gattr =  float(self.gattr[index])

        # DEMD codebase only needs group label
        # return torch.tensor(x).float(), torch.tensor(y).long(), torch.tensor(c).long(), torch.tensor(gattr).float()
        return torch.tensor(x).float(), torch.tensor((torch.tensor(y), torch.tensor(c)))
        

if __name__ == "__main__":
    def test_adult_age_proc():
        dataset = BaselineDataset(data_path='./data/adult_age_proc.z', split='val',
                                  continuous_confound=True)
        print(dataset.__getitem__(0))
        print(len(dataset))

    test_adult_age_proc()