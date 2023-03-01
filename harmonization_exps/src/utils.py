import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

class ConfusionMatrix:
    # self.stat = [#TP, #TN, #FP, #FN]
    def __init__(self):
        self.stat = [0] * 4

    def update(self, new_stat, as_dict=True):
        if as_dict:
            self.stat[0] += new_stat['TP']
            self.stat[1] += new_stat['TN']
            self.stat[2] += new_stat['FP']
            self.stat[3] += new_stat['FN']
        else:
            if len(new_stat) != 4:
                raise ValueError("argument new_stat must be an iterable of size 4")
            self.stat = [s + new_s for s, new_s in zip(self.stat, new_stat)]

    def get_accuracy(self):
        divisor = sum(self.stat)
        if divisor == 0: return math.nan
        return (self.stat[0] + self.stat[1]) / divisor

    def get_precision(self):
        divisor = self.stat[0] + self.stat[2]
        if divisor == 0: return math.nan
        return self.stat[0] / (self.stat[0] + self.stat[2])

    def get_recall(self):
        divisor = self.stat[0] + self.stat[3]
        if divisor == 0: return math.nan
        return self.stat[0] / divisor

    def get_f1(self):
        p = self.get_precision()
        r = self.get_recall()
        if p + r == 0: return math.nan
        assert not (math.isnan(p) or math.isnan(r))
        return 2 * (p * r) / (p + r)

    def get_summary(self):
        return {'accuracy': self.get_accuracy(),
                'precision': self.get_precision(),
                'recall': self.get_recall(),
                'f1': self.get_f1()}

    def reset(self):
        self.stat = [0] * 4

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    
    def forward(self, x):
        return x * F.sigmoid(x)