import numpy as np
import torch
from torch import nn


class Model:

    def __init__(self, train_set: np.ndarray = None, test_set: np.ndarray = None):
        self.train = train_set
        self.test = test_set

        self.X_train = self.train[:, :-1]
        self.y_train = self.train[:, -1]

    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass

    def relu(self):
        pass

    def export(self):
        pass
