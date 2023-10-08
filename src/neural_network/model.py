import numpy as np


class Model:

    def __init__(self, train_set: np.ndarray = None, test_set: np.ndarray = None):
        self.train = train_set
        self.test = test_set

        self.x_train = self.train[:, 0]
        self.y_train = self.train[:, 1]

        self.x_test = self.test[:, 0]
        self.y_test = self.test[:, 1]

    def train(self):
        for i in range(len(self.x_train)):
            pass

    def test(self):
        pass

    def predict(self):
        pass

    def relu(self):
        pass

    def export(self):
        pass
