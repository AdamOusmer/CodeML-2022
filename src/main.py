import pandas as pd
from tkinter import filedialog
import src.neural_network.neuralNetwork as nN
from PIL import Image
import numpy as np


class Main:
    def __init__(self):
        self.neural_network = None
        self.data = None
        self.data_normalized = self.data
        self.data_numpy = None
        self.prediction = None

        self.path = filedialog.askopenfilename()

    def load_data(self):
        self.path: str = ""
        try:
            self.path = filedialog.askopenfilename()
        except Exception as e:
            print("No file selected")
            exit(1)

        self.data = pd.read_csv(self.path)

        self.filter_data()
        self.preprocess()

        self.data_numpy = self.data.to_numpy()

    def filter_data(self):
        """
        Filter the data to replace words with numbers
        :return:
        """

        def replace_words():

            dic = {}
            counter = 0

            for i in self.data[2]:
                if i not in dic:
                    dic[i] = counter
                    counter += 1

            self.data_normalized[2] = self.data[2].map(dic)

        def remove_nan():
            """
            Remove all rows with NaN values
            :return:
            """
            self.data_normalized.dropna(inplace=True)

        replace_words()
        remove_nan()
        self.data_normalized.drop(self.data_numpy.columns[0], axis=1)

    def preprocess(self):
        """
        Load the images and preprocess them
        :return:
        """

        for i in range(self.data_normalized[1]):
            with Image.open(self.path + self.data_normalized[1, i]) as img:
                img.resize((224, 224), Image.ANTIALIAS)

                array = np.array(img)
                array = array / 255

                self.data_normalized[1, i] = array.flatten()

    def train(self):
        pass

    def test(self):
        pass

    def predict(self):
        pass

    def export_data(self):
        pass

    def main(self):
        self.load_data()
        self.neural_network = nN.NeuralNetwork(dataset=self.data_numpy)


if __name__ == "__main__":
    main = Main()
    main.main()
