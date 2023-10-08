import pandas as pd

from PIL.Image import Resampling
from torchvision import transforms

import src.neural_network.neuralNetwork as nN
from PIL import Image


class Main:
    def __init__(self):
        self.neural_network = None
        self.data = None
        self.data_normalized = self.data
        self.data_numpy = None
        self.prediction = None

        self.folder = "../files"
        self.path = "../data_participant.csv"

    def load_data(self):

        self.data = pd.read_csv(self.path)

        self.data.drop('ID', axis=1, inplace=True)

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

            for i in self.data.iloc[:, 1]:
                if i not in dic:
                    dic[i] = counter
                    counter += 1

            self.data_normalized = self.data

            self.data_normalized.iloc[:, 1] = self.data.iloc[:, 1].map(dic)

        def remove_nan():
            """
            Remove all rows with NaN values
            :return:
            """
            self.data.dropna(inplace=True)

        remove_nan()
        replace_words()

    def preprocess(self):
        """
        Load the images and preprocess them
        :return:
        """

        for i in range(len(self.data_normalized.iloc[:, 1])):
            with Image.open(self.folder + "/" + self.data_normalized.iloc[i, 0]) as img:
                img.resize((224, 224), Resampling.LANCZOS)

                img = img.convert("RGB")

                img = transforms.RandomResizedCrop((224, 224))(img)
                img = transforms.RandomHorizontalFlip()(img)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

                img = img.numpy()

                self.data_normalized.iloc[i, 0] = img

    def train(self):
        self.neural_network.train_nn()

    def test(self):
        pass

    def predict(self):

        self.data = pd.read_csv(self.path)
        self.data = self.data.dropna(subset=['label'], how='any')

        self.data = self.data.fillna(0, inplace=True)

        self.preprocess()

        for i in range(len(self.data_normalized.iloc[:, 1])):

            value = self.neural_network.predict_nn(self.data_normalized.iloc[i, 1])

            match value:
                case 0:
                    self.data_normalized.iloc[i, 2] = "angry"
                case 1:
                    self.data_normalized.iloc[i, 2] = "disgusted"
                case 2:
                    self.data_normalized.iloc[i, 2] = "fearful"
                case 3:
                    self.data_normalized.iloc[i, 2] = "happy"
                case 4:
                    self.data_normalized.iloc[i, 2] = "neutral"
                case 5:
                    self.data_normalized.iloc[i, 2] = "sad"
                case 6:
                    self.data_normalized.iloc[i, 2] = "surprised"

    def export_data(self):

        self.data = pd.read_csv(self.path)

        self.data = self.data.dropna()

        self.data = self.data.concat(self.data_normalized)

        self.data.to_csv("../data_participant_predicted.csv")

    def main(self):
        self.load_data()
        self.neural_network = nN.NeuralNetwork(dataset=self.data_numpy)
        self.train()
        self.predict()


if __name__ == "__main__":
    main = Main()
    main.main()
