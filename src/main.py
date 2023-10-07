import pandas as pd
from tkinter import filedialog
import src.neural_network.neuralNetwork as nN


class Main:
    def __init__(self):
        self.neural_network = None
        self.data = None
        self.data_numpy = None
        self.prediction = None

    def load_data(self):
        path_to_data: str = ""
        try:
            path_to_data = filedialog.askopenfilename()
        except Exception as e:
            print("No file selected")
            exit(1)

        self.data = pd.read_csv(path_to_data)

        self.data_numpy = self.data.to_numpy()

    def preprocessing(self):
        pass

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
