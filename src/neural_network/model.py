import numpy as np
import torch
from torch import nn
from torchvision import models


class DynamicNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DynamicNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        self.layer4 = nn.Linear(hidden_size, output_size)
        self.layer5 = nn.Linear(hidden_size, output_size)
        self.layer6 = nn.Linear(hidden_size, output_size)
        self.layer7 = nn.Linear(hidden_size, output_size)
        self.layer8 = nn.Linear(hidden_size, output_size)
        self.layer9 = nn.Linear(hidden_size, output_size)
        self.layer10 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = torch.relu(self.layer1(x))
        out = torch.relu(self.layer2(out))
        out = torch.relu(self.layer3(out))
        out = torch.relu(self.layer4(out))
        out = torch.relu(self.layer5(out))
        out = torch.relu(self.layer6(out))
        out = torch.relu(self.layer7(out))
        out = torch.relu(self.layer8(out))
        out = torch.relu(self.layer9(out))
        out = self.layer10(out)
        return out


class Model:
    def __init__(self, train_set: np.ndarray = None, test_set: np.ndarray = None):
        self.train = train_set
        self.test = test_set

        self.x_train = self.train[:, 0]
        self.y_train = self.train[:, 1]

        self.x_test = self.test[:, 0]
        self.y_test = self.test[:, 1]

        # Initialize the neural network with the dynamic input size and output size of 8
        # self.model = DynamicNeuralNetwork(self.size, 64, 8)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(weights=None)

        num_ftrs = self.model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
        self.model.fc = nn.Linear(num_ftrs, 7)

        self.model = self.model.to(device)

    # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()  # Changed the loss function
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train_model(self, num_epochs=1):
        for epoch in range(num_epochs):
            for i in range(len(self.x_train)):
                inputs = torch.from_numpy(self.x_train[i]).float().unsqueeze(
                    0)  # Convert to tensor and add batch dimension
                label = int(self.y_train[i])  # Convert label to integer

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, torch.tensor(label).unsqueeze(0))

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if (epoch + 1) % 2 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Sample {i + 1}/{len(self.x_train)}, Loss: {loss.item():.4f}')

    def test_model(self):
        inputs = torch.from_numpy(self.x_test).float()
        labels = torch.from_numpy(self.y_test).long()  # Changed the data type to long

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)

        print(f'Test Loss: {loss.item():.4f}')

    def predict_model(self, input_data):
        input_data = torch.from_numpy(input_data).float()
        return torch.argmax(self.model(input_data)).item()  # Return the predicted class
