
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn.base

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits




class PyTorchMLP():
    def __init__(self, device = None, neural = None, loss_fn = None, optimizer= None):
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #print(f"Using {self.device} device")

        #self.neural = NeuralNetwork().to(self.device)
        #self.loss_fn = nn.CrossEntropyLoss()
        #self.optimizer = torch.optim.SGD(self.neural.parameters(), lr=1e-3)
        #print(self.nn)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if neural is None:
            self.neural = NeuralNetwork().to(self.device)
        else:
            self.neural = neural

        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

        if optimizer is None:
            self.optimizer = torch.optim.SGD(self.neural.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer


    def fit(self, X,y, sample_weight=None):
        X = torch.tensor(X)
        y = torch.tensor(y)
        epochs = 5
        for t in range(epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            self.train(X,y)
            self.test(y)

        print("Done!")
        return

    def predict(self, X):
        return np.array([random.uniform(0,1) for x in range(len(X))])

    def __str__(self):
        return "NN"

    def train(self, X,y):
        self.neural.train()
        for i in range(len(X)):
            #X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = self.neural(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss, current = loss.item(), i * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{len(X):>5d}]")

    def test(self, dataloader):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.neural.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.neural(X)
                test_loss += self.loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def get_params(self, deep=False):
        return {
            'neural': self.neural,
            'device': self.device,
            'loss_fn': self.loss_fn,
            'optimizer': self.optimizer
        }
    def set_params(self, **params):
        return