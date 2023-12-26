import torch.nn as nn
import torch


class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()
        # Define the model parameters using Pytorch modules here
        # FOR MNIST
        # self.weight = torch.nn.Linear(28*28, 10, bias=True)
        # FOR CIFAR10
        self.weight = torch.nn.Linear(28 * 28, 10, bias=True)

    def forward(self, x):
        # x = x.view(-1, 28*28)
        x = x.view(-1, 28 * 28)  # Reshaping the data so that it becomes a vector
        # Fill in the rest here
        x = self.weight(x)
        return x
