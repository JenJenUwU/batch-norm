import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from data import train_loader, test_loader
from hyper_parameters import *


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # Define the model parameters using Pytorch modules here
        self.fc1 = torch.nn.Linear(28 * 28, 300)
        self.fc2 = torch.nn.Linear(300, 100)
        self.fc3 = torch.nn.Linear(100, 10)
        self.bnm1 = torch.nn.BatchNorm1d(300, momentum=0.1)
        self.bnm2 = torch.nn.BatchNorm1d(100, momentum=0.1)

        self.addNoise1 = AddNoise(0.5, 1.25, 0.1, 0.1)
        self.addNoise2 = AddNoise(0.5, 1.25, 0.1, 0.1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Reshaping the data so that it becomes a vector
        x = self.fc1(x)
        x = F.relu(x)

        x = self.bnm1(x)
        x = self.addNoise1(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.bnm2(x)
        x = self.addNoise2(x)
        x = self.fc3(x)
        # Fill in the rest here

        return x


class AddNoise(nn.Module):
    def __init__(self, n_mu, n_sigma, r_mu, r_sigma):
        super().__init__()
        self.n_mu = n_mu
        self.n_sigma = n_sigma
        self.r_mu = r_mu
        self.r_sigma = r_sigma

    def forward(self, input):
        batchSize = input.shape[0]
        numOfEntries = input.shape[1]

        mu = np.random.uniform(low=(-1) * self.n_mu, high=self.n_mu, size=numOfEntries)
        sigma = np.random.uniform(low=1, high=self.n_sigma, size=numOfEntries)

        for j in range(numOfEntries):
            if j == 0:
                mMat = np.random.uniform(low=mu[j] - self.r_mu, high=mu[j] + self.r_mu, size=batchSize)
                sMat = np.random.normal(loc=sigma[j], scale=self.r_sigma, size=batchSize)
                mMat = np.expand_dims(mMat, axis=1)
                sMat = np.expand_dims(sMat, axis=1)
            else:
                m = np.random.uniform(low=mu[j] - self.r_mu, high=mu[j] + self.r_mu, size=batchSize)
                s = np.random.normal(loc=sigma[j], scale=self.r_sigma, size=batchSize)
                m = np.expand_dims(m, axis=1)
                s = np.expand_dims(s, axis=1)
                mMat = np.concatenate((mMat, m), axis=1)
                sMat = np.concatenate((sMat, s), axis=1)
        return torch.from_numpy(sMat).float().to(device) * input + torch.from_numpy(mMat).float().to(device)


def noisy_batch_norm():
    loss_function = torch.nn.CrossEntropyLoss()

    train_accuracy = []
    test_accuracy = []

    model = LeNet()
    model.to(device)

    # Define the optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=momentum,
                          weight_decay=weight_decay)
    # iterate over epochs
    for epoch in range(1, epochs + 1):
        # train phase
        model.train()
        accuracy = 0
        N = 0

        # iterate over train data
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            # forward pass
            logits = model(images)
            loss = loss_function(logits, labels)

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # check if predicted labels are equal to true labels
            predicted_labels = torch.argmax(logits, dim=1)
            accuracy += torch.sum((predicted_labels == labels).float()).item()
            N += images.shape[0]

        print("Epoch: " + str(epoch) + " -- Avg. Accuracy: " + str(100. * accuracy / N))
        train_accuracy.append(100. * accuracy / N)

        # test phase
        model.eval()
        accuracy = 0
        N = 0

        # iterate over test data
        for batch_idx, (images, labels) in enumerate(test_loader, start=1):
            images, labels = images.to(device), labels.to(device)

            # forward pass
            logits = model(images)

            # check if predicted labels are equal to true labels
            predicted_labels = torch.argmax(logits, dim=1)
            accuracy += torch.sum((predicted_labels == labels).float()).item()
            N += images.shape[0]
        test_accuracy.append(100. * accuracy / N)
        print(test_accuracy[-1])

    # plot results
    plt.title('Accuracy Versus Epoch')
    plt.plot(range(1, epochs + 1), train_accuracy, label='Train')
    plt.plot(range(1, epochs + 1), test_accuracy, label='Test')
    plt.legend()
