import torch

from torchvision import datasets, transforms
from hyper_parameters import *

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.1307, 0.3081)])

train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', download=True, train=True, transform=transform),
                                           batch_size=batch_size, shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', download=True, train=False, transform=transform),
                                          batch_size=batch_size, shuffle=False, drop_last=False)
