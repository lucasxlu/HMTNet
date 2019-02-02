"""
Model Zoo for other state-of-the-arts in FBP
unfinished yet!!!!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SCUTNet(nn.Module):
    """
    Paper: SCUT-FBP: A Benchmark Dataset for Facial Beauty Perception
    Link: https://arxiv.org/ftp/arxiv/papers/1511/1511.02459.pdf
    Note: input size is 227*227
    Performance:

    """

    def __init__(self):
        super(SCUTNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(50, 100, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(100, 150, 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(150, 200, 4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(200, 250, 4)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(250, 300, 2)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(300 * 1 * 1, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.pool6(F.relu(self.conv6(x)))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PICNN(nn.Module):
    """
    Paper: Facial attractiveness prediction using psychologically inspired convolutional neural network (PI-CNN)
    Link: https://ieeexplore.ieee.org/abstract/document/7952438/
    Note: input size is 227*227
    """

    def __init__(self):
        super(PICNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 5)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(50, 100, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(100, 150, 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv4 = nn.Conv2d(150, 200, 4)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.conv5 = nn.Conv2d(200, 250, 3)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(250, 300, 2)
        self.pool6 = nn.AvgPool2d(kernel_size=2, stride=2)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(300 * 2 * 2, 500)
        self.fc2 = nn.Linear(500, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.pool6(F.relu(self.conv6(x)))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
