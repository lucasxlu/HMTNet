import sys

import torch.nn as nn

sys.path.append('../')


class GenderBranch(nn.Module):
    """
    Branch layers for handling three different tasks
    Input: BATCH*512*13*13
    """

    def __init__(self, output_num=2):
        super(GenderBranch, self).__init__()
        self.gconv1 = nn.Conv2d(512, 256, 3)
        self.gbn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.grelu1 = nn.ReLU()

        self.gconv2 = nn.Conv2d(256, 128, 3)
        self.gbn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.grelu2 = nn.ReLU()
        self.gpool2 = nn.MaxPool2d(3)

        self.gconv3 = nn.Conv2d(128, output_num, 1, stride=2)
        self.gbn3 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)
        self.grelu3 = nn.ReLU()
        # self.gpool3 = nn.MaxPool2d(2)
        self.gpool3 = nn.AvgPool2d(2)

    def forward(self, x):
        x1 = self.gconv1(x)
        x2 = self.gbn1(x1)
        x3 = self.grelu1(x2)
        x4 = self.gconv2(x3)
        x5 = self.gbn2(x4)
        x6 = self.grelu2(x5)
        x7 = self.gpool2(x6)
        x8 = self.gconv3(x7)
        x9 = self.gbn3(x8)
        x10 = self.grelu3(x9)
        x11 = self.gpool3(x10)

        return x11

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class RaceBranch(nn.Module):
    """
    Branch layers for handling three different tasks
    Input: BATCH*512*13*13
    """

    # def __init__(self, output_num=2):
    #     super(RaceBranch, self).__init__()
    #
    #     self.rconv1 = nn.Conv2d(512, 256, 5, 2, 1)
    #     self.rbn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
    #     self.rrelu1 = nn.ReLU()
    #
    #     self.rconv2 = nn.Conv2d(256, 2, 3)
    #     self.rbn2 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)
    #     self.rrelu2 = nn.ReLU()
    #
    #     self.rconv3 = nn.Conv2d(2, output_num, 3)
    #     self.rbn3 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)
    #     self.rrelu3 = nn.ReLU()
    #     # self.rpool3 = nn.MaxPool2d(2)
    #     self.rpool3 = nn.AvgPool2d(2)
    #
    # def forward(self, x):
    #     x1 = self.rconv1(x)
    #     x2 = self.rbn1(x1)
    #     x3 = self.rrelu1(x2)
    #     x4 = self.rconv2(x3)
    #     x5 = self.rbn2(x4)
    #     x6 = self.rrelu2(x5)
    #     x7 = self.rconv3(x6)
    #     x8 = self.rbn3(x7)
    #     x9 = self.rrelu3(x8)
    #     x10 = self.rpool3(x9)
    #
    #     return x10

    def __init__(self, output_num=2):
        super(RaceBranch, self).__init__()
        self.rconv1 = nn.Conv2d(512, 256, 3)
        self.rbn1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu1 = nn.ReLU()

        self.rconv2 = nn.Conv2d(256, 128, 3)
        self.rbn2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu2 = nn.ReLU()
        self.rpool2 = nn.MaxPool2d(3)

        self.rconv3 = nn.Conv2d(128, output_num, 1, stride=2)
        self.rbn3 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu3 = nn.ReLU()
        # self.rpool3 = nn.MaxPool2d(2)
        self.rpool3 = nn.AvgPool2d(2)

    def forward(self, x):
        x1 = self.rconv1(x)
        x2 = self.rbn1(x1)
        x3 = self.rrelu1(x2)
        x4 = self.rconv2(x3)
        x5 = self.rbn2(x4)
        x6 = self.rrelu2(x5)
        x7 = self.rpool2(x6)
        x8 = self.rconv3(x7)
        x9 = self.rbn3(x8)
        x10 = self.rrelu3(x9)
        x11 = self.rpool3(x10)

        return x11

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class HMTNet(nn.Module):
    """
    definition of HMTNet
    """

    def __init__(self):
        super(HMTNet, self).__init__()
        # self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
        #              'std': [1, 1, 1],
        #              'imageSize': [224, 224, 3]}

        self.conv1 = nn.Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2))
        self.bn49 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1))
        self.bn50 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn51 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn52 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn53 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=(6, 6), stride=(1, 1))
        self.bn54 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=False)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1))
        self.bn55 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Conv2d(4096, 1, kernel_size=(1, 1), stride=(1, 1))

        # self.fc8 = nn.Conv2d(4096, 1024, kernel_size=[1, 1], stride=(1, 1))
        # self.relu8 = nn.ReLU()
        # self.fc9 = nn.Conv2d(1024, 1, kernel_size=1)

        self.gbranch = GenderBranch()
        self.rbranch = RaceBranch()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn49(x1)
        x3 = self.relu1(x2)
        x4 = self.pool1(x3)
        x5 = self.conv2(x4)
        x6 = self.bn50(x5)
        x7 = self.relu2(x6)
        x8 = self.pool2(x7)
        x9 = self.conv3(x8)
        x10 = self.bn51(x9)
        x11 = self.relu3(x10)
        x12 = self.conv4(x11)
        x13 = self.bn52(x12)
        x14 = self.relu4(x13)
        x15 = self.conv5(x14)
        x16 = self.bn53(x15)
        x17 = self.relu5(x16)

        x17 = (x9 + x11 + x17) / 3  # Avg Fusion
        # x17 = torch.cat((x13, x17), 1)  # Depth-Wise Concatenate Fusion

        x18 = self.pool5(x17)
        x19 = self.fc6(x18)
        x20 = self.bn54(x19)
        x21 = self.relu6(x20)
        x22 = self.fc7(x21)
        x23 = self.bn55(x22)
        x24 = self.relu7(x23)

        x24 = (x19 + x22 + x24) / 3  # Avg Fusion

        x25 = self.fc8(x24)
        # x25 = self.fc9(self.relu8(self.fc8(x24)))

        g_pred = self.gbranch(x11)
        r_pred = self.rbranch(x17)
        a_pred = x25

        return g_pred, r_pred, a_pred

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class GNet(nn.Module):
    """
    definition of GenderNet
    """

    def __init__(self):
        super(GNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(512, 256, 5, 2, 1)
        self.bn4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(256, 128, 3)
        self.bn5 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(128, 2, 3)
        self.bn6 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.AvgPool2d(2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        x4 = self.pool1(x3)
        x5 = self.conv2(x4)
        x6 = self.bn2(x5)
        x7 = self.relu2(x6)
        x8 = self.pool2(x7)
        x9 = self.conv3(x8)
        x10 = self.bn3(x9)
        x11 = self.relu3(x10)
        x12 = self.conv4(x11)
        x13 = self.bn4(x12)
        x14 = self.relu4(x13)
        x15 = self.conv5(x14)
        x16 = self.bn5(x15)
        x17 = self.relu5(x16)
        x18 = self.conv6(x17)
        x19 = self.bn6(x18)
        x20 = self.relu6(x19)
        x21 = self.pool6(x20)

        return x21

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class RNet(nn.Module):
    """
    definition of RaceNet
    """

    def __init__(self):
        super(RNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=True)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.bn5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)

        self.conv6 = nn.Conv2d(512, 256, 5, 2, 1)
        self.bn6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256, 128, 3)
        self.bn7 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(128, 2, 3)
        self.bn8 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.ReLU()
        self.pool8 = nn.AvgPool2d(2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.bn1(x1)
        x3 = self.relu1(x2)
        x4 = self.pool1(x3)
        x5 = self.conv2(x4)
        x6 = self.bn2(x5)
        x7 = self.relu2(x6)
        x8 = self.pool2(x7)
        x9 = self.conv3(x8)
        x10 = self.bn3(x9)
        x11 = self.relu3(x10)
        x12 = self.conv4(x11)
        x13 = self.bn4(x12)
        x14 = self.relu4(x13)
        x15 = self.conv5(x14)
        x16 = self.bn5(x15)
        x17 = self.conv6(x16)
        x18 = self.bn6(x17)
        x19 = self.relu6(x18)
        x20 = self.conv7(x19)
        x21 = self.bn7(x20)
        x22 = self.relu7(x21)
        x23 = self.conv8(x22)
        x24 = self.bn8(x23)
        x25 = self.relu8(x24)
        x26 = self.pool8(x25)

        return x26

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
