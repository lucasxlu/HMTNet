import sys

import torch.nn as nn

sys.path.append('../')


class GenderBranch(nn.Module):
    """
    Branch layers for handling gender classification task
    Input: BATCH*512*14*14
    """

    def __init__(self, output_num=3):
        super(GenderBranch, self).__init__()
        self.gconv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.gbn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.grelu1 = nn.ReLU()
        self.gconv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.gbn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.grelu2 = nn.ReLU()
        self.gconv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.gbn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.grelu3 = nn.ReLU()
        self.gconv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.gbn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.grelu4 = nn.ReLU()
        self.gpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gconv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.gbn5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.grelu5 = nn.ReLU()
        self.gconv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.gbn6 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.grelu6 = nn.ReLU()
        self.gpool6 = nn.MaxPool2d(2)

        self.gconv7 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.gbn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.grelu7 = nn.ReLU()
        self.gconv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.gbn8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.grelu8 = nn.ReLU()
        self.gpool8 = nn.MaxPool2d(2)

        self.gconv9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.gbn9 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.grelu9 = nn.ReLU()
        self.gconv10 = nn.Conv2d(128, output_num, kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x1 = self.grelu1(self.gbn1(self.gconv1(x)))
        x2 = self.grelu2(self.gbn2(self.gconv2(x1)))
        x2 = (x1 + x2) / 2
        x3 = self.grelu3(self.gbn3(self.gconv3(x2)))
        x4 = self.grelu4(self.gbn4(self.gconv4(x3)))
        x4 = (x3 + x4) / 2
        x4 = self.gpool4(x4)

        x5 = self.grelu5(self.gbn5(self.gconv5(x4)))
        x6 = self.grelu6(self.gbn6(self.gconv6(x5)))
        x6 = (x5 + x6) / 2
        x6 = self.gpool6(x6)

        x7 = self.grelu7(self.gbn7(self.gconv7(x6)))
        x8 = self.grelu8(self.gbn8(self.gconv8(x7)))
        x8 = (x7 + x8) / 2
        x8 = self.gpool8(x8)

        x9 = self.grelu9(self.gbn9(self.gconv9(x8)))
        x10 = self.gconv10(x9)

        return x10

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class RaceBranch(nn.Module):
    """
    Branch layers for race classification
    Input: BATCH*512*14*14
    """

    def __init__(self, output_num=3):
        super(RaceBranch, self).__init__()

        self.rconv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.rbn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu1 = nn.ReLU()
        self.rconv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.rbn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu2 = nn.ReLU()
        self.rconv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.rbn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu3 = nn.ReLU()
        self.rconv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.rbn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu4 = nn.ReLU()
        self.rpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rconv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.rbn5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu5 = nn.ReLU()
        self.rconv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.rbn6 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu6 = nn.ReLU()
        self.rpool6 = nn.MaxPool2d(2)

        self.rconv7 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.rbn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu7 = nn.ReLU()
        self.rconv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.rbn8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu8 = nn.ReLU()
        self.rpool8 = nn.MaxPool2d(2)

        self.rconv9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.rbn9 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.rrelu9 = nn.ReLU()
        self.rconv10 = nn.Conv2d(128, output_num, kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x1 = self.rrelu1(self.rbn1(self.rconv1(x)))
        x2 = self.rrelu2(self.rbn2(self.rconv2(x1)))
        x2 = (x1 + x2) / 2
        x3 = self.rrelu3(self.rbn3(self.rconv3(x2)))
        x4 = self.rrelu4(self.rbn4(self.rconv4(x3)))
        x4 = (x3 + x4) / 2
        x4 = self.rpool4(x4)

        x5 = self.rrelu5(self.rbn5(self.rconv5(x4)))
        x6 = self.rrelu6(self.rbn6(self.rconv6(x5)))
        x6 = (x5 + x6) / 2
        x6 = self.rpool6(x6)

        x7 = self.rrelu7(self.rbn7(self.rconv7(x6)))
        x8 = self.rrelu8(self.rbn8(self.rconv8(x7)))
        x8 = (x7 + x8) / 2
        x8 = self.rpool8(x8)

        x9 = self.rrelu9(self.rbn9(self.rconv9(x8)))
        x10 = self.rconv10(x9)

        return x10

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class AgeBranch(nn.Module):
    """
    Branch layers for age classification
    Input: BATCH*512*14*14
    """

    def __init__(self, output_num=5):
        super(AgeBranch, self).__init__()

        self.aconv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.abn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.arelu1 = nn.ReLU()
        self.aconv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.abn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.arelu2 = nn.ReLU()
        self.aconv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.abn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.arelu3 = nn.ReLU()
        self.aconv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.abn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.arelu4 = nn.ReLU()
        self.apool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.aconv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.abn5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.arelu5 = nn.ReLU()
        self.aconv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.abn6 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.arelu6 = nn.ReLU()
        self.apool6 = nn.MaxPool2d(2)

        self.aconv7 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.abn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.arelu7 = nn.ReLU()
        self.aconv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.abn8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.arelu8 = nn.ReLU()
        self.apool8 = nn.MaxPool2d(2)

        self.aconv9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.abn9 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.arelu9 = nn.ReLU()
        self.aconv10 = nn.Conv2d(128, output_num, kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        x1 = self.arelu1(self.abn1(self.aconv1(x)))
        x2 = self.arelu2(self.abn2(self.aconv2(x1)))
        x2 = (x1 + x2) / 2
        x3 = self.arelu3(self.abn3(self.aconv3(x2)))
        x4 = self.arelu4(self.abn4(self.aconv4(x3)))
        x4 = (x3 + x4) / 2
        x4 = self.apool4(x4)

        x5 = self.arelu5(self.abn5(self.aconv5(x4)))
        x6 = self.arelu6(self.abn6(self.aconv6(x5)))
        x6 = (x5 + x6) / 2
        x6 = self.apool6(x6)

        x7 = self.arelu7(self.abn7(self.aconv7(x6)))
        x8 = self.arelu8(self.abn8(self.aconv8(x7)))
        x8 = (x7 + x8) / 2
        x8 = self.apool8(x8)

        x9 = self.arelu9(self.abn9(self.aconv9(x8)))
        x10 = self.aconv10(x9)

        return x10

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class EmotionBranch(nn.Module):
    """
    Branch layers for emotion classification
    Input: BATCH*512*14*14
    """

    def __init__(self, output_num=7):
        super(EmotionBranch, self).__init__()

        self.econv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.ebn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.erelu1 = nn.ReLU()
        self.econv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.ebn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.erelu2 = nn.ReLU()
        self.econv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.ebn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.erelu3 = nn.ReLU()
        self.econv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.ebn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.erelu4 = nn.ReLU()
        self.epool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.econv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.ebn5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.erelu5 = nn.ReLU()
        self.econv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.ebn6 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.erelu6 = nn.ReLU()
        self.epool6 = nn.MaxPool2d(2, stride=2)

        self.econv7 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.ebn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.erelu7 = nn.ReLU()
        self.econv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.ebn8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.erelu8 = nn.ReLU()
        self.epool8 = nn.MaxPool2d(2, stride=2)

        self.econv9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.ebn9 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.erelu9 = nn.ReLU()
        self.econv10 = nn.Conv2d(128, output_num, kernel_size=3, stride=1, padding=1)
        # self.ebn10 = nn.BatchNorm2d(output_num, eps=1e-05, momentum=0.1, affine=True)
        # self.erelu10 = nn.ReLU()
        # self.epool10 = nn.AvgPool2d(3)

    def forward(self, x):
        # x17 = torch.cat((x13, x17), 1)  # Depth-Wise Concatenate Fusion
        x1 = self.erelu1(self.ebn1(self.econv1(x)))
        x2 = self.erelu2(self.ebn2(self.econv2(x1)))
        x2 = (x1 + x2) / 2  # Avg Fusion
        x3 = self.erelu3(self.ebn3(self.econv3(x2)))
        x4 = self.erelu4(self.ebn4(self.econv4(x3)))
        x4 = (x3 + x4) / 2
        x4 = self.epool4(x4)

        x5 = self.erelu5(self.ebn5(self.econv5(x4)))
        x6 = self.erelu6(self.ebn6(self.econv6(x5)))
        x6 = (x5 + x6) / 2
        x6 = self.epool6(x6)

        x7 = self.erelu7(self.ebn7(self.econv7(x6)))
        x8 = self.erelu8(self.ebn8(self.econv8(x7)))
        x8 = (x7 + x8) / 2
        x8 = self.epool8(x8)

        x9 = self.erelu9(self.ebn9(self.econv9(x8)))
        x10 = self.econv10(x9)
        # x10 = self.erelu10(self.ebn10(self.econv10(x9)))
        # x10 = self.epool10(x10)

        return x10

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class CoordinateBranch(nn.Module):
    """
    Branch layers for facial landmarks localisation
    Input: BATCH*512*14*14
    """

    def __init__(self, output_num=10):
        super(CoordinateBranch, self).__init__()

        self.cconv1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.cbn1 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.crelu1 = nn.ReLU()
        self.cconv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.cbn2 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.crelu2 = nn.ReLU()
        self.cconv3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.cbn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.crelu3 = nn.ReLU()
        self.cconv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.cbn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.crelu4 = nn.ReLU()
        self.cpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.cconv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.cbn5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.crelu5 = nn.ReLU()
        self.cconv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.cbn6 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.crelu6 = nn.ReLU()
        self.cpool6 = nn.MaxPool2d(2)

        self.cconv7 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.cbn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.crelu7 = nn.ReLU()
        self.cconv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.cbn8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.crelu8 = nn.ReLU()
        self.cpool8 = nn.MaxPool2d(2)

        self.cconv9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.cbn9 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.crelu9 = nn.ReLU()
        self.cconv10 = nn.Conv2d(128, output_num, kernel_size=2, stride=2, padding=1)

    def forward(self, x):
        # x17 = (x9 + x11 + x17) / 3  # Avg Fusion
        # x17 = torch.cat((x13, x17), 1)  # Depth-Wise Concatenate Fusion
        x1 = self.crelu1(self.cbn1(self.cconv1(x)))
        x2 = self.crelu2(self.cbn2(self.cconv2(x1)))
        x3 = self.crelu3(self.cbn3(self.cconv3(x2)))
        x4 = self.cpool4(self.crelu4(self.cbn4(self.cconv4(x3))))

        x5 = self.crelu5(self.cbn5(self.cconv5(x4)))
        x6 = self.cpool6(self.crelu6(self.cbn6(self.cconv6(x5))))

        x7 = self.crelu7(self.cbn7(self.cconv7(x6)))
        x8 = self.cpool8(self.crelu8(self.cbn8(self.cconv8(x7))))

        x9 = self.crelu9(self.cbn9(self.cconv9(x8)))
        # x10 = self.cpool10(self.crelu10(self.cbn10(self.cconv10(x9))))
        x10 = self.cconv10(x9)

        return x10

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
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu8 = nn.ReLU()
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn9 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu9 = nn.ReLU()
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn10 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu10 = nn.ReLU()
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn11 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn12 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu12 = nn.ReLU()
        self.pool12 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.gbranch = GenderBranch()
        self.rbranch = RaceBranch()
        self.abranch = AgeBranch()
        self.ebranch = EmotionBranch()
        # self.cbranch = CoordinateBranch()

    def forward(self, x):
        x1 = self.relu1(self.bn1(self.conv1(x)))
        x2 = self.pool2(self.relu2(self.bn2(self.conv2(x1))))

        x3 = self.relu3(self.bn3(self.conv3(x2)))
        x4 = self.pool4(self.relu4(self.bn4(self.conv4(x3))))

        x5 = self.relu5(self.bn5(self.conv5(x4)))
        x6 = self.relu6(self.bn6(self.conv6(x5)))
        x7 = self.relu7(self.bn7(self.conv7(x6)))
        x8 = self.pool8(self.relu8(self.bn8(self.conv8(x7))))

        x9 = self.relu9(self.bn9(self.conv9(x8)))
        x10 = self.relu10(self.bn10(self.conv10(x9)))
        x11 = self.relu11(self.bn11(self.conv11(x10)))
        x12 = self.pool12(self.relu12(self.bn12(self.conv12(x11))))

        g_pred = self.gbranch(x12)
        r_pred = self.rbranch(x12)
        a_pred = self.abranch(x12)
        e_pred = self.ebranch(x12)
        # c_pred = self.cbranch(x12)

        return e_pred, a_pred, r_pred, g_pred

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
