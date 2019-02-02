import sys

import torch
import torch.nn as nn

sys.path.append('../')
from config.cfg import cfg


class VggM(nn.Module):
    """
    VGGM Model with BatchNorm
    """

    def __init__(self):
        super(VggM, self).__init__()
        self.meta = {'mean': [131.45376586914062, 103.98748016357422, 91.46234893798828],
                     'std': [1, 1, 1],
                     'imageSize': [224, 224, 3]}
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=(2, 2))
        self.bn49 = nn.BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, dilation=(1, 1), ceil_mode=False)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=1)
        self.bn50 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, dilation=(1, 1), ceil_mode=True)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn51 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn52 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn53 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, dilation=(1, 1), ceil_mode=False)
        self.fc6 = nn.Conv2d(512, 4096, kernel_size=6, stride=1)
        self.bn54 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=False)
        self.relu6 = nn.ReLU()
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1, stride=1)
        self.bn55 = nn.BatchNorm2d(4096, eps=1e-05, momentum=0.1, affine=True)
        self.relu7 = nn.ReLU()
        self.fc8 = nn.Conv2d(4096, 2622, kernel_size=1, stride=1)

    def forward(self, x0):
        x1 = self.conv1(x0)
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
        x18 = self.pool5(x17)
        x19 = self.fc6(x18)
        x20 = self.bn54(x19)
        x21 = self.relu6(x20)
        x22 = self.fc7(x21)
        x23 = self.bn55(x22)
        x24 = self.relu7(x23)
        x25 = self.fc8(x24)

        return x25

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


def load_vgg_m_face_bn_dag(weights_path=cfg['pretrained_vgg_face']):
    """
    load imported model instance

    Args:
        weights_path (str): If set, loads model weights from the given path
    """
    model = VggM()
    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)

    return model
