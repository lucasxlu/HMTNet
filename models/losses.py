from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F


class HMTLoss(nn.Module):

    def __init__(self, weight_g=1, weight_r=1, weight_a=2):
        super(HMTLoss, self).__init__()

        self.weight_g = weight_g
        self.weight_r = weight_r
        self.weight_a = weight_a

        self.g_criterion = nn.CrossEntropyLoss()
        self.r_criterion = nn.CrossEntropyLoss()
        self.a_criterion = nn.L1Loss()
        # self.a_criterion = HuberLoss()
        # self.a_criterion = SmoothHuberLoss()

    def forward(self, g_pred, g_gt, r_pred, r_gt, a_pred, a_gt):
        g_loss = self.g_criterion(g_pred, g_gt)
        r_loss = self.r_criterion(r_pred, r_gt)
        a_loss = self.a_criterion(a_pred, a_gt)

        hmt_loss = self.weight_g * g_loss + self.weight_r * r_loss + self.weight_a * a_loss

        return hmt_loss


# class LogCoshLoss(_Loss):
#     """
#     LogCosh Loss
#     """
#
#     def __init__(self, size_average=False, reduce=True, epsilon=1e-4):
#         super(LogCoshLoss, self).__init__(size_average, reduce)
#         self.epsilon = epsilon
#
#     def forward(self, input, target):
#
#         return torch.log(torch.cosh(target - input) + self.epsilon)


class LogCoshLoss(nn.Module):
    """
    Definition of LogCosh Loss
    """

    def __init__(self, epsilon=0):
        super(LogCoshLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input, target):
        return torch.log(torch.cosh(target - input) + self.epsilon)


class HuberLoss(_Loss):
    """
    Huber Loss
    if |y-\hat{y}| < \delta, return \frac{1}{2}MSE
    else return \delta MAE - \frac{1}{2}\delta ** 2
    """

    def __init__(self, size_average=True, reduce=True, delta=0.1):
        super(HuberLoss, self).__init__(size_average, reduce)
        self.delta = delta

    def forward(self, input, target):
        if F.l1_loss(input, target) < self.delta:
            return 0.5 * F.mse_loss(input, target, size_average=self.size_average, reduce=self.reduce)
        else:
            return self.delta * F.l1_loss(input, target, size_average=self.size_average,
                                          reduce=self.reduce) - 0.5 * self.delta * self.delta


class SmoothHuberLoss(nn.Module):
    """
    SmoothHuberLoss
    if |y-\hat{y}| < \delta, return \frac{1}{2}LogCosh
    else return \delta MAE - \frac{1}{2}\delta ** 2
    """

    def __init__(self, delta=0.1):
        super(SmoothHuberLoss, self).__init__()
        self.delta = delta

    def forward(self, input, target):
        if nn.L1Loss(input, target) < self.delta:
            return 0.5 * LogCoshLoss(input, target)
        else:
            return self.delta * nn.L1Loss(input, target) - 0.5 * self.delta * self.delta


class WingLoss(_Loss):
    """
    WingLoss
    first introduced at CVPR2018
    http://openaccess.thecvf.com/content_cvpr_2018/papers/Feng_Wing_Loss_for_CVPR_2018_paper.pdf
    """

    def __init__(self, size_average=True, reduce=True, epsilon=2, w=10):
        super(WingLoss, self).__init__(size_average, reduce)
        self.epsilon = epsilon
        self.w = w
        self.C = self.w - self.w * math.log(1 + abs(self.w) / self.epsilon)

    def forward(self, input, target):

        y_hat = []
        for _ in torch.abs(input):
            if _ < self.epsilon:
                y_hat.append(self.w * torch.log(1 + _ / self.epsilon))
            else:
                y_hat.append(_ - self.C - target)

        return F.l1_loss(torch.Tensor(y_hat).to(torch.device("cuda:0")), target)


class HMTFERLoss(nn.Module):
    """
    HMTLossRaf definition
    """

    def __init__(self, emotion_branch_w=0.7, age_branch_w=0.1, race_branch_w=0.1, gender_branch_w=0.1):
        super(HMTFERLoss, self).__init__()

        self.emotion_branch_w = emotion_branch_w
        self.age_branch_w = age_branch_w
        self.race_branch_w = race_branch_w
        self.gender_branch_w = gender_branch_w

        self.emotion_criterion = nn.CrossEntropyLoss()
        self.age_criterion = nn.CrossEntropyLoss()
        self.race_criterion = nn.CrossEntropyLoss()
        self.gender_criterion = nn.CrossEntropyLoss()

    def forward(self, gender_pred, gender_gt, race_pred, race_gt, age_pred, age_gt, emotion_pred, emotion_gt):
        gender_loss = self.gender_criterion(gender_pred, gender_gt)
        race_loss = self.race_criterion(race_pred, race_gt)
        emotion_loss = self.emotion_criterion(emotion_pred, emotion_gt)
        age_loss = self.age_criterion(age_pred, age_gt)

        hmt_fer_loss = self.emotion_branch_w * emotion_loss + self.age_branch_w * age_loss + self.race_branch_w * race_loss \
                       + self.gender_branch_w * gender_loss

        return hmt_fer_loss
