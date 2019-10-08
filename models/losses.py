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
        # self.a_criterion = nn.L1Loss()
        # self.a_criterion = HuberLoss()
        self.a_criterion = SmoothHuberLoss()

    def forward(self, g_pred, g_gt, r_pred, r_gt, a_pred, a_gt):
        g_loss = self.g_criterion(g_pred, g_gt)
        r_loss = self.r_criterion(r_pred, r_gt)
        a_loss = self.a_criterion(a_pred, a_gt)

        hmt_loss = self.weight_g * g_loss + self.weight_r * r_loss + self.weight_a * a_loss

        return hmt_loss


def log_cosh_loss(input, target, epsilon=0):
    """
    Definition of LogCosh Loss
    """
    return torch.log(torch.cosh(target - input) + epsilon)


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


class SmoothHuberLoss(_Loss):
    """
    SmoothHuberLoss
    if |y-\hat{y}| < \delta, return log(\frac{1}{2}LogCosh(y-\hat{y}))
    else return |y-\hat{y}|
    """

    def __init__(self, reduction='mean', delta=0.6):
        super(SmoothHuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, input, target):
        t = torch.abs(input - target)

        return torch.mean(torch.where(t < self.delta, log_cosh_loss(input, target), F.l1_loss(input, target)))


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
