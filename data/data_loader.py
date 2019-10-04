import sys

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.transforms import Lambda

sys.path.append('../')
from data.datasets import FaceDataset, FDataset, RafFaceDataset, ScutFBP
from config.cfg import cfg

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomRotation(30),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.FiveCrop(224),
        Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(crop) for crop in crops])),
    ]),
}


def split_train_and_test_with_py_datasets(data_set, batch_size=cfg['batch_size'], test_size=0.2, num_works=4,
                                          pin_memory=True):
    """
    split datasets into train and test loader
    :param data_set:
    :param batch_size:
    :param test_size:
    :param num_works:
    :param pin_memory:
    :return:
    """
    num_dataset = len(data_set)
    indices = list(range(num_dataset))
    split = int(np.floor(test_size * num_dataset))

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset=data_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_works,
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=data_set, batch_size=batch_size, sampler=test_sampler, num_workers=num_works,
        pin_memory=pin_memory
    )

    return train_loader, test_loader


def load_raf_db():
    """
    load RAF-Db dataset
    :param batch_size:
    :return:
    """
    print('loading RAF-Db dataset...')
    train_dataset = RafFaceDataset(train=True, type='basic',
                                   transform=transforms.Compose([
                                       transforms.Resize(227),
                                       transforms.RandomCrop(224),
                                       transforms.ColorJitter(),
                                       transforms.RandomRotation(30),
                                       transforms.ToTensor(),
                                       transforms.Normalize(
                                           mean=[131.45376586914062, 103.98748016357422, 91.46234893798828],
                                           std=[1, 1, 1])
                                   ]))
    trainloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    test_dataset = RafFaceDataset(train=False, type='basic',
                                  transform=transforms.Compose([
                                      transforms.Resize(227),
                                      transforms.RandomCrop(224),
                                      transforms.ColorJitter(),
                                      transforms.RandomRotation(30),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[131.45376586914062, 103.98748016357422, 91.46234893798828],
                                          std=[1, 1, 1])
                                  ]))
    testloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=4, drop_last=True)

    return trainloader, testloader


def load_scutfbp5500_cv(cv_index=1):
    """
    load SCUT-FBP5500 with Cross Validation Index
    :return:
    """
    train_loader = torch.utils.data.DataLoader(FaceDataset(cv_index=cv_index, train=True, transform=data_transforms[
        'train']),
                                               batch_size=cfg['batch_size'], shuffle=True, num_workers=4,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(FaceDataset(cv_index=cv_index, train=False, transform=data_transforms[
        'test']),
                                              batch_size=cfg['batch_size'], shuffle=False, num_workers=4,
                                              drop_last=True)

    return train_loader, test_loader


def load_scutfbp5500_64():
    """
    load Face Dataset for SCUT-FBP5500 with 6/4 split CV
    :return:
    """
    train_loader = torch.utils.data.DataLoader(FDataset(train=True, transform=data_transforms['train']),
                                               batch_size=cfg['batch_size'], shuffle=True, num_workers=50,
                                               drop_last=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(FDataset(train=False, transform=data_transforms['test']),
                                              batch_size=cfg['batch_size'], shuffle=False, num_workers=50,
                                              drop_last=True, pin_memory=True)

    return train_loader, test_loader


def load_scutfbp():
    """
    load Face Dataset for SCUT-FBP
    :return:
    """
    train_loader = torch.utils.data.DataLoader(ScutFBP(transform=data_transforms['train']), batch_size=cfg[
        'batch_size'],
                                               shuffle=True, num_workers=4, drop_last=True)
    test_loader = torch.utils.data.DataLoader(ScutFBP(transform=data_transforms['test']), batch_size=cfg['batch_size'],
                                              shuffle=False, num_workers=4, drop_last=True)

    return train_loader, test_loader
