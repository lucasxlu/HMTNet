"""
HMT-Net on RAF-DB
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, mean_absolute_error
from torch.optim import lr_scheduler

sys.path.append('../')
from config.cfg import cfg
from models.hmtnet_fer import HMTNet
from models.losses import HMTFERLoss
from util.file_utils import mkdirs_if_not_exist
from data.data_loader import load_raf_db


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=80,
                inference=False):
    """
    train and eval HMT-Net
    :param model:
    :param train_dataloader:
    :param test_dataloader:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :param inference:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if not inference:
        print('Start training HMT-Net...')
        model.train()

        for epoch in range(num_epochs):
            scheduler.step()

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                # inputs, gender, race, age, emotion, coordinate, = data['image'], data['gender'], data['race'], \
                #                                                   data['age'], data['emotion'], data['coordinate']

                inputs, gender, race, age, emotion, = data['image'], data['gender'], data['race'], \
                                                      data['age'], data['emotion']

                inputs = inputs.to(device)
                gender = gender.to(device)
                race = race.to(device)
                age = age.to(device)
                emotion = emotion.to(device)
                # coordinate = coordinate.to(device)

                optimizer.zero_grad()

                e_pred, a_pred, r_pred, g_pred = model(inputs)
                e_pred = e_pred.view(cfg['batch_size'], -1)
                a_pred = a_pred.view(cfg['batch_size'], -1)
                g_pred = g_pred.view(cfg['batch_size'], -1)
                r_pred = r_pred.view(cfg['batch_size'], -1)
                # c_pred = c_pred.view(cfg['batch_size'], -1)

                # loss = criterion(g_pred, gender, r_pred, race, a_pred, age, e_pred, emotion, c_pred, coordinate)
                loss = criterion(g_pred, gender, r_pred, race, a_pred, age, e_pred, emotion)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        print('Finished training HMT-Net...\n')
        print('Saving trained model...')
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'hmt-net-fer.pth'))
        print('HMT-Net has been saved successfully~')

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/hmt-net-fer.pth')))

    model.eval()

    g_correct = 0
    r_correct = 0
    a_correct = 0
    e_correct = 0
    total = 0
    e_predicted_list = []
    e_gt_list = []
    # c_predicted_list = []
    # c_gt_list = []
    for data in test_dataloader:
        images, gender, race, age, emotion = data['image'], data['gender'], data['race'], data['age'], data['emotion']
        gender = gender.to(device)
        race = race.to(device)
        age = age.to(device)
        emotion = emotion.to(device)
        e_pred, a_pred, r_pred, g_pred = model.forward(images)

        g_pred = g_pred.view(cfg['batch_size'], 3)
        r_pred = r_pred.view(cfg['batch_size'], 3)
        a_pred = a_pred.view(cfg['batch_size'], 5)
        e_pred = e_pred.view(cfg['batch_size'], 7)
        # c_pred = c_pred.view(cfg['batch_size'], 10)

        _, g_predicted = torch.max(g_pred.data, -1)
        _, r_predicted = torch.max(r_pred.data, -1)
        _, a_predicted = torch.max(a_pred.data, -1)
        _, e_predicted = torch.max(e_pred.data, -1)

        total += emotion.size(0)

        e_predicted_list += e_predicted.to("cpu").data.numpy().tolist()
        e_gt_list += emotion.to("cpu").numpy().tolist()
        # c_predicted_list += c_pred.to("cpu").data.numpy().tolist()
        # c_gt_list += coordinate.to("cpu").numpy().tolist()

        g_correct += (g_predicted == gender).sum().item()
        r_correct += (r_predicted == race).sum().item()
        a_correct += (a_predicted == age).sum().item()
        e_correct += (e_predicted == emotion).sum().item()

    print('Race Accuracy of HMT-Net: %f' % (r_correct / total))
    print('Gender Accuracy of HMT-Net: %f' % (g_correct / total))
    print('Age Accuracy of HMT-Net: %f' % (a_correct / total))
    print('Emotion Accuracy of HMT-Net: %f' % (e_correct / total))
    # print('Coordinates MAE of HMT-Net: %f' % mean_absolute_error(np.array(c_gt_list), np.array(c_predicted_list)))
    print('Confusion Matrix on FER: ')
    print(confusion_matrix(np.array(e_gt_list).ravel().tolist(), np.array(e_predicted_list).ravel().tolist()))


def run_hmtnet(train_dataloader, test_dataloader, epoch=80):
    """
    run HMT-Net
    :param train_dataloader:
    :param test_dataloader:
    :param epoch:
    :return:
    """
    hmtnet = HMTNet()
    criterion = HMTFERLoss()

    optimizer = optim.SGD(hmtnet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-6)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    # lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer)

    train_model(model=hmtnet, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler, num_epochs=epoch, inference=False)


if __name__ == '__main__':
    trainloader, testloader = load_raf_db()
    run_hmtnet(trainloader, testloader, epoch=80)
