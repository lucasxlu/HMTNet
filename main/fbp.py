import copy
import time
import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets
from sklearn.metrics import mean_absolute_error, mean_squared_error

# os.environ['CUDA_VISIBLE_DEVICES'] = 'gpu02'

sys.path.append('../')
from models.vggm import VggM
from util import file_utils
from models.losses import HMTLoss
from models.hmtnet_fbp import RNet, GNet, HMTNet
from data.datasets import FaceGenderDataset, FaceRaceDataset, FaceDataset, FDataset
from config.cfg import cfg
from data import data_loader


def train_gnet(model, train_loader, test_loader, criterion, optimizer, num_epochs=200, inference=False):
    """
    train GNet
    :param model:
    :param train_loader:
    :param test_loader:
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :return:
    """
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if not inference:
        exp_lr_scheduler.step()
        model.train()

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                # inputs, labels = data
                inputs, labels = data['image'], data['gender']

                model = model.to(device)
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model.forward(inputs)

                outputs = outputs.view(cfg['batch_size'], -1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        print('Finished Training')
        print('Save trained model...')

        model_path_dir = './model'
        file_utils.mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'gnet.pth'))

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/gnet.pth')))

    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        # images, labels = data
        images, labels = data['image'], data['gender']
        model = model.to(device)
        labels = labels.to(device)
        outputs = model.forward(images.to(device))

        outputs = outputs.view(cfg['batch_size'], 2)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('correct = %d ...' % correct)
    print('total = %d ...' % total)
    print('Accuracy of the network on test images: %f' % (correct / total))


def train_rnet(model, train_loader, test_loader, criterion, optimizer, num_epochs=200, inference=False):
    """
    train GNet
    :param model:
    :param train_loader:
    :param test_loader:
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :return:
    """
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    if not inference:
        exp_lr_scheduler.step()
        model.train()

        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                # inputs, labels = data
                inputs, labels = data['image'], data['race']

                model = model.to(device)
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model.forward(inputs)
                outputs = outputs.view(cfg['batch_size'], 2)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        print('Finished Training')
        print('Save trained model...')

        model_path_dir = './model'
        file_utils.mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'rnet.pth'))

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/rnet.pth')))

    model.eval()
    correct = 0
    total = 0
    for data in test_loader:
        # images, labels = data
        images, labels = data['image'], data['race']
        model = model.to(device)
        labels = labels.to(device)
        outputs = model.forward(images.to(device))

        outputs = outputs.view(cfg['batch_size'], 2)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('correct = %d ...' % correct)
    print('total = %d ...' % total)
    print('Accuracy of the RNet on test images: %f' % (correct / total))


def finetune_vgg_m_model(model_ft, train_loader, test_loader, criterion, num_epochs=200, inference=False):
    """
    fine-tune VGG M Face Model
    :param model_ft:
    :param train_loader:
    :param test_loader:
    :param criterion:
    :param num_epochs:
    :param inference:
    :return:
    """
    num_ftrs = model_ft.fc8.in_channels
    model_ft.fc8 = nn.Conv2d(num_ftrs, 2, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_ft = nn.DataParallel(model_ft)

    model_ft = model_ft.to(device)

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    if not inference:
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            exp_lr_scheduler.step()
            model_ft.train()

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                # inputs, labels = data
                inputs, labels = data['image'], data['attractiveness'].float()

                model_ft = model_ft.to(device)
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward + backward + optimize
                outputs = model_ft.forward(inputs)
                outputs = (torch.sum(outputs, dim=1) / 2).view(cfg['batch_size'], 1)
                outputs = outputs.view(cfg['batch_size'])

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_ft.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        print('Finished Training')
        print('Save trained model...')

        model_path_dir = './model'
        file_utils.mkdirs_if_not_exist(model_path_dir)
        torch.save(model_ft.state_dict(), os.path.join(model_path_dir, 'ft_vgg_m.pth'))

    else:
        print('Loading pre-trained model...')
        model_ft.load_state_dict(torch.load(os.path.join('./model/ft_vgg_m.pth')))

    model_ft.eval()
    correct = 0
    total = 0

    # for data in test_loader:
    for i, data in enumerate(test_loader, 0):
        # images, labels = data
        images, labels = data['image'], data['attractiveness'].float()
        model_ft = model_ft.to(device)
        labels = labels.to(device)
        outputs = model_ft.forward(images.to(device))
        outputs = outputs.view(cfg['batch_size'])

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('correct = %d ...' % correct)
    print('total = %d ...' % total)
    print('Accuracy of the network on the test images: %f' % (correct / total))


def train_anet(model_ft, train_loader, test_loader, criterion, num_epochs=200, inference=False):
    """
    train ANet
    :param model_ft:
    :param train_loader:
    :param test_loader:
    :param criterion:
    :param num_epochs:
    :param inference:
    :return:
    """
    num_ftrs = model_ft.fc8.in_channels
    model_ft.fc8 = nn.Conv2d(num_ftrs, 1, 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        model_ft = nn.DataParallel(model_ft)

    model_ft = model_ft.to(device)

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    if not inference:
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            exp_lr_scheduler.step()
            model_ft.train()

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data['image'], data['attractiveness']

                model_ft = model_ft.to(device)
                inputs, labels = inputs.to(device), labels.float().to(device)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward + backward + optimize
                outputs = model_ft(inputs)
                outputs = outputs.view(cfg['batch_size'])

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_ft.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        print('Finished Training')
        print('Save trained model...')

        model_path_dir = './model'
        file_utils.mkdirs_if_not_exist(model_path_dir)
        torch.save(model_ft.state_dict(), os.path.join(model_path_dir, 'anet.pth'))

    else:
        print('Loading pre-trained model...')
        model_ft.load_state_dict(torch.load(os.path.join('./model/anet.pth')))

    model_ft.eval()
    predicted_labels = []
    gt_labels = []

    for i, data in enumerate(test_loader, 0):
        images, labels = data['image'], data['attractiveness']
        model_ft = model_ft.to(device)
        labels = labels.to(device)
        outputs = model_ft.forward(images.to(device))

        predicted_labels += outputs.to("cpu").data.numpy().tolist()
        gt_labels += labels.to("cpu").numpy().tolist()

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae_lr = round(mean_absolute_error(np.array(gt_labels), np.array(predicted_labels).ravel()), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(gt_labels), np.array(predicted_labels).ravel())), 4)
    pc = round(np.corrcoef(np.array(gt_labels), np.array(predicted_labels).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of ANet is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of ANet is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of ANet is {0}===================='.format(pc))


def train_hmtnet(hmt_net, train_loader, test_loader, num_epochs, inference=False):
    """
    train HMT-Net
    :param hmt_net:
    :param train_loader:
    :param test_loader:
    :param num_epochs:
    :param inference:
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        hmt_net = nn.DataParallel(hmt_net)

    hmt_net = hmt_net.to(device)

    criterion = HMTLoss()
    optimizer = optim.SGD(hmt_net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-2)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    if not inference:
        hmt_net.train()
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            exp_lr_scheduler.step()

            running_loss = 0.0

            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, gender, race, attractiveness = data['image'], data['gender'], data['race'], \
                                                       data['attractiveness']

                hmt_net = hmt_net.cuda()
                inputs, gender, race, attractiveness = inputs.to(device), gender.to(device), race.to(
                    device), attractiveness.float().to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                g_pred, r_pred, a_pred = hmt_net.forward(inputs)

                g_pred = g_pred.view(cfg['batch_size'], -1)
                r_pred = r_pred.view(cfg['batch_size'], -1)
                a_pred = a_pred.view(cfg['batch_size'])

                loss = criterion(g_pred, gender, r_pred, race, a_pred, attractiveness)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

            hmt_net.eval()

            predicted_attractiveness_values = []
            gt_attractiveness_values = []

            total = 0
            g_correct = 0
            r_correct = 0
            for data in test_loader:
                images, g_gt, r_gt, a_gt = data['image'], data['gender'], data['race'], \
                                           data['attractiveness']
                hmt_net = hmt_net.to(device)
                g_gt = g_gt.to(device)
                r_gt = r_gt.to(device)
                a_gt = a_gt.to(device)

                g_pred, r_pred, a_pred = hmt_net.forward(images.to(device))

                predicted_attractiveness_values += a_pred.to("cpu").data.numpy().tolist()
                gt_attractiveness_values += a_gt.to("cpu").numpy().tolist()

                g_pred = g_pred.view(cfg['batch_size'], -1)
                r_pred = r_pred.view(cfg['batch_size'], -1)

                _, g_predicted = torch.max(g_pred.data, 1)
                _, r_predicted = torch.max(r_pred.data, 1)
                total += g_gt.size(0)
                g_correct += (g_predicted == g_gt).sum().item()
                r_correct += (r_predicted == r_gt).sum().item()

            print('total = %d ...' % total)
            print('Gender correct sample = %d ...' % g_correct)
            print('Race correct sample = %d ...' % r_correct)
            print('Accuracy of Race Classification: %.4f' % (r_correct / total))
            print('Accuracy of Gender Classification: %.4f' % (g_correct / total))

            mae_lr = round(
                mean_absolute_error(np.array(gt_attractiveness_values),
                                    np.array(predicted_attractiveness_values).ravel()), 4)
            rmse_lr = round(np.math.sqrt(
                mean_squared_error(np.array(gt_attractiveness_values),
                                   np.array(predicted_attractiveness_values).ravel())), 4)
            pc = round(
                np.corrcoef(np.array(gt_attractiveness_values), np.array(predicted_attractiveness_values).ravel())[
                    0, 1], 4)

            print('===============The Mean Absolute Error of HMT-Net is {0}===================='.format(mae_lr))
            print('===============The Root Mean Square Error of HMT-Net is {0}===================='.format(rmse_lr))
            print('===============The Pearson Correlation of HMT-Net is {0}===================='.format(pc))

            model_path_dir = './model'
            file_utils.mkdirs_if_not_exist(model_path_dir)
            torch.save(hmt_net.state_dict(), os.path.join(model_path_dir, 'hmt-net-fbp-epoch-%d.pth' % (epoch + 1)))

            hmt_net.train()

        print('Finished Training')
        print('Save trained model...')

        model_path_dir = './model'
        file_utils.mkdirs_if_not_exist(model_path_dir)
        torch.save(hmt_net.state_dict(), os.path.join(model_path_dir, 'hmt-net-fbp.pth'))

    else:
        print('Loading pre-trained model...')
        hmt_net.load_state_dict(torch.load(os.path.join('./model/hmt-net-fbp.pth')))

    hmt_net.eval()

    predicted_attractiveness_values = []
    gt_attractiveness_values = []

    total = 0
    g_correct = 0
    r_correct = 0

    for data in test_loader:
        images, g_gt, r_gt, a_gt = data['image'], data['gender'], data['race'], \
                                   data['attractiveness']
        hmt_net = hmt_net.to(device)
        g_gt = g_gt.to(device)
        r_gt = r_gt.to(device)
        a_gt = a_gt.to(device)

        g_pred, r_pred, a_pred = hmt_net.forward(images.to(device))

        predicted_attractiveness_values += a_pred.to("cpu").data.numpy().tolist()
        gt_attractiveness_values += a_gt.to("cpu").numpy().tolist()

        # g_pred = g_pred.view(-1, g_pred.numel())
        # r_pred = r_pred.view(-1, r_pred.numel())

        g_pred = g_pred.view(cfg['batch_size'], -1)
        r_pred = r_pred.view(cfg['batch_size'], -1)

        _, g_predicted = torch.max(g_pred.data, 1)
        _, r_predicted = torch.max(r_pred.data, 1)
        total += g_gt.size(0)
        g_correct += (g_predicted == g_gt).sum().item()
        r_correct += (r_predicted == r_gt).sum().item()

    print('total = %d ...' % total)
    print('Gender correct sample = %d ...' % g_correct)
    print('Race correct sample = %d ...' % r_correct)
    print('Accuracy of Race Classification: %.4f' % (r_correct / total))
    print('Accuracy of Gender Classification: %.4f' % (g_correct / total))

    mae_lr = round(
        mean_absolute_error(np.array(gt_attractiveness_values), np.array(predicted_attractiveness_values).ravel()), 4)
    rmse_lr = round(np.math.sqrt(
        mean_squared_error(np.array(gt_attractiveness_values), np.array(predicted_attractiveness_values).ravel())), 4)
    pc = round(np.corrcoef(np.array(gt_attractiveness_values), np.array(predicted_attractiveness_values).ravel())[0, 1],
               4)

    print('===============The Mean Absolute Error of HMT-Net is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of HMT-Net is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of HMT-Net is {0}===================='.format(pc))


if __name__ == '__main__':
    print('***************************start training GNet***************************')
    gnet = GNet()
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = data_loader.load_scutfbp5500_64()
    optimizer = optim.SGD(gnet.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-2)
    train_gnet(gnet, train_loader, test_loader, criterion, optimizer, num_epochs=170, inference=False)
    print('***************************finish training GNet***************************')

    # print('###############################start training RNet###############################')
    # rnet = RNet()
    # criterion = nn.CrossEntropyLoss()
    # train_loader, test_loader = data_loader.load_scutfbp5500_64()
    # optimizer = optim.SGD(rnet.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-2)
    # train_rnet(rnet, train_loader, test_loader, criterion, optimizer, num_epochs=170, inference=False)
    # print('###############################finish training RNet###############################')

    # print('***************************start training ANet***************************')
    # train_loader, test_loader = data_loader.load_scutfbp5500_64()
    # train_anet(VggM(), train_loader, test_loader, nn.MSELoss(), 200, False)
    # print('***************************end training ANet***************************')

    # print('+++++++++++++++++++++++++++++++++start training HMT-Net on SCUT-FBP5500+++++++++++++++++++++++++++++++++')
    # hmtnet = HMTNet()
    # # train_loader, test_loader = data_loader.load_scutfbp5500_cv(cv_index=1)
    # train_loader, test_loader = data_loader.load_scutfbp5500_64()
    #
    # train_hmtnet(hmtnet, train_loader, test_loader, 250, False)
    # print('+++++++++++++++++++++++++++++++++finish training HMT-Net SCUT-FBP5500+++++++++++++++++++++++++++++++++')
