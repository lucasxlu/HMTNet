import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

sys.path.append('../')
from models.baselines import SCUTNet, PICNN
from util.file_utils import mkdirs_if_not_exist
from data import data_loader


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs,
                inference=False):
    """
    train model
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
        print('Start training Model...')
        model.train()

        for epoch in range(num_epochs):
            scheduler.step()

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                images, label = data['image'], data['attractiveness']

                images = images.to(device)
                label = label.to(device).float()

                optimizer.zero_grad()

                pred = model(images)
                loss = criterion(pred, label.unsqueeze(-1))
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        print('Finished training Model...\n')
        print('Saving trained model...')
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'scutnet.pth'))
        print('Model has been saved successfully~')

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/scutnet.pth')))

    model.eval()

    predicted_attractiveness_values = []
    gt_attractiveness_values = []

    for data in test_dataloader:
        images, a_gt = data['image'], data['attractiveness']
        model = model.to(device)
        a_gt = a_gt.to(device)

        a_pred = model.forward(images.to(device))

        predicted_attractiveness_values += a_pred.to("cpu").data.numpy().tolist()
        gt_attractiveness_values += a_gt.to("cpu").numpy().tolist()

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae_lr = round(
        mean_absolute_error(np.array(gt_attractiveness_values), np.array(predicted_attractiveness_values).ravel()), 4)
    rmse_lr = round(np.math.sqrt(
        mean_squared_error(np.array(gt_attractiveness_values), np.array(predicted_attractiveness_values).ravel())), 4)
    pc = round(np.corrcoef(np.array(gt_attractiveness_values), np.array(predicted_attractiveness_values).ravel())[0, 1],
               4)

    print('===============The Mean Absolute Error of Model is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Model is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Model is {0}===================='.format(pc))


def run_scutnet(train_dataloader, test_dataloader):
    """
    train and eval on SCUTNet
    :param train_dataloader:
    :param test_dataloader:
    :return:
    """
    scutnet = SCUTNet()
    criterion = nn.MSELoss()

    optimizer = optim.SGD(scutnet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_model(model=scutnet, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler,
                num_epochs=200,
                inference=False)


def run_picnn(train_dataloader, test_dataloader):
    """
    train and eval on PICNN
    :param train_dataloader:
    :param test_dataloader:
    :return:
    """
    picnn = PICNN()
    criterion = nn.MSELoss()

    optimizer = optim.SGD(picnn.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train_model(model=picnn, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                criterion=criterion, optimizer=optimizer, scheduler=exp_lr_scheduler,
                num_epochs=200,
                inference=False)


if __name__ == '__main__':
    trainloader, testloader = data_loader.load_scutfbp5500_64()
    run_scutnet(trainloader, testloader)
    # run_picnn(trainloader, testloader)
