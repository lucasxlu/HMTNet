"""
cross-dataset test on SCUT-FBP dataset
the results of 5-fold CV are:
0.8582, 0.8607, 0.8662, 0.8709, 0.8977
"""

import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from skimage import io
from skimage.transform import resize
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append('../')
from config.cfg import cfg

from models.hmtnet_fbp import HMTNet


def extract_df(img_list, layer="relu5", hmt_fbp_model='./model/hmt-net-fbp.pth'):
    """
    extract deep features from pretrained HMT-Net
    :param img_list:
    :param hmt_fbp_model:
    :return:
    """
    X = []
    hmt_net = HMTNet()
    print("loading pre-trained model...")
    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        hmt_net = nn.DataParallel(hmt_net)
        hmt_net.load_state_dict(torch.load(hmt_fbp_model))
    else:
        state_dict = torch.load(hmt_fbp_model)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
        hmt_net.load_state_dict(new_state_dict)

    hmt_net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hmt_net = hmt_net.to(device)

    print("extracting deep features...")
    for img in img_list:
        print('extracting image %s ...' % str(img))
        x = resize(io.imread(img), (224, 224), mode='constant')

        x[0] -= 131.45376586914062
        x[1] -= 103.98748016357422
        x[2] -= 91.46234893798828

        x = np.transpose(x, [2, 0, 1])

        x = torch.from_numpy(x).unsqueeze(0).float()
        x = x.to(device)

        for name, module in hmt_net.named_children():
            if name != layer:
                x = module.forward(x)
            else:
                break

        ft = x.to("cpu").data.numpy().ravel().tolist()
        X.append(ft)

    return X


def main_scut(filenames, X, y):
    """
    train and eval on SCUT-FBP benchmark with HMTNet descriptor and Ridge Regression
    :param filenames:
    :param X:
    :param y:
    :return:
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)
    reg = linear_model.BayesianRidge()
    # reg = linear_model.Ridge(alpha=50.0)
    # reg = linear_model.Lasso(alpha=0.005)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)

    mae_lr = round(mean_absolute_error(np.array(y_test), np.array(y_pred).ravel()), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(y_test), np.array(y_pred).ravel())), 4)
    pc = round(np.corrcoef(np.array(y_test), np.array(y_pred).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of Trans HMT-Net is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Trans HMT-Net is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Trans HMT-Net is {0}===================='.format(pc))

    col = ['filename', 'gt', 'pred']
    rows = []
    for i in range(len(y_test)):
        rows.append([filenames[i], y_test.tolist()[i], y_pred.tolist()[i]])

    df = pd.DataFrame(rows, columns=col)
    df.to_excel("./scutfbp_output.xlsx", sheet_name='Output', index=False)
    print('Output Excel has been generated~')


def main_eccv(filenames, X_train, y_train, X_test, y_test):
    """
    train and test on ECCV HotOrNot DataSet
    :param filenames:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    # reg = linear_model.Ridge(alpha=50.0)
    reg = linear_model.BayesianRidge()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)

    mae_lr = round(mean_absolute_error(np.array(y_test), np.array(y_pred).ravel()), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(y_test), np.array(y_pred).ravel())), 4)
    pc = round(np.corrcoef(np.array(y_test), np.array(y_pred).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of Trans HMT-Net is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Trans HMT-Net is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Trans HMT-Net is {0}===================='.format(pc))

    col = ['filename', 'gt', 'pred']
    rows = []
    for i in range(len(y_test)):
        rows.append([filenames[i], y_test[i], y_pred.tolist()[i]])

    df = pd.DataFrame(rows, columns=col)
    df.to_excel("./eccv_output.xlsx", sheet_name='Output', index=False)
    print('Output Excel has been generated~')


if __name__ == '__main__':
    print('Performing on SCUT-FBP DataSet...')
    df = pd.read_excel(cfg['scutfbp_excel'], sheet_name='Sheet1')
    img_list = [os.path.join(cfg['scutfbp_images_dir'], 'SCUT-FBP-%d.jpg' % _) for _ in df['Image']]
    print(img_list)
    X = extract_df(img_list)
    y = df['Attractiveness label']
    main_scut(['SCUT-FBP-%d.jpg' % _ for _ in df['Image']], X, y)

    # print('Performing on ECCV HotOrNot DataSet...')
    # cv_split = 1
    # df = pd.read_csv(
    #     os.path.join(os.path.split(os.path.abspath(cfg['hotornot_dir']))[0], 'eccv2010_split%d.csv' % cv_split),
    #     header=None)
    #
    # filenames = [os.path.join(cfg['hotornot_dir'], _.replace('.bmp', '.jpg')) for
    #              _ in df.iloc[:, 0].tolist()]
    # scores = df.iloc[:, 1].tolist()
    # flags = df.iloc[:, 2].tolist()
    #
    # train_set = OrderedDict()
    # test_set = OrderedDict()
    #
    # for i in range(len(flags)):
    #     if flags[i] == 'train':
    #         train_set[filenames[i]] = scores[i]
    #     elif flags[i] == 'test':
    #         test_set[filenames[i]] = scores[i]
    #
    # test_filenames = [_ for _ in test_set.keys()]
    # X_train = extract_df(train_set.keys(), layer='conv2')
    # X_test = extract_df(test_set.keys(), layer='conv2')
    # main_eccv(test_filenames, X_train, list(train_set.values()), X_test, list(test_set.values()))

    # X = extract_df(["E:\DataSet\Face\SCUT-FBP\Crop\SCUT-FBP-2.jpg", "E:\DataSet\Face\SCUT-FBP\Crop\SCUT-FBP-1.jpg"])
    # print(np.array(X))
