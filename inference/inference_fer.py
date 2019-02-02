import sys
import time
from pprint import pprint

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage import io
from skimage.transform import resize

sys.path.append('../')
from models.hmtnet_fer import HMTNet


def inference(img, hmtnet_model_file='../main/model/hmt-net-fer.pth'):
    """
    inference with pre-trained HMT-Net
    :param img: an image filepath or image numpy array
    :param hmtnet_model_file:
    :return:
    """
    hmtnet = HMTNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        hmtnet = nn.DataParallel(hmtnet)

    hmtnet.load_state_dict(torch.load(hmtnet_model_file))
    hmtnet.eval()

    if type(img) is str:
        image = resize(io.imread(img), (224, 224), mode='constant')
    else:
        img = cv2.resize(img, (224, 224))
        image = img.astype(np.float64)

    image[:, :, 0] -= np.mean(image[:, :, 0])
    image[:, :, 1] -= np.mean(image[:, :, 1])
    image[:, :, 2] -= np.mean(image[:, :, 2])

    image = np.transpose(image, [2, 0, 1])

    input = torch.from_numpy(image).unsqueeze(0).float()

    hmtnet = hmtnet.to(device)
    input = input.to(device)

    tik = time.time()
    e_pred, a_pred, r_pred, g_pred = hmtnet.forward(input)
    tok = time.time()

    _, e_predicted = torch.max(e_pred.data, 1)
    _, a_predicted = torch.max(a_pred.data, 1)
    _, r_predicted = torch.max(r_pred.data, 1)
    _, g_predicted = torch.max(g_pred.data, 1)

    if int(g_predicted.to("cpu")) == 0:
        g_pred = 'male'
    elif int(g_predicted.to("cpu")) == 1:
        g_pred = 'female'
    elif int(g_predicted.to("cpu")) == 2:
        g_pred = 'unsure'

    if int(r_predicted.to("cpu")) == 0:
        r_pred = 'Caucasian'
    elif int(r_predicted.to("cpu")) == 1:
        r_pred = 'African-American'
    elif int(r_predicted.to("cpu")) == 2:
        r_pred = 'Asian'

    if int(a_predicted.to("cpu")) == 0:
        a_pred = '0-3'
    elif int(a_predicted.to("cpu")) == 1:
        a_pred = '4-19'
    elif int(a_predicted.to("cpu")) == 2:
        a_pred = '20-39'
    elif int(a_predicted.to("cpu")) == 3:
        a_pred = '40-69'
    elif int(a_predicted.to("cpu")) == 4:
        a_pred = '70+'

    if int(e_predicted.to("cpu")) == 0:
        e_pred = 'Surprise'
    elif int(e_predicted.to("cpu")) == 1:
        e_pred = 'Fear'
    elif int(e_predicted.to("cpu")) == 2:
        e_pred = 'Disgust'
    elif int(e_predicted.to("cpu")) == 3:
        e_pred = 'Happiness'
    elif int(e_predicted.to("cpu")) == 4:
        e_pred = 'Sadness'
    elif int(e_predicted.to("cpu")) == 5:
        e_pred = 'Anger'
    elif int(e_predicted.to("cpu")) == 6:
        e_pred = 'Neutral'

    # coord = c_pred.data.to("cpu").view(-1).tolist()
    # landmarks = [[coord[i], coord[i + 5]] for i in range(5)]

    return {'gender': g_pred, 'emotion': e_pred, 'race': r_pred, 'age': a_pred, 'elapse': tok - tik}


if __name__ == '__main__':
    pprint(inference('./xl.jpg'))

    # image = cv2.imread('./xl.jpg')
    # image = cv2.resize(image, (224, 224))
    # landmarks = [[25.154125213623047, 26.930830001831055],
    #              [31.520950317382812, 26.71474838256836],
    #              [26.985374450683594, 29.277803421020508],
    #              [25.51268196105957, 33.68495559692383],
    #              [31.69681739807129, 33.485130310058594]]
    #
    # for ldmk in landmarks:
    #     cv2.circle(image, (int(ldmk[0]), int(ldmk[1])), 2, (255, 245, 0), -1)
    #
    # cv2.imshow('img', image)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
