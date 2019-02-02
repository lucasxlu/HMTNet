import os
import sys

import cv2
import numpy as np

sys.path.append('../')
from util.file_utils import mkdirs_if_not_exist
from config.cfg import cfg


def det_landmarks(image_path):
    """
    detect faces in one image, return face bbox and landmarks
    :param image_path:
    :return:
    """
    import dlib
    predictor = dlib.shape_predictor(cfg['dlib_model'])
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(image_path)
    faces = detector(img, 1)

    result = {}
    if len(faces) > 0:
        for k, d in enumerate(faces):
            shape = predictor(img, d)
            result[k] = {"bbox": [d.left(), d.top(), d.right(), d.bottom()],
                         "landmarks": [[shape.part(i).x, shape.part(i).y] for i in range(68)]}

    return result


def crop_faces(img_dir, type='MTCNN'):
    """
    crop face region and show image window
    :param img_dir:
    :param type" MTCNN or dlib
    :return:
    """
    from mtcnn.mtcnn import MTCNN
    detector = MTCNN()
    fail_list = []

    crop_dir = 'E:/DataSet/Face/SCUT-FBP5500/Crop'
    mkdirs_if_not_exist(crop_dir)

    for img_file in os.listdir(img_dir):
        print('process image %s ...' % str(img_file))

        if type == 'dlib':
            res = det_landmarks(os.path.join(img_dir, img_file))
            for i in range(len(res)):
                bbox = res[i]['bbox']
                image = cv2.imread(os.path.join(img_dir, img_file))
                # cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

                face_region = image[bbox[0]: bbox[2], bbox[1]:bbox[3]]
                cv2.imwrite(os.path.join(crop_dir, img_file), face_region)
        elif type == 'MTCNN':
            img = cv2.imread(os.path.join(img_dir, img_file))
            try:
                result = detector.detect_faces(img)
                # cv2.rectangle(img, (result[0]['box'][0], result[0]['box'][1]),
                #               (result[0]['box'][0] + result[0]['box'][2], result[0]['box'][1] + result[0]['box'][3]),
                #               (0, 155, 255), 2)
                face_region = img[result[0]['box'][0]: result[0]['box'][0] + result[0]['box'][2],
                              result[0]['box'][1]: result[0]['box'][1] + result[0]['box'][3]]
                cv2.imwrite(os.path.join(crop_dir, img_file), face_region)
            except:
                fail_list.append(img_file)

        # cv2.imshow('face_region', face_region)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    print(fail_list)


def img_seqs_to_video(seq_dir):
    """
    Define the codec and create VideoWriter object
    :param seq_dir:
    :return:
    """
    print('start to generate video from %s' % seq_dir)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('./%s.avi' % seq_dir.split(os.sep)[-1], fourcc, 30, (1080, 590))

    seq_indices = [int(_.split('.')[0].replace('frame', '')) for _ in os.listdir(seq_dir)]
    seq_indices.sort()

    filelist = [os.path.join(seq_dir, 'frame{0}.jpg'.format(_)) for _ in seq_indices]

    for file in filelist:
        frame = cv2.imread(file)
        # frame = cv2.flip(frame, 0)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    out.release()

    print('Video has been generated...')


def video_to_imgs(video_file):
    """
    write video frames to local images
    :param video_file:
    :return:
    """
    dir_name = './{0}'.format(video_file.split('/')[-1].split('.')[0])
    mkdirs_if_not_exist(dir_name)
    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()
    count = 0
    while success:
        cv2.imwrite(os.path.join(dir_name, "frame%d.jpg" % count), image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1

    print('Images have been written successfully~~~')


if __name__ == '__main__':
    # crop_faces('E:\DataSet\Face\SCUT-FBP5500\Images')

    img_seqs_to_video('D:\TikTok_with_Annotation')
