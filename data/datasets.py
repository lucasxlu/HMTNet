import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

sys.path.append('../')
from config.cfg import cfg


class RafFaceDataset(Dataset):
    """
    RAF-Face dataset for Face Expression Recognition
    """

    def __init__(self, train=True, type='basic', transform=None):
        manual_annotation_dir = os.path.join(cfg['raf_root'], '%s/Annotation/manual' % type)
        emotion_label_txt_path = os.path.join(cfg['raf_root'], "%s/EmoLabel/list_patition_label.txt" % type)

        emotion_dict = dict(np.loadtxt(emotion_label_txt_path, dtype=np.str))

        if train:
            face_files = []
            genders = []
            races = []
            ages = []
            emotions = []
            coordinates = []
            for _ in os.listdir(manual_annotation_dir):
                if _.startswith('train_'):
                    face_fname = _.replace('_manu_attri', '_aligned').replace('.txt', '.jpg')
                    face_files.append(os.path.join(cfg['raf_root'], '%s/Image/aligned' % type, face_fname))
                    with open(os.path.join(manual_annotation_dir, _), mode='rt') as f:
                        manu_info_list = f.readlines()
                    genders.append(int(manu_info_list[5]))
                    races.append(int(manu_info_list[6]))
                    ages.append(int(manu_info_list[7]))
                    emotions.append(int(emotion_dict[face_fname.replace('_aligned', '')].strip()) - 1)
                    coordinates.append([float(manu_info_list[i].split('\t')[0]) for i in range(5)] +
                                       [float(manu_info_list[i].split('\t')[1]) for i in range(5)])

        else:
            face_files = []
            genders = []
            races = []
            ages = []
            emotions = []
            coordinates = []
            for _ in os.listdir(manual_annotation_dir):
                if _.startswith('test_'):
                    face_fname = _.replace('_manu_attri', '_aligned').replace('.txt', '.jpg')
                    face_files.append(os.path.join(cfg['raf_root'], '%s/Image/aligned' % type, face_fname))
                    with open(os.path.join(manual_annotation_dir, _), mode='rt') as f:
                        manu_info_list = f.readlines()
                    genders.append(int(manu_info_list[5]))
                    races.append(int(manu_info_list[6]))
                    ages.append(int(manu_info_list[7]))
                    emotions.append(int(emotion_dict[face_fname.replace('_aligned', '')].strip()) - 1)
                    coordinates.append([float(manu_info_list[i].split('\t')[0]) for i in range(5)] +
                                       [float(manu_info_list[i].split('\t')[1]) for i in range(5)])

        self.face_files = face_files
        self.genders = genders
        self.races = races
        self.ages = ages
        self.emotions = emotions
        self.coordinates = coordinates

        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(self.face_files[idx])
        gender = self.genders[idx]
        race = self.races[idx]
        age = self.ages[idx]
        emotion = self.emotions[idx]
        coordinate = self.coordinates[idx]

        sample = {'image': image, 'gender': gender, 'race': race, 'age': age, 'emotion': emotion,
                  'coordinate': np.array(coordinate, np.float32)}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class FaceGenderDataset(Dataset):
    """
    Face Gender dataset with hierarchical sampling strategy
    """

    def __init__(self, csv_file=cfg['SCUT_FBP5500_csv'], root_dir=cfg['gender_base_dir'], transform=None,
                 male_shuffled_indices=None, female_shuffled_indices=None, train=True):
        self.root_dir = root_dir
        self.img_index = pd.read_csv(csv_file, header=None, sep=',').iloc[:, 2]
        self.img_label = pd.DataFrame(np.array([1 if _ == 'm' else 0 for _ in
                                                pd.read_csv(csv_file, header=None, sep=',').iloc[:,
                                                0].values.tolist()]).ravel())

        def get_fileindex_and_label():
            fileindex_and_label = {}
            for i in range(len(self.img_index.tolist())):
                fileindex_and_label[self.img_index.values.tolist()[i]] = self.img_label.values.tolist()[i]

            return fileindex_and_label

        m_fileindex_list = os.listdir(os.path.join(cfg['gender_base_dir'], 'M'))
        f_fileindex_list = os.listdir(os.path.join(cfg['gender_base_dir'], 'F'))

        tmp = get_fileindex_and_label()
        m_label = [tmp[_][0] for _ in m_fileindex_list]
        f_label = [tmp[_][0] for _ in f_fileindex_list]

        male_train_set_size = int(len(m_fileindex_list) * 0.6)
        female_train_set_size = int(len(f_fileindex_list) * 0.6)

        male_train_indices = male_shuffled_indices[:male_train_set_size]
        male_test_indices = male_shuffled_indices[male_train_set_size:]
        female_train_indices = female_shuffled_indices[:female_train_set_size]
        female_test_indices = female_shuffled_indices[female_train_set_size:]

        if train:
            self.image_files = pd.concat(
                [pd.DataFrame(m_fileindex_list).iloc[male_train_indices],
                 pd.DataFrame(f_fileindex_list).iloc[female_train_indices]])

            self.image_labels = pd.concat(
                [pd.DataFrame(pd.DataFrame(m_label).iloc[male_train_indices].values.ravel().tolist()),
                 pd.DataFrame(pd.DataFrame(f_label).iloc[female_train_indices].values.ravel().tolist())])
        else:
            self.image_files = pd.concat(
                [pd.DataFrame(m_fileindex_list).iloc[male_test_indices],
                 pd.DataFrame(f_fileindex_list).iloc[female_test_indices]])

            self.image_labels = pd.concat(
                [pd.DataFrame(pd.DataFrame(m_label).iloc[male_test_indices].values.ravel().tolist()),
                 pd.DataFrame(pd.DataFrame(f_label).iloc[female_test_indices].values.ravel().tolist())])

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        label = self.image_labels.values.ravel().tolist()[idx]
        img_name = os.path.join(self.root_dir, 'M' if label == 1 else 'F', self.image_files.values.tolist()[idx][0])
        image = io.imread(img_name)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class FaceRaceDataset(Dataset):
    """
    Face Race dataset with hierarchical sampling strategy
    """

    def __init__(self, csv_file=cfg['SCUT_FBP5500_csv'], root_dir=cfg['race_base_dir'], transform=None,
                 yellow_shuffled_indices=None, white_shuffled_indices=None, train=True):
        self.root_dir = root_dir
        self.img_index = pd.read_csv(csv_file, header=None, sep=',').iloc[:, 2]
        self.img_label = pd.DataFrame(np.array([1 if _ == 'w' else 0 for _ in
                                                pd.read_csv(csv_file, header=None, sep=',').iloc[:,
                                                1].values.tolist()]).ravel())

        def get_fileindex_and_label():
            fileindex_and_label = {}
            for i in range(len(self.img_index.tolist())):
                fileindex_and_label[self.img_index.values.tolist()[i]] = self.img_label.values.tolist()[i]

            return fileindex_and_label

        y_fileindex_list = os.listdir(os.path.join(cfg['race_base_dir'], 'Y'))
        w_fileindex_list = os.listdir(os.path.join(cfg['race_base_dir'], 'W'))

        tmp = get_fileindex_and_label()
        y_label = [tmp[_][0] for _ in y_fileindex_list]
        w_label = [tmp[_][0] for _ in w_fileindex_list]

        yellow_train_set_size = int(len(y_fileindex_list) * 0.6)
        white_train_set_size = int(len(w_fileindex_list) * 0.6)

        yellow_train_indices = yellow_shuffled_indices[:yellow_train_set_size]
        yellow_test_indices = yellow_shuffled_indices[yellow_train_set_size:]
        white_train_indices = white_shuffled_indices[:white_train_set_size]
        white_test_indices = white_shuffled_indices[white_train_set_size:]

        if train:
            self.image_files = pd.concat(
                [pd.DataFrame(y_fileindex_list).iloc[yellow_train_indices],
                 pd.DataFrame(w_fileindex_list).iloc[white_train_indices]])

            self.image_labels = pd.concat(
                [pd.DataFrame(pd.DataFrame(y_label).iloc[yellow_train_indices].values.ravel().tolist()),
                 pd.DataFrame(pd.DataFrame(w_label).iloc[white_train_indices].values.ravel().tolist())])
        else:
            self.image_files = pd.concat(
                [pd.DataFrame(y_fileindex_list).iloc[yellow_test_indices],
                 pd.DataFrame(w_fileindex_list).iloc[white_test_indices]])

            self.image_labels = pd.concat(
                [pd.DataFrame(pd.DataFrame(y_label).iloc[yellow_test_indices].values.ravel().tolist()),
                 pd.DataFrame(pd.DataFrame(w_label).iloc[white_test_indices].values.ravel().tolist())])

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        label = self.image_labels.values.ravel().tolist()[idx]
        img_name = os.path.join(self.root_dir, 'W' if label == 1 else 'Y', self.image_files.values.tolist()[idx][0])
        image = io.imread(img_name)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class FBPDataset(Dataset):
    """
    SCUT-FBP5500 dataset
    """

    def __init__(self, train=True, transform=None):
        if train:
            self.face_img = pd.read_csv(os.path.join(cfg['4_6_split_dir'], 'train.txt'), sep=' ', header=None).iloc[:,
                            0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['4_6_split_dir'], 'train.txt'), sep=' ', header=None).iloc[:,
                              1].astype(np.float).tolist()
        else:
            self.face_img = pd.read_csv(os.path.join(cfg['4_6_split_dir'], 'test.txt'), sep=' ', header=None).iloc[:,
                            0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['4_6_split_dir'], 'test.txt'), sep=' ', header=None).iloc[:,
                              1].astype(np.float).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.face_img)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scutfbp5500_images_dir'], self.face_img[idx]))
        score = self.face_score[idx]
        sample = {'image': image, 'score': score}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class FaceDataset(Dataset):
    """
    Face Dataset for SCUT-FBP5500 with 5-Fold CV
    """

    def __init__(self, cv_index=1, train=True, transform=None):
        if train:
            self.face_img = pd.read_csv(
                os.path.join(cfg['cv_split_base_dir'], 'cross_validation_%d' % cv_index, 'train_%d.txt' % cv_index),
                sep=' ', header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['cv_split_base_dir'], 'cross_validation_%d' % cv_index,
                                                       'train_%d.txt' % cv_index),
                                          sep=' ', header=None).iloc[:, 1].astype(np.float).tolist()
        else:
            self.face_img = pd.read_csv(
                os.path.join(cfg['cv_split_base_dir'], 'cross_validation_%d' % cv_index, 'test_%d.txt' % cv_index),
                sep=' ',
                header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['cv_split_base_dir'], 'cross_validation_%d' % cv_index,
                                                       'test_%d.txt' % cv_index), sep=' ', header=None).iloc[:, 1] \
                .astype(np.float).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.face_img)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scutfbp5500_images_dir'], self.face_img[idx]))
        attractiveness = self.face_score[idx]
        gender = 1 if self.face_img[idx].split('.')[0][0] == 'm' else 0
        race = 1 if self.face_img[idx].split('.')[0][2] == 'w' else 0

        sample = {'image': image, 'attractiveness': attractiveness, 'gender': gender, 'race': race}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class FDataset(Dataset):
    """
    Face Dataset for SCUT-FBP5500 with 6/4 split CV
    """

    def __init__(self, train=True, transform=None):
        if train:
            self.face_img = pd.read_csv(
                os.path.join(cfg['4_6_split_dir'], 'train.txt'),
                sep=' ', header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['4_6_split_dir'], 'train.txt'), sep=' ', header=None).iloc[:,
                              1].astype(np.float).tolist()
        else:
            self.face_img = pd.read_csv(
                os.path.join(cfg['4_6_split_dir'], 'test.txt'),
                sep=' ', header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['4_6_split_dir'], 'test.txt'), sep=' ', header=None).iloc[:,
                              1].astype(np.float).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.face_img)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scutfbp5500_images_dir'], self.face_img[idx]))
        attractiveness = self.face_score[idx]
        gender = 1 if self.face_img[idx].split('.')[0][0] == 'm' else 0
        race = 1 if self.face_img[idx].split('.')[0][2] == 'w' else 0

        sample = {'image': image, 'attractiveness': attractiveness, 'gender': gender, 'race': race}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class ScutFBP(Dataset):
    """
    SCUT-FBP dataset
    """

    def __init__(self, f_list, f_labels, transform=None):
        self.face_files = f_list
        self.face_score = f_labels.tolist()

        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scutfbp_images_dir'], self.face_files[idx]))
        score = self.face_score[idx]

        sample = {'image': image, 'attractiveness': score, 'gender': 0, 'race': 0}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class HotOrNotDataset(Dataset):
    """
    ECCV HorOrNot Dataset
    """

    def __init__(self, cv_split=1, train=True, transform=None):
        df = pd.read_csv(
            os.path.join(os.path.split(os.path.abspath(cfg['hotornot_dir']))[0], 'eccv2010_split%d.csv' % cv_split),
            header=None)

        filenames = [os.path.join(cfg['hotornot_dir'], _.replace('.bmp', '.jpg')) for
                     _ in df.iloc[:, 0].tolist()]
        scores = df.iloc[:, 1].tolist()
        flags = df.iloc[:, 2].tolist()

        train_set = OrderedDict()
        test_set = OrderedDict()

        for i in range(len(flags)):
            if flags[i] == 'train':
                train_set[filenames[i]] = scores[i]
            elif flags[i] == 'test':
                test_set[filenames[i]] = scores[i]

        if train:
            self.face_files = list(train_set.keys())
            self.face_scores = list(train_set.values())
        else:
            self.face_files = list(test_set.keys())
            self.face_scores = list(test_set.values())

        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(self.face_files[idx])
        score = self.face_scores[idx]

        sample = {'image': image, 'score': score, 'filename': os.path.basename(self.face_files[idx])}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
