"""
script for data preprocessing
"""

import os
import shutil

import pandas as pd

from config.cfg import cfg


def split_by_attribute(attr_name='gender'):
    df = pd.read_csv(cfg['SCUT_FBP5500_csv'], index_col=False, header=None)
    genders = df[0].tolist()
    races = df[1].tolist()
    files = df[2].tolist()

    if attr_name == 'gender':
        f_filenames = []
        m_filenames = []
        for i, gender in enumerate(genders):
            if gender == 'f':
                f_filenames.append(os.path.join(cfg['scutfbp5500_images_dir'], files[i]))
            elif gender == 'm':
                m_filenames.append(os.path.join(cfg['scutfbp5500_images_dir'], files[i]))

        return m_filenames, f_filenames

    elif attr_name == 'race':
        w_filenames = []
        y_filenames = []
        for i, race in enumerate(races):
            if race == 'w':
                w_filenames.append(os.path.join(cfg['scutfbp5500_images_dir'], files[i]))
            elif race == 'y':
                y_filenames.append(os.path.join(cfg['scutfbp5500_images_dir'], files[i]))

        return w_filenames, y_filenames

    else:
        print('Invalid Attribute Param!!')


def process_gender_imgs():
    """
    process gender images
    :return:
    """
    m_filenames, f_filenames = split_by_attribute('gender')
    if not os.path.exists(os.path.join(cfg['gender_base_dir'], 'M')):
        os.makedirs(os.path.join(cfg['gender_base_dir'], 'M'))
    if not os.path.exists(os.path.join(cfg['gender_base_dir'], 'F')):
        os.makedirs(os.path.join(cfg['gender_base_dir'], 'F'))

    for m_f in m_filenames:
        shutil.copy(m_f, os.path.join(cfg['gender_base_dir'], 'M', os.path.basename(m_f)))

    for f_f in f_filenames:
        shutil.copy(f_f, os.path.join(cfg['gender_base_dir'], 'F', os.path.basename(f_f)))


def process_race_imgs():
    """
    process race images
    :return:
    """
    w_filenames, y_filenames = split_by_attribute('race')
    if not os.path.exists(os.path.join(cfg['race_base_dir'], 'W')):
        os.makedirs(os.path.join(cfg['race_base_dir'], 'W'))
    if not os.path.exists(os.path.join(cfg['race_base_dir'], 'Y')):
        os.makedirs(os.path.join(cfg['race_base_dir'], 'Y'))

    for w_f in w_filenames:
        shutil.copy(w_f, os.path.join(cfg['race_base_dir'], 'W', os.path.basename(w_f)))

    for y_f in y_filenames:
        shutil.copy(y_f, os.path.join(cfg['race_base_dir'], 'Y', os.path.basename(y_f)))
