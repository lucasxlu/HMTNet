import os
from collections import OrderedDict

cfg = OrderedDict()

cfg['raf_root'] = 'E:/DataSet/CV/TreeCNN/RAF-Face'

cfg['scut_fbp5500_root'] = 'E:/DataSet/Face/SCUT-FBP5500/'

cfg['scutfbp_images_dir'] = os.path.join(os.path.abspath(os.path.dirname(
    cfg['scut_fbp5500_root']) + os.path.sep + "..") + '/SCUT-FBP/Crop')
cfg['scutfbp_excel'] = os.path.join(os.path.abspath(os.path.dirname(
    cfg['scut_fbp5500_root']) + os.path.sep + "..") + '/SCUT-FBP/Rating_Collection/AttractivenessLabel.xlsx')
cfg['hotornot_dir'] = 'E:/DataSet/CV/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/hotornot_face'

cfg['scutfbp5500_images_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'Images')
cfg['gender_base_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'Gender')
cfg['race_base_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'Race')

cfg['SCUT_FBP5500_csv'] = os.path.join(cfg['scut_fbp5500_root'], 'train_test_files/SCUT-FBP5500.csv')
cfg['cv_split_base_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'train_test_files/5_folders_cross_validations_files')
cfg['4_6_split_dir'] = os.path.join(cfg['scut_fbp5500_root'], 'train_test_files/split_of_60%training and 40%testing')
cfg['batch_size'] = 32
cfg['pretrained_vgg_face'] = 'E:/ModelZoo/vgg_m_face_bn_dag.pth'
cfg['dlib_model'] = os.path.join(
    os.path.dirname(cfg['pretrained_vgg_face'])) + '/shape_predictor_68_face_landmarks.dat'
