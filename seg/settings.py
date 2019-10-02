import os.path as osp

ROOT_DIR = '/mnt/chicm/data/open-images'
DATA_DIR = osp.join(ROOT_DIR, 'segmentation')
IMG_DIR = osp.join(ROOT_DIR, 'train', 'imgs')
TEST_IMG_DIR = osp.join(ROOT_DIR, 'test')
MASK_DIR = osp.join(ROOT_DIR, 'masks', 'train')
