import os
import numpy as np
import pandas as pd
import os.path as osp
import json
import pandas as pd
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
import glob
import pickle
import pycocotools.mask as mutils

from settings import DATA_DIR, TEST_IMG_DIR
from utils import get_image_size, parallel_apply, encode_binary_mask

preds = None
classes = None

def get_top_classes(start_index, end_index, class_file='top_classes_level1.csv'):
    df = pd.read_csv(osp.join(DATA_DIR, class_file))
    c = df['class'].values[start_index:end_index]
    #print(df.head())
    stoi = { c[i]: i for i in range(len(c)) }
    return c, stoi

def get_fn(img_id):
    return TEST_IMG_DIR + '/' + img_id + '.jpg'

def get_mask(idx):
    if idx >= len(preds):
        return [], [], []

    box_pred, mask_pred = preds[idx]
    masks, labels, confs = [], [], []
    for i in range(len(classes)):
        for encoded_mask, bbox in zip(mask_pred[i], box_pred[i]):
            if True: #bbox[4] > 0.01:
                mask = mutils.decode(encoded_mask)
                masks.append(mask)
                labels.append(classes[i])
                confs.append(bbox[4])
                
                # extend to parent
                #for parent_class in parent_dict[classes[i]]:
                #    if parent_class != '/m/0bl9f':
                #        masks.append(mask)
                #        labels.append(classes[i])
                #        confs.append(bbox[4])
                
    return masks, labels, confs

def get_pred_str(idx):
    masks, labels, confs = get_mask(idx)
    #if len(masks) < 1:
    #    return ''
    res = []
    for mask, label, conf in zip(masks, labels, confs):
        res.append(label)
        res.append('{:.7f}'.format(conf))
        res.append(encode_binary_mask(mask.astype(np.bool)))
    
    return ' '.join(res)

def set_pred_str(df):
    df['PredictionString'] = df.img_index.map(lambda x: get_pred_str(x))
    return df

def submit(args):
    global preds, classes

    classes, _ = get_top_classes(args.start_index, args.end_index, args.class_file)
    print('loading {}...'.format(args.pred_file))
    with open(args.pred_file, 'rb') as f:
        preds = pickle.load(f)

    print('len(preds):', len(preds))
    print('num classes of preds:', len(preds[0][1]))
    print('specified num classes:', len(classes))
    #assert len(preds[0][1]) == len(classes)
    
    print('creating submission...')
    df_test = pd.read_csv(osp.join(DATA_DIR, 'sample_empty_submission.csv'))
    df_test.ImageWidth = df_test.ImageID.map(lambda x: get_image_size(get_fn(x))[0])
    df_test.ImageHeight = df_test.ImageID.map(lambda x: get_image_size(get_fn(x))[1])
    df_test['img_index'] = df_test.index
    df_test = parallel_apply(df_test, set_pred_str)
    df_test = df_test.drop(columns=['img_index'], axis=1)

    df_test.to_csv(args.out, index=False)
    print('done')

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create submission from pred file')
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--th', type=float, default=0.)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=275)
    parser.add_argument('--class_file', type=str, default='top_classes_level1.csv')
    args = parser.parse_args()
    print(args)

    submit(args)
