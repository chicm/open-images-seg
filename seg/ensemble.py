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
from multiprocessing import Pool

from settings import DATA_DIR, TEST_IMG_DIR
from utils import get_image_size, parallel_apply, encode_binary_mask, general_ensemble

preds1, preds2, classes, ens_dets = None, None, None, []
preds3 = None

def get_top_classes(start_index, end_index, class_file='top_classes_level1.csv'):
    df = pd.read_csv(osp.join(DATA_DIR, class_file))
    c = df['class'].values[start_index:end_index]
    #print(df.head())
    stoi = { c[i]: i for i in range(len(c)) }
    return c, stoi

def get_fn(img_id):
    return TEST_IMG_DIR + '/' + img_id + '.jpg'

def get_dets(preds, idx):
    box_pred, mask_pred = preds[idx]
    #masks, labels, confs = [], [], []
    dets = []
    for i in range(len(classes)):
        for encoded_mask, bbox in zip(mask_pred[i], box_pred[i]):
            if True:
                mask = mutils.decode(encoded_mask)
                #masks.append(mask)
                #labels.append(classes[i])
                #confs.append(bbox[4])
                dets.append([mask, classes[i], bbox[4]])
                
    return dets

def get_ens_det(idx):
    det1 = get_dets(preds1, idx)
    det2 = get_dets(preds2, idx)
    det3 = get_dets(preds3, idx)
    
    ens_det = general_ensemble([det1, det2, det3], weights=[0.5, 0.3, 0.2])
    ens_det = [[encode_binary_mask(x[0].astype(np.bool)), x[1], x[2]] for x in ens_det]

    #del det1, det2
    return sorted(ens_det, key=lambda x: x[2], reverse=True)[:50]

def get_pred_str(idx):
    #masks, labels, confs = get_mask(idx)
    res = []
    #for mask, label, conf in zip(masks, labels, confs):
    for det in ens_dets[idx]:
        res.append(det[1])
        res.append('{:.7f}'.format(det[2]))
        #res.append(encode_binary_mask(det[0].astype(np.bool)))
        res.append(det[0])
    
    return ' '.join(res)

def set_pred_str(df):
    df['PredictionString'] = df.img_index.map(lambda x: get_pred_str(x))
    return df

def submit(args):
    global preds1, preds2, preds3, classes, ens_dets

    classes, _ = get_top_classes(args.start_index, args.end_index, args.class_file)
    #print('loading {}...'.format(args.pred_file))
    #with open(args.pred_file, 'rb') as f:
    #    preds = pickle.load(f)
    print('loading...')
    with open('../work_dirs/htc_level1_275/preds_0913pm_all_lb4343.pkl', 'rb') as f:
        preds1 = pickle.load(f)
    with open('../preds_0902_3_50_all_lb04195.pkl', 'rb') as f:
        preds2 = pickle.load(f)
    with open('../work_dirs/htc_level1_275/preds_0913am_all_lb4304.pkl', 'rb') as f:
        preds3 = pickle.load(f)

    print('len(preds):', len(preds1))
    print('num classes of preds:', len(preds1[0][1]))
    print('specified num classes:', len(classes))
    #assert len(preds[0][1]) == len(classes)

    with Pool(24) as p:
        num_imgs = len(preds1)
        #ens_dets = list(tqdm(iterable=p.map(get_ens_det, list(range(num_imgs))), total=num_imgs))
        ens_dets = p.map(get_ens_det, range(num_imgs))
    #num_imgs = len(preds1)
    #for idx in tqdm(range(num_imgs), total=num_imgs):
    #    ens_dets.append(get_ens_det(idx))

    print('getting img size...')
    df_test = pd.read_csv(osp.join(DATA_DIR, 'sample_empty_submission.csv'))
    df_test.ImageWidth = df_test.ImageID.map(lambda x: get_image_size(get_fn(x))[0])
    df_test.ImageHeight = df_test.ImageID.map(lambda x: get_image_size(get_fn(x))[1])

    print('creating submission...')

    df_test['img_index'] = df_test.index
    df_test = parallel_apply(df_test, set_pred_str)
    df_test = df_test.drop(columns=['img_index'], axis=1)

    df_test.to_csv(args.out, index=False)
    print('done')

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create submission from pred file')
    #parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--th', type=float, default=0.)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=275)
    parser.add_argument('--class_file', type=str, default='top_classes_level1.csv')
    args = parser.parse_args()
    print(args)

    submit(args)
