import pandas as pd
import numpy as np
import os
import time
import base64
import numpy as np
import typing as t
import zlib
import cv2
from tqdm import tqdm
import pycocotools.mask as mutils
from pycocotools import mask as coco_mask_d
from multiprocessing import Pool, Value

from settings import DATA_DIR, TEST_IMG_DIR
from utils import get_image_size, parallel_apply, encode_binary_mask, general_ensemble

# lb 0.4712 = 0.5050 - 0.0338
#csv_files = [
#    '../sub_htc275_lb4560_top100_iou03_mask42_lb4619.csv',
#    '../sub_cas275_0919pm_all_lb4351.csv',
#    '../sub_od_convert_0928_top300_275.csv'
#]
#ens_weights = [0.5, 0.3, 0.2]


# top145 lb: 0.5088 - 0.331
#csv_files = [
#    '../sub_htc275_lb4560_rpn1500_iou25_mask50_top100_lb4632.csv',
#    '../sub_htc275_0927am_lb4533.csv',
#    '../sub_htc275_0917pm_683_lb4436.csv',
#    '../sub_cas275_lb4351_rpn1500_iou25_mask50_top100_lb4433.csv',
#    '../sub_cas275_0925am_lb4341.csv'
#]
#ens_weights = [0.4, 0.15, 0.15, 0.2, 0.1]

#lb
csv_files = [
    '../sub_htc275_lb4560_rpn1500_iou25_mask50_top100_lb4632.csv',
    '../sub_htc275_0927am_lb4533_rpn1500_iou25_mask50_top50.csv',
    '../sub_htc275_0919pm_lb4496_863_rpn1500_iou25_mask50_top50.csv',
    '../sub_htc275_0921_lb4521.csv',
    '../sub_cas275_lb4351_rpn1500_iou25_mask50_top100_lb4433.csv',
    '../sub_cas275_0925am_lb4341.csv',
    '../sub_cas275_0917pm_lb4248.csv'
]
ens_weights = [0.3, 0.15, 0.1, 0.1, 0.15, 0.1, 0.1]


dfs, ens_dets = [], []
bg_time = None

MAX_NUM = 150

counter = None

def init(init_value):
    ''' store the counter for later use '''
    global counter
    counter = init_value

def decode_mask(mask, h, w):
    mask = base64.b64decode(mask)
    mask = zlib.decompress(mask)
    decoding_dict = {
          'size': [h, w],
          'counts': mask
    }
    mask_tensor = coco_mask_d.decode(decoding_dict)
    return mask_tensor

def get_dets(df, i):
    row = df.iloc[i]
    items = row.PredictionString.strip().split(' ')
    if len(items) < 3:
        #print(items)
        return []
    dets = []
    det = []
    for i, item in enumerate(items):
        if i % 3 == 0:
            det = []
        det.append(item)
        
        if (i+1) % 3 == 0:
            mask = decode_mask(det[2], row.ImageHeight, row.ImageWidth)
            dets.append([mask, det[0], float(det[1])])

    return dets

def get_ens_det(idx):
    dets = [get_dets(df, idx) for df in dfs]

    ens_det = general_ensemble(dets, weights=ens_weights)
    ens_det = [[encode_binary_mask(x[0].astype(np.bool)), x[1], x[2]] for x in ens_det]

    res = sorted(ens_det, key=lambda x: x[2], reverse=True)[:MAX_NUM]

    global counter
    # += operation is not atomic, so we need to get a lock:
    with counter.get_lock():
        counter.value += 1
    if counter.value % 100 == 0:
        print('counter: {}, {:.2f}'.format(counter.value, (time.time()-bg_time)/60))

    return res

def get_pred_str(idx):
    if idx >= len(ens_dets):
        return ''
    res = []
    for det in ens_dets[idx]:
        res.append(det[1])
        res.append('{:.7f}'.format(det[2]))
        #res.append(encode_binary_mask(det[0].astype(np.bool)))
        res.append(det[0])

    return ' '.join(res)

def set_pred_str(df):
    df['PredictionString'] = df.img_index.map(lambda x: get_pred_str(x))
    return df

def ensemble(args):
    global dfs, ens_dets, MAX_NUM, bg_time
    MAX_NUM = args.max_num

    df_test = pd.read_csv(os.path.join(DATA_DIR, 'sample_empty_submission.csv'))
    df_test.PredictionString = df_test.PredictionString.fillna('')
    print(df_test.head())

    print('loading {} ...'.format(csv_files))
    dfs = [pd.read_csv(fn) for fn in csv_files]
    for df in dfs:
        df.PredictionString = df.PredictionString.fillna('')
    #assert len(preds[0][1]) == len(classes)
    for i in range(1, len(dfs)):
        dfs[i] = dfs[i].set_index('ImageID')
        dfs[i] = dfs[i].reindex(index=dfs[0]['ImageID'])
        dfs[i] = dfs[i].reset_index()

    print('ensembling...')
    bg_time = time.time()
    counter = Value('i', 0)
    with Pool(24, initializer=init, initargs=(counter, )) as p:
        num_imgs = len(dfs[0])
        #ens_dets = list(tqdm(iterable=p.map(get_ens_det, list(range(num_imgs))), total=num_imgs))
        ens_dets = p.map(get_ens_det, range(num_imgs))

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
    parser.add_argument('--max_num', type=int, default=150)

    args = parser.parse_args()
    print(args)

    ensemble(args)
