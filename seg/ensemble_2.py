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
from multiprocessing import Pool, Value

from settings import DATA_DIR, TEST_IMG_DIR
from utils import get_image_size, parallel_apply, encode_binary_mask, general_ensemble

# lb4590
#pred_files = [
#    '../work_dirs/htc_level1_275/preds_0918pm_all_lb4478.pkl',
#    '../work_dirs/htc_level1_275/preds_0916pm_all_lb4456.pkl',
#    '../preds_cas275_0918pm_lb4279.pkl'
#]
#ens_weights = [0.5, 0.2, 0.3]

# lb 4607
#pred_files = [
#    '../work_dirs/htc_level1_275/preds_0918pm_all_lb4478.pkl',
#    '../work_dirs/htc_level1_275/preds_0916pm_all_lb4456.pkl',
#    '../preds_0902_3_50_all_lb04195.pkl',
#    '../preds_cas275_0918pm_lb4279.pkl',
#    '../work_dirs/htc_level1_275/preds_0913pm_all_lb4343.pkl'
#]
#ens_weights = [0.35, 0.22, 0.15, 0.18, 0.1]

#lb4645
#pred_files = [
#    '../work_dirs/htc_level1_275/preds_0919pm_all_lb4495.pkl',
#    '../work_dirs/htc_level1_275/preds_0918pm_all_lb4478.pkl',
#    '../work_dirs/htc_level1_275/preds_0916pm_all_lb4456.pkl',
#    '../preds_cas275_0918pm_lb4279.pkl',
#    '../preds_cas275_0919pm_all.pkl'
#]
#ens_weights = [0.35, 0.2, 0.12, 0.15, 0.18]

#pred_files = [
#    '../work_dirs/htc_level1_275/preds_0919pm_all_lb4495.pkl',
#    '../work_dirs/htc_level1_275/preds_0918pm_all_lb4478.pkl',
#    '../work_dirs/htc_level1_275/preds_0916pm_all_lb4456.pkl',
#    '../work_dirs/htc_level1_275/preds_0917pm_all_683_lb4436.pkl',
#    '../preds_cas275_0919pm_all_lb4351.pkl',
#    '../preds_cas275_0918pm_lb4279.pkl',
#    '../preds_cas275_0917pm_lb4248.pkl'
#]
#ens_weights = [0.3, 0.13, 0.12, 0.1, 0.15, 0.1, 0.1]

#pred_files = [
#    '../work_dirs/htc_level1_275/preds_0921_all_lb4521.pkl',
#    '../work_dirs/htc_level1_275/preds_0919pm_all_lb4495.pkl',
#    '../work_dirs/htc_level1_275/preds_0918pm_all_lb4478.pkl',
#    '../work_dirs/htc_level1_275/preds_0916pm_all_lb4456.pkl',
#    '../preds_cas275_0919pm_all_lb4351.pkl',
#    '../work_dirs/cas_level1_275/preds_cas275_0921_all_lb4349.pkl',
#    '../preds_cas275_0918pm_lb4279.pkl'
#]
#ens_weights = [0.3, 0.13, 0.12, 0.1, 0.15, 0.1, 0.1]

# lb5026 - 0.0335 =  4691
#pred_files = [
#    '../work_dirs/htc_level1_275/preds_0922pm_all_lb4560.pkl',
#    '../work_dirs/htc_level1_275/preds_0921_all_lb4521.pkl',
#    '../work_dirs/htc_level1_275/preds_0919pm_all_lb4495.pkl',
#    '../work_dirs/htc_level1_275/preds_0918pm_all_lb4478.pkl',
#    '../preds_cas275_0919pm_all_lb4351.pkl',
#    '../preds_cas275_0918pm_lb4279.pkl',
#    '../preds_cas275_0917pm_lb4248.pkl'
#]
#ens_weights = [0.25, 0.20, 0.15, 0.1, 0.1, 0.1, 0.1]

# no memory to ensmble 9 models
pred_files = [
    '../work_dirs/htc_level1_275/preds_0922pm_all_lb4560.pkl',
    '../work_dirs/htc_level1_275/preds_0921_all_lb4521.pkl',
    '../work_dirs/htc_level1_275/preds_0919pm_all_lb4495.pkl',
    '../work_dirs/htc_level1_275/preds_0918pm_all_lb4478.pkl',
    '../work_dirs/htc_level1_275/preds_0917pm_all_683_lb4436.pkl',
    '../preds_cas275_0919pm_all_lb4351.pkl',
    '../preds_cas275_0918pm_lb4279.pkl',
    '../preds_cas275_0917pm_lb4248.pkl',
    '../preds_0902_3_50_all_lb04195.pkl'
]
ens_weights = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# lb4593
#pred_files = [
#    '../work_dirs/htc_level1_275/preds_0917pm_all_683_lb4436.pkl',
#    '../preds_cas275_0919pm_all_lb4351.pkl',
#    '../preds_cas275_0918pm_lb4279.pkl',
#    '../preds_cas275_0917pm_lb4248.pkl',
#    '../preds_0902_3_50_all_lb04195.pkl'
#]
#ens_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

#lb 4574
#pred_files = [
#    '../work_dirs/htc_level1_275/preds_0922pm_all_lb4560.pkl',
#    '../work_dirs/htc_level1_275/preds_0921_all_lb4521.pkl',
#    '../work_dirs/htc_level1_275/preds_0919pm_all_lb4495.pkl',
#    '../work_dirs/htc_level1_275/preds_0918pm_all_lb4478.pkl',
#    '../work_dirs/htc_level1_275/preds_0925am_all_lb4549.pkl'
#]
#ens_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

#lb0.4693
#pred_files = [
#    '../work_dirs/htc_level1_275/preds_0922pm_all_lb4560.pkl',
#    '../work_dirs/htc_level1_275/preds_0921_all_lb4521.pkl',
#    '../work_dirs/htc_level1_275/preds_0919pm_all_lb4495.pkl',
#    '../work_dirs/htc_level1_275/preds_0917pm_all_683_lb4436.pkl',
#    '../preds_cas275_0919pm_all_lb4351.pkl',
#    '../preds_cas275_0918pm_lb4279.pkl',
#    '../preds_0902_3_50_all_lb04195.pkl'
#]
#ens_weights = [0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.1]




all_preds, ens_dets, classes = [], [], None
bg_time = None
counter = None

MAX_NUM = 160

def init(init_value):
    ''' store the counter for later use '''
    global counter
    counter = init_value


def get_top_classes(start_index, end_index, class_file='top_classes_level1.csv'):
    df = pd.read_csv(osp.join(DATA_DIR, class_file))
    c = df['class'].values[start_index:end_index]
    #print(df.head())
    stoi = { c[i]: i for i in range(len(c)) }
    return c, stoi

def get_fn(img_id):
    return TEST_IMG_DIR + '/' + img_id + '.jpg'

def get_dets(preds, idx):
    print('index:', idx)
    box_pred, mask_pred = preds[idx]
    dets = []
    for i in range(len(classes)):
        for encoded_mask, bbox in zip(mask_pred[i], box_pred[i]):
            if True:
                mask = mutils.decode(encoded_mask)
                # bbox[4] is confidence
                dets.append([mask, classes[i], bbox[4]])
                
    return dets

def get_ens_det(idx):
    dets = [get_dets(p, idx) for p in all_preds]
    
    ens_det = general_ensemble(dets, weights=ens_weights)
    ens_det = [[encode_binary_mask(x[0].astype(np.bool)), x[1], x[2]] for x in ens_det]

    res = sorted(ens_det, key=lambda x: x[2], reverse=True)[:MAX_NUM]
    del dets

    global counter
    # += operation is not atomic, so we need to get a lock:
    with counter.get_lock():
        counter.value += 1
    if counter.value % 100 == 0:
        print('counter: {}, {:.2f}'.format(counter.value, (time.time()-bg_time)/60))
    return res

def get_pred_str(idx):
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
    global all_preds, classes, ens_dets, MAX_NUM, bg_time
    MAX_NUM = args.max_num

    #print('getting img size...')
    df_test = pd.read_csv(osp.join(DATA_DIR, 'sample_empty_submission.csv'))
    print(df_test.head())
    #df_test.ImageWidth = df_test.ImageID.map(lambda x: get_image_size(get_fn(x))[0])
    #df_test.ImageHeight = df_test.ImageID.map(lambda x: get_image_size(get_fn(x))[1])

    classes, _ = get_top_classes(args.start_index, args.end_index, args.class_file)
    for fn in pred_files:
        print('loading {} ...'.format(fn))
        with open(fn, 'rb') as f:
            all_preds.append(pickle.load(f))

    print('len(preds):', len(all_preds[0]))
    print('num classes of preds:', len(all_preds[0][0][1]))
    print('specified num classes:', len(classes))
    #assert len(preds[0][1]) == len(classes)

    print('ensembling...')
    bg_time = time.time()
    counter = Value('i', 0)

    with Pool(24, initializer=init, initargs=(counter, )) as p:
        num_imgs = len(all_preds[0])
        #ens_dets = list(tqdm(iterable=p.map(get_ens_det, list(range(num_imgs))), total=num_imgs))
        ens_dets = p.map(get_ens_det, range(num_imgs))
    #num_imgs = len(preds1)
    #for idx in tqdm(range(num_imgs), total=num_imgs):
    #    ens_dets.append(get_ens_det(idx))

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
    parser.add_argument('--max_num', type=int, default=160)

    args = parser.parse_args()
    print(args)

    ensemble(args)
