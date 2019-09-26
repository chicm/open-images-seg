import os
import numpy as np
import pandas as pd
import argparse
from utils import get_top_classes


def get_filtered_pred_string(pred_str, selected_classes):
    if len(pred_str) < 1:
        return ''
    dets = []
    det = []
    for i, e in enumerate(pred_str.split(' ')):
        if i % 3 == 0:
            det = []
        det.append(e)
        if (i+1) % 3 == 0 and det[0] in selected_classes:
            dets.append(' '.join(det))
                
    return ' '.join(dets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='filter classes')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=275)
    parser.add_argument('--class_file', type=str, default='top_classes_level1.csv')
    args = parser.parse_args()
    print(args)

    selected_classes = set(get_top_classes(args.start_index, args.end_index, args.class_file)[0])

    df = pd.read_csv(args.input_file)
    df.PredictionString = df.PredictionString.fillna('')
    df.PredictionString = df.PredictionString.map(lambda x: get_filtered_pred_string(x, selected_classes))

    df.to_csv(args.out, index=False)
