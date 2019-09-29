import pandas as pd

def merge_df(args):
    # lb5050
    #csv_files = [
    #    'ens_0928_htc275_cas275_od275.csv',
    #    '../sub_htc_parent_0927am_lb0338.csv'
    #]
    
    csv_files = [
        'ens_0929_1_5models_top145.csv',
        '../sub_htc_parent_0929am_iou02_rpn2000_top20_lb0331.csv'
    ]

    print('loading {} ...'.format(csv_files))
    dfs = [pd.read_csv(x) for x in csv_files]

    for df in dfs:
        df.PredictionString = df.PredictionString.fillna('')

    print('merging...')

    x = None
    for d in dfs:
        if x is None:
            x = d.PredictionString
        else:
            x = x.str.cat(d.PredictionString, sep=' ')
    
    p = x.map(lambda s: ' '.join(s.split()))
    dfs[0].PredictionString = p

    print('saving {} ...'.format(args.out))
    dfs[0].to_csv(args.out, index=False)

    msg = 'kaggle competitions submit -c open-images-2019-instance-segmentation -f {} -m "submit"'.format(args.out)
    print(msg)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='merge df')
    parser.add_argument('--out', type=str, required=True)

    args = parser.parse_args()
    print(args)

    merge_df(args)
