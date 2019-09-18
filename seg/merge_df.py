import pandas as pd

def merge_df(args):
    csv_files = [
        'ens_0918_100.csv', # lb4590
        '../sub_htc_parent_0918pm_all.csv' #,
        #'sub_0902_3_top50.csv.zip',
        #'sub_parent_0903_1_top50.csv'
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

    dfs[0].PredictionString = x

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
