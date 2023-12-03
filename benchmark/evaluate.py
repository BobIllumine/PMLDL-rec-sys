import os
import sys
from pathlib import Path
src_path = str(Path(__file__).parent.parent.resolve())
if src_path not in sys.path:
    sys.path.append(src_path)
from src.models.wide_deep import WideDeepModel
from src.utils.data import load_movielens
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--user_id', type=int, required=True)
    parser.add_argument('-k', type=int, default=10)
    parser.add_argument('--use_checkpoints', action='store_true')
    args = parser.parse_args()
    out_path, user, k, checkpoints = args.output, args.user_id, args.k, args.use_checkpoints
    data = load_movielens('100k')
    model = WideDeepModel(data, split_ratio=0.9)
    
    if not checkpoints:
        model.train()
    
    prediction, metrics = model.predict(k = k)

    topk_items = prediction[prediction['userID'] == user].sort_values(by='prediction', ascending=False)[:k]
    titles = load_movielens('100k', True)[['itemID', 'title']]
    result = pd.merge(topk_items, titles, how="inner", on="itemID").drop_duplicates('itemID').reset_index(drop=True)
    os.makedirs(Path(__file__).parent / 'results', exist_ok=True)
    result.to_csv(Path(__file__).parent / 'results' / f'top-{k}_{user}.csv', sep='|', header=True, index=False)
    print(metrics)
    
