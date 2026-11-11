"""Run PVBoost on TSB-AD-U / TSB-AD-M with precomputed source-detector scores.

Writes one CSV per (detector, seed) with TSB-AD's metrics on the returned score (VUS-PR, AUC-PR, ...)
and the point-wise F1 of PVBoost's own decisions.

Example:
    python benchmark/run_tsbad.py --benchmark U --detector Sub_IForest --seed 0 \
        --dataset_dir TSB-AD/Datasets/TSB-AD-U --score_dir scores/uni \
        --file_list TSB-AD/Datasets/File_List/TSB-AD-U-Eva.csv --save_dir results
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pvboost import PVBoost, PVBoostConfig  # noqa: E402


def str2bool(x):
    return x.lower() in ('true', '1', 'yes')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--benchmark', required=True, choices=['U', 'M'])
    p.add_argument('--detector', required=True,
                   help='source detector; scores are read from <score_dir>/<detector>/<series>.npy')
    p.add_argument('--dataset_dir', required=True)
    p.add_argument('--score_dir', required=True)
    p.add_argument('--file_list', required=True, help="CSV with a 'file_name' column")
    p.add_argument('--save_dir', default='results')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default=None, help="e.g. 'cuda:0' or 'cpu'; default: cuda:0 if available")
    p.add_argument('--refine_select', type=str2bool, default=True,
                   help='select the refinement strength (default); False returns the fully refined score')
    p.add_argument('--threshold_rule', default='frag_knee', choices=['frag_knee', 'posterior_mass'])
    p.add_argument('--save_scores', type=str2bool, default=False,
                   help='also save the returned score of every series under <save_dir>/scores/')
    p.add_argument('--resume', type=str2bool, default=True, help='skip series already in the result CSV')
    args = p.parse_args()

    from TSB_AD.evaluation.metrics import get_metrics  # imported here so --help works without TSB-AD

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    tag = f'PVBoost_{args.detector}_seed{args.seed}'
    if not args.refine_select:
        tag += '_full'
    if args.threshold_rule != 'frag_knee':
        tag += f'_{args.threshold_rule}'
    out_dir = os.path.join(args.save_dir, f'TSB-AD-{args.benchmark}')
    os.makedirs(out_dir, exist_ok=True)
    result_csv = os.path.join(out_dir, f'{tag}.csv')
    score_out = os.path.join(out_dir, 'scores', tag)
    if args.save_scores:
        os.makedirs(score_out, exist_ok=True)

    rows = pd.read_csv(result_csv).to_dict('records') if args.resume and os.path.exists(result_csv) else []
    done = {r['file'] for r in rows}
    cfg = PVBoostConfig(seed=args.seed, device=args.device, refine_select=args.refine_select,
                        threshold_rule=args.threshold_rule)

    files = pd.read_csv(args.file_list)['file_name'].tolist()
    for i, filename in enumerate(files):
        if filename in done:
            continue
        stem = filename.split('.')[0]
        score_path = os.path.join(args.score_dir, args.detector, stem + '.npy')
        if not os.path.exists(score_path):
            raise FileNotFoundError(f'missing source score {score_path}')
        df = pd.read_csv(os.path.join(args.dataset_dir, filename)).dropna()
        X = df.iloc[:, 0:-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        source = np.load(score_path)

        t0 = time.time()
        res = PVBoost(cfg).fit_predict(X, source)
        run_time = time.time() - t0

        score = np.nan_to_num(res.score)
        metrics = get_metrics(score, label, slidingWindow=100)
        row = {'file': filename, 'Time': run_time, **metrics,
               'F1': f1_score(label, res.labels, zero_division=0),
               'pi_hat': res.anomaly_rate, 'pi_star': float(label.mean()), 'a_star': res.a_star,
               'n_rounds': res.n_rounds}
        rows.append(row)
        pd.DataFrame(rows).to_csv(result_csv, index=False)
        if args.save_scores:
            np.save(os.path.join(score_out, stem + '.npy'), score.astype(np.float32))
        print(f'[{i + 1}/{len(files)}] {filename}: VUS-PR {metrics["VUS-PR"]:.4f}  F1 {row["F1"]:.4f}  '
              f'a*={res.a_star}  ({run_time:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
