"""Summarize run_tsbad.py results: per-detector VUS-PR, AUC-PR and F1 (mean over series, then over seeds).

    python benchmark/summarize.py --save_dir results
"""
import argparse
import glob
import os
import re

import pandas as pd

# TSB-AD score-directory name -> name used in the paper.
DISPLAY = {'Sub_LOF': 'LOF', 'Sub_KNN': 'KNN', 'KMeansAD_U': 'KMeansAD', 'Sub_OCSVM': 'OCSVM',
           'Sub_HBOS': 'HBOS', 'Sub_IForest': 'IForest', 'Sub_PCA': 'PCA',
           'AnomalyTransformer': 'AnomalyTrans.', 'Chronos2': 'Chronos-2', 'Toto2': 'Toto-2.0'}
METRICS = ['VUS-PR', 'AUC-PR', 'F1']


def summarize(bench_dir, suffix=''):
    pat = re.compile(rf'^PVBoost_(.+)_seed(\d+){re.escape(suffix)}\.csv$')
    per_seed = []
    for f in sorted(glob.glob(os.path.join(bench_dir, 'PVBoost_*.csv'))):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        df = pd.read_csv(f)
        per_seed.append({'detector': DISPLAY.get(m.group(1), m.group(1)), 'seed': int(m.group(2)),
                         'n_series': len(df), **df[METRICS].mean().to_dict()})
    if not per_seed:
        return None
    P = pd.DataFrame(per_seed)
    T = P.groupby('detector').agg(seeds=('seed', 'nunique'), n_series=('n_series', 'min'),
                                  **{m: (m, 'mean') for m in METRICS})
    T.loc['Avg'] = [T.seeds.min(), T.n_series.min(), *T[METRICS].mean()]
    return T


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--save_dir', default='results')
    p.add_argument('--suffix', default='', help="e.g. '_full' for --refine_select False runs")
    a = p.parse_args()
    for b in ('U', 'M'):
        T = summarize(os.path.join(a.save_dir, f'TSB-AD-{b}'), a.suffix)
        if T is None:
            continue
        print(f'\nTSB-AD-{b}')
        print(T.to_string(float_format=lambda v: f'{v:.3f}'))
        T.to_csv(os.path.join(a.save_dir, f'summary_TSB-AD-{b}{a.suffix}.csv'))


if __name__ == '__main__':
    main()
