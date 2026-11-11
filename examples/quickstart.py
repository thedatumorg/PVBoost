"""PVBoost on a synthetic series, with Isolation Forest on sliding windows as the source detector.

    python examples/quickstart.py
"""
import os
import sys

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pvboost import PVBoost  # noqa: E402

rng = np.random.default_rng(0)
T, period = 4000, 50
t = np.arange(T)
x = np.sin(2 * np.pi * t / period) + 0.1 * rng.standard_normal(T)
label = np.zeros(T, dtype=int)
for start in (900, 2100, 3300):                      # three anomalous segments
    x[start:start + 40] += 0.8 * rng.standard_normal(40)
    label[start:start + 40] = 1

# Source detector: Isolation Forest on windows of one period, score assigned to the window's last point.
W = period
windows = np.lib.stride_tricks.sliding_window_view(x, W)
s = -IsolationForest(random_state=0).fit(windows).score_samples(windows)
source = np.concatenate([np.full(W - 1, s[0]), s])

# Source decisions with the usual 3-sigma rule, for comparison.
src_pred = (source > source.mean() + 3 * source.std()).astype(int)

res = PVBoost(seed=0).fit_predict(x, source)

print(f'source : AUC-PR {average_precision_score(label, source):.3f}   F1 (3-sigma) {f1_score(label, src_pred):.3f}')
print(f'PVBoost: AUC-PR {average_precision_score(label, res.score):.3f}   F1           {f1_score(label, res.labels):.3f}')
print(f'pi_hat = {res.anomaly_rate:.4f} (true {label.mean():.4f}), a* = {res.a_star}, rounds = {res.n_rounds}')
