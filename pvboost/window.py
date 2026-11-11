"""Period-based window length for the weak learner's lagged features.

Adapted from TSB-AD (``TSB_AD.utils.slidingWindows.find_length_rank``,
https://github.com/thedatumorg/TSB-AD, Apache-2.0), with ``rank=1``.
"""
import numpy as np
from scipy.signal import argrelextrema
from statsmodels.tsa.stattools import acf


def find_period(x):
    """Dominant period of a 1-d series from its autocorrelation; 125 if none is found or it exceeds 300."""
    x = np.asarray(x).squeeze()
    if x.ndim > 1:
        return 0
    x = x[:min(20000, len(x))]
    base = 3
    auto_corr = acf(x, nlags=400, fft=True)[base:]
    local_max = argrelextrema(auto_corr, np.greater)[0]
    try:
        best = np.argsort([auto_corr[i] for i in local_max])[::-1][0]
        period = local_max[best] + base
        return 125 if period > 300 else int(period)
    except Exception:
        return 125
