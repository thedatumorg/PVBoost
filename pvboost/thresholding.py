"""Anomaly-rate estimation and refinement-strength selection.

Two label-free decisions are made here:

* ``estimate_rate`` sets the anomaly rate pi_hat used for the threshold. The posterior
  proposes a mass, mean(q), and a temporal fragmentation test on the score caps it:
  pi_hat = min(clip(mean(q), lo, hi), r_frag(s)).
* ``select_refinement`` decides how much of the refinement to keep. It scores blends of
  the source and refined scores and keeps the one whose thresholded decisions are the
  most temporally coherent.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Candidate rates for the fragmentation test: 36 log-spaced rates in [0.002, 0.40].
RATE_GRID = np.logspace(np.log10(0.002), np.log10(0.40), 36)

RULES = ('frag_knee', 'posterior_mass')

# Refinement strengths a; a = 0 is the source score and a = 1 the fully refined score.
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def zscore(s):
    s = np.asarray(s, np.float64)
    sd = s.std()
    return (s - s.mean()) / (sd if sd > 1e-12 else 1.0)


def mmz(x):
    """Min-max scaling followed by z-scoring."""
    x = np.asarray(x, np.float64).reshape(-1, 1)
    return zscore(MinMaxScaler().fit_transform(x).reshape(-1))


def n_runs(mask):
    """Number of maximal runs of True in a boolean vector."""
    m = mask.astype(np.int8)
    return int(np.count_nonzero(np.diff(np.concatenate(([0], m, [0]))) == 1))


def frag_curve(s, grid=RATE_GRID):
    """Observed number of runs C(r) and its random-placement expectation E_null(r) on the grid."""
    n = s.size
    thr = np.quantile(s, 1.0 - grid)
    C = np.empty(grid.size)
    E = np.empty(grid.size)
    for i, t in enumerate(thr):
        m = s >= t
        k = int(m.sum())
        C[i] = n_runs(m)
        E[i] = k * (n - k + 1) / n if k else 1.0
    return C, np.maximum(E, 1e-9)


def frag_knee_rate(s, lo=0.001, grid=RATE_GRID):
    """Rate at which the flagged points are the least fragmented relative to chance."""
    C, E = frag_curve(s, grid)
    return float(np.clip(grid[int(np.argmin(C / E))], lo, 1.0))


def estimate_rate(q, s, rule='frag_knee', lo=0.001, hi=0.15):
    """Return (pi_hat, parts), where parts holds the two terms.

    q : anomaly posterior per time step.
    s : score in time order (any affine scale).
    rule : 'frag_knee' (default) or 'posterior_mass' (posterior term only).
    """
    if rule not in RULES:
        raise ValueError(f'unknown threshold rule {rule!r}; expected one of {RULES}')
    mass = float(np.clip(float(np.mean(q)), lo, hi))
    parts = {'pi_mass': mass, 'pi_frag': None}
    if rule == 'posterior_mass':
        return mass, parts
    frag = frag_knee_rate(zscore(s), lo)
    parts['pi_frag'] = frag
    return min(mass, frag), parts


def decision_ce(mask):
    """C / E_null of a decision set: its number of runs over the random-placement expectation."""
    n, k = mask.size, int(mask.sum())
    E = k * (n - k + 1) / n if k else 1.0
    return n_runs(mask) / max(E, 1e-9)


def select_refinement(source, refined, q_source, q_final, rule='frag_knee', lo=0.001, hi=0.15,
                      alphas=ALPHAS):
    """Choose the refinement strength a* (the gate).

    Candidates are b_a = (1 - a) mmz(source) + a mmz(refined), with the raw source (a = 0) and
    raw refined (a = 1) scores at the endpoints. Each is thresholded with ``estimate_rate``;
    the candidate with the lowest C / E_null is kept, and ties go to the larger a.

    source, refined : source and refined scores in time order, same length.
    q_source        : anomaly posterior of the source score (used for a = 0).
    q_final         : PVBoost's final posterior (used for every a > 0).
    Returns dict(a_star, score, rate, preds, ce).
    """
    source = np.asarray(source, np.float64)
    refined = np.asarray(refined, np.float64)
    zs, zr = mmz(source), mmz(refined)
    best = None
    ces = {}
    for a in alphas:
        b = source if a == 0 else refined if a == 1 else (1 - a) * zs + a * zr
        rate, _ = estimate_rate(q_source if a == 0 else q_final, b, rule, lo, hi)
        m = b >= np.quantile(b, 1 - rate)
        ce = round(decision_ce(m), 9)
        ces[a] = ce
        if best is None or ce <= best[0]:
            best = (ce, a, b, rate, m)
    _, a, b, rate, m = best
    return dict(a_star=a, score=b, rate=float(rate), preds=m.astype(int), ce=ces)
