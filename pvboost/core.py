"""PVBoost: posterior-calibrated variance boosting for time-series anomaly detection."""
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler

from .model import fit_weak_learners, predict
from .thresholding import estimate_rate, mmz, select_refinement
from .window import find_period

logger = logging.getLogger(__name__)


@dataclass
class PVBoostConfig:
    seed: int = 0
    device: Optional[str] = None        # None: 'cuda:0' if available, else 'cpu'
    # Boosting loop
    n_rounds: int = 10                  # maximum boosting rounds T
    eta: float = 0.1                    # step size of f_{t+1} = f_t + eta * h_t
    ema_beta: float = 0.7               # EMA momentum of the posterior across rounds
    patience: int = 3                   # early stop after this many rounds of posterior drift < 1e-3
    gmm_k: int = 2                      # mixture components; the highest-mean one is the anomaly component
    # Weak learner
    window: Optional[int] = None        # lagged-feature window; None: dominant period of channel 0
    hidden_dim: int = 128
    max_epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3
    samples_per_epoch: int = 10000
    # Threshold
    threshold_rule: str = 'frag_knee'   # 'frag_knee' or 'posterior_mass'
    min_rate: float = 0.001
    max_rate: float = 0.15
    # Refinement strength (gate): keep the blend of source and refined score whose
    # decisions are most temporally coherent. False always returns the fully refined score.
    refine_select: bool = True


@dataclass
class PVBoostResult:
    score: np.ndarray                   # returned anomaly score (the selected blend if refine_select)
    labels: np.ndarray                  # binary decisions, 1 = anomaly
    anomaly_rate: float                 # pi_hat of the returned decisions
    threshold: float                    # score threshold of the returned decisions
    a_star: Optional[float]             # selected refinement strength (None if refine_select is off)
    refined_score: np.ndarray           # fully refined score (a = 1)
    posterior: np.ndarray               # final anomaly posterior q_final
    n_rounds: int                       # boosting rounds that ran
    info: dict = field(default_factory=dict)


def _zscore(s):
    s = np.asarray(s, dtype=np.float64).ravel()
    if not np.all(np.isfinite(s)):
        s = np.where(np.isfinite(s), s, np.nan)
        s = np.where(np.isnan(s), np.nanmedian(s) if np.any(np.isfinite(s)) else 0.0, s)
    sigma = s.std()
    return np.zeros_like(s) if sigma < 1e-12 else (s - s.mean()) / sigma


def _sigmoid_norm(s):
    s = np.asarray(s, dtype=np.float64).ravel()
    if not np.all(np.isfinite(s)):
        s = np.where(np.isfinite(s), s, np.nan)
        s = np.where(np.isnan(s), np.nanmedian(s) if np.any(np.isfinite(s)) else 0.0, s)
    std = s.std()
    if std < 1e-10:
        return np.full_like(s, 0.5)
    return 1.0 / (1.0 + np.exp(-(s - s.mean()) / std))


def lagged_features(X, window):
    """Row t becomes [x[t-window+1], ..., x[t]] flattened, zero-padded at the start. Shape (T, window * F)."""
    T, F = X.shape
    padded = np.concatenate([np.zeros((window - 1, F)), X], axis=0)
    idx = np.arange(window)[None, :] + np.arange(T)[:, None]
    return padded[idx].reshape(T, window * F)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PVBoost:
    """Refine a black-box detector's anomaly score and threshold it, without labels.

    Example
    -------
    >>> result = PVBoost(seed=0).fit_predict(X, source_score)
    >>> result.score, result.labels
    """

    def __init__(self, config: Optional[PVBoostConfig] = None, **kwargs):
        self.config = config if config is not None else PVBoostConfig(**kwargs)
        if config is not None and kwargs:
            raise ValueError('pass either a PVBoostConfig or keyword arguments, not both')

    def fit_predict(self, X, source_score) -> PVBoostResult:
        """
        X            : (T,) or (T, F) time series.
        source_score : (T,) anomaly score of any detector, higher = more anomalous.
        """
        cfg = self.config
        device = torch.device(cfg.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        source_score = np.asarray(source_score)
        if source_score.shape[0] != X.shape[0]:
            raise ValueError(f'source_score has length {source_score.shape[0]}, X has {X.shape[0]} steps')
        _set_seed(cfg.seed)

        window = cfg.window if cfg.window is not None else find_period(X[:, 0].reshape(-1, 1))
        feats = MinMaxScaler().fit_transform(X)
        window = max(1, window)
        if window > 1:
            feats = lagged_features(feats, window)

        # History of [f_0, h_1, ..., h_t]; its per-step spread is the variance signal v_t.
        # It keeps the source score's dtype, as the reference implementation does.
        history = source_score.copy().reshape(-1, 1)
        f_t = MinMaxScaler().fit_transform(history[:, 0].reshape(-1, 1)).reshape(-1)
        history[:, 0] = f_t
        f_src = f_t.copy()

        gmm = GaussianMixture(n_components=cfg.gmm_k, random_state=cfg.seed, max_iter=30, warm_start=True)
        beta = cfg.ema_beta
        q_bar = q_prev = None
        no_improve = rounds = 0
        for rnd in range(cfg.n_rounds):
            # Variance across the source score and the weak learners fitted so far.
            v_t = np.zeros(f_t.shape[0]) if history.shape[1] == 1 else history.std(axis=1)
            s_t = _zscore(f_t)
            # Anomaly posterior from a mixture on the current score.
            gmm.fit(s_t.reshape(-1, 1))
            anomaly = int(np.argmax(gmm.means_.reshape(-1)))
            q_t = gmm.predict_proba(s_t.reshape(-1, 1))[:, anomaly]
            q_bar = q_t.copy() if q_bar is None else beta * q_bar + (1.0 - beta) * q_t
            # Posterior-gated variance target and its residual.
            r_t = _sigmoid_norm(f_t + q_bar * v_t) - f_t

            if q_prev is not None:
                if np.mean(np.abs(q_t - q_prev)) < 1e-3:
                    no_improve += 1
                    if no_improve >= cfg.patience:
                        break
                else:
                    no_improve = 0
            q_prev = q_t.copy()

            history = np.concatenate([history, r_t.reshape(-1, 1)], axis=-1)
            models = fit_weak_learners(feats, history[:, -1], cfg, device)
            h_t = predict(models, feats, cfg.batch_size, device)
            history[:, -1] = h_t
            f_t = f_t + cfg.eta * h_t
            rounds = rnd + 1
            logger.info('PVBoost round %d: anomaly weight %.3f', rnd, gmm.weights_[anomaly])

        refined = _zscore(f_t)
        final_gmm = GaussianMixture(n_components=cfg.gmm_k, random_state=cfg.seed, max_iter=100)
        final_gmm.fit(refined.reshape(-1, 1))
        q_final = final_gmm.predict_proba(refined.reshape(-1, 1))[:, int(np.argmax(final_gmm.means_.reshape(-1)))]
        q_final = beta * q_bar + (1.0 - beta) * q_final

        rate, parts = estimate_rate(q_final, refined, cfg.threshold_rule, cfg.min_rate, cfg.max_rate)
        threshold = float(np.quantile(refined, 1.0 - rate))
        score, labels, a_star, ce = refined, (refined >= threshold).astype(int), None, None

        if cfg.refine_select:
            z0 = mmz(f_src).reshape(-1, 1)
            g0 = GaussianMixture(n_components=cfg.gmm_k, random_state=cfg.seed, max_iter=30,
                                 warm_start=True).fit(z0)
            q_src = g0.predict_proba(z0)[:, int(np.argmax(g0.means_.reshape(-1)))]
            sel = select_refinement(f_src, refined, q_src, q_final, cfg.threshold_rule,
                                    cfg.min_rate, cfg.max_rate)
            score, labels, rate, a_star, ce = sel['score'], sel['preds'], sel['rate'], sel['a_star'], sel['ce']
            threshold = float(np.quantile(score, 1.0 - rate))

        return PVBoostResult(
            score=score, labels=labels, anomaly_rate=float(rate), threshold=threshold, a_star=a_star,
            refined_score=refined, posterior=q_final, n_rounds=rounds,
            info={'window': window, 'pi_mass': parts['pi_mass'], 'pi_frag': parts['pi_frag'], 'ce': ce})
