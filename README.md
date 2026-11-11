<p align="center">
<img width="500" src="assets/overview.png"/>
</p>

<h2 align="center">PVBoost: Posterior-Calibrated Variance Boosting for Time-Series Anomaly Detection</h2>

PVBoost takes the anomaly scores of any black-box detector and, **without labels**, (1) refines the scores and (2) selects a decision threshold.

- **Score refinement by boosting.** Weak learners repeatedly fit the current score. Time steps whose predictions vary most across rounds are raised, since anomalies are harder to model than normal patterns. Because noise also produces variance, the variance is weighted by each time step's *anomaly posterior*: its probability of belonging to the anomalous component of a Gaussian mixture fitted to the current score. Refined scores sharpen the posteriors in turn, forming a posterior–variance feedback loop.
- **Threshold from the posterior.** The final posterior proposes an anomaly rate. A fragmentation test, which checks whether the flagged time steps form contiguous segments rather than scattered points, caps it.
- **Refinement strength.** PVBoost compares blends of the source and refined scores and keeps the one whose decisions are most temporally coherent. Strong source scores are therefore left largely intact.

On TSB-AD (530 series, nine domains) with 16 statistical, neural and foundation-model detectors, PVBoost improves average VUS-PR over the source scores by 6.9% (univariate) and 15.6% (multivariate). It also improves average F-score over 3σ thresholding of the source scores by 71.9% and 132.1%.

## Installation

```bash
git clone https://github.com/thedatumorg/PVBoost && cd PVBoost
pip install -r requirements.txt
pip install TSB-AD
```

PVBoost needs Python ≥ 3.9 and PyTorch ≥ 2.0. A GPU is used when available, but it is not required.

## Quick start

```python
from pvboost import PVBoost

# X: (T,) or (T, F) time series; source_score: (T,) score of any detector, higher = more anomalous
result = PVBoost(seed=0).fit_predict(X, source_score)

result.score          # refined anomaly score
result.labels         # binary decisions (1 = anomaly)
result.anomaly_rate   # estimated anomaly rate used for the threshold
result.a_star         # selected refinement strength (0 = source score, 1 = fully refined)
```

`python examples/quickstart.py` runs PVBoost end to end on a synthetic series, using Isolation Forest as the source detector.

### Main options (`PVBoostConfig`)

| Option | Default | Meaning |
|---|---|---|
| `seed` | `0` | Random seed (weak learners, mixtures) |
| `device` | `None` | `'cuda:0'` if available, else `'cpu'` |
| `n_rounds` | `10` | Maximum boosting rounds; stops early once the posterior stabilizes |
| `eta` | `0.1` | Boosting step size |
| `ema_beta` | `0.7` | Momentum of the posterior across rounds |
| `gmm_k` | `2` | Mixture components (the highest-mean one is the anomaly component) |
| `window` | `None` | Lagged-feature window of the weak learner; `None` uses the dominant period |
| `refine_select` | `True` | Select the refinement strength; `False` always returns the fully refined score |

Pass options either as keyword arguments, `PVBoost(seed=1, refine_select=False)`, or as a `PVBoostConfig`.

## Reproducing the TSB-AD experiments

**1. Data.** Download TSB-AD-U and TSB-AD-M as described in the [TSB-AD repository](https://github.com/thedatumorg/TSB-AD). The evaluation lists are `Datasets/File_List/TSB-AD-{U,M}-Eva.csv` (350 univariate and 180 multivariate series).

**2. Source scores.** PVBoost reads one precomputed score per series from `<score_dir>/<detector>/<series>.npy`. Generate them with TSB-AD's own scripts, which use its tuned hyperparameters:

```bash
cd TSB-AD/benchmark_exp
python Run_Detector_U.py --AD_Name Sub_IForest --dataset_dir ../Datasets/TSB-AD-U/ \
    --file_lsit ../Datasets/File_List/TSB-AD-U-Eva.csv --score_dir <SCORE_ROOT>/uni/
python Run_Detector_M.py --AD_Name IForest --dataset_dir ../Datasets/TSB-AD-M/ \
    --file_lsit ../Datasets/File_List/TSB-AD-M-Eva.csv --score_dir <SCORE_ROOT>/multi/
```

**3. Run PVBoost.** For one detector and seed:

```bash
python benchmark/run_tsbad.py --benchmark U --detector Sub_IForest --seed 0 \
    --dataset_dir <TSB-AD>/Datasets/TSB-AD-U --score_dir <SCORE_ROOT>/uni \
    --file_list <TSB-AD>/Datasets/File_List/TSB-AD-U-Eva.csv --save_dir results
```

For all 16 detectors × 3 seeds on both benchmarks, followed by the summary table:

```bash
TSBAD_ROOT=<TSB-AD> SCORE_ROOT=<SCORE_ROOT> JOBS=4 bash benchmark/run_all.sh
```

`run_all.sh` expects the datasets under `<TSB-AD>/Datasets/TSB-AD-{U,M}`. Each run writes `results/TSB-AD-{U,M}/PVBoost_<detector>_seed<s>.csv`, with per-series VUS-PR, AUC-PR and the other TSB-AD metrics of the returned score. It also records `F1`, the point-wise F-score of PVBoost's own decisions, plus the estimated rate `pi_hat` and the selected strength `a_star`. `python benchmark/summarize.py --save_dir results` averages these over series and seeds per detector.

Results are deterministic for a given seed and device. GPU and CPU runs can differ slightly in floating point.

## Repository layout

```
pvboost/
  core.py           PVBoost: boosting loop, final posterior, threshold, refinement strength
  model.py          weak learner (MLP) and its K-fold training
  thresholding.py   anomaly-rate estimation and refinement-strength selection
  window.py         period-based window length
benchmark/
  run_tsbad.py      PVBoost on TSB-AD-U / TSB-AD-M with precomputed source scores
  run_all.sh        all detectors x seeds, then summary
  summarize.py      per-detector table
examples/
  quickstart.py
```

## Acknowledgement

PVBoost builds on the following repositories:

* [UADB](https://github.com/HangtingYe/UADB)
* [TSB-AD](https://github.com/thedatumorg/TSB-AD) (datasets, metrics, and detector implementations)

## License

Apache-2.0. See [LICENSE](LICENSE).
