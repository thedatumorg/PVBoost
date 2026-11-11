import copy
from itertools import cycle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset


class WeakLearner(nn.Module):
    """Three-layer MLP regressor fitted to the boosting residual."""

    def __init__(self, input_dim, hidden_dim, lr):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Linear(hidden_dim, 1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.out(self.encoder(x))


def _loader(X, y, batch_size, shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float()),
                      batch_size=batch_size, shuffle=shuffle)


def fit_weak_learners(X, target, cfg, device):
    """Train one weak learner per K-fold split (K = 3) on ``target`` and return them.

    Each learner trains for ``cfg.max_epochs`` epochs of ``cfg.samples_per_epoch // cfg.batch_size``
    steps; the checkpoint with the lowest validation MSE is kept.
    """
    num_iters = cfg.samples_per_epoch // cfg.batch_size
    target = target.reshape(-1, 1)
    models = []
    for train_idx, val_idx in KFold(n_splits=3).split(X):
        train_loader = _loader(X[train_idx], target[train_idx], cfg.batch_size, shuffle=True)
        val_loader = _loader(X[val_idx], target[val_idx], cfg.batch_size, shuffle=False)
        model = WeakLearner(X.shape[1], cfg.hidden_dim, cfg.learning_rate).to(device)
        best_model, best_val = None, np.inf
        for _ in range(cfg.max_epochs):
            model.train()
            for it, (xb, yb) in enumerate(cycle(train_loader)):
                if it >= num_iters:
                    break
                loss = F.mse_loss(model(xb.to(device)), yb.to(device))
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += F.mse_loss(model(xb.to(device)), yb.to(device)).item()
            val_loss /= len(val_loader)
            if val_loss < best_val:
                best_val = val_loss
                best_model = copy.deepcopy(model)
        models.append(best_model)
    return models


def predict(models, X, batch_size, device):
    """Average prediction of the fold models over the whole series, in time order.

    Iterating a DataLoader draws from the global torch RNG even without shuffling; predicting
    through one keeps the random stream, and hence results, identical across implementations.
    """
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=batch_size, shuffle=False)
    parts = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            h = 0
            for m in models:
                h += m(xb).to('cpu')
            h /= len(models)
            parts.append(h)
    return torch.cat(parts).numpy().reshape(-1)
