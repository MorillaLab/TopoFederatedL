"""
baselines.py â€” FedProx, SCAFFOLD, pFedMe implementations for TopoFederatedL
============================================================================
All implemented from scratch with numpy/scipy/sklearn,
matching the logistic regression local model used throughout.

References:
  FedProx : Li et al. 2020 (MLSys) â€” proximal regularisation
  SCAFFOLD: Karimireddy et al. 2020 (ICML)  control variates for drift
  pFedMe  : T Dinh et al. 2020 (NeurIPS) €” Moreau envelope personalisation
"""

import numpy as np
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score


# #############################################################################
# Shared utilities
# #############################################################################

def _params_to_vec(coef, intercept):
    return np.concatenate([coef.ravel(), intercept.ravel()])

def _vec_to_params(vec, coef_shape, intercept_shape):
    n_coef = int(np.prod(coef_shape))
    coef = vec[:n_coef].reshape(coef_shape)
    intercept = vec[n_coef:n_coef + int(np.prod(intercept_shape))].reshape(intercept_shape)
    return coef, intercept

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _lr_gradient(X, y, coef, intercept, C=1.0):
    """Logistic regression gradient (coef, intercept) with L2 regularisation."""
    n = len(y)
    logits = X @ coef.T + intercept          # (n, n_classes)
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)

    # Binary case
    probs = _sigmoid(logits[:, 0])
    err = probs - y                           # (n,)
    grad_coef = (X.T @ err) / n + coef / (C * n)   # (d,)
    grad_int  = np.array([err.mean()])
    return grad_coef.reshape(coef.shape), grad_int

def _evaluate(X, y, coef, intercept, scaler):
    Xs = scaler.transform(X)
    logits = Xs @ coef.T + intercept
    if logits.ndim == 2:
        logits = logits[:, 0]
    probs = _sigmoid(logits)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = 0.5
    return {'accuracy': acc, 'auc': auc}


# #############################################################################
# FedProx Client
# #############################################################################

class FedProxClient:
    """
    FedProx (Li et al. 2020): adds proximal term ||theta - theta_global||^2 * mu/2
    to local objective, limiting client drift on heterogeneous data.
    """

    def __init__(self, client_id, X_train, y_train, X_test=None, y_test=None,
                 mu=0.1, lr=0.05, n_steps=30, random_state=42):
        self.client_id = client_id
        self.X_train = X_train
        self.y_train = y_train.astype(float)
        self.X_test  = X_test
        self.y_test  = y_test
        self.mu      = mu          # proximal coefficient
        self.lr      = lr
        self.n_steps = n_steps
        self.scaler  = StandardScaler()
        self.coef_       = None
        self.intercept_  = None
        self.classes_    = np.array([0, 1])
        self.rng         = np.random.RandomState(random_state + client_id)

    def _init_params(self, d):
        if self.coef_ is None:
            self.coef_      = self.rng.randn(1, d) * 0.01
            self.intercept_ = np.zeros(1)

    def train_local(self, global_params=None):
        X = self.scaler.fit_transform(self.X_train)
        n, d = X.shape
        self._init_params(d)

        # Global params for proximal term
        if global_params is not None:
            coef_g = global_params['coef'].copy()
            int_g  = global_params['intercept'].copy()
        else:
            coef_g = self.coef_.copy()
            int_g  = self.intercept_.copy()

        coef = self.coef_.copy()
        intercept = self.intercept_.copy()

        for _ in range(self.n_steps):
            gc, gi = _lr_gradient(X, self.y_train, coef, intercept)
            # Proximal gradient step
            coef      -= self.lr * (gc + self.mu * (coef - coef_g))
            intercept -= self.lr * (gi + self.mu * (intercept - int_g))

        self.coef_      = coef
        self.intercept_ = intercept
        return self

    def get_model_params(self):
        if self.coef_ is None:
            return None
        return {
            'coef':      self.coef_.copy(),
            'intercept': self.intercept_.copy(),
            'classes':   self.classes_,
            'n_samples': len(self.X_train),
        }

    def set_global_params(self, params):
        if params is not None:
            self.coef_      = params['coef'].copy()
            self.intercept_ = params['intercept'].copy()

    def evaluate(self, X=None, y=None):
        if self.coef_ is None:
            return {'accuracy': 0.0, 'auc': 0.5}
        X = X if X is not None else self.X_test
        y = y if y is not None else self.y_test
        if X is None or len(X) == 0:
            return {'accuracy': 0.0, 'auc': 0.5}
        return _evaluate(X, y, self.coef_, self.intercept_, self.scaler)


class FedProxServer:
    def __init__(self):
        self.global_params = None

    def aggregate(self, clients, round_num=0):
        params = [c.get_model_params() for c in clients]
        valid  = [p for p in params if p is not None]
        if not valid:
            return self.global_params, {}

        if len(set(p['coef'].shape for p in valid)) > 1:
            self.global_params = valid[0]
            return self.global_params, {}

        n_total = sum(p['n_samples'] for p in valid)
        agg_coef = sum(p['n_samples'] / n_total * p['coef']      for p in valid)
        agg_int  = sum(p['n_samples'] / n_total * p['intercept'] for p in valid)
        self.global_params = {
            'coef': agg_coef, 'intercept': agg_int, 'classes': valid[0]['classes']
        }
        return self.global_params, {}


# #############################################################################
# SCAFFOLD Client & Server
# #############################################################################
class SCAFFOLDClient:
    """
    SCAFFOLD (Karimireddy et al. 2020): each client maintains a control variate
    c_i that corrects for client drift. Server maintains global control variate c.
    Update rule:
        y_i  <- theta - lr * (grad_i - c_i + c)
        delta_c_i <- (theta_old - y_i) / (K * lr) - c
    """

    def __init__(self, client_id, X_train, y_train, X_test=None, y_test=None,
                 lr=0.05, n_steps=30, random_state=42):
        self.client_id   = client_id
        self.X_train     = X_train
        self.y_train     = y_train.astype(float)
        self.X_test      = X_test
        self.y_test      = y_test
        self.lr          = lr
        self.n_steps     = n_steps
        self.scaler      = StandardScaler()
        self.coef_       = None
        self.intercept_  = None
        self.classes_    = np.array([0, 1])
        self.c_i_coef    = None   # client control variate
        self.c_i_int     = None
        self.rng         = np.random.RandomState(random_state + client_id)

    def _init_params(self, d):
        if self.coef_ is None:
            self.coef_      = self.rng.randn(1, d) * 0.01
            self.intercept_ = np.zeros(1)
            self.c_i_coef   = np.zeros((1, d))
            self.c_i_int    = np.zeros(1)

    def train_local(self, global_params=None, global_control=None):
        X = self.scaler.fit_transform(self.X_train)
        n, d = X.shape
        self._init_params(d)

        if global_params is not None:
            coef      = global_params['coef'].copy()
            intercept = global_params['intercept'].copy()
        else:
            coef      = self.coef_.copy()
            intercept = self.intercept_.copy()

        # Global control variate
        if global_control is not None:
            c_coef = global_control['coef']
            c_int  = global_control['intercept']
        else:
            c_coef = np.zeros_like(coef)
            c_int  = np.zeros_like(intercept)

        old_coef = coef.copy()
        old_int  = intercept.copy()

        for _ in range(self.n_steps):
            gc, gi = _lr_gradient(X, self.y_train, coef, intercept)
            # SCAFFOLD correction: subtract client drift, add global drift
            coef      -= self.lr * (gc - self.c_i_coef + c_coef)
            intercept -= self.lr * (gi - self.c_i_int  + c_int)

        # Update client control variate
        delta_c_coef = (old_coef - coef) / (self.n_steps * self.lr) - c_coef
        delta_c_int  = (old_int  - intercept) / (self.n_steps * self.lr) - c_int
        self.c_i_coef += delta_c_coef
        self.c_i_int  += delta_c_int

        self.coef_      = coef
        self.intercept_ = intercept
        return self

    def get_model_params(self):
        if self.coef_ is None:
            return None
        return {
            'coef':         self.coef_.copy(),
            'intercept':    self.intercept_.copy(),
            'classes':      self.classes_,
            'n_samples':    len(self.X_train),
            'delta_c_coef': self.c_i_coef.copy(),
            'delta_c_int':  self.c_i_int.copy(),
        }

    def set_global_params(self, params):
        if params is not None:
            self.coef_      = params['coef'].copy()
            self.intercept_ = params['intercept'].copy()

    def evaluate(self, X=None, y=None):
        if self.coef_ is None:
            return {'accuracy': 0.0, 'auc': 0.5}
        X = X if X is not None else self.X_test
        y = y if y is not None else self.y_test
        if X is None or len(X) == 0:
            return {'accuracy': 0.0, 'auc': 0.5}
        return _evaluate(X, y, self.coef_, self.intercept_, self.scaler)


class SCAFFOLDServer:
    def __init__(self, n_features=20):
        self.global_params  = None
        self.global_control = None  # server-side control variate c

    def aggregate(self, clients, round_num=0):
        params = [c.get_model_params() for c in clients]
        valid  = [p for p in params if p is not None]
        if not valid:
            return self.global_params, {}

        if len(set(p['coef'].shape for p in valid)) > 1:
            self.global_params = valid[0]
            return self.global_params, {}

        n_total = sum(p['n_samples'] for p in valid)
        agg_coef = sum(p['n_samples'] / n_total * p['coef']      for p in valid)
        agg_int  = sum(p['n_samples'] / n_total * p['intercept'] for p in valid)

        # Update global control variate
        K = len(valid)
        if self.global_control is None:
            self.global_control = {
                'coef': np.zeros_like(agg_coef),
                'intercept': np.zeros_like(agg_int),
            }
        self.global_control['coef'] += (
            sum(p['delta_c_coef'] for p in valid) / K
        )
        self.global_control['intercept'] += (
            sum(p['delta_c_int'] for p in valid) / K
        )

        self.global_params = {
            'coef': agg_coef, 'intercept': agg_int, 'classes': valid[0]['classes']
        }
        return self.global_params, {}


# #############################################################################
# pFedMe Client & Server
# #############################################################################”€

class pFedMeClient:
    """
    pFedMe (T Dinh et al. 2020): personalised FL via Moreau envelopes.
    Each client has a personalised model theta_i minimising:
        F_i(theta_i) + lambda/2 * ||theta_i - w||^2
    where w is the global model (Moreau envelope proximal term).
    Inner loop: optimise personalised theta_i (given w).
    Outer update: w <- w - eta * lambda * (w - theta_i)
    """

    def __init__(self, client_id, X_train, y_train, X_test=None, y_test=None,
                 lam=15.0, lr=0.01, lr_personal=0.01, n_steps=20,
                 random_state=42):
        self.client_id     = client_id
        self.X_train       = X_train
        self.y_train       = y_train.astype(float)
        self.X_test        = X_test
        self.y_test        = y_test
        self.lam           = lam        # Moreau envelope parameter
        self.lr            = lr         # global update lr
        self.lr_personal   = lr_personal
        self.n_steps       = n_steps
        self.scaler        = StandardScaler()
        self.coef_         = None       # personalised model theta_i
        self.intercept_    = None
        self.w_coef_       = None       # global model w (local copy)
        self.w_int_        = None
        self.classes_      = np.array([0, 1])
        self.rng           = np.random.RandomState(random_state + client_id)

    def _init_params(self, d):
        if self.coef_ is None:
            self.coef_      = self.rng.randn(1, d) * 0.01
            self.intercept_ = np.zeros(1)
            self.w_coef_    = self.coef_.copy()
            self.w_int_     = self.intercept_.copy()

    def train_local(self, global_params=None):
        X = self.scaler.fit_transform(self.X_train)
        n, d = X.shape
        self._init_params(d)

        if global_params is not None:
            self.w_coef_ = global_params['coef'].copy()
            self.w_int_  = global_params['intercept'].copy()

        # Inner loop: optimise personalised model theta_i
        coef      = self.coef_.copy()
        intercept = self.intercept_.copy()

        for _ in range(self.n_steps):
            gc, gi = _lr_gradient(X, self.y_train, coef, intercept)
            # Moreau envelope gradient: grad_F + lambda*(theta - w)
            coef      -= self.lr_personal * (gc + self.lam * (coef - self.w_coef_))
            intercept -= self.lr_personal * (gi + self.lam * (intercept - self.w_int_))

        self.coef_      = coef
        self.intercept_ = intercept

        # Outer update: w <- w - eta * lambda * (w - theta_i)
        self.w_coef_ -= self.lr * self.lam * (self.w_coef_ - coef)
        self.w_int_  -= self.lr * self.lam * (self.w_int_  - intercept)
        return self

    def get_model_params(self):
        """Return updated global model w (not personalised theta_i)."""
        if self.w_coef_ is None:
            return None
        return {
            'coef':      self.w_coef_.copy(),
            'intercept': self.w_int_.copy(),
            'classes':   self.classes_,
            'n_samples': len(self.X_train),
        }

    def set_global_params(self, params):
        if params is not None:
            self.w_coef_ = params['coef'].copy()
            self.w_int_  = params['intercept'].copy()

    def evaluate(self, X=None, y=None):
        """Evaluate with personalised model theta_i."""
        if self.coef_ is None:
            return {'accuracy': 0.0, 'auc': 0.5}
        X = X if X is not None else self.X_test
        y = y if y is not None else self.y_test
        if X is None or len(X) == 0:
            return {'accuracy': 0.0, 'auc': 0.5}
        return _evaluate(X, y, self.coef_, self.intercept_, self.scaler)


class pFedMeServer:
    def __init__(self):
        self.global_params = None

    def aggregate(self, clients, round_num=0):
        params = [c.get_model_params() for c in clients]
        valid  = [p for p in params if p is not None]
        if not valid:
            return self.global_params, {}

        if len(set(p['coef'].shape for p in valid)) > 1:
            self.global_params = valid[0]
            return self.global_params, {}

        n_total  = sum(p['n_samples'] for p in valid)
        agg_coef = sum(p['n_samples'] / n_total * p['coef']      for p in valid)
        agg_int  = sum(p['n_samples'] / n_total * p['intercept'] for p in valid)
        self.global_params = {
            'coef': agg_coef, 'intercept': agg_int, 'classes': valid[0]['classes']
        }
        return self.global_params, {}
