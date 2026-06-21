"""
smm_gd.py - Vectorised SMM (Stabilization Matrix Method) engine.

Same algorithm as the course gradient-descent SMM. The only change is that
the inner per-weight Python loop:

    do = O - t                                  # error with CURRENT weights
    for i in range(len(w)):
        de_dw_i = do*x[i] + 2*lamb_N*w[i]
        w[i]   -= eps * de_dw_i

is replaced by the single NumPy expression it is mathematically equal to:

    do  = x.dot(w) - t
    w  -= eps * (do*x + 2*lamb_N*w)

Each weight's gradient depends ONLY on its own current value, the shared
scalar error `do`, and its own input feature - there is no cross-weight
dependency inside one update - so the vectorised form produces the same
weights to floating-point precision. (Updates ACROSS peptides stay in a
Python loop on purpose: those are sequential and vectorising them would turn
stochastic GD into batch GD, which is a different algorithm.)
"""

import numpy as np
from scipy.stats import pearsonr


#Encoding

def load_encoding(alphabet_path, sparse_path):
    """Load the amino-acid alphabet and an encoding matrix (sparse/one-hot,
    or BLOSUM, etc.). Returns (alphabet_list, scheme_dict)."""
    alphabet = np.loadtxt(alphabet_path, dtype=str)
    mat = np.loadtxt(sparse_path, dtype=float)
    scheme = {a: {b: mat[i, j] for j, b in enumerate(alphabet)}
              for i, a in enumerate(alphabet)}
    return list(alphabet), scheme


def encode(peptides, scheme, alphabet):
    """Encode peptides to fixed-length numeric vectors (len = L * |alphabet|)."""
    rowvec = {a: np.array([scheme[a][b] for b in alphabet], dtype=float)
              for a in alphabet}
    return np.array([np.concatenate([rowvec[aa] for aa in pep])
                     for pep in peptides], dtype=float)


def load_dataset(path, scheme, alphabet):
    """Read a peptide file (peptide, target, [allele]); return (X, y, peptides)."""
    data = np.loadtxt(path, dtype=str)
    if data.ndim == 1:                       # a file with a single line
        data = data.reshape(1, -1)
    peptides = data[:, 0]
    y = data[:, 1].astype(float)
    X = encode(peptides, scheme, alphabet)
    return X, y, peptides


#Training / Prediction

def train_smm(X, y, lamb, epsilon, epochs, seed):
    """Stochastic GD for SMM with L2 regularisation.

    Uses a private RandomState(seed) so runs are reproducible and independent
    across parallel workers.
    """
    X = np.ascontiguousarray(X, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    N, dim = X.shape
    two_lamb_N = 2.0 * (lamb / N)

    rng = np.random.RandomState(seed)
    w = rng.uniform(-0.1, 0.1, size=dim)

    for _ in range(epochs):
        for _ in range(N):
            ix = rng.randint(0, N)
            xi = X[ix]
            do = xi.dot(w) - y[ix]
            w -= epsilon * (do * xi + two_lamb_N * w)
    return w


def predict(X, w):
    """Vectorised prediction: a single matrix-vector product."""
    return np.asarray(X, dtype=float).dot(w)


#Metrics

def pcc(y_true, y_pred):
    """Pearson correlation; NaN-safe (returns nan for degenerate inputs)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))