"""
ptopopfl/descriptor.py
======================
48-dimensional Persistent Homology descriptor (Eq. 5, paper §3.1).

Components (concatenation order, total = 48):
  betti_curve_H0   20 values  – alive H0 features at 20 filtration thresholds
  betti_curve_H1   20 values  – alive H1 features at 20 filtration thresholds
  β0, β1            2 values  – Betti numbers
  H0, H1            2 values  – persistence entropy
  A0, A1            2 values  – L2 diagram amplitude
  n0, n1            2 values  – count of finite birth–death pairs

Key corrections vs. the Appendix D sketch
------------------------------------------
* H0 uses deterministic elder-rule union-find (lower index = elder),
  eliminating the random root choice that made results non-reproducible.
* H1 uses proper boundary-matrix reduction (∂2 over Z2) applied to the
  filtered triangle complex, replacing the zero-placeholder.
* Descriptor is assembled and checked to be exactly 48-dimensional.
* Subsampling uses a seeded RNG so results are fully reproducible.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform


# ─────────────────────────────────────────────────────────────────────────────
# H0 — connected components via deterministic union-find with elder rule
# ─────────────────────────────────────────────────────────────────────────────

def _compute_h0(dist_matrix):
    """
    Vietoris–Rips H0 persistence via MST union-find.

    Elder rule: when two components merge at edge weight w, the component
    with the *higher* root index dies (born = 0, death = w for the younger
    component).  This is fully deterministic for a given distance matrix.

    Returns
    -------
    pairs : ndarray, shape (n-1, 2)
        Finite (birth=0, death=w) pairs.  The last surviving component
        has infinite lifetime and is excluded per convention.
    """
    n = dist_matrix.shape[0]
    parent = np.arange(n)

    def find(x):
        # Iterative path compression
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    # All edges sorted by weight (stable sort for reproducibility)
    rows, cols = np.triu_indices(n, k=1)
    weights = dist_matrix[rows, cols]
    order = np.argsort(weights, kind='stable')

    pairs = []
    for idx in order:
        i, j = int(rows[idx]), int(cols[idx])
        w = weights[idx]
        ri, rj = find(i), find(j)
        if ri == rj:
            continue
        # Elder rule: lower index survives, higher index dies
        elder, younger = (ri, rj) if ri < rj else (rj, ri)
        pairs.append((0.0, w))
        parent[younger] = elder

    return np.array(pairs, dtype=float) if pairs else np.zeros((0, 2))


# ─────────────────────────────────────────────────────────────────────────────
# H1 — independent loops via triangle-filtration boundary-matrix reduction
# ─────────────────────────────────────────────────────────────────────────────

def _compute_h1(dist_matrix, max_scale_percentile=95):
    """
    Vietoris–Rips H1 persistence via boundary-matrix reduction (Z2).

    Algorithm
    ---------
    1. Build all edges up to max_scale = percentile(weights, max_scale_percentile).
    2. Enumerate all triangles; each appears at filtration value = max edge weight.
    3. Reduce the boundary matrix ∂2 (columns = triangles, rows = edges) over Z2.
       A zero column after reduction means a triangle kills an H1 cycle; the
       pivot edge records when that cycle was born.

    Complexity: O(|edges|·|triangles|/|edges|) ≈ O(n^3) for n=80, tractable.

    Returns
    -------
    pairs : ndarray, shape (k, 2)
        Finite (birth, death) pairs for H1 features with death > birth.
    """
    n = dist_matrix.shape[0]

    # ── 1. Collect edges up to max_scale ────────────────────────────────────
    rows, cols = np.triu_indices(n, k=1)
    weights = dist_matrix[rows, cols]
    max_scale = np.percentile(weights, max_scale_percentile)

    mask = weights <= max_scale
    edge_i = rows[mask].astype(int)
    edge_j = cols[mask].astype(int)
    edge_w = weights[mask]

    order = np.argsort(edge_w, kind='stable')
    edge_i = edge_i[order]
    edge_j = edge_j[order]
    edge_w = edge_w[order]
    n_edges = len(edge_i)

    if n_edges == 0:
        return np.zeros((0, 2))

    # Fast edge lookup: (min_idx, max_idx) → (array_idx, weight)
    edge_dict = {}
    for k in range(n_edges):
        a, b = int(edge_i[k]), int(edge_j[k])
        if a > b:
            a, b = b, a
        edge_dict[(a, b)] = (k, edge_w[k])

    # ── 2. Enumerate triangles ───────────────────────────────────────────────
    # Build adjacency for fast triangle enumeration
    adj = {v: set() for v in range(n)}
    for k in range(n_edges):
        a, b = int(edge_i[k]), int(edge_j[k])
        adj[a].add(b)
        adj[b].add(a)

    seen_tri = set()
    triangles = []  # (filtration_value, (i, j, k))

    for a in range(n):
        for b in adj[a]:
            if b <= a:
                continue
            for c in adj[a]:
                if c <= b:
                    continue
                if c not in adj[b]:
                    continue
                tri = (a, b, c)  # already sorted since a<b<c
                if tri in seen_tri:
                    continue
                seen_tri.add(tri)
                w_ab = edge_dict[(a, b)][1]
                w_ac = edge_dict[(a, c)][1]
                w_bc = edge_dict[(b, c)][1]
                filt_val = max(w_ab, w_ac, w_bc)
                triangles.append((filt_val, tri))

    if not triangles:
        return np.zeros((0, 2))

    # Sort triangles by filtration value (stable)
    triangles.sort(key=lambda x: x[0])

    # ── 3. Boundary-matrix reduction over Z2 ────────────────────────────────
    # Each column = set of edge indices bounding a triangle (∂2 column).
    boundary = []
    for filt_val, (a, b, c) in triangles:
        e_ab = edge_dict.get((a, b), (None,))[0]
        e_ac = edge_dict.get((a, c), (None,))[0]
        e_bc = edge_dict.get((b, c), (None,))[0]
        col = {e for e in (e_ab, e_ac, e_bc) if e is not None}
        boundary.append(col)

    # Standard column-reduction: pivot = max row index in column
    pivot_owner = {}   # pivot row → column index
    h1_pairs = []

    for col_idx, col in enumerate(boundary):
        working = set(col)
        while working:
            pivot = max(working)
            if pivot not in pivot_owner:
                pivot_owner[pivot] = col_idx
                # Triangle col_idx kills the H1 cycle born at edge 'pivot'
                birth_val = edge_w[pivot]
                death_val = triangles[col_idx][0]
                if death_val > birth_val:
                    h1_pairs.append((birth_val, death_val))
                break
            # Add (XOR) with the column that currently owns this pivot
            working ^= boundary[pivot_owner[pivot]]

    return np.array(h1_pairs, dtype=float) if h1_pairs else np.zeros((0, 2))


# ─────────────────────────────────────────────────────────────────────────────
# Scalar statistics and Betti curves
# ─────────────────────────────────────────────────────────────────────────────

def _scalar_stats(pairs):
    """
    From a (birth, death) pair array compute:
      beta      – Betti number (count of finite pairs)
      entropy   – persistence entropy
      amplitude – L2 diagram amplitude
      n_pairs   – count of finite pairs (same as beta for PH)
    """
    if len(pairs) == 0:
        return 0, 0.0, 0.0, 0

    pers = pairs[:, 1] - pairs[:, 0]
    finite = pers[np.isfinite(pers) & (pers > 0)]

    if len(finite) == 0:
        return 0, 0.0, 0.0, 0

    total = finite.sum()
    if total > 0:
        p = finite / total
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
    else:
        entropy = 0.0

    amplitude = float(np.sqrt(np.sum(finite ** 2)))
    beta = int(len(finite))

    return beta, entropy, amplitude, beta


def _betti_curve(pairs, n_thresholds=20):
    """
    Betti curve: count of alive features at n_thresholds linearly
    spaced between 0 and the 95th percentile of death values.
    """
    if len(pairs) == 0:
        return np.zeros(n_thresholds)

    finite_mask = np.isfinite(pairs[:, 1]) & (pairs[:, 1] > pairs[:, 0])
    if not finite_mask.any():
        return np.zeros(n_thresholds)

    births = pairs[finite_mask, 0]
    deaths = pairs[finite_mask, 1]
    max_val = float(np.percentile(deaths, 95)) if len(deaths) > 0 else 1.0
    if max_val == 0:
        max_val = 1.0

    thresholds = np.linspace(0.0, max_val, n_thresholds)
    curve = np.array(
        [float(np.sum((births <= t) & (deaths > t))) for t in thresholds]
    )
    return curve


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

class PHDescriptor:
    """
    Compute the 48-dimensional persistent homology descriptor φ_k.

    Parameters
    ----------
    n_sub : int
        Maximum number of points to subsample (default 80, as in paper).
    n_thresholds : int
        Number of Betti-curve thresholds (default 20).
    random_state : int or None
        Seed for reproducible subsampling.
    """

    DIM = 48

    def __init__(self, n_sub=80, n_thresholds=20, random_state=None):
        self.n_sub = n_sub
        self.n_thresholds = n_thresholds
        self.rng = np.random.default_rng(random_state)

    def _subsample(self, X):
        X = np.asarray(X, dtype=float)
        if len(X) > self.n_sub:
            idx = self.rng.choice(len(X), self.n_sub, replace=False)
            return X[idx]
        return X

    def compute(self, X):
        """
        Compute φ_k ∈ R^48 for point cloud X.

        Returns
        -------
        phi : ndarray, shape (48,)
        """
        X = self._subsample(X)

        if len(X) < 3:
            return np.zeros(self.DIM)

        # Normalise to unit box for scale invariance across clients
        X_range = X.max(axis=0) - X.min(axis=0)
        X_range[X_range == 0] = 1.0
        X_norm = (X - X.min(axis=0)) / X_range

        dist_matrix = squareform(pdist(X_norm))

        h0_pairs = _compute_h0(dist_matrix)
        h1_pairs = _compute_h1(dist_matrix)

        bc_h0 = _betti_curve(h0_pairs, self.n_thresholds)   # 20
        bc_h1 = _betti_curve(h1_pairs, self.n_thresholds)   # 20

        b0, e0, a0, n0 = _scalar_stats(h0_pairs)
        b1, e1, a1, n1 = _scalar_stats(h1_pairs)

        phi = np.concatenate([
            bc_h0,          # 20
            bc_h1,          # 20
            [b0, b1],       #  2
            [e0, e1],       #  2
            [a0, a1],       #  2
            [n0, n1],       #  2
        ])

        assert len(phi) == self.DIM, f"Descriptor dimension {len(phi)} ≠ 48"
        return phi

    def compute_batch(self, datasets):
        """Compute descriptors for a list of datasets (one per client)."""
        return np.array([self.compute(X) for X in datasets])
