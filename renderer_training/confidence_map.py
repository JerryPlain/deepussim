#!/usr/bin/env python
"""Ultrasound confidence maps via random walks (Karamalis et al., MedIA 2012).

Per-pixel confidence in [0, 1]: high where the signal is well transmitted (near field),
low in shadow / attenuated / deep regions. We use it to crop segmentation masks down to
the US-visible part -- removing mask pixels that fall in low-confidence (shadow) regions
where the CBCT-projected label claims tissue but the US shows nothing reliable.

Model (paper Eqs. 5-9), 8-connected lattice over the fan-content pixels:
    c_i   = g_i * exp(-alpha * l_i)              depth-attenuated intensity (Beer-Lambert)
    w_ij  = exp(-beta * (|c_i - c_j|_norm + p))  p = 0 (vertical), gamma (horiz), sqrt2*gamma (diag)
Seeds: per column, top content pixel = 1 (virtual transducer), bottom = 0 (absorption).
Solve the Dirichlet system L_U x_U = -B^T x_M for the reach-the-transducer probability.

Defaults alpha=2, beta=90, gamma=0.05 are the paper's. Computed on a downsampled grid for
speed, then upsampled back to the frame size.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def _norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanpercentile(x, [1, 99])
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def confidence_map(img: np.ndarray, *, alpha: float = 1.0, beta: float = 10.0,
                   gamma: float = 0.05, max_rows: int = 160, presmooth: float = 2.0) -> np.ndarray:
    """Random-walks confidence map for one US frame. Returns float array in [0, 1].

    ``presmooth`` (Gaussian sigma on the downsampled grid) suppresses speckle before the
    gradient weights, else per-pixel speckle underflows the exp weights and the map collapses.
    """
    full = _norm01(img)
    H, W = full.shape
    scale = min(1.0, max_rows / H)
    g = ndimage.zoom(full, scale, order=1) if scale < 1.0 else full.copy()
    if presmooth > 0:
        g = ndimage.gaussian_filter(g, presmooth)
    h, w = g.shape

    content = g > 0.02                                   # fan pixels (background is ~0)
    content = ndimage.binary_erosion(content, iterations=1)
    if content.sum() < 16:
        return np.ones_like(full)

    # normalised depth per column (0 at the transducer edge, 1 at the deepest content)
    depth = np.zeros_like(g)
    top = np.full(w, -1); bot = np.full(w, -1)
    for c in range(w):
        rows = np.where(content[:, c])[0]
        if len(rows) == 0:
            continue
        top[c], bot[c] = rows[0], rows[-1]
        span = max(bot[c] - top[c], 1)
        depth[rows, c] = (rows - top[c]) / span
    ci = g * np.exp(-alpha * depth)

    idx = -np.ones((h, w), dtype=int)
    nodes = np.argwhere(content)
    idx[content] = np.arange(len(nodes))
    N = len(nodes)

    # 8-connectivity: 4 unique offsets (E, S, SE, SW); each with its penalty class
    offsets = [((0, 1), gamma), ((1, 0), 0.0), ((1, 1), np.sqrt(2) * gamma), ((1, -1), np.sqrt(2) * gamma)]
    rr, cc, grad, pen = [], [], [], []
    for (dr, dc), p in offsets:
        a = content.copy()
        b = np.zeros_like(content)
        r0, r1 = max(0, -dr), h - max(0, dr)
        c0, c1 = max(0, -dc), w - max(0, dc)
        b[r0:r1, c0:c1] = content[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        both = a & b
        ii = idx[both]
        shifted = np.zeros((h, w), dtype=int)
        shifted[r0:r1, c0:c1] = idx[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        jj = shifted[both]
        gd = np.abs(ci[both] - np.roll(np.roll(ci, -dr, 0), -dc, 1)[both])
        rr.append(ii); cc.append(jj); grad.append(gd); pen.append(np.full(len(ii), p))
    ii = np.concatenate(rr); jj = np.concatenate(cc)
    gd = np.concatenate(grad); pn = np.concatenate(pen)
    gd = gd / max(gd.max(), 1e-6)
    wij = np.exp(-beta * (gd + pn)) + 1e-6

    # symmetric weight matrix
    data = np.concatenate([wij, wij])
    row = np.concatenate([ii, jj]); col = np.concatenate([jj, ii])
    Wm = csr_matrix((data, (row, col)), shape=(N, N))
    d = np.asarray(Wm.sum(1)).ravel()
    L = csr_matrix((d, (np.arange(N), np.arange(N))), shape=(N, N)) - Wm

    # seeds: top content pixel per column = 1, bottom = 0
    seed_val = np.full(N, np.nan)
    for c in range(w):
        if top[c] >= 0:
            seed_val[idx[top[c], c]] = 1.0
            seed_val[idx[bot[c], c]] = 0.0
    marked = ~np.isnan(seed_val)
    unk = ~marked
    xm = seed_val[marked]

    Lu = L[unk][:, unk]
    Bum = L[unk][:, marked]
    rhs = -Bum @ xm
    # tiny diagonal regularisation: grounds nodes in fan fragments disconnected from any
    # seed, so L_U is non-singular (otherwise spsolve returns NaN for those components).
    nu = Lu.shape[0]
    Lu = Lu + csr_matrix((np.full(nu, 1e-6), (np.arange(nu), np.arange(nu))), shape=(nu, nu))
    xu = spsolve(Lu.tocsc(), rhs)

    x = np.zeros(N); x[marked] = xm; x[unk] = np.clip(xu, 0.0, 1.0)
    conf = np.zeros((h, w)); conf[content] = x
    if scale < 1.0:
        conf = ndimage.zoom(conf, (H / h, W / w), order=1)[:H, :W]
        if conf.shape != full.shape:
            out = np.zeros_like(full); out[:conf.shape[0], :conf.shape[1]] = conf; conf = out
    return np.clip(conf, 0.0, 1.0)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    pairs = np.load("data/renderer_lc2_pairs/pairs.npz", allow_pickle=True)
    c = confidence_map(pairs["us"][0])
    print("confidence map", c.shape, "range", round(float(c.min()), 3), round(float(c.max()), 3),
          "mean", round(float(c.mean()), 3))
