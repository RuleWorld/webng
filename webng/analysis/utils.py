"""
Helpers shared by WEBNG's analysis tools:

  load_mapper   the bin mapper a run stored in its west.h5 (used by average
                through weAnalysis.load_run_mapper)
  voronoi       Voronoi cell edges for drawing adaptive bins (average)
"""
import os
import pickle

import h5py
import matplotlib
import matplotlib.tri
import numpy as np


def load_mapper(west_path, sim_path, n_iter=None):
    """
    The bin mapper a run used at iteration `n_iter` (default: its last
    finished iteration), as WESTPA stored it in west.h5 (bin_topologies),
    falling back to the iteration before if none was stored there.

    A stored mapper refers to classes and functions in the sim's system.py,
    so unpickling imports system.py (the sim folder must be on sys.path, as
    weAnalysis arranges). That's done from inside `sim_path`, where WESTPA
    always runs system.py, so relative paths in it (say, a file of bin
    centers opened at import) resolve exactly as during the simulation.
    System.initialize() is never called.
    """
    with h5py.File(west_path, "r") as f:
        if n_iter is None:
            n_iter = int(f.attrs["west_current_iteration"]) - 1
        index = f["bin_topologies/index"][...]
        data = None
        for it in (n_iter, n_iter - 1):
            group = "iterations/iter_{:08d}".format(it)
            if it < 1 or group not in f or "binhash" not in f[group].attrs:
                continue
            binhash = f[group].attrs["binhash"]
            if isinstance(binhash, str):
                binhash = binhash.encode("utf-8")
            rows = [r for r, entry in enumerate(index)
                    if bytes(entry["hash"]).rstrip(b"\0") == binhash.rstrip(b"\0")]
            if rows:
                length = int(index[rows[0]]["pickle_len"])
                data = np.asarray(f["bin_topologies/pickles"][rows[0], :length]).tobytes()
                break
        if data is None:
            raise ValueError("no stored bin mapper for iteration {} in {}".format(n_iter, west_path))
    here = os.getcwd()
    os.chdir(sim_path)
    try:
        return pickle.loads(data)
    finally:
        os.chdir(here)


# Next two functions by:
# Voronoi diagram from a list of points
# Copyright (C) 2011  Nicolas P. Rougier


def circumcircle(P1, P2, P3):
    """
    Used for plotting voronoi center in `average` analysis.

    Adapted from:
    http://local.wasp.uwa.edu.au/~pbourke/geometry/circlefrom3/Circle.cpp
    """
    delta_a = P2 - P1
    delta_b = P3 - P2
    if np.abs(delta_a[0]) <= 0.000000001 and np.abs(delta_b[1]) <= 0.000000001:
        center_x = 0.5 * (P2[0] + P3[0])
        center_y = 0.5 * (P1[1] + P2[1])
    else:
        aSlope = delta_a[1] / delta_a[0]
        bSlope = delta_b[1] / delta_b[0]

        if aSlope == 0.0:
            aSlope = 1e-6

        if bSlope == 0.0:
            bSlope = 1e-6

        if np.isinf(aSlope):
            aSlope = 1e6

        if np.isinf(bSlope):
            bSlope = 1e6

        if np.abs(aSlope - bSlope) <= 0.000000001:
            return None
        center_x = (
            aSlope * bSlope * (P1[1] - P3[1])
            + bSlope * (P1[0] + P2[0])
            - aSlope * (P2[0] + P3[0])
        ) / (2 * (bSlope - aSlope))
        center_y = -1 * (center_x - (P1[0] + P2[0]) / 2) / aSlope + (P1[1] + P2[1]) / 2
    return center_x, center_y


def voronoi(X, Y):
    """
    Used for plotting voronoi center in `average` analysis.
    """
    P = np.zeros((X.size + 4, 2))
    P[: X.size, 0], P[: Y.size, 1] = X, Y
    # We add four points at "infinity"
    m = max(np.abs(X).max(), np.abs(Y).max()) * 1e5
    P[X.size :, 0] = [-m, -m, +m, +m]
    P[Y.size :, 1] = [-m, +m, -m, +m]
    D = matplotlib.tri.Triangulation(P[:, 0], P[:, 1])
    T = D.triangles
    n = T.shape[0]
    C = np.zeros((n, 2))
    for i in range(n):
        C[i] = circumcircle(P[T[i, 0]], P[T[i, 1]], P[T[i, 2]])
    X, Y = C[:, 0], C[:, 1]
    segments = []
    for i in range(n):
        for j in range(3):
            k = D.neighbors[i][j]
            if k != -1:
                segments.append([(X[i], Y[i]), (X[k], Y[k])])
    return segments