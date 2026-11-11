"""
Bin-level Markov-model math shared by the analysis tools: `cluster` with
method gpcca (states from the bin-to-bin transition matrix) and `network`
(transition matrix and MFPTs between the states of a states.yaml).

The transition matrix T is at lag tau, one WE iteration: T_ij is the
probability that a walker starting a segment in bin i ends it in bin j,
estimated from the walkers' weights. Rows are bins; states are sets of bins.
"""
import h5py
import numpy as np
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# transition counts from w_assign output
# ---------------------------------------------------------------------------

def iter_bin_flux(assign_path, west_path, first_iter, last_iter, with_points=False):
    """
    Weighted bin-to-bin transitions at lag tau, one WE iteration at a time,
    from a w_assign file (with or without states) and the run's walker weights:
    every segment adds its weight to C[bin at its first point, bin at its last
    point]. (The plain bin-level version of w_reweight's flux matrices, without
    that deprecated tool.)

    Yields (n_iter, C_iter), or with with_points (n_iter, C_iter, end_bins,
    end_points): each segment's last bin and its progress coordinate there.
    """
    with h5py.File(assign_path, "r") as af, h5py.File(west_path, "r") as wf:
        nbins = int(af.attrs["nbins"])
        a_start = int(af.attrs.get("iter_start", 1))
        assignments = af["assignments"]
        nsegs = af["nsegs"][...]
        npts = af["npts"][...]
        for n_iter in range(int(first_iter), int(last_iter) + 1):
            i = n_iter - a_start
            if i < 0 or i >= len(nsegs):
                raise ValueError("{} has no assignments for iteration {} (it covers {}-{})"
                                 .format(assign_path, n_iter, a_start, a_start + len(nsegs) - 1))
            ns, npt = int(nsegs[i]), int(npts[i])
            C = np.zeros((nbins, nbins))
            if ns == 0:
                yield (n_iter, C, np.zeros(0, np.int64), None) if with_points else (n_iter, C)
                continue
            asg = assignments[i, :ns, :npt]
            b0, b1 = asg[:, 0].astype(np.int64), asg[:, npt - 1].astype(np.int64)
            ok = (b0 < nbins) & (b1 < nbins)        # w_assign's "no bin" sentinel, if any
            g = wf["iterations/iter_{:08d}".format(n_iter)]
            w = g["seg_index"]["weight"][:ns]
            C += np.bincount(b0[ok] * nbins + b1[ok], weights=w[ok],
                             minlength=nbins * nbins).reshape(nbins, nbins)
            if with_points:
                yield n_iter, C, b1[ok], g["pcoord"][:ns, npt - 1, :][ok]
            else:
                yield n_iter, C


def bin_flux_from_assignments(assign_path, west_path, first_iter, last_iter):
    """
    The transitions of iter_bin_flux summed over first_iter..last_iter, and
    for each bin one progress coordinate the run actually had in that bin (the
    last point of a segment w_assign assigned there), so states can name their
    bins by a coordinate inside them, whatever the binning scheme.

    Returns (C, rep_points, visited): C is nbins x nbins; rep_points is
    nbins x ndim, NaN for bins no segment ended in; visited marks those bins.
    """
    C, rep = None, None
    for _, C_i, b1, ends in iter_bin_flux(assign_path, west_path, first_iter, last_iter, with_points=True):
        C = C_i if C is None else C + C_i
        if ends is None:
            continue
        if rep is None:
            rep = np.full((len(C_i), ends.shape[1]), np.nan)
        rep[b1] = ends
    if rep is None:
        raise RuntimeError("no segments in iterations {}-{} of {}".format(first_iter, last_iter, west_path))
    return C, rep, ~np.isnan(rep[:, 0])


# ---------------------------------------------------------------------------
# matrix basics
# ---------------------------------------------------------------------------

def active_set(C):
    """Indices of the bins in the largest strongly connected set of observed
    transitions: the bins between which the chain can go back and forth."""
    seen = np.nonzero((C.sum(axis=1) > 0) | (C.sum(axis=0) > 0))[0]
    if len(seen) == 0:
        return seen
    _, lab = connected_components(C[np.ix_(seen, seen)] > 0, directed=True, connection="strong")
    return seen[lab == np.argmax(np.bincount(lab))]


def row_normalize(C):
    """Rows divided by their sums; rows with no counts stay zero."""
    C = np.asarray(C, dtype=float)
    rs = C.sum(axis=1)
    return np.divide(C, rs[:, None], out=np.zeros_like(C), where=rs[:, None] > 0)


def stationary(T):
    """Stationary distribution of a row-stochastic matrix (left eigenvector of 1)."""
    ev, evec = np.linalg.eig(np.asarray(T, dtype=float).T)
    p = np.abs(np.real(evec[:, np.argmin(np.abs(ev - 1.0))]))
    return p / p.sum()


def reversible_by_flux(T):
    """
    Reversible (detailed-balance) estimate of a transition matrix that keeps
    its stationary distribution: symmetrize the stationary flux
    F_ij = pi_i T_ij as (F + F^T) / 2 and renormalize each row.

    T is row-normalized here first (rows of a submatrix need not sum to 1).
    Returns (T_rev, pi), with pi T_rev = pi and pi_i T_rev_ij = pi_j T_rev_ji.
    Rows whose stationary flux is zero (bins the chain never returns to) keep
    their original transition probabilities.

    Averaging T with its transpose instead, (T + T^T) / 2, also gives a
    symmetric matrix, but its stationary distribution is no longer pi: it
    weights every bin about equally, which can merge or split metastable
    states regardless of how much probability they hold.
    """
    T = row_normalize(T)
    pi = stationary(T)
    F = pi[:, None] * T
    F = (F + F.T) / 2.0
    fs = F.sum(axis=1)
    T_rev = np.divide(F, fs[:, None], out=T.copy(), where=fs[:, None] > 0)
    return T_rev, pi


def mfpt(T, pi, source, target):
    """
    Mean first passage time from a set of bins into another, in steps of the
    matrix's lag (WE iterations): the expected number of steps to first reach
    `target` from each bin outside it, averaged over `source`'s bins weighted
    by their stationary probability. `source` and `target` are boolean masks
    over T's rows. NaN if either is empty or source has no probability.
    """
    source, target = np.asarray(source, bool), np.asarray(target, bool)
    if not source.any() or not target.any() or pi[source].sum() <= 0:
        return float("nan")
    outside = ~target
    A = np.eye(int(outside.sum())) - T[np.ix_(outside, outside)]
    try:
        m_out = np.linalg.solve(A, np.ones(int(outside.sum())))
    except np.linalg.LinAlgError:
        m_out = np.linalg.lstsq(A, np.ones(int(outside.sum())), rcond=None)[0]
    m = np.zeros(len(T))
    m[outside] = m_out
    return float((pi[source] * m[source]).sum() / pi[source].sum())


def coarse_grain(T, pi, labels, n_states):
    """
    State-to-state transition matrix at the same lag: T_IJ = sum over i in I
    of pi_i sum over j in J of T_ij, divided by pi_I. `labels` gives each bin's
    state (0..n_states-1, anything else: no state). Returns (T_states, pi_states,
    bins_per_state); rows of states holding no probability stay zero.
    """
    labels = np.asarray(labels)
    P = np.zeros((n_states, n_states))
    p = np.zeros(n_states)
    nb = np.zeros(n_states, dtype=int)
    for I in range(n_states):
        bi = labels == I
        nb[I] = int(bi.sum())
        p[I] = pi[bi].sum()
        if p[I] <= 0:
            continue
        flux = pi[bi][:, None] * T[bi]
        for J in range(n_states):
            P[I, J] = flux[:, labels == J].sum() / p[I]
    return P, p, nb


# ---------------------------------------------------------------------------
# GPCCA+
# ---------------------------------------------------------------------------

def gpcca_states(C, n_clusters=None, max_clusters=10):
    """
    Metastable states from weighted bin-to-bin transition counts by GPCCA+
    (pygpcca), on the flux-symmetrized (reversible) estimate of the
    transition matrix over the active bins.

    n_clusters: number of states, or None to let GPCCA+ choose the number
    between 2 and max_clusters with the crispest decomposition.

    Returns a dict: active (bin indices), labels (state of each active bin,
    numbered by decreasing stationary probability), n_states, crispness,
    pi (stationary distribution over the active bins), T (row-normalized
    matrix over the active bins, not symmetrized).
    """
    import pygpcca as gp
    act = active_set(C)
    if len(act) < 3:
        raise ValueError("only {} bins have transitions back and forth; GPCCA+ needs more"
                         .format(len(act)))
    T = row_normalize(C[np.ix_(act, act)])
    T_rev, pi = reversible_by_flux(T)
    g = gp.GPCCA(T_rev)
    if n_clusters is None:
        hi = int(min(max_clusters, len(act) - 1))
        if hi <= 2:
            g.optimize(2)
        else:
            g.optimize({"m_min": 2, "m_max": hi})
    else:
        g.optimize(int(n_clusters))
    raw = np.asarray(g.macrostate_assignment)
    # number states by decreasing stationary probability (state0 = most probable)
    weight = {c: pi[raw == c].sum() for c in np.unique(raw)}
    order = sorted(weight, key=lambda c: -weight[c])
    renumber = {c: k for k, c in enumerate(order)}
    labels = np.array([renumber[c] for c in raw])
    return {"active": act, "labels": labels, "n_states": len(order),
            "crispness": float(g.optimal_crispness), "pi": pi, "T": T}
