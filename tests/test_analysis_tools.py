"""
Tests for the analysis tools' own logic -- cluster (dbscan and gpcca),
network (w_direct and the transition matrix), flux, and the Markov-model math
they share -- with no WESTPA tools and no simulation.

cluster, network, and flux run their real run() methods: each test writes the
file the WESTPA tool would have produced (pdist.h5, assign_bins.h5, assign.h5,
direct.h5, fluxanl.h5), stamped as built from the current inputs, so
the analysis cache reuses it instead of running the tool. The math in
webng.analysis.markov is checked directly, against answers worked out by hand.
"""
import os
import pickle
import sys

import h5py
import numpy as np
import pytest
import yaml

from webng.analysis.analysis import STAMP_ATTR, weAnalysis
from webng.analysis.average import weAverage
from webng.analysis.cluster import weCluster
from webng.analysis.flux import weFlux
from webng.analysis.network import weNetwork
from webng.analysis import markov
from conftest import CI_DTYPE, write_stamped, write_west


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

# Twelve bins: group A (bins 0-3, near (100, 100), probability 0.2 each), group
# B (bins 4-7, near (3000, 3000), 0.04 each, so B is 2% of the total per bin)
# and four scattered bins (0.01 each). Walkers jump between bins with these
# probabilities (every row of T is the same), so each bin's average
# probability is exactly its entry here.
DBSCAN_P = np.array([0.2] * 4 + [0.04] * 4 + [0.01] * 4)
DBSCAN_POINTS = np.array([[100 + 10 * k, 100.0] for k in range(4)]
                         + [[3000 + 10 * k, 3000.0] for k in range(4)]
                         + [[1500, 500], [500, 2500], [2500, 700], [800, 1800.0]])


def _dbscan_cluster(analysis_sim, **opts):
    sim = analysis_sim()
    T = np.tile(DBSCAN_P, (12, 1))
    asg, nsegs, npts = _write_run(str(sim), T, 10, DBSCAN_POINTS)
    c = weCluster({"sim_name": "sim", "work-path": None, "pcoords": ["A", "B"],
                   "method": "dbscan", "first-iter": None, "last-iter": None, **opts})
    _stamp_bins("assign_bins.h5", c.stamp("w_assign bins"), asg, nsegs, npts, 12)
    return sim, c


def _states(sim):
    with open(sim / "analysis" / "states.yaml") as f:
        return yaml.safe_load(f)["states"]


@pytest.mark.parametrize("eps", [None, 0.05])
def test_dbscan_keeps_the_probable_bins_and_groups_nearby_ones(analysis_sim, eps):
    """probability-mass 0.95 keeps A and B (0.96 of the probability) and drops
    the scattered bins; DBSCAN then finds A and B, A first (more probable)."""
    sim, c = _dbscan_cluster(analysis_sim, **{"probability-mass": 0.95, "eps": eps})
    c.run()
    states = _states(sim)
    assert [s["label"] for s in states] == ["state0", "state1"]
    assert sorted(map(tuple, states[0]["coords"])) == sorted(map(tuple, DBSCAN_POINTS[:4]))
    assert sorted(map(tuple, states[1]["coords"])) == sorted(map(tuple, DBSCAN_POINTS[4:8]))
    for state in states:
        for point in state["coords"]:
            assert all(isinstance(v, float) for v in point), point   # numbers, not text
    report = (sim / "analysis" / "cluster_dbscan.txt").read_text()
    assert "kept 8 of 12 bins" in report and "2 states" in report


def test_dbscan_probability_mass_decides_which_bins_count(analysis_sim):
    """With 0.8 only group A's bins are kept: one state."""
    sim, c = _dbscan_cluster(analysis_sim, **{"probability-mass": 0.8})
    c.run()
    states = _states(sim)
    assert len(states) == 1
    assert sorted(map(tuple, states[0]["coords"])) == sorted(map(tuple, DBSCAN_POINTS[:4]))


@pytest.mark.parametrize("units", [1.0, 1000.0])
def test_dbscan_eps_is_a_fraction_of_each_coordinates_range(analysis_sim, units, monkeypatch):
    """The same eps finds the same states whatever the coordinates' units."""
    monkeypatch.setattr(sys.modules[__name__], "DBSCAN_POINTS", DBSCAN_POINTS * units)
    sim, c = _dbscan_cluster(analysis_sim, **{"probability-mass": 0.95, "eps": 0.05})
    c.run()
    states = _states(sim)
    assert len(states) == 2
    scaled = sys.modules[__name__].DBSCAN_POINTS       # already in the new units
    assert sorted(map(tuple, states[0]["coords"])) == sorted(map(tuple, scaled[:4]))


def test_scaling_and_automatic_eps(analysis_sim):
    sim, c = _dbscan_cluster(analysis_sim)
    X = c.scaled_points(DBSCAN_POINTS, DBSCAN_POINTS)
    assert X.min(axis=0) == pytest.approx([0, 0]) and X.max(axis=0) == pytest.approx([1, 1])
    c.log_scale = True
    L = c.scaled_points(np.array([[9.0, 0.0], [99.0, 0.0], [999.0, 1.0]]),
                        np.array([[9.0, 0.0], [999.0, 1.0]]))
    assert L[:, 0] == pytest.approx([0.0, 0.5, 1.0])          # log10(x + 1): 1, 2, 3
    # twice the median nearest-neighbour distance: neighbours here are 0.1 apart
    assert c.auto_eps(np.array([[0.0], [0.1], [0.2], [0.3]])) == pytest.approx(0.2)
    os.chdir(c.curr_path)


@pytest.mark.parametrize("opts, match", [
    ({"threshold": 90}, "probability-mass"),        # the old option, replaced
    ({"eps": 1.5}, "fraction"),                      # the old meaning: histogram cells
    ({"probability-mass": 0}, "probability-mass"),
])
def test_dbscan_rejects_old_style_options(analysis_sim, opts, match):
    analysis_sim()
    with pytest.raises(ValueError, match=match):
        weCluster({"sim_name": "sim", "work-path": None, "pcoords": ["A", "B"],
                   "method": "dbscan", "first-iter": None, "last-iter": None, **opts})


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

def test_network_plots_named_fields_at_each_windows_end(analysis_sim):
    sim = analysis_sim()
    with open(sim / "analysis" / "states.yaml", "w") as f:
        f.write("states:\n  - label: SC\n    coords:\n      - [0.0, 0.0]\n"
                "  - label: TE\n    coords:\n      - [9.0, 9.0]\n")
    n = weNetwork({"sim_name": "sim", "work-path": None, "pcoords": ["A", "B"],
                   "tau": 10, "step-iter": 5, "matrix": False,
                   "first-iter": None, "last-iter": None})

    # two evaluation windows, deliberately uneven: ending at iterations 4 and 10
    rates = np.zeros((2, 2, 2), dtype=CI_DTYPE)
    rates["iter_start"] = 1
    rates["iter_stop"][0], rates["iter_stop"][1] = 4, 10
    rates["expected"][:, 0, 1], rates["ci_lbound"][:, 0, 1], rates["ci_ubound"][:, 0, 1] = [.1, .2], [.05, .15], [.15, .25]
    rates["expected"][:, 1, 0], rates["ci_lbound"][:, 1, 0], rates["ci_ubound"][:, 1, 0] = [.3, .4], [.25, .35], [.35, .45]

    assign_stamp = n.stamp("w_assign", states_yaml=n.file_digest("states.yaml"))
    write_stamped("assign.h5", assign_stamp)
    write_stamped("direct.h5", n.stamp("w_direct", assign=assign_stamp, first_iter=int(n.first_iter),
                                  last_iter=int(n.last_iter), step_iter=n.step_iter),
             rate_evolution=rates)
    figs = n.run()

    assert [f.axes[0].get_title() for f in figs] == ["SC$\\rightarrow$TE", "TE$\\rightarrow$SC"]
    for fig, (a, b) in zip(figs, [(0, 1), (1, 0)]):
        line = fig.axes[0].lines[0]
        assert list(line.get_xdata()) == [4, 10]                         # real window ends
        assert list(line.get_ydata()) == pytest.approx(list(rates["expected"][:, a, b]))
        band = fig.axes[0].collections[0].get_paths()[0].vertices[:, 1]
        assert min(band) == pytest.approx(rates["ci_lbound"][:, a, b].min())
        assert max(band) == pytest.approx(rates["ci_ubound"][:, a, b].max())

# ===========================================================================
# flux (treats every line of tstates.txt as one sink)
# ===========================================================================

N_ITER = 40

def _flux_analysis(analysis_sim, per_target, separate_cis):
    """
    weFlux over a fluxanl.h5 with one target per entry of `per_target`
    (per-iteration flux series). `separate_cis` are the per-target
    (mean, lo, hi) that w_fluxanl would have reported for each.
    """
    analysis_sim(current_iteration=N_ITER + 1, ndim=1)
    fx = weFlux({"sim_name": "sim", "work-path": None, "pcoords": ["A"], "step-iter": 5,
                 "output": "flux.png", "first-iter": None, "last-iter": None})

    n_iter = np.arange(1, N_ITER + 1)
    with h5py.File("fluxanl.h5", "w") as f:      # cwd is the analysis folder now
        avg = np.zeros(len(per_target), dtype=CI_DTYPE)
        for i, (fluxes, (mean, lo, hi)) in enumerate(zip(per_target, separate_cis)):
            g = f.create_group("target_flux/target_{}".format(i))
            g["n_iter"], g["flux"] = n_iter, np.asarray(fluxes)
            evol = np.zeros(3, dtype=CI_DTYPE)
            evol["iter_start"], evol["iter_stop"] = 1, [10, 20, 40]
            evol["expected"], evol["ci_lbound"], evol["ci_ubound"] = mean, lo, hi
            g["flux_evolution"] = evol
            avg[i] = (1, N_ITER + 1, mean, lo, hi, 0.0, 0)
        f["avg_flux"] = avg
        f.attrs[STAMP_ATTR] = fx.stamp("w_fluxanl", first_iter=int(fx.first_iter),
                                       last_iter=int(fx.last_iter), step_iter=fx.step_iter)
    return fx


def test_one_target_uses_w_fluxanls_own_results(analysis_sim):
    fluxes = np.random.default_rng(1).uniform(0, 1e-3, N_ITER)
    fx = _flux_analysis(analysis_sim, [fluxes], [(5e-4, 4e-4, 6e-4)])
    fig, ax = fx.run()
    assert fx.avg_flux == pytest.approx((5e-4, 4e-4, 6e-4))
    line = ax.lines[0]
    assert list(line.get_xdata()) == [10, 20, 40]
    assert list(line.get_ydata()) == pytest.approx([5e-4] * 3)


def test_several_target_lines_are_one_sink(analysis_sim):
    """
    Two bins of one sink whose fluxes swing in opposite directions: the total
    is nearly constant, so its CI must be nearly zero width -- while adding
    the two separate CIs' bounds would give a wide interval.
    """
    rng = np.random.default_rng(2)
    swing = rng.uniform(-0.5, 0.5, N_ITER)
    a = 1.0 + swing + rng.normal(0, 1e-3, N_ITER)
    b = 1.0 - swing + rng.normal(0, 1e-3, N_ITER)
    fx = _flux_analysis(analysis_sim, [a, b],
                        [(a.mean(), a.mean() - 0.2, a.mean() + 0.2),
                         (b.mean(), b.mean() - 0.2, b.mean() + 0.2)])
    fig, ax = fx.run()

    # WESTPA's bootstrap, when it detects correlation between iterations,
    # averages in blocks and drops leftover iterations that don't fill a final
    # block (w_fluxanl behaves identically), so its "mean" can differ from the
    # exact mean by a little. Whether it detects correlation depends on its own
    # random draws, so compare to well within the total's noise, not exactly.
    close = dict(abs=1e-3)
    total = a + b
    mean, lo, hi = fx.avg_flux
    assert mean == pytest.approx(a.mean() + b.mean(), **close)    # means do add
    assert lo <= mean <= hi
    assert hi - lo < 0.02                                      # vs 0.8 from adding the bounds

    # evolution: cumulative mean of the TOTAL, in blocks of step-iter (5), as w_fluxanl does
    # (w_fluxanl counts iterations as last - first = 39 here, so 8 blocks, each
    # ending at min(first + 5*(k+1), last), each averaging the first 5*(k+1) iterations)
    line = ax.lines[0]
    assert list(line.get_xdata()) == [min(1 + 5 * (k + 1), N_ITER) for k in range(8)]
    assert list(line.get_ydata()) == pytest.approx(
        [total[:min(5 * (k + 1), N_ITER)].mean() for k in range(8)], **close)

# ===========================================================================
# Markov-model math (webng.analysis.markov)
# ===========================================================================

def test_stationary_two_state():
    # leave state 0 with p=0.2, leave state 1 with p=0.1  ->  pi = (1/3, 2/3)
    T = np.array([[0.8, 0.2],
                  [0.1, 0.9]])
    assert markov.stationary(T) == pytest.approx([1 / 3, 2 / 3])


# 0 -> 1 -> 2, each step taken with probability 1/2, absorbing at 2.
# By hand: m1 = 1 + m1/2  ->  m1 = 2;   m0 = 1 + m0/2 + m1/2  ->  m0 = 4.
CHAIN = np.array([[0.5, 0.5, 0.0],
                  [0.0, 0.5, 0.5],
                  [0.0, 0.0, 1.0]])
UNIFORM = np.full(3, 1 / 3)


@pytest.mark.parametrize("source, expected", [(0, 4.0), (1, 2.0)])
def test_mfpt_hand_solved_chain(source, expected):
    src = np.arange(3) == source
    assert markov.mfpt(CHAIN, UNIFORM, src, np.arange(3) == 2) == pytest.approx(expected)


def test_mfpt_source_spread_over_bins_is_stationary_weighted():
    # source covers bins 0 and 1, weighted 1:3 by stationary probability
    pi = np.array([0.1, 0.3, 0.6])
    m = markov.mfpt(CHAIN, pi, np.array([True, True, False]), np.array([False, False, True]))
    assert m == pytest.approx(0.25 * 4.0 + 0.75 * 2.0)


def test_mfpt_missing_target_is_nan():
    assert np.isnan(markov.mfpt(CHAIN, UNIFORM, np.arange(3) == 0, np.zeros(3, bool)))


def test_active_set_drops_bins_the_chain_never_returns_from():
    C = np.array([[5.0, 1.0, 0.0, 0.0],
                  [1.0, 5.0, 0.0, 0.0],
                  [1.0, 0.0, 0.0, 0.0],      # bin 2 only ever leaves
                  [0.0, 0.0, 0.0, 0.0]])     # bin 3 never visited
    assert list(markov.active_set(C)) == [0, 1]


def test_coarse_graining_onto_states():
    # symmetric with unit row sums, so every bin's stationary probability is 1/4
    T = np.array([[0.7, 0.2, 0.1, 0.0],
                  [0.2, 0.6, 0.1, 0.1],
                  [0.1, 0.1, 0.6, 0.2],
                  [0.0, 0.1, 0.2, 0.7]])
    P, p, nb = markov.coarse_grain(T, np.full(4, 0.25), np.array([0, 0, 1, 1]), 2)
    # A -> B = (T02 + T03 + T12 + T13) / 2 = (0.1 + 0.0 + 0.1 + 0.1) / 2
    assert P == pytest.approx(np.array([[0.85, 0.15], [0.15, 0.85]]))
    assert p == pytest.approx([0.5, 0.5])
    assert list(nb) == [2, 2]


def test_bins_outside_every_state_belong_to_none():
    T = np.full((4, 4), 0.25)
    P, p, nb = markov.coarse_grain(T, np.full(4, 0.25), np.array([0, -1, 1, 1]), 2)
    assert list(nb) == [1, 2]
    assert p == pytest.approx([0.25, 0.5])


def _nonreversible_T(n=6, seed=3):
    rng = np.random.default_rng(seed)
    T = rng.random((n, n)) ** 3          # uneven, so pi is far from uniform
    return T / T.sum(axis=1, keepdims=True)


def test_reversible_by_flux_keeps_pi_and_detailed_balance():
    T = _nonreversible_T()
    T_rev, pi = markov.reversible_by_flux(T)
    assert pi == pytest.approx(markov.stationary(T))
    assert T_rev.sum(axis=1) == pytest.approx(np.ones(len(T)))
    assert pi @ T_rev == pytest.approx(pi)                      # same stationary distribution
    F = pi[:, None] * T_rev
    assert F == pytest.approx(F.T)                              # detailed balance


def test_averaging_with_transpose_loses_pi():
    """Why GPCCA+ no longer uses (T + T^T) / 2: its stationary distribution is not pi."""
    T = _nonreversible_T()
    S = (T + T.T) / 2.0
    S = S / S.sum(axis=1, keepdims=True)
    assert np.abs(markov.stationary(S) - markov.stationary(T)).max() > 0.05


def _three_metastable_groups():
    """Nine bins in three metastable groups of three; group 2 holds ~1e-4 of
    the probability (like LN2 in the 8-gene model). Returns (T, groups)."""
    groups = np.repeat([0, 1, 2], 3)
    out_rate = {0: 2e-3, 1: 5e-3, 2: 5e-2}              # group 2 is left quickly ...
    T = np.zeros((9, 9))
    for i in range(9):
        g = groups[i]
        T[i, groups == g] = (1 - out_rate[g]) / 3
        others = groups != g
        T[i, others] = out_rate[g] / others.sum()
    T[np.ix_(groups != 2, groups == 2)] *= 1e-2          # ... and entered rarely
    return T / T.sum(axis=1, keepdims=True), groups


@pytest.mark.parametrize("n_clusters", [3, None])
def test_gpcca_states_separate_a_rare_metastable_state(n_clusters):
    """GPCCA+ must give each group its own state, the rare one included, with
    its true (tiny) stationary probability; with n_clusters None it should
    choose three by crispness."""
    pytest.importorskip("pygpcca")
    T, groups = _three_metastable_groups()
    pi = markov.stationary(T)
    assert pi[groups == 2].sum() < 1e-3
    g = markov.gpcca_states(pi[:, None] * T, n_clusters=n_clusters)    # counts ~ stationary flux
    assert g["n_states"] == 3
    labels = g["labels"]
    for grp in range(3):
        assert len(set(labels[groups == grp])) == 1                    # one state per group
    assert len({labels[groups == grp][0] for grp in range(3)}) == 3
    assert labels[groups == 2][0] == 2                                  # numbered by probability
    _, p, _ = markov.coarse_grain(g["T"], g["pi"], labels, 3)
    assert p[2] == pytest.approx(pi[groups == 2].sum(), rel=1e-6)


# ---------------------------------------------------------------------------
# bin-to-bin transition counts from w_assign output
# ---------------------------------------------------------------------------

def _write_run(sim, T, n_iter, rep):
    """
    A west.h5 and the matching bin assignments of a run that moves exactly as
    the transition matrix T says: every iteration, one segment per (i, j)
    with T_ij > 0, weighted pi_i T_ij, from bin i to bin j. A walker in bin b
    sits at the progress coordinate rep[b]. Returns the assignments, nsegs,
    npts that w_assign would write.
    """
    pi = markov.stationary(T)
    pairs = [(i, j) for i in range(len(T)) for j in range(len(T)) if T[i, j] > 0]
    w = np.array([pi[i] * T[i, j] for i, j in pairs])
    seg_dtype = np.dtype([("weight", np.float64), ("parent_id", np.int64)])
    with h5py.File(os.path.join(sim, "west.h5"), "w") as f:
        f.attrs["west_current_iteration"] = n_iter + 1
        for n in range(1, n_iter + 1):
            g = f.create_group("iterations/iter_{:08d}".format(n))
            seg = np.zeros(len(pairs), dtype=seg_dtype)
            seg["weight"] = w
            g["seg_index"] = seg
            g["pcoord"] = np.array([[rep[i], rep[j]] for i, j in pairs], dtype=np.float32)
    asg = np.array([[[i, j] for i, j in pairs]] * n_iter, dtype=np.uint16)
    return asg, np.full(n_iter, len(pairs)), np.full(n_iter, 2)


def _stamp_bins(path, stamp, asg, nsegs, npts, nbins):
    write_stamped(path, stamp, assignments=asg, nsegs=nsegs, npts=npts)
    with h5py.File(path, "a") as f:
        f.attrs["nbins"] = nbins
        f.attrs["iter_start"] = 1


def test_bin_flux_from_assignments(tmp_path):
    T = CHAIN.copy()
    T[2] = [0.5, 0.0, 0.5]                    # make it recurrent
    rep = np.array([[0.0, 0.0], [1.0, 10.0], [2.0, 20.0]])
    asg, nsegs, npts = _write_run(str(tmp_path), T, 4, rep)
    _stamp_bins(str(tmp_path / "assign.h5"), "x", asg, nsegs, npts, 3)
    C, points, visited = markov.bin_flux_from_assignments(
        str(tmp_path / "assign.h5"), str(tmp_path / "west.h5"), 2, 4)
    pi = markov.stationary(T)
    assert C == pytest.approx(3 * pi[:, None] * T)          # 3 iterations of stationary flux
    assert markov.row_normalize(C) == pytest.approx(T)
    assert visited.all()
    assert points == pytest.approx(rep)                     # a real point inside each bin


# ---------------------------------------------------------------------------
# cluster, method gpcca
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_clusters", [3, None])
def test_cluster_gpcca_writes_states_of_whole_bins(analysis_sim, n_clusters):
    pytest.importorskip("pygpcca")
    sim = analysis_sim()
    T, groups = _three_metastable_groups()
    rep = np.array([[float(b), 10.0 * b] for b in range(9)])
    asg, nsegs, npts = _write_run(str(sim), T, 10, rep)
    c = weCluster({"sim_name": "sim", "work-path": None, "pcoords": ["A", "B"],
                   "method": "gpcca", "n-clusters": n_clusters,
                   "first-iter": None, "last-iter": None})
    # cwd is now the analysis folder (weAnalysis.__init__ moved there)
    _stamp_bins("assign_bins.h5", c.stamp("w_assign bins"), asg, nsegs, npts, 9)
    c.run()

    with open(sim / "analysis" / "states.yaml") as f:
        states = yaml.safe_load(f)["states"]
    assert [s["label"] for s in states] == ["state0", "state1", "state2"]
    found = sorted(sorted(map(tuple, s["coords"])) for s in states)
    assert found == sorted(sorted(tuple(rep[b]) for b in np.nonzero(groups == g)[0]) for g in range(3))
    # the least probable state is the rare group, and it is reported as such
    assert sorted(map(tuple, states[2]["coords"])) == sorted(tuple(rep[b]) for b in (6, 7, 8))
    report = (sim / "analysis" / "cluster_gpcca.txt").read_text()
    assert "3 states" in report and "network" in report


def test_cluster_rejects_an_unknown_method(analysis_sim):
    analysis_sim()
    with pytest.raises(ValueError, match="method"):
        weCluster({"sim_name": "sim", "work-path": None, "pcoords": ["A", "B"],
                   "method": "kmeans", "first-iter": None, "last-iter": None})


# ---------------------------------------------------------------------------
# network: the transition matrix (from assign.h5) and its MFPTs
# ---------------------------------------------------------------------------

def _network_with_matrix(analysis_sim, T, state_map, labels=("A", "B"), tau=10, n_iter=10):
    """
    weNetwork over a run that moves exactly as T says (see _write_run), with
    w_assign's assign.h5 (bins mapped to states by `state_map`) and w_direct's
    direct.h5 prepared, so neither tool runs.
    """
    sim = analysis_sim(current_iteration=n_iter + 1)
    with open(sim / "analysis" / "states.yaml", "w") as f:
        yaml.safe_dump({"states": [{"label": l, "coords": [[float(k), 0.0]]}
                                   for k, l in enumerate(labels)]}, f)
    rep = np.array([[float(b), 0.0] for b in range(len(T))])
    asg, nsegs, npts = _write_run(str(sim), T, n_iter, rep)
    n = weNetwork({"sim_name": "sim", "work-path": None, "pcoords": ["A", "B"],
                   "tau": tau, "step-iter": 1, "first-iter": None, "last-iter": None})
    ns = len(labels)
    assign_stamp = n.stamp("w_assign", states_yaml=n.file_digest("states.yaml"))
    write_stamped("assign.h5", assign_stamp, assignments=asg, nsegs=nsegs, npts=npts,
                  state_map=np.array(list(state_map) + [ns]),
                  state_labels=np.array([l.encode() for l in labels]))
    with h5py.File("assign.h5", "a") as f:
        f.attrs["nstates"], f.attrs["nbins"], f.attrs["iter_start"] = ns, len(T), 1
    rates = np.zeros((2, ns, ns), dtype=CI_DTYPE)
    rates["iter_start"] = 1
    rates["iter_stop"][0], rates["iter_stop"][1] = 5, 10      # two evaluation windows
    rates["expected"] = 0.05
    write_stamped("direct.h5", n.stamp("w_direct", assign=assign_stamp, first_iter=int(n.first_iter),
                                       last_iter=int(n.last_iter), step_iter=n.step_iter),
                  rate_evolution=rates)
    return sim, n


# a 3-bin cycle, each step taken with probability 1/2: pi uniform
CYCLE = np.array([[0.5, 0.5, 0.0],
                  [0.0, 0.5, 0.5],
                  [0.5, 0.0, 0.5]])


def test_network_matrix_mfpts_hand_solved(analysis_sim):
    """A = bins {0, 1}, B = bin 2. To B: m1 = 2, m0 = 1 + m0/2 + m1/2 -> m0 = 4,
    so A -> B = (4 + 2)/2 = 3 steps; B -> A: m2 = 1 + m2/2 = 2 steps."""
    sim, n = _network_with_matrix(analysis_sim, CYCLE, state_map=[0, 0, 1], tau=10)
    n.run()
    m = n.result["matrix"]
    assert m["mfpt"][(0, 1)][0] == pytest.approx(30.0)
    assert m["mfpt"][(1, 0)][0] == pytest.approx(20.0)
    assert m["state_prob"] == pytest.approx([2 / 3, 1 / 3])
    # A leaves to B only from bin 1 (half its probability) with p = 1/2
    assert m["transition_matrix"] == pytest.approx(np.array([[0.75, 0.25], [0.5, 0.5]]))
    report = (sim / "analysis" / "network.txt").read_text()
    assert "A -> B" in report and "30" in report
    assert "200" in report                               # direct: tau / rate = 10 / 0.05


def test_network_matrix_uses_only_the_analysis_iterations(analysis_sim):
    """first-iter/last-iter bound the matrix's iterations as they do w_direct's."""
    sim, n = _network_with_matrix(analysis_sim, CYCLE, state_map=[0, 0, 1], tau=10)
    seen = [i for i, _ in markov.iter_bin_flux("assign.h5", n.west_path, 3, 7)]
    assert seen == [3, 4, 5, 6, 7]
    os.chdir(n.curr_path)


def test_state_assignment_reads_state_map(tmp_path):
    with h5py.File(tmp_path / "assign.h5", "w") as f:
        f["state_labels"] = np.array([b"SC", b"TE"])
        # w_assign marks bins outside every state with an index past the last
        # state; state_map can also run longer than the number of bins
        f["state_map"] = np.array([0, 2, 1, 1, 0, 2])
    labels, names = weNetwork.load_bin_state_assignment(str(tmp_path / "assign.h5"), 5)
    assert list(labels) == [0, -1, 1, 1, 0]
    assert names == {0: "SC", 1: "TE"}


# ---------------------------------------------------------------------------
# the analyses dispatcher
# ---------------------------------------------------------------------------

def test_a_pcca_block_says_where_pcca_went(tmp_path):
    from webng.core.weAnalysis import weAnalysis as Dispatcher
    opts = {"analyses": {"enabled": True, "analysis_bins": 10, "first-iter": None,
                         "last-iter": None, "pcca": {"enabled": True}},
            "sampling_options": {"tau": 10}}
    (tmp_path / "opts.yaml").write_text(yaml.safe_dump(opts))

    class Args:
        pass
    args = Args()
    args.opts = str(tmp_path / "opts.yaml")
    with pytest.raises(SystemExit, match="method: gpcca"):
        Dispatcher(args).run()

# ===========================================================================
# stored bin mappers (weAnalysis.load_run_mapper), used by average
# ===========================================================================

SYSTEM_PY = """
import os
{top}

# stand-ins for WESTPA's mapper classes, defined here like a custom mapper
class Rect:
    def __init__(self, boundaries): self.boundaries = boundaries
class Voro:
    def __init__(self, centers): self.centers = centers
class Single:
    pass

class System:
    def initialize(self):
        {init}
"""

# a system.py reading a file next to it (like west_movement.h5), either when
# imported or in initialize(); initialize() must never be called either way
PLACEMENTS = {
    "at import": dict(top="CENTERS = open('west_movement.dat').read()",
                      init="raise RuntimeError('initialize() should never be called')"),
    "in initialize": dict(top="",
                          init="open('west_movement.dat').read(); raise RuntimeError('called')"),
}


def _sim_with_stored_mapper(tmp_path, placement, make_mapper):
    """A sim whose west.h5 stores make_mapper(system_module) for iterations 1-2."""
    sim = tmp_path / "sim"
    (sim / "analysis").mkdir(parents=True)
    (sim / "system.py").write_text(SYSTEM_PY.format(**PLACEMENTS[placement]))
    (sim / "west_movement.dat").write_text("centers")
    write_west(sim, 3, ndim=2)
    # pickle a mapper defined in this system.py, the way WESTPA stores one
    here = os.getcwd()
    sys.path.insert(0, str(sim))
    os.chdir(sim)
    sys.modules.pop("system", None)
    try:
        import system
        data = np.frombuffer(pickle.dumps(make_mapper(system)), dtype=np.uint8)
    finally:
        sys.modules.pop("system", None)
        sys.path.remove(str(sim))
        os.chdir(here)
    index = np.zeros(1, dtype=[("hash", "S64"), ("pickle_len", "i8")])
    index[0] = (b"abc123", len(data))
    with h5py.File(sim / "west.h5", "a") as f:
        f.create_dataset("bin_topologies/index", data=index)
        f.create_dataset("bin_topologies/pickles", data=data[None, :])
        for n in (1, 2):
            f["iterations/iter_{:08d}".format(n)].attrs["binhash"] = "abc123"
    return sim


def _analysis_elsewhere(tmp_path, monkeypatch, cls=None, **opts):
    """An analysis of tmp_path/sim working from a folder outside the sim folder."""
    monkeypatch.chdir(tmp_path)
    base = {"sim_name": "sim", "work-path": "elsewhere", "pcoords": ["A", "B"],
            "first-iter": None, "last-iter": None}
    base.update(opts)
    return (cls or weAnalysis)(base)


@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
def test_stored_mapper_loads_from_outside_the_sim_folder(tmp_path, monkeypatch, placement):
    _sim_with_stored_mapper(tmp_path, placement, lambda s: s.Voro(np.array([[1.0, 2.0]])))
    an = _analysis_elsewhere(tmp_path, monkeypatch)
    try:
        mapper = an.load_run_mapper()
        assert type(mapper).__name__ == "Voro" and mapper.centers.tolist() == [[1.0, 2.0]]
        assert os.getcwd() == str(tmp_path / "elsewhere")        # back where it was
    finally:
        an.h5file.close()
        os.chdir(an.curr_path)
        sys.modules.pop("system", None)


@pytest.mark.parametrize("make, style", [
    (lambda s: s.Rect([[0, 10, 20], [0, 5]]), "regular"),
    (lambda s: s.Voro(np.array([[1.0, 2.0]])), "adaptive"),
    (lambda s: s.Single(), "none"),
])
def test_average_reads_the_binning_style_from_the_stored_mapper(tmp_path, monkeypatch, capsys, make, style):
    _sim_with_stored_mapper(tmp_path, "at import", make)
    av = _analysis_elsewhere(tmp_path, monkeypatch, cls=weAverage, **{"plot-boundaries": True})
    try:
        assert av.style == style
        assert "Warning" not in capsys.readouterr().out
        if style == "regular":
            assert av.boundaries == [[0, 10, 20], [0, 5]]
    finally:
        av.h5file.close()
        os.chdir(av.curr_path)
        sys.modules.pop("system", None)
