"""
Tests for the analysis tools' own logic -- cluster, network, flux, and pcca's
math -- with no WESTPA tools and no simulation.

cluster, network, and flux run their real run() methods: each test writes the
file the WESTPA tool would have produced (pdist.h5, assign.h5, direct.h5,
fluxanl.h5), stamped as built from the current inputs, so the analysis cache
reuses it instead of running the tool. pcca's numerical pieces are checked
directly, each against an answer worked out by hand.
"""
import h5py
import numpy as np
import pytest
import yaml

from webng.analysis.analysis import STAMP_ATTR
from webng.analysis.cluster import weCluster
from webng.analysis.flux import weFlux
from webng.analysis.network import weNetwork
from webng.analysis.pcca import wePCCA
from conftest import CI_DTYPE, write_stamped


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

def test_cluster_writes_one_state_per_dense_region(analysis_sim):
    sim = analysis_sim()
    c = weCluster({"sim_name": "sim", "work-path": None, "pcoords": ["A", "B"],
                   "bins": 10, "threshold": 90, "min-samples": 2, "eps": 1.5,
                   "first-iter": None, "last-iter": None})
    hist = np.zeros((10, 10))
    hist[1:3, 1:3] = 1.0                       # region 1: 2x2 bins
    hist[7:9, 6:9] = 1.0                       # region 2: 2x3 bins, far away
    mid_a, mid_b = np.arange(10) + 0.5, (np.arange(10) + 0.5) * 10
    # cwd is now the analysis folder (weAnalysis.__init__ moved there)
    write_stamped("pdist.h5", c.stamp("w_pdist", first_iter=c.first_iter, last_iter=c.last_iter, bins=c.bins),
             histograms=np.stack([hist, hist]), midpoints_0=mid_a, midpoints_1=mid_b)
    c.run()

    with open(sim / "analysis" / "states.yaml") as f:
        states = yaml.safe_load(f)["states"]
    assert len(states) == 2
    for state in states:
        for point in state["coords"]:
            assert all(isinstance(v, float) for v in point), point   # numbers, not text
    found = sorted(sorted(map(tuple, s["coords"])) for s in states)
    expected = sorted([
        sorted((mid_a[i], mid_b[j]) for i in (1, 2) for j in (1, 2)),
        sorted((mid_a[i], mid_b[j]) for i in (7, 8) for j in (6, 7, 8)),
    ])
    assert found == expected


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

def test_network_plots_named_fields_at_each_windows_end(analysis_sim):
    sim = analysis_sim()
    with open(sim / "analysis" / "states.yaml", "w") as f:
        f.write("states:\n  - label: SC\n    coords:\n      - [0.0, 0.0]\n"
                "  - label: TE\n    coords:\n      - [9.0, 9.0]\n")
    n = weNetwork({"sim_name": "sim", "work-path": None, "pcoords": ["A", "B"],
                   "tau": 10, "step-iter": 5, "first-iter": None, "last-iter": None})

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
# pcca math
# ===========================================================================

@pytest.fixture
def pcca():
    # the math methods don't depend on __init__ (which needs a full sim)
    return wePCCA.__new__(wePCCA)


# ---------------------------------------------------------------------------
# stationary probabilities
# ---------------------------------------------------------------------------

def test_stationary_prob_two_state(pcca):
    # leave state 0 with p=0.2, leave state 1 with p=0.1  ->  pi = (1/3, 2/3)
    T = np.array([[0.8, 0.2],
                  [0.1, 0.9]])
    assert pcca.stationary_prob(T, 2) == pytest.approx([1 / 3, 2 / 3])


# ---------------------------------------------------------------------------
# mean first passage times
# ---------------------------------------------------------------------------

# 0 -> 1 -> 2, each step taken with probability 1/2, absorbing at 2.
# By hand: m1 = 1 + m1/2  ->  m1 = 2;   m0 = 1 + m0/2 + m1/2  ->  m0 = 4.
CHAIN = np.array([[0.5, 0.5, 0.0],
                  [0.0, 0.5, 0.5],
                  [0.0, 0.0, 1.0]])
EACH_BIN_ITS_OWN_STATE = np.array([0, 1, 2])
IDENTITY_MAP = {0: 0, 1: 1, 2: 2}


@pytest.mark.parametrize("source, expected", [(0, 4.0), (1, 2.0)])
def test_mfpt_hand_solved_chain(pcca, source, expected):
    stat = np.full(3, 1 / 3)
    mfpt = pcca.compute_mfpt_pair(CHAIN, 3, IDENTITY_MAP, EACH_BIN_ITS_OWN_STATE,
                                  source, 2, stat)
    assert mfpt == pytest.approx(expected)


def test_mfpt_source_spread_over_bins_is_stationary_weighted(pcca):
    # source state covers bins 0 and 1, weighted 1:3 by stationary probability
    assignments = np.array([0, 0, 1])
    stat = np.array([0.1, 0.3, 0.6])
    mfpt = pcca.compute_mfpt_pair(CHAIN, 3, IDENTITY_MAP, assignments, 0, 1, stat)
    assert mfpt == pytest.approx(0.25 * 4.0 + 0.75 * 2.0)


def test_mfpt_missing_target_is_nan(pcca):
    mfpt = pcca.compute_mfpt_pair(CHAIN, 3, IDENTITY_MAP, EACH_BIN_ITS_OWN_STATE,
                                  0, 7, np.full(3, 1 / 3))
    assert np.isnan(mfpt)


# ---------------------------------------------------------------------------
# transition matrix from w_reweight's tmat.h5
# ---------------------------------------------------------------------------

def _write_tmat(path, per_iteration, nbins):
    """
    tmat.h5 in w_reweight's layout: each bin is a 2x2 block of rows/cols
    (w_reweight's own doubling, unrelated to macrostates). `per_iteration`
    maps n_iter -> dense nbins x nbins flux matrix; each bin-level entry is
    split across its 2x2 block so the block sums back to it.
    """
    with h5py.File(path, "w") as f:
        f.attrs["nrows"] = 2 * nbins
        f.attrs["ncols"] = 2 * nbins
        iters = sorted(per_iteration)
        f.attrs["iter_start"] = iters[0]
        f.attrs["iter_stop"] = iters[-1] + 1
        for n_iter, flux in per_iteration.items():
            rows, cols, vals = [], [], []
            for bi in range(nbins):
                for bj in range(nbins):
                    if flux[bi, bj]:
                        for r, c, share in [(0, 0, 0.5), (1, 1, 0.3), (0, 1, 0.2)]:
                            rows.append(2 * bi + r)
                            cols.append(2 * bj + c)
                            vals.append(share * flux[bi, bj])
            g = f.create_group("iterations/iter_{:08d}".format(n_iter))
            g["rows"], g["cols"], g["flux"] = np.array(rows), np.array(cols), np.array(vals)


def test_transition_matrix_averaged_and_row_normalized(pcca, tmp_path):
    it1 = np.array([[3.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0]])      # bin 2 never visited
    it2 = np.array([[1.0, 3.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0]])
    _write_tmat(tmp_path / "tmat.h5", {1: it1, 2: it2}, nbins=3)
    tm_norm, all_iter_tms, nbins = pcca.build_transition_matrix(str(tmp_path / "tmat.h5"))

    assert nbins == 3
    assert all_iter_tms[1] == pytest.approx(it1)      # 2x2 blocks summed back to bins
    avg = (it1 + it2) / 2
    assert tm_norm[0] == pytest.approx(avg[0] / avg[0].sum())
    assert tm_norm[1] == pytest.approx(avg[1] / avg[1].sum())
    assert tm_norm[:2].sum(axis=1) == pytest.approx([1.0, 1.0])
    assert not tm_norm[2].any()                        # unvisited stays zero


# ---------------------------------------------------------------------------
# coarse-graining onto the states.yaml partition
# ---------------------------------------------------------------------------

def test_states_yaml_coarse_graining(pcca):
    # symmetric with unit row sums, so every bin's stationary probability is 1/4
    T = np.array([[0.7, 0.2, 0.1, 0.0],
                  [0.2, 0.6, 0.1, 0.1],
                  [0.1, 0.1, 0.6, 0.2],
                  [0.0, 0.1, 0.2, 0.7]])
    assignments = np.array([0, 0, 1, 1])      # bins 0,1 -> A; bins 2,3 -> B
    result, _ = pcca.summarize_states_yaml_partition(T, assignments, {0: "A", 1: "B"})
    # A -> B = (T02 + T03 + T12 + T13) / 2 = (0.1 + 0.0 + 0.1 + 0.1) / 2
    assert result["transition_matrix"] == pytest.approx(np.array([[0.85, 0.15], [0.15, 0.85]]))
    assert result["stationary_probability"] == pytest.approx([0.5, 0.5])
    assert list(result["bin_counts"]) == [2, 2]
    assert result["state_order"] == ["A", "B"]


def test_unlabeled_bins_belong_to_no_state(pcca):
    T = np.array([[0.7, 0.2, 0.1, 0.0],
                  [0.2, 0.6, 0.1, 0.1],
                  [0.1, 0.1, 0.6, 0.2],
                  [0.0, 0.1, 0.2, 0.7]])
    assignments = np.array([0, -999, 1, 1])  # bin 1 outside every state
    result, _ = pcca.summarize_states_yaml_partition(T, assignments, {0: "A", 1: "B"})
    assert list(result["bin_counts"]) == [1, 2]
    assert result["stationary_probability"] == pytest.approx([0.25, 0.5])


# ---------------------------------------------------------------------------
# bin -> state assignment from w_assign's assign.h5
# ---------------------------------------------------------------------------

def test_state_assignment_reads_state_map(pcca, tmp_path):
    with h5py.File(tmp_path / "assign.h5", "w") as f:
        f["state_labels"] = np.array([b"SC", b"TE"])
        # w_assign marks bins outside every state with an index past the last
        # state; state_map can also run longer than the number of bins
        f["state_map"] = np.array([0, 2, 1, 1, 0, 2])
    assignments, labels = pcca.load_bin_state_assignment(str(tmp_path / "assign.h5"), 5)
    assert list(assignments) == [0, -999, 1, 1, 0]
    assert labels == {0: "SC", 1: "TE"}