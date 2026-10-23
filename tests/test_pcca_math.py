"""
Unit tests for pcca.py's numerical pieces, each checked against an answer
worked out by hand. None of them need a simulation: the HDF5 inputs
(w_reweight's tmat.h5, w_assign's assign.h5) are written synthetically in
the same layout those tools produce.
"""
import h5py
import numpy as np
import pytest

from webng.analysis.pcca import wePCCA


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
    assert result["transition_matrix"] == pytest.approx([[0.85, 0.15], [0.15, 0.85]])
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
