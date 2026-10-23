"""
Tests for cluster.py and network.py's own logic, running their real run()
methods with no WESTPA tools: each test writes the file the WESTPA tool would
have produced (pdist.h5, assign.h5, direct.h5), stamped as built from the
current inputs, so the analysis cache reuses it instead of running the tool.
"""
import os

import h5py
import numpy as np
import pytest
import yaml

from webng.analysis.analysis import STAMP_ATTR
from webng.analysis.cluster import weCluster
from webng.analysis.network import weNetwork


def _sim(tmp_path, monkeypatch, ndim=2):
    """A minimal sim folder (just west.h5), with cwd at its parent."""
    sim = tmp_path / "sim"
    (sim / "analysis").mkdir(parents=True)
    with h5py.File(sim / "west.h5", "w") as f:
        f.attrs["west_current_iteration"] = 11
        f.create_dataset("iterations/iter_00000001/pcoord", data=np.zeros((2, 3, ndim)))
    monkeypatch.chdir(tmp_path)
    return sim


def _stamped(path, stamp, **datasets):
    with h5py.File(path, "w") as f:
        for name, data in datasets.items():
            f[name] = data
        f.attrs[STAMP_ATTR] = stamp


# ---------------------------------------------------------------------------
# cluster
# ---------------------------------------------------------------------------

def test_cluster_writes_one_state_per_dense_region(tmp_path, monkeypatch):
    sim = _sim(tmp_path, monkeypatch)
    c = weCluster({"sim_name": "sim", "work-path": None, "pcoords": ["A", "B"],
                   "bins": 10, "threshold": 90, "min-samples": 2, "eps": 1.5,
                   "first-iter": None, "last-iter": None})
    hist = np.zeros((10, 10))
    hist[1:3, 1:3] = 1.0                       # region 1: 2x2 bins
    hist[7:9, 6:9] = 1.0                       # region 2: 2x3 bins, far away
    mid_a, mid_b = np.arange(10) + 0.5, (np.arange(10) + 0.5) * 10
    # cwd is now the analysis folder (weAnalysis.__init__ moved there)
    _stamped("pdist.h5", c.stamp("w_pdist", first_iter=c.first_iter, last_iter=c.last_iter, bins=c.bins),
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

# WESTPA's iter_block_ci_dtype (westpa/tools/dtypes.py), what w_direct writes
CI_DTYPE = np.dtype([("iter_start", np.uint32), ("iter_stop", np.uint32),
                     ("expected", np.float64), ("ci_lbound", np.float64),
                     ("ci_ubound", np.float64), ("sterr", np.float64),
                     ("corr_len", np.uint32)])


def test_network_plots_named_fields_at_each_windows_end(tmp_path, monkeypatch):
    sim = _sim(tmp_path, monkeypatch)
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
    _stamped("assign.h5", assign_stamp)
    _stamped("direct.h5", n.stamp("w_direct", assign=assign_stamp, first_iter=int(n.first_iter),
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
