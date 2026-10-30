"""
Tests for how analyses reuse the files WESTPA tools build (pdist.h5,
assign.h5, direct.h5, ...): reuse only when built from the same inputs,
rebuild otherwise, and fail loudly when a tool fails.

The WESTPA tools are replaced by tiny fake executables on PATH that log
each call, so these run fast and don't need a real simulation.
"""
import json
import os
import sys
import time

import h5py
import numpy as np
import pytest

from webng.analysis.analysis import weAnalysis, STAMP_ATTR
from conftest import write_west

FAKE_TOOL = """#!{python}
import json, os, sys, h5py
name = os.path.basename(sys.argv[0])
with open(os.environ["WEBNG_TEST_CALLS"], "a") as f:
    f.write(name + "\\n")
with open(os.environ["WEBNG_TEST_CALLS"] + ".cwd", "a") as f:
    f.write(name + " " + os.getcwd() + "\\n")
with open(os.environ["WEBNG_TEST_CALLS"] + ".args", "a") as f:
    f.write(json.dumps([name] + sys.argv[1:]) + "\\n")
if os.environ.get("WEBNG_TEST_FAIL") == name:
    sys.exit(3)
out = sys.argv[sys.argv.index("-o") + 1] if "-o" in sys.argv else "direct.h5"
print("Loading data...")
print(name + " done")
h5py.File(out, "w").close()
"""


@pytest.fixture
def sim(tmp_path, monkeypatch):
    """A sim folder with west.h5 and analysis/, fake tools on PATH, cwd = analysis/."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ["w_pdist", "w_assign", "w_direct", "w_reweight", "w_fluxanl"]:
        tool = bin_dir / name
        tool.write_text(FAKE_TOOL.format(python=sys.executable))
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    calls = tmp_path / "calls.log"
    monkeypatch.setenv("WEBNG_TEST_CALLS", str(calls))

    sim_dir = tmp_path / "sim"
    (sim_dir / "analysis").mkdir(parents=True)
    write_west(sim_dir, 101, ndim=1)
    monkeypatch.chdir(sim_dir / "analysis")

    an = weAnalysis.__new__(weAnalysis)
    an.sim_path, an.west_path = str(sim_dir), str(sim_dir / "west.h5")
    an.first_iter, an.last_iter, an.bins = 1, 100, 30

    def take_calls():
        names = calls.read_text().split() if calls.exists() else []
        if calls.exists():
            calls.unlink()
        return names

    return sim_dir, an, take_calls


def test_first_call_builds_and_stamps(sim):
    _, an, take_calls = sim
    an.ensure_pdist().close()
    assert take_calls() == ["w_pdist"]
    with h5py.File("pdist.h5", "r") as f:
        assert STAMP_ATTR in f.attrs


def test_same_inputs_are_reused(sim):
    _, an, take_calls = sim
    an.ensure_pdist().close()
    take_calls()
    an.ensure_pdist().close()
    assert take_calls() == []


@pytest.mark.parametrize("change", ["bins", "first_iter", "last_iter"])
def test_changed_settings_rebuild(sim, change):
    _, an, take_calls = sim
    an.ensure_pdist().close()
    take_calls()
    # (last_iter moves down: the run only has 100 finished iterations)
    setattr(an, change, getattr(an, change) + (-7 if change == "last_iter" else 7))
    an.ensure_pdist().close()
    assert take_calls() == ["w_pdist"]


def test_continued_or_replaced_run_rebuilds(sim):
    sim_dir, an, take_calls = sim
    an.ensure_pdist().close()
    take_calls()
    write_west(sim_dir, 160, ndim=1)            # run continued
    an.ensure_pdist().close()
    assert take_calls() == ["w_pdist"]
    time.sleep(0.01)
    write_west(sim_dir, 160, ndim=1, padding=50)   # different run, same iteration count
    an.ensure_pdist().close()
    assert take_calls() == ["w_pdist"]


def test_unstamped_old_file_rebuilds(sim):
    _, an, take_calls = sim
    h5py.File("pdist.h5", "w").close()          # e.g. made before stamping existed
    an.ensure_pdist().close()
    assert take_calls() == ["w_pdist"]


def test_failing_tool_raises_then_reruns(sim, monkeypatch):
    _, an, take_calls = sim
    monkeypatch.setenv("WEBNG_TEST_FAIL", "w_pdist")
    with pytest.raises(RuntimeError, match="w_pdist failed"):
        an.ensure_pdist()
    monkeypatch.delenv("WEBNG_TEST_FAIL")
    take_calls()
    an.ensure_pdist().close()
    assert take_calls() == ["w_pdist"]


def test_assign_shared_and_dependents_follow_states_yaml(sim):
    sim_dir, an, take_calls = sim
    with open("states.yaml", "w") as f:
        f.write("states:\n  - label: SC\n")
    first = an.ensure_assign()
    assert take_calls() == ["w_assign"]
    assert an.ensure_assign() == first          # second tool reuses it
    assert take_calls() == []

    def direct(assign_stamp):
        an.run_westpa_tool("direct.h5", ["w_direct", "all"], an.stamp("w_direct", assign=assign_stamp),
                           cwd="../", produced_at=os.path.join("..", "direct.h5"),
                           log_path="direct_output.txt")

    direct(first)
    assert take_calls() == ["w_direct"]
    assert os.path.isfile("direct.h5") and not (sim_dir / "direct.h5").exists()
    assert open("direct_output.txt").read() == "w_direct done\n"   # "..." lines dropped

    with open("states.yaml", "w") as f:
        f.write("states:\n  - label: TE\n")
    second = an.ensure_assign()
    assert take_calls() == ["w_assign"] and second != first
    direct(second)
    assert take_calls() == ["w_direct"]                            # dependent rebuilt too


def test_work_path_outside_sim_folder(sim, tmp_path, monkeypatch):
    """The analysis folder doesn't have to sit inside the sim folder."""
    sim_dir, _, take_calls = sim
    monkeypatch.chdir(tmp_path)
    an = weAnalysis({"sim_name": "sim", "work-path": "elsewhere", "pcoords": ["A"],
                     "first-iter": 5, "last-iter": None})
    try:
        assert an.work_path == "elsewhere"
        assert os.getcwd() == str(tmp_path / "elsewhere")
        assert str(sim_dir) in sys.path                         # absolute, for system.py
        assert (an.first_iter, an.last_iter) == (5, 100)        # last finished iteration
        an.bins = 30
        an.ensure_pdist().close()
        with open("states.yaml", "w") as f:
            f.write("states:\n  - label: SC\n")
        an.ensure_assign()
        assert take_calls() == ["w_pdist", "w_assign"]
        assert os.path.isfile("pdist.h5") and os.path.isfile("assign.h5")
        cwds = dict(line.split(" ", 1) for line in
                    open(os.environ["WEBNG_TEST_CALLS"] + ".cwd").read().splitlines())
        assert cwds["w_assign"] == str(sim_dir)                 # needs west.cfg, so runs there
    finally:
        an.h5file.close()
        os.chdir(an.curr_path)


def test_each_sim_gets_its_own_system_py(tmp_path, monkeypatch):
    """
    Analyzing two sims in one process (a notebook, or the test suite) must
    import each sim's own system.py, e.g. when unpickling its bin mapper.
    Previously the first sim's system.py silently won for every later one.
    """
    monkeypatch.chdir(tmp_path)
    for name in ["sim_a", "sim_b"]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "system.py").write_text("WHICH = {!r}\n".format(name))
        write_west(tmp_path / name, 11)
    for name in ["sim_a", "sim_b", "sim_a"]:
        an = weAnalysis({"sim_name": name, "work-path": None, "pcoords": ["A"]})
        try:
            import system
            assert system.WHICH == name
        finally:
            an.h5file.close()
            os.chdir(an.curr_path)
    sys.modules.pop("system", None)


# ---------------------------------------------------------------------------
# pcoord names
# ---------------------------------------------------------------------------

def _named_sim(tmp_path, ndim, cfg_names=None, section="bng"):
    """A sim whose west.h5 has `ndim` pcoord dimensions, optionally with a west.cfg naming them."""
    sim_dir = tmp_path / "sim"
    write_west(sim_dir, 3, ndim=ndim)
    if cfg_names is not None:
        (sim_dir / "west.cfg").write_text(
            "# vi: set filetype=yaml :\n---\nwest:\n  {}:\n    data:\n      pcoords: {}\n"
            .format(section, json.dumps(cfg_names)))


def _names(tmp_path, monkeypatch, option_names):
    monkeypatch.chdir(tmp_path)
    an = weAnalysis({"sim_name": "sim", "work-path": None, "pcoords": option_names})
    try:
        return [an.names[i] for i in sorted(an.names)]
    finally:
        an.h5file.close()
        os.chdir(an.curr_path)


@pytest.mark.parametrize("section", ["bng", "librr"])
def test_pcoord_names_come_from_west_cfg(tmp_path, monkeypatch, section):
    """west.cfg records what the propagator wrote; analysis options listing
    other names or another order (e.g. all 8 genes) must not win."""
    _named_sim(tmp_path, 4, ["NANOG", "GATA6", "CDX2", "OCT4"], section)
    all_genes = ["OCT4", "SOX2", "NANOG", "KLF4", "GATA6", "CDX2", "PBX1", "GCNF"]
    assert _names(tmp_path, monkeypatch, all_genes) == ["NANOG", "GATA6", "CDX2", "OCT4"]


def test_pcoord_names_fall_back_to_analysis_options(tmp_path, monkeypatch):
    _named_sim(tmp_path, 2)        # no names in west.cfg, as for executable sims
    assert _names(tmp_path, monkeypatch, ["Atot", "Btot"]) == ["Atot", "Btot"]


def test_pcoord_name_count_mismatch_labels_by_number(tmp_path, monkeypatch):
    _named_sim(tmp_path, 3)
    assert _names(tmp_path, monkeypatch, ["Atot", "Btot"]) == ["0", "1", "2"]

# ---------------------------------------------------------------------------
# histogram bins (pdist_bin_edges)
# ---------------------------------------------------------------------------

def _binned(tmp_path, values, bins):
    """
    weAnalysis over a sim whose pcoord takes `values` (array, shape
    (n, ndim)), spread over 3 iterations; returns (analysis, edges).
    """
    values = np.asarray(values, dtype=np.float32)
    chunks = np.array_split(values, 3)
    write_west(tmp_path / "s", 4, ndim=values.shape[1],
               pcoord=lambda n: np.repeat(chunks[n - 1][:, None, :], 2, axis=1))
    an = weAnalysis.__new__(weAnalysis)
    an.sim_path, an.west_path = str(tmp_path / "s"), str(tmp_path / "s" / "west.h5")
    an.first_iter, an.last_iter, an.bins = 1, 3, bins
    return an, an.pdist_bin_edges()


def test_whole_number_pcoords_get_one_bin_per_integer(tmp_path):
    """Counts 0..52 with 100 bins asked for: 53 bins, one per count, none empty
    (equal-width bins of ~0.5 left every other bin empty: the comb artifact)."""
    counts = np.column_stack([np.arange(53), np.arange(53) % 51])
    _, edges = _binned(tmp_path, counts, bins=100)
    assert edges[0] == [v - 0.5 for v in range(54)]
    assert edges[1] == [v - 0.5 for v in range(52)]
    for d in range(2):
        filled, _ = np.histogram(counts[:, d], bins=edges[d])
        assert (filled > 0).all()


def test_wide_integer_range_groups_whole_integers(tmp_path):
    """Counts 0..3999 with 100 bins: 40 integers in every bin, all equally filled."""
    counts = np.arange(4000)[:, None]
    _, edges = _binned(tmp_path, counts, bins=100)
    assert len(edges[0]) == 101
    assert np.allclose(np.diff(edges[0]), 40) and edges[0][0] == -0.5
    filled, _ = np.histogram(counts[:, 0], bins=edges[0])
    assert (filled == 40).all()


def test_integer_range_not_starting_at_zero(tmp_path):
    counts = np.arange(50, 61)[:, None]                # a gene that never drops below 50
    _, edges = _binned(tmp_path, counts, bins=30)
    assert edges[0] == [v - 0.5 for v in range(50, 62)]


def test_non_integer_pcoords_bin_as_w_pdist_would(tmp_path):
    values = np.linspace(0.25, 7.75, 60)[:, None]
    _, edges = _binned(tmp_path, values, bins=30)
    lo, hi = float(np.float32(0.25)), float(np.float32(7.75))
    assert edges[0] == pytest.approx(list(np.linspace(lo, hi * 1.01, 31)))


def test_each_dimension_is_binned_on_its_own(tmp_path):
    mixed = np.column_stack([np.arange(20), np.linspace(0.1, 2.0, 20)])
    _, edges = _binned(tmp_path, mixed, bins=10)
    assert np.allclose(np.diff(edges[0]), 2) and edges[0][0] == -0.5     # whole integers, 2 per bin
    assert len(edges[1]) == 11 and edges[1][0] == pytest.approx(0.1)     # 10 equal bins


def test_pdist_gets_the_edges_and_old_style_files_rebuild(sim):
    """w_pdist is handed explicit edges; a pdist.h5 stamped by the old
    plain-bin-count method is rebuilt rather than reused."""
    _, an, take_calls = sim
    with h5py.File("pdist.h5", "w") as f:              # made before integer-aware bins
        f.attrs[STAMP_ATTR] = an.stamp("w_pdist", first_iter=1, last_iter=100, bins=30)
    an.ensure_pdist().close()
    assert take_calls() == ["w_pdist"]
    args = [json.loads(l) for l in open(os.environ["WEBNG_TEST_CALLS"] + ".args")]
    pdist_args = [a for a in args if a[0] == "w_pdist"][-1]
    assert json.loads(pdist_args[pdist_args.index("-b") + 1]) == [[-0.5, 0.5]]   # all-zero pcoord