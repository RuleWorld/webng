"""
Tests for how analyses reuse the files WESTPA tools build (pdist.h5,
assign.h5, direct.h5, ...): reuse only when built from the same inputs,
rebuild otherwise, and fail loudly when a tool fails.

The WESTPA tools are replaced by tiny fake executables on PATH that log
each call, so these run fast and don't need a real simulation.
"""
import os
import sys
import time

import h5py
import pytest

from webng.analysis.analysis import weAnalysis, STAMP_ATTR

FAKE_TOOL = """#!{python}
import os, sys, h5py
name = os.path.basename(sys.argv[0])
with open(os.environ["WEBNG_TEST_CALLS"], "a") as f:
    f.write(name + "\\n")
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
    set_west(sim_dir, 101)
    monkeypatch.chdir(sim_dir / "analysis")

    an = weAnalysis.__new__(weAnalysis)
    an.first_iter, an.last_iter, an.bins = 1, 100, 30

    def take_calls():
        names = calls.read_text().split() if calls.exists() else []
        if calls.exists():
            calls.unlink()
        return names

    return sim_dir, an, take_calls


def set_west(sim_dir, iteration, extra=0):
    with h5py.File(sim_dir / "west.h5", "w") as f:
        f.attrs["west_current_iteration"] = iteration
        if extra:
            f["padding"] = list(range(extra))   # changes the file's size


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
    setattr(an, change, getattr(an, change) + 7)
    an.ensure_pdist().close()
    assert take_calls() == ["w_pdist"]


def test_continued_or_replaced_run_rebuilds(sim):
    sim_dir, an, take_calls = sim
    an.ensure_pdist().close()
    take_calls()
    set_west(sim_dir, 160)                      # run continued
    an.ensure_pdist().close()
    assert take_calls() == ["w_pdist"]
    time.sleep(0.01)
    set_west(sim_dir, 160, extra=50)            # different run, same iteration count
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


def test_custom_work_path_and_absolute_sys_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mysim").mkdir()
    an = weAnalysis({"sim_name": "mysim", "work-path": "elsewhere", "pcoords": ["A"]})
    assert an.work_path == "elsewhere"
    assert os.getcwd() == str(tmp_path / "elsewhere")
    assert str(tmp_path / "mysim") in sys.path
    os.chdir(an.curr_path)
