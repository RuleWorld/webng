"""
Shared pytest fixtures and helpers for the WEBNG test suite.
"""
import ctypes
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest

from webng.core.weConvert import weConvert

# The tests folder: test.bngl lives here, and end-to-end tests build their
# simulation folders here.
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


# A small hand-built .net file used across unit tests that need something
# to parse/simulate without running BNG2.pl. It deliberately includes a
# weighted, multi-species observable (WeightedObs = 2*B + 3*C) — the case
# that used to be silently mishandled before the group-parsing fix.
SYNTHETIC_NET = """\
begin parameters
  1 k1  0.01
end parameters
begin species
  1 A() 20
  2 B() 10
  3 C() 5
end species
begin reactions
  1 1 2 k1
  2 2 3 k1
end reactions
begin groups
  1 SimpleObs 1
  2 WeightedObs 2*2,3*3
end groups
"""


@pytest.fixture
def synthetic_net(tmp_path):
    """Writes SYNTHETIC_NET to <tmp_path>/bngl_conf/init.net, returns its path."""
    net_dir = tmp_path / "bngl_conf"
    net_dir.mkdir()
    net_path = net_dir / "init.net"
    net_path.write_text(SYNTHETIC_NET)
    return str(net_path)


@pytest.fixture
def sim_root(tmp_path, monkeypatch):
    """
    A tmp directory laid out like a WEBNG sim folder (bngl_conf/, bstates/),
    with SYNTHETIC_NET already written to bngl_conf/init.net, and cwd
    changed into it — matching what weConvert's writer/sampling methods
    expect (they write relative paths like "west.cfg", "bstates/...").
    """
    monkeypatch.chdir(tmp_path)
    os.makedirs("bngl_conf")
    os.makedirs("bstates")
    with open("bngl_conf/init.net", "w") as f:
        f.write(SYNTHETIC_NET)
    return tmp_path


def make_bare_weconvert(**attrs):
    """
    Build a weConvert instance without going through __init__/_parse_opts
    (which needs a full opts.yaml), for isolated unit tests of individual
    writer/parsing methods. Defaults describe a bng+mabl+recycling setup;
    override whatever a given test needs.
    """
    wc = weConvert.__new__(weConvert)
    defaults = dict(
        propagator_type="bng",
        tau=50.0,
        plen=2,
        max_iter=100,
        block_size=500,
        binning_style="mabl",
        do_recycling=True,
        pcoord_list=["NANOG", "GATA6"],
        gillespie_update_interval=1,
        libssa_path=None,
        max_wallclock="72:00:00",
    )
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(wc, k, v)
    return wc


@pytest.fixture
def bare_weconvert():
    return make_bare_weconvert


def _bng_lib_loadable():
    """
    Actually try to dlopen the resolved libssa.so, rather than just
    checking platform.system() — this is the real test (an x86-64 Linux
    build won't load on an ARM Linux host either, for example), and it
    stays correct automatically once a matching build exists for another
    platform, with no changes needed here.
    """
    path = make_bare_weconvert()._resolve_bng_lib_path()
    try:
        ctypes.CDLL(path)
        return True
    except OSError:
        return False


BNG_LIB_LOADABLE = _bng_lib_loadable()

requires_bng_lib = pytest.mark.skipif(
    not BNG_LIB_LOADABLE,
    reason=(
        "libssa.so could not be loaded on this platform/architecture "
        "(e.g. a Linux build on macOS or Windows). This is expected on "
        "machines where bng hasn't been built yet — see build_libssa.sh. "
        "CI (ubuntu-latest) runs these tests for real."
    ),
)


# ---------------------------------------------------------------------------
# Running real commands (end-to-end tests)
# ---------------------------------------------------------------------------

# Markers that mean the run failed even when the exit code is 0. WESTPA has
# been observed to catch an internal exception, print "exception caught;
# shutting down", and still exit 0 -- checking the return code alone lets that
# kind of failure silently report as a passing test.
_FAILURE_MARKERS = ("exception caught", "Traceback (most recent call last)")


def _dump_seg_logs():
    """
    WESTPA's executable propagator redirects each segment's runseg.sh
    stdout/stderr to its own file under seg_logs/{iter:06d}-{seg:06d}.log
    (see _executable_westcfg) -- none of that ever reaches this process's
    own stdout/stderr, so every diagnostic runseg.sh prints is otherwise
    invisible here. Dump them explicitly so they actually show up.
    """
    if not os.path.isdir("seg_logs"):
        return
    log_files = sorted(os.listdir("seg_logs"))
    if not log_files:
        print("seg_logs/ exists but is empty")
        return
    for name in log_files:
        path = os.path.join("seg_logs", name)
        print("--- seg_logs/{} ---".format(name))
        try:
            with open(path) as f:
                print(f.read())
        except Exception as e:
            print("could not read {}: {}".format(path, e))
        print("--- end seg_logs/{} ---".format(name))


def run_and_check(cmd, step_name):
    """
    Run a subprocess, print its output (so -s still shows it), and fail
    the test if either the exit code is nonzero OR the output contains a
    failure marker WESTPA prints even on an exit code of 0.
    """
    rc = subprocess.run(cmd, capture_output=True, text=True)
    if rc.stdout:
        print(rc.stdout)
    if rc.stderr:
        print(rc.stderr, file=sys.stderr)
    _dump_seg_logs()
    combined = (rc.stdout or "") + (rc.stderr or "")
    assert rc.returncode == 0, "{} exited with code {}".format(step_name, rc.returncode)
    for marker in _FAILURE_MARKERS:
        assert marker not in combined, (
            "{} exited 0 but its output contains {!r} -- this masks a "
            "real failure inside the run (see captured output above)".format(
                step_name, marker
            )
        )


# ---------------------------------------------------------------------------
# Synthetic simulation data (analysis tests that don't run WESTPA)
# ---------------------------------------------------------------------------

# WESTPA's iter_block_ci_dtype (westpa/tools/dtypes.py): one record per
# evaluation window, as w_direct and w_fluxanl write them.
CI_DTYPE = np.dtype([("iter_start", np.uint32), ("iter_stop", np.uint32),
                     ("expected", np.float64), ("ci_lbound", np.float64),
                     ("ci_ubound", np.float64), ("sterr", np.float64),
                     ("corr_len", np.uint32)])


def write_west(sim_dir, current_iteration, ndim=None, padding=0):
    """
    A minimal west.h5 in `sim_dir`: just the current-iteration attribute,
    plus (if `ndim`) an iteration-1 pcoord with that many dimensions, plus
    (if `padding`) some extra data, which changes the file's size.
    """
    os.makedirs(sim_dir, exist_ok=True)
    with h5py.File(os.path.join(sim_dir, "west.h5"), "w") as f:
        f.attrs["west_current_iteration"] = current_iteration
        if ndim is not None:
            f.create_dataset("iterations/iter_00000001/pcoord", data=np.zeros((2, 3, ndim)))
        if padding:
            f["padding"] = list(range(padding))


def write_stamped(path, stamp, **datasets):
    """
    An HDF5 file holding `datasets`, stamped as built from `stamp`, so an
    analysis's cache reuses it instead of running the WESTPA tool that would
    normally produce it.
    """
    from webng.analysis.analysis import STAMP_ATTR
    with h5py.File(path, "w") as f:
        for name, data in datasets.items():
            f[name] = data
        f.attrs[STAMP_ATTR] = stamp


@pytest.fixture
def analysis_sim(tmp_path, monkeypatch):
    """
    analysis_sim(current_iteration=11, ndim=2) -> a minimal sim folder
    (tmp_path/sim, with a synthetic west.h5 and an empty analysis/), with cwd
    at tmp_path, where an analysis's options would name it "sim".
    """
    def make(current_iteration=11, ndim=2):
        sim = tmp_path / "sim"
        (sim / "analysis").mkdir(parents=True)
        write_west(sim, current_iteration, ndim=ndim)
        monkeypatch.chdir(tmp_path)
        return sim
    return make


# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _close_figures():
    """
    The analysis tools return their figures (so notebooks can use them) and
    leave them open; close them after each test so they don't pile up.
    """
    yield
    plt = sys.modules.get("matplotlib.pyplot")      # only if a test used it
    if plt is not None:
        plt.close("all")
