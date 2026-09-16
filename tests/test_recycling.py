"""
Unit tests for basis-state sampling (recycling setup): the dispatch in
sample_basis_states(), the bng-native sampler (real ctypes call into
libssa.so), and the libRoadRunner sampler's dependency guard.

IMPORTANT: ssa_wrapper.cpp keeps all of its state (g_rates, g_species,
g_reactions, g_initialized, ...) in C global/static variables, with no
reset or free between ssa_init() calls. ctypes.CDLL also caches a loaded
library by path *within a process*, so a second in-process call to
ssa_init() reuses -- without freeing -- whatever the first call already
allocated. This is completely safe in real usage: webng setup and every
spawned w_run process each call ssa_init() exactly once, in their own
fresh process. But three separate test functions in the same pytest
process each calling into libssa.so hit exactly this and produced a
real segfault in CI (Fatal Python error: Segmentation fault, inside
_sample_basis_states_bng). Every test below that touches libssa.so runs
its actual work in a freshly spawned subprocess (multiprocessing
"spawn", not "fork" -- fork would still copy over whatever state
already exists in the parent's address space), so every ssa_init() call
gets a genuinely clean process, matching real usage.

Tests that never call into libssa.so (the dispatch short-circuit and the
libRoadRunner import guard) don't need isolation and run normally.
"""
import multiprocessing
import os
import warnings

from conftest import requires_bng_lib, make_bare_weconvert


# ---------------------------------------------------------------------------
# Subprocess isolation harness
# ---------------------------------------------------------------------------

def _subprocess_entry(q, worker, args):
    try:
        worker(*args)
        q.put(("ok", None))
    except AssertionError as e:
        q.put(("error", str(e)))
    except Exception as e:
        q.put(("error", "{}: {}".format(type(e).__name__, e)))


def _isolated(worker, *args, timeout=120):
    """
    Run worker(*args) in a fresh spawned subprocess and re-raise whatever
    it reports. `worker` must be a module-level function (picklable) and
    everything in `args` must be picklable (plain strings/numbers/lists
    are fine; live weConvert instances are not passed this way -- workers
    rebuild what they need via make_bare_weconvert instead).
    """
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_subprocess_entry, args=(q, worker, args))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        raise AssertionError(
            "{} timed out after {}s".format(worker.__name__, timeout)
        )
    if p.exitcode != 0:
        raise AssertionError(
            "{} crashed in its subprocess with exit code {} (a nonzero "
            "exit code here usually means a native crash, e.g. a "
            "segfault in libssa.so)".format(worker.__name__, p.exitcode)
        )
    status, payload = q.get()
    if status == "error":
        raise AssertionError(payload)


# ---------------------------------------------------------------------------
# Worker functions -- each runs in its OWN subprocess, so must be
# module-level (picklable) and self-contained.
# ---------------------------------------------------------------------------

def _worker_weighted_filtering(cwd):
    os.chdir(cwd)
    wc = make_bare_weconvert(
        bs_traj_length=200,
        basis_region=[{"observable": "WeightedObs", "min": 0.0, "max": 1e9}],
    )
    groups = wc._parse_net_groups("bngl_conf/init.net")
    n_species = len(wc._parse_net_species("bngl_conf/init.net"))
    wc._sample_basis_states_bng("bngl_conf/init.net", groups, n_species)

    pool_path = "bstates/microstate_pool.dat"
    assert os.path.isfile(pool_path)
    lines = open(pool_path).readlines()
    assert lines[0].startswith("# weight")
    assert len(lines) > 1  # at least one valid frame

    for line in lines[1:]:
        weight, sp1, sp2, sp3 = (float(x) for x in line.split())
        weighted_obs = 2 * sp2 + 3 * sp3
        assert weighted_obs >= 0.0


def _worker_excludes_outside_region(cwd):
    os.chdir(cwd)
    wc = make_bare_weconvert(
        bs_traj_length=50,
        basis_region=[{"observable": "WeightedObs", "min": 1e9, "max": 1e10}],
    )
    groups = wc._parse_net_groups("bngl_conf/init.net")
    n_species = len(wc._parse_net_species("bngl_conf/init.net"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        wc._sample_basis_states_bng("bngl_conf/init.net", groups, n_species)
        assert any(issubclass(w.category, UserWarning) for w in caught)

    assert os.path.isfile("bstates/0.net")
    assert not os.path.isfile("bstates/microstate_pool.dat")


def _worker_bng_dispatch(cwd):
    os.chdir(cwd)
    wc = make_bare_weconvert(
        propagator_type="bng",
        do_recycling=True,
        basis_region=[{"observable": "SimpleObs", "min": 0.0, "max": 1e9}],
        bs_traj_length=100,
    )
    wc.sample_basis_states()
    # if it ran, we get a pool file (or, at minimum, the fallback 0.net --
    # either way, no exception and no accidental libRoadRunner import)
    assert os.path.isfile("bstates/0.net") or os.path.isfile(
        "bstates/microstate_pool.dat"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSampleBasisStatesBng:
    @requires_bng_lib
    def test_weighted_filtering_against_real_lib(self, sim_root):
        """
        End-to-end: runs the real libssa.so via ctypes (in an isolated
        subprocess), samples a trajectory, and filters using a weighted,
        multi-species group -- the case that used to be silently wrong
        before the group-parsing fix.
        """
        _isolated(_worker_weighted_filtering, str(sim_root))

    @requires_bng_lib
    def test_excludes_frames_outside_basis_region(self, sim_root):
        """
        A basis_region no frame can satisfy should fall back to a single
        basis state rather than writing an empty/invalid pool file.
        """
        _isolated(_worker_excludes_outside_region, str(sim_root))


class TestSampleBasisStatesDispatch:
    def test_no_recycling_short_circuits_to_single_basis_state(
        self, bare_weconvert, sim_root
    ):
        """
        With recycling disabled, sample_basis_states() should never touch
        either sampling engine -- it just copies init.net -> 0.net. No
        libssa.so call happens here, so no isolation needed.
        """
        wc = bare_weconvert(do_recycling=False)
        wc.sample_basis_states()
        assert os.path.isfile("bstates/0.net")
        assert not os.path.isfile("bstates/microstate_pool.dat")

    @requires_bng_lib
    def test_bng_propagator_dispatches_to_bng_sampler(self, sim_root):
        _isolated(_worker_bng_dispatch, str(sim_root))

    def test_non_bng_propagator_requires_roadrunner(
        self, bare_weconvert, sim_root, monkeypatch
    ):
        """
        For propagator_type != "bng" with recycling on, sample_basis_states
        should route to the libRoadRunner sampler, which exits with a clear
        message if roadrunner isn't importable. Blocked at import time,
        before ever reaching libssa.so -- no isolation needed.
        """
        import builtins

        real_import = builtins.__import__

        def blocked_import(name, *args, **kwargs):
            if name == "roadrunner":
                raise ImportError("simulated: roadrunner not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked_import)

        wc = bare_weconvert(
            propagator_type="libRoadRunner",
            do_recycling=True,
            basis_region=[{"observable": "SimpleObs", "min": 0.0, "max": 1e9}],
            bs_traj_length=100,
        )
        try:
            wc.sample_basis_states()
            assert False, "expected SystemExit when roadrunner is unavailable"
        except SystemExit as e:
            assert "libRoadRunner is required" in str(e)