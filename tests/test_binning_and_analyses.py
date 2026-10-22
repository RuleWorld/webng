"""
End-to-end tests for each binning style and each analysis tool.

One short simulation per binning style (regular, adaptive Voronoi, MABL) is
built once and shared by the tests in this file. Each analysis then runs
through the real `webng analysis --opts` command (so the dispatcher in
weAnalysis.py is exercised too) on the simulation that suits it:

  regular   average, evolution, cluster
  adaptive  network, pcca, and average with Voronoi boundaries drawn
            (pcca needs Voronoi bin centers)
  mabl      flux (MABL always recycles, so it has a target state)

These check that everything sets up, runs, and produces its outputs without
errors -- not that the numbers are scientifically meaningful, which a
5-iteration toy run can't establish.
"""
import os
import shutil
import traceback

import h5py
import numpy as np
import pytest
import yaml

from webng.main import weBNGTest
from conftest import requires_bng_lib
from test_cli import _run_and_check

pytestmark = requires_bng_lib       # the template's default propagator is bng

tfold = os.path.dirname(os.path.abspath(__file__))
N_ITER = 5
ANALYSES = ["average", "evolution", "cluster", "network", "flux", "pcca"]


def _names(style):
    return "test_{}_binning.yaml".format(style), "test_{}_binning".format(style)


def _remove(style):
    os.chdir(tfold)
    yaml_name, sim_name = _names(style)
    if os.path.isfile(yaml_name):
        os.remove(yaml_name)
    if os.path.isdir(sim_name):
        shutil.rmtree(sim_name)


def _configure(opts, style):
    opts["path_options"]["sim_name"] = _names(style)[1]
    opts["sampling_options"]["max_iter"] = N_ITER
    if style == "mabl":
        # The template leaves both regions unbounded, which would recycle every
        # walker. Start walkers at low first-observable values and recycle them
        # once it reaches 8: easy to reach in a few iterations from the model's
        # starting state (A = 4), so the flux analysis has events to count.
        first_obs = opts["propagator_options"]["pcoords"][0]
        for region, bound, value in [("basis_region", "max", 6), ("recycle_region", "min", 8)]:
            for entry in opts["recycling"][region]:
                if entry["observable"] == first_obs:
                    entry[bound] = value


@pytest.fixture(scope="module")
def build_sim():
    """
    build_sim(style) -> (absolute yaml path, absolute sim folder), built once
    per style. A style that fails to build is not retried: later tests for it
    fail straight away with the original error.
    """
    built, failed = {}, {}

    def build(style):
        if style in built:
            return built[style]
        if style in failed:
            pytest.fail("the {} simulation failed to build earlier:\n{}".format(style, failed[style]))
        _remove(style)
        yaml_name, sim_name = _names(style)
        yaml_path, sim_path = os.path.join(tfold, yaml_name), os.path.join(tfold, sim_name)
        try:
            argv = ["template", "-i", "test.bngl", "-o", yaml_name, "--bins", style]
            with weBNGTest(argv=argv) as app:
                app.run()
            with open(yaml_path) as f:
                opts = yaml.safe_load(f)
            _configure(opts, style)
            with open(yaml_path, "w") as f:
                yaml.dump(opts, f)

            # `webng setup` runs in its own process, as it does for users. With
            # recycling it samples basis states through libssa, which can only
            # be initialized once per process (its C globals are never freed;
            # see test_recycling.py), so running it in this pytest process
            # would crash the next build that samples. It also chdirs into the
            # sim folder and doesn't come back, which is why paths here are absolute.
            _run_and_check(["webng", "setup", "--opts", yaml_path], "webng setup")
            os.chdir(sim_path)
            _run_and_check(["bash", "-c", "source env.sh && ./init.sh"], "init.sh")
            _run_and_check(["bash", "-c", "source env.sh && w_run --serial"], "w_run --serial")
        except BaseException:
            failed[style] = traceback.format_exc()
            print(failed[style])     # show it now, in case something later stops pytest early
            raise
        finally:
            os.chdir(tfold)
        built[style] = (yaml_path, sim_path)
        return built[style]

    yield build
    for style in ["regular", "adaptive", "mabl"]:
        _remove(style)


def _run_analyses(yaml_path, enable, settings=None):
    """Turn on exactly `enable`, apply per-tool `settings`, run `webng analysis`."""
    with open(yaml_path) as f:
        opts = yaml.safe_load(f)
    an = opts["analyses"]
    an["enabled"] = True
    for name in ANALYSES:
        an[name]["enabled"] = name in enable
    for name, extra in (settings or {}).items():
        an[name].update(extra)
    with open(yaml_path, "w") as f:
        yaml.dump(opts, f)
    os.chdir(tfold)      # sim_name in the yaml is relative to here
    try:
        with weBNGTest(argv=["analysis", "--opts", yaml_path]) as app:
            app.run()
    finally:
        os.chdir(tfold)  # analyses chdir into their work path; don't leak it on failure


def _assert_outputs(sim, *names):
    missing = [n for n in names if not os.path.isfile(os.path.join(sim, "analysis", n))]
    assert not missing, "analysis outputs not produced: {}".format(missing)


# ---------------------------------------------------------------------------
# binning styles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", ["regular", "adaptive", "mabl"])
def test_binning_style_sets_up_and_runs(build_sim, style):
    _, sim = build_sim(style)
    with h5py.File(os.path.join(sim, "west.h5"), "r") as f:
        assert int(f.attrs["west_current_iteration"]) == N_ITER + 1
        for i in range(1, N_ITER + 1):
            assert len(f["iterations/iter_{:08d}/seg_index".format(i)]) > 0


# ---------------------------------------------------------------------------
# analysis tools
# ---------------------------------------------------------------------------

def test_average_evolution_cluster(build_sim):
    yaml_path, sim = build_sim("regular")
    _run_analyses(yaml_path, ["average", "evolution", "cluster"])
    _assert_outputs(sim, "pdist.h5", "average.png", "evolution.png", "states.yaml")


def _states_from_run(sim):
    """
    A states.yaml with two single-point states: the walkers with the lowest
    and highest first observable in the last finished iteration. Taken from
    the run itself so each point is guaranteed to fall in a bin that exists.
    """
    with h5py.File(os.path.join(sim, "west.h5"), "r") as f:
        last = int(f.attrs["west_current_iteration"]) - 1
        ends = f["iterations/iter_{:08d}/pcoord".format(last)][:, -1, :]
    low, high = ends[np.argmin(ends[:, 0])], ends[np.argmax(ends[:, 0])]
    os.makedirs(os.path.join(sim, "analysis"), exist_ok=True)
    with open(os.path.join(sim, "analysis", "states.yaml"), "w") as f:
        f.write("states:\n")
        for label, point in [("low", low), ("high", high)]:
            f.write("  - label: {}\n    coords:\n      - {}\n".format(
                label, [float(v) for v in point]))


def test_network_pcca_and_voronoi_boundaries(build_sim):
    yaml_path, sim = build_sim("adaptive")
    _states_from_run(sim)
    _run_analyses(
        yaml_path, ["average", "network", "pcca"],
        settings={"average": {"plot-boundaries": True},
                  "network": {"step-iter": 1},
                  "pcca": {"step-iter": 1}},
    )
    _assert_outputs(sim, "average.png", "assign.h5", "direct.h5", "tmat.h5",
                    "output_pcca.txt", "mfpt_evolution.png", "voronoi_states.png")


def test_flux(build_sim):
    yaml_path, sim = build_sim("mabl")
    _run_analyses(yaml_path, ["flux"])
    _assert_outputs(sim, "fluxanl.h5", "flux.png")


def test_rerun_reuses_cached_files(build_sim):
    """A second identical run reuses every WESTPA output instead of rebuilding."""
    yaml_path, sim = build_sim("regular")
    _run_analyses(yaml_path, ["average"])
    before = os.path.getmtime(os.path.join(sim, "analysis", "pdist.h5"))
    _run_analyses(yaml_path, ["average", "evolution"])
    assert os.path.getmtime(os.path.join(sim, "analysis", "pdist.h5")) == before