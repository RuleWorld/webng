"""
CLI-level integration tests: exercises `webng template`, `webng setup`,
and an actual simulation run through the real cement app, for each
propagator_type. These need the real dependency stack installed
(bionetgen/BNG2.pl, westpa, roadrunner, and — for propagator_type "bng"
— libssa.so) since they run genuine simulations, not mocked ones.

Historically this file (test_webng.py) only exercised whatever
propagator_type happened to be the default in weTemplater — it never
pinned propagator_type explicitly, so a change to the default (as
happened when "bng" became the default) silently changes which engine
gets tested here without anyone deciding that on purpose. The
TestSetupAndRunByPropagator class below fixes that by parametrizing
explicitly over all three engines, so all three stay covered regardless
of which one is the current default.
"""
import os
import shutil
import subprocess

import pytest
import yaml
from pytest import raises

from webng.main import weBNGTest
from conftest import requires_bng_lib, BNG_LIB_LOADABLE

tfold = os.path.dirname(__file__)


def test_webng():
    # test webng without any subcommands or arguments
    with weBNGTest() as app:
        app.run()
        assert app.exit_code == 0


def test_webng_debug():
    # test that debug mode is functional
    argv = ["--debug"]
    with weBNGTest(argv=argv) as app:
        app.run()
        assert app.debug is True


class TestTemplate:
    def test_template_help(self):
        os.chdir(tfold)
        with raises(SystemExit):
            argv = ["template", "-h"]
            with weBNGTest(argv=argv) as app:
                app.run()
                assert app.exit_code == 0

    def test_template_writes_file(self):
        os.chdir(tfold)
        fpath = os.path.join(tfold, "test.bngl")
        opath = os.path.join(tfold, "test_setup.yaml")
        argv = ["template", "-i", fpath, "-o", opath]
        with weBNGTest(argv=argv) as app:
            app.run()
            assert os.path.isfile(opath)
        os.remove(opath)

    def test_template_default_propagator_is_documented(self):
        """
        The template output should spell out propagator_type and its
        associated options explicitly (pcoords, libssa_path,
        gillespie_update_interval) rather than requiring the user to
        already know those keys exist.
        """
        os.chdir(tfold)
        fpath = os.path.join(tfold, "test.bngl")
        opath = os.path.join(tfold, "test_template_check.yaml")
        argv = ["template", "-i", fpath, "-o", opath]
        with weBNGTest(argv=argv) as app:
            app.run()
        with open(opath) as f:
            opts = yaml.safe_load(f)
        prop_opts = opts["propagator_options"]
        assert "propagator_type" in prop_opts
        assert "pcoords" in prop_opts
        assert "libssa_path" in prop_opts
        assert "gillespie_update_interval" in prop_opts
        os.remove(opath)


class TestSetupAndRunByPropagator:
    """
    For each propagator_type, generate a template, force that
    propagator_type, run `webng setup`, then actually run the simulation
    (init.sh + w_run --serial) — mirroring what the original test_setup /
    test_simrun did, but explicitly for all three engines rather than
    only the current default.
    """

    @pytest.mark.parametrize(
        "propagator_type",
        [
            "executable",
            "libRoadRunner",
            pytest.param("bng", marks=requires_bng_lib),
        ],
    )
    def test_setup_and_run(self, propagator_type):
        os.chdir(tfold)
        yaml_name = "test_setup_{}.yaml".format(propagator_type)
        sim_name = "test_{}".format(propagator_type)
        fpath = os.path.join(tfold, yaml_name)
        opath = os.path.join(tfold, sim_name)

        # clean up any stale leftovers from a previous crashed run before
        # starting, not just after — otherwise a leftover sim folder
        # causes a FileExistsError on this run too
        if os.path.isfile(fpath):
            os.remove(fpath)
        if os.path.isdir(opath):
            shutil.rmtree(opath)

        try:
            # 1. template
            argv = ["template", "-i", "test.bngl", "-o", yaml_name]
            with weBNGTest(argv=argv) as app:
                app.run()
            assert os.path.isfile(fpath)

            # 2. force propagator_type, point sim_name at a unique folder,
            #    keep the run short
            with open(fpath, "r") as f:
                opts = yaml.safe_load(f)
            opts["propagator_options"]["propagator_type"] = propagator_type
            opts["path_options"]["sim_name"] = sim_name
            opts["sampling_options"]["max_iter"] = 5
            with open(fpath, "w") as f:
                yaml.dump(opts, f)

            # 3. setup
            argv = ["setup", "--opts", fpath]
            with weBNGTest(argv=argv) as app:
                app.run()
            assert os.path.isdir(opath)

            # 4. actually run it
            os.chdir(opath)
            rc = subprocess.run(["./init.sh"])
            assert rc.returncode == 0
            rc = subprocess.run(["w_run", "--serial"])
            assert rc.returncode == 0
        finally:
            os.chdir(tfold)
            if os.path.isfile(fpath):
                os.remove(fpath)
            if os.path.isdir(opath):
                shutil.rmtree(opath)


# Kept for parity with the original test file's naming/behavior — exercises
# whichever propagator_type is the current template default end-to-end.
class TestDefaultSetupAndRun:
    @staticmethod
    def _remove_default_artifacts():
        # ALWAYS move to a known-safe directory before deleting anything —
        # if the process's cwd is currently inside opath (e.g. a previous
        # step chdir'd into the sim folder and a failure left it there),
        # rmtree-ing the directory we're standing in corrupts os.getcwd()
        # for the rest of the pytest session.
        os.chdir(tfold)
        fpath = os.path.join(tfold, "test_setup.yaml")
        opath = os.path.join(tfold, "test")
        if os.path.isfile(fpath):
            os.remove(fpath)
        if os.path.isdir(opath):
            shutil.rmtree(opath)

    @pytest.fixture(autouse=True, scope="class")
    def _cleanup(self):
        # class-scoped: runs ONCE before the first test in this class and
        # ONCE after the last one — NOT between test_setup and
        # test_simrun, which share state (test_setup creates tests/test/,
        # test_simrun depends on it still being there). A per-test-method
        # scope here would delete tests/test/ right after test_setup
        # finishes, before test_simrun ever runs.
        self._remove_default_artifacts()
        yield
        self._remove_default_artifacts()

    def test_setup(self):
        os.chdir(tfold)
        fpath = os.path.join(tfold, "test_setup.yaml")
        opath = os.path.join(tfold, "test")

        argv = ["template", "-i", "test.bngl", "-o", "test_setup.yaml"]
        with weBNGTest(argv=argv) as app:
            app.run()

        with open(fpath, "r") as f:
            opts_yaml = yaml.safe_load(f)
        opts_yaml["sampling_options"]["max_iter"] = 5
        with open(fpath, "w") as f:
            yaml.dump(opts_yaml, f)

        argv = ["setup", "--opts", fpath]
        with weBNGTest(argv=argv) as app:
            app.run()
            assert os.path.isdir(opath)

    @pytest.mark.skipif(
        not BNG_LIB_LOADABLE,
        reason=(
            "template's current default propagator_type is 'bng', which "
            "needs a working libssa.so on this platform to actually run "
            "(setup itself is fine — this only affects executing the sim)."
        ),
    )
    def test_simrun(self):
        fpath = os.path.join(tfold, "test")
        os.chdir(fpath)
        rc = subprocess.run(["./init.sh"])
        assert rc.returncode == 0
        rc = subprocess.run(["w_run", "--serial"])
        assert rc.returncode == 0


# TODO: Write tests for each analysis module (average/evolution/cluster/
# network/flux) — still no coverage here. Tracked separately.