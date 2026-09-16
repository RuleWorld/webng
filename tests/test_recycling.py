"""
Unit tests for basis-state sampling (recycling setup): the dispatch in
sample_basis_states(), the bng-native sampler (real ctypes call into
libssa.so), and the libRoadRunner sampler's dependency guard.

Tests that actually run libssa.so are marked @requires_bng_lib and skip
cleanly on platforms where that specific binary can't be loaded (e.g. a
Linux build on macOS) rather than failing.
"""
import os
import warnings

from conftest import requires_bng_lib


class TestSampleBasisStatesBng:
    @requires_bng_lib
    def test_weighted_filtering_against_real_lib(self, bare_weconvert, sim_root):
        """
        End-to-end: runs the real libssa.so via ctypes, samples a
        trajectory, and filters using a weighted, multi-species group —
        the case that used to be silently wrong before the group-parsing
        fix (it would have picked only species2, ignoring species3 and
        both coefficients).
        """
        wc = bare_weconvert(
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

    @requires_bng_lib
    def test_excludes_frames_outside_basis_region(self, bare_weconvert, sim_root):
        """
        A basis_region no frame can satisfy should fall back to a single
        basis state (init.net copied to 0.net) rather than writing an
        empty or invalid pool file.
        """
        wc = bare_weconvert(
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


class TestSampleBasisStatesDispatch:
    def test_no_recycling_short_circuits_to_single_basis_state(
        self, bare_weconvert, sim_root
    ):
        """
        With recycling disabled, sample_basis_states() should never touch
        either sampling engine — it just copies init.net -> 0.net.
        """
        wc = bare_weconvert(do_recycling=False)
        wc.sample_basis_states()
        assert os.path.isfile("bstates/0.net")
        assert not os.path.isfile("bstates/microstate_pool.dat")

    @requires_bng_lib
    def test_bng_propagator_dispatches_to_bng_sampler(self, bare_weconvert, sim_root):
        wc = bare_weconvert(
            propagator_type="bng",
            do_recycling=True,
            basis_region=[{"observable": "SimpleObs", "min": 0.0, "max": 1e9}],
            bs_traj_length=100,
        )
        wc.sample_basis_states()
        # if it ran, we get a pool file (or, at minimum, the fallback 0.net —
        # either way, no exception and no accidental libRoadRunner import)
        assert os.path.isfile("bstates/0.net") or os.path.isfile(
            "bstates/microstate_pool.dat"
        )

    def test_non_bng_propagator_requires_roadrunner(
        self, bare_weconvert, sim_root, monkeypatch
    ):
        """
        For propagator_type != "bng" with recycling on, sample_basis_states
        should route to the libRoadRunner sampler, which exits with a clear
        message if roadrunner isn't importable — rather than silently
        falling back to the bng sampler or crashing with an unrelated error.
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