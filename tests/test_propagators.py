"""
Unit tests for the propagator writer methods in weConvert: the generated
propagator scripts and west.cfg blocks for each of the three
propagator_type options (executable, libRoadRunner, bng).

These test the *generated output*, not a running simulation — each test
instantiates a bare weConvert (bypassing YAML loading), calls a writer
method, and checks the file it produced. This catches the kind of bug we
found by hand during development: wrong config namespace, stale method
names after a rename, syntactically invalid generated Python, etc.
"""
import ast

import pytest


# ---------------------------------------------------------------------------
# bng
# ---------------------------------------------------------------------------

class TestWriteBngPropagator:
    def test_generates_valid_python(self, bare_weconvert, tmp_path, monkeypatch):
        wc = bare_weconvert()
        monkeypatch.chdir(tmp_path)
        wc._write_bngPropagator()
        out = tmp_path / "bng_propagator.py"
        assert out.is_file()
        ast.parse(out.read_text())  # raises SyntaxError if malformed

    def test_class_and_config_namespace(self, bare_weconvert, tmp_path, monkeypatch):
        wc = bare_weconvert()
        monkeypatch.chdir(tmp_path)
        wc._write_bngPropagator()
        src = (tmp_path / "bng_propagator.py").read_text()
        assert "class bngPropagator(WESTPropagator):" in src
        assert "'west', 'bng'," in src
        # only the literal filenames libssa.so and build_libssa.sh should
        # still say "libssa" — everything else should be "bng"-branded
        stripped = src.replace("libssa.so", "").replace("build_libssa.sh", "")
        assert "libssa" not in stripped


class TestBngPropagatorStartStates:
    """
    Loads the *generated* bng_propagator.py and drives propagate() with a
    fake simulator that behaves like the real library (one global state;
    each run adds +10 to species A per sub-interval), so these run on any
    platform without the real shared library.

    Regression coverage for bugs found in the generator:
      - iteration-1 walkers only reseeded and never set a start state, so
        each started where the previous one in the same worker ended
      - basis states were ignored (every bstate got init.net's pcoord)
      - with num_time_step == 2, pcoord[0] was always init.net's pcoord
    """
    NET = ("begin species\n1 A() {a}\n2 B() {b}\n3 C() {c}\nend species\n"
           "begin groups\n1 Atot 1\n2 Weighted 2*2,3*3\nend groups\n")

    class FakeLib:
        def __init__(self, n):
            import numpy as np
            self.np, self.n, self.state = np, n, np.zeros(n)
        def _arr(self, ptr, n):
            return self.np.ctypeslib.as_array(ptr, shape=(n,))
        def ssa_set_state(self, ptr, n):
            self.state = self._arr(ptr, n).copy(); return 0
        def ssa_reseed(self, seed):
            pass
        def ssa_run(self, t0, t1):
            self.state[0] += 10; return 0
        def ssa_get_state(self, ptr, n):
            self._arr(ptr, n)[:] = self.state; return 0
        def ssa_run_multi(self, t0, t1, nts, ptr, n):
            out = self.np.ctypeslib.as_array(ptr, shape=(nts * n,)).reshape(nts, n)
            out[0] = self.state
            for i in range(1, nts):
                self.state[0] += 10; out[i] = self.state
            return 0
        def ssa_get_last_error(self):
            return b""

    class Seg:
        SEG_STATUS_COMPLETE = 2
        def __init__(self, seg_id, n_iter, parent_id, data=None):
            self.seg_id, self.n_iter, self.parent_id = seg_id, n_iter, parent_id
            self.data = data or {}

    class State:
        def __init__(self, auxref=None, basis_state_id=None):
            self.auxref, self.basis_state_id = auxref, basis_state_id

    def _prop(self, bare_weconvert, tmp_path, monkeypatch, num_ts):
        import importlib.util
        import numpy as np
        monkeypatch.chdir(tmp_path)
        bare_weconvert()._write_bngPropagator()
        for name, (a, b, c) in {"init.net": (1, 2, 3), "SC.net": (800, 0, 0),
                                "IM.net": (100, 5, 5)}.items():
            (tmp_path / name).write_text(self.NET.format(a=a, b=b, c=c))
        spec = importlib.util.spec_from_file_location("gen_bng", tmp_path / "bng_propagator.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        P = mod.bngPropagator.__new__(mod.bngPropagator)   # skip __init__ (needs the real library)
        P.net_file = str(tmp_path / "init.net")
        P.t_start, P.t_end, P.num_ts = 0.0, 10.0, num_ts
        P.pcoord_keys, P._n_species = ["Atot", "Weighted"], 3
        P._lib = self.FakeLib(3)
        groups = mod._parse_net_groups(P.net_file)
        P._pcoord_terms = [groups[k] for k in P.pcoord_keys]
        P._init_counts = mod._parse_net_species_counts(P.net_file, 3)
        P._initial_pcoord = P._weighted_pcoord(P._init_counts)
        P._multi_buf, P._state_buf = np.zeros(num_ts * 3), np.zeros(3)
        P.basis_states = {0: self.State(auxref="SC.net"), 1: self.State(auxref="IM.net")}
        P.initial_states = {0: self.State(basis_state_id=0), 1: self.State(basis_state_id=0),
                            2: self.State(basis_state_id=1)}
        return P

    @pytest.mark.parametrize("num_ts", [2, 11])
    def test_iteration1_starts_from_basis_states_not_chained(self, bare_weconvert, tmp_path, monkeypatch, num_ts):
        P = self._prop(bare_weconvert, tmp_path, monkeypatch, num_ts)
        segs = [self.Seg(0, 1, -1), self.Seg(1, 1, -2), self.Seg(2, 1, -3)]  # SC, SC, IM, same worker
        P.propagate(segs)
        assert [s.pcoord[0][0] for s in segs] == [800, 800, 100]
        assert segs[2].pcoord[0][1] == 2 * 5 + 3 * 5       # weighted observable

    @pytest.mark.parametrize("num_ts", [2, 11])
    def test_continuing_walker_records_its_own_start(self, bare_weconvert, tmp_path, monkeypatch, num_ts):
        import numpy as np
        P = self._prop(bare_weconvert, tmp_path, monkeypatch, num_ts)
        seg = self.Seg(3, 2, 0, {"restart_state": np.array([42.0, 1.0, 1.0])})
        P.propagate([seg])
        assert seg.pcoord[0][0] == 42                      # not init.net's value
        assert seg.data["final_state"][0] == 42 + 10 * (num_ts - 1)

    def test_recycled_walkers_use_pool_then_basis_state(self, bare_weconvert, tmp_path, monkeypatch):
        import numpy as np
        P = self._prop(bare_weconvert, tmp_path, monkeypatch, 11)
        pooled = self.Seg(4, 3, -3, {"restart_state": [-1, -1, -1], "pool_microstate": np.array([7.0, 0, 0])})
        unpooled = self.Seg(5, 3, -3, {"restart_state": [-1, -1, -1], "pool_microstate": None})
        P.propagate([pooled, unpooled])
        assert pooled.pcoord[0][0] == 7
        assert unpooled.pcoord[0][0] == 100                # IM basis state, not chained from `pooled`

    def test_continuing_walker_without_restart_state_raises(self, bare_weconvert, tmp_path, monkeypatch):
        P = self._prop(bare_weconvert, tmp_path, monkeypatch, 11)
        with pytest.raises(RuntimeError, match="restart_state"):
            P.propagate([self.Seg(6, 2, 0, {})])

    def test_basis_state_pcoord_comes_from_its_net_file(self, bare_weconvert, tmp_path, monkeypatch):
        P = self._prop(bare_weconvert, tmp_path, monkeypatch, 11)
        bstate = self.State(auxref="SC.net")
        P.get_pcoord(bstate)
        assert list(bstate.pcoord) == [800, 0]

    def test_no_file_based_istate_hack(self, bare_weconvert, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        bare_weconvert()._write_bngPropagator()
        src = (tmp_path / "bng_propagator.py").read_text()
        for leftover in ["istate_net_map", "threading", "bstate_lookup_debug"]:
            assert leftover not in src


class TestBngWestcfg:
    def test_uses_bng_namespace_not_librr(self, bare_weconvert, tmp_path, monkeypatch):
        wc = bare_weconvert()
        monkeypatch.chdir(tmp_path)
        wc._bng_westcfg()
        cfg = (tmp_path / "west.cfg").read_text()
        assert "bng_propagator.bngPropagator" in cfg
        assert "  bng:" in cfg
        assert "librr" not in cfg

    def test_gen_istates_reflects_recycling(self, bare_weconvert, tmp_path, monkeypatch):
        for do_recycling, expected in [(True, "true"), (False, "false")]:
            wc = bare_weconvert(do_recycling=do_recycling)
            monkeypatch.chdir(tmp_path)
            wc._bng_westcfg()
            cfg = (tmp_path / "west.cfg").read_text()
            assert "gen_istates:          {}".format(expected) in cfg


# ---------------------------------------------------------------------------
# libRoadRunner — parity coverage: this propagator had no test coverage
# at all before, despite being one of only two options prior to bng.
# ---------------------------------------------------------------------------

class TestWriteLibrrPropagator:
    def test_generates_valid_python(self, bare_weconvert, tmp_path, monkeypatch):
        wc = bare_weconvert(propagator_type="libRoadRunner")
        monkeypatch.chdir(tmp_path)
        wc._write_librrPropagator()
        out = tmp_path / "libRR_propagator.py"
        assert out.is_file()
        ast.parse(out.read_text())


class TestLibRRWestcfg:
    def test_uses_librr_propagator_and_namespace(self, bare_weconvert, tmp_path, monkeypatch):
        wc = bare_weconvert(propagator_type="libRoadRunner")
        monkeypatch.chdir(tmp_path)
        wc._libRR_westcfg()
        cfg = (tmp_path / "west.cfg").read_text()
        assert "libRR_propagator.librrPropagator" in cfg
        assert "  librr:" in cfg


# ---------------------------------------------------------------------------
# executable
# ---------------------------------------------------------------------------

class TestExecutableWestcfg:
    def test_uses_executable_propagator(self, bare_weconvert, tmp_path, monkeypatch):
        wc = bare_weconvert(propagator_type="executable")
        monkeypatch.chdir(tmp_path)
        wc._executable_westcfg()
        cfg = (tmp_path / "west.cfg").read_text()
        assert "propagator:           executable" in cfg
        assert "gen_istates:          false" in cfg  # executable never recycles via this path


# ---------------------------------------------------------------------------
# max_run_wallclock (sampling_options.max_wallclock)
# ---------------------------------------------------------------------------

class TestWallclock:
    @pytest.mark.parametrize("value, expected", [
        ("72:00:00", "72:00:00"),
        ("720:00:00", "720:00:00"),
        ("1:30:00", "1:30:00"),
        (259200, "72:00:00"),        # an unquoted 72:00:00 in YAML arrives as base-60 seconds
        (2592000, "720:00:00"),
    ])
    def test_accepted_forms(self, bare_weconvert, value, expected):
        assert bare_weconvert()._parse_wallclock(value) == expected

    def test_yaml_unquoted_value_round_trips(self, bare_weconvert):
        import yaml
        value = yaml.safe_load("max_wallclock: 720:00:00")["max_wallclock"]
        assert bare_weconvert()._parse_wallclock(value) == "720:00:00"

    @pytest.mark.parametrize("bad", ["3 days", "72:00", "1:75:00", "0:00:00", None, True])
    def test_rejected_forms(self, bare_weconvert, bad):
        with pytest.raises(SystemExit, match="max_wallclock"):
            bare_weconvert()._parse_wallclock(bad)

    @pytest.mark.parametrize("writer, propagator", [
        ("_bng_westcfg", "bng"),
        ("_libRR_westcfg", "libRoadRunner"),
        ("_executable_westcfg", "executable"),
    ])
    def test_every_west_cfg_uses_it(self, bare_weconvert, tmp_path, monkeypatch, writer, propagator):
        monkeypatch.chdir(tmp_path)
        getattr(bare_weconvert(propagator_type=propagator, max_wallclock="720:00:00"), writer)()
        cfg = (tmp_path / "west.cfg").read_text()
        assert "max_run_wallclock:    720:00:00" in cfg
        assert "72:00:00" not in cfg.replace("720:00:00", "")


# ---------------------------------------------------------------------------
# bng lib path resolution (per-platform bundled builds vs. explicit override)
# ---------------------------------------------------------------------------

class TestResolveBngLibPath:
    def test_explicit_override_wins(self, bare_weconvert):
        wc = bare_weconvert(libssa_path="/some/custom/path/libssa.so")
        assert wc._resolve_bng_lib_path() == "/some/custom/path/libssa.so"

    @pytest.mark.parametrize("system, machine, expected", [
        ("Linux", "x86_64", "libssa-linux-x86_64.so"),
        ("Linux", "aarch64", "libssa-linux-aarch64.so"),
        ("Darwin", "arm64", "libssa-macos-arm64.dylib"),
        ("Darwin", "x86_64", "libssa-macos-x86_64.dylib"),
    ])
    def test_bundled_build_named_by_platform(self, bare_weconvert, system, machine, expected):
        assert bare_weconvert()._bng_lib_name(system, machine) == expected

    def test_falls_back_to_this_platforms_bundled_build(self, bare_weconvert):
        import os
        wc = bare_weconvert(libssa_path=None)
        resolved = wc._resolve_bng_lib_path()
        assert resolved.endswith(os.path.join("assets", wc._bng_lib_name()))
        assert os.path.isfile(resolved), (
            "no bundled libssa build for this platform ({}) -- build one with "
            "libssa/build_libssa.sh and add it under webng/assets/".format(wc._bng_lib_name())
        )

    def test_platform_without_a_build_gives_a_clear_error(self, bare_weconvert, tmp_path, monkeypatch):
        import platform
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(platform, "system", lambda: "Windows")
        monkeypatch.setattr(platform, "machine", lambda: "AMD64")
        wc = bare_weconvert(libssa_path=None)
        with pytest.raises(SystemExit) as exit_info:
            wc._copy_bng_lib()
        message = str(exit_info.value)
        assert "Windows-AMD64" in message and "WSL" in message
        assert "libssa-linux-x86_64.so" in message        # lists the builds that do exist
        assert not (tmp_path / "libssa.so").exists()

    def test_setup_copies_the_build_as_libssa_so(self, bare_weconvert, tmp_path, monkeypatch):
        """The generated propagator loads 'libssa.so' whatever the platform."""
        monkeypatch.chdir(tmp_path)
        wc = bare_weconvert(libssa_path=None)
        wc._copy_bng_lib()
        assert (tmp_path / "libssa.so").read_bytes() == open(wc._resolve_bng_lib_path(), "rb").read()
