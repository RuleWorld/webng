"""
Unit tests for the propagators: the propagator scripts and west.cfg blocks
weConvert writes for each propagator_type (executable, libRoadRunner, bng,
ezgillespz), and the propagators' own behavior.

Most of these test the *generated output*, not a running simulation — each
test instantiates a bare weConvert (bypassing YAML loading), calls a writer
method, and checks the file it produced. This catches the kind of bug we
found by hand during development: wrong config namespace, stale method
names after a rename, syntactically invalid generated Python, etc.

The ezgillespz section also drives the propagator itself with fake WESTPA
segments and checks its statistics against exact answers and against libssa.
Those tests need ezgillespz installed (it is not on PyPI, so CI skips them);
the ones that only check what `webng setup` writes run everywhere.
"""
import ast
import json
import math
import subprocess
import sys

import numpy as np
import pytest
import yaml

from conftest import make_bare_weconvert, requires_bng_lib, requires_ezgillespz
from webng.core.weConvert import weConvert


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
# Files copied from webng/assets (the same for every simulation)
# ---------------------------------------------------------------------------

STATIC_FILES = [
    # writer, asset in webng/assets, where setup puts it, executable
    ("_write_bngPropagator", "bng_propagator.py", "bng_propagator.py", False),
    ("_write_librrPropagator", "libRR_propagator.py", "libRR_propagator.py", False),
    ("_write_ezgillespzPropagator", "ezgillespz_propagator.py", "ezgillespz_propagator.py", False),
    ("_write_restartDriver", "restart_plugin.py", "restart_plugin.py", False),
    ("_write_auxfuncs", "aux_functions.py", "aux_functions.py", False),
    ("_write_getpcoord_sh", "get_pcoord.sh", "westpa_scripts/get_pcoord.sh", True),
    ("_write_postiter_sh", "post_iter.sh", "westpa_scripts/post_iter.sh", True),
]


@pytest.mark.parametrize("writer, asset, dest, executable", STATIC_FILES)
def test_static_files_are_copied_from_assets(bare_weconvert, tmp_path, monkeypatch, writer, asset, dest, executable):
    import os
    monkeypatch.chdir(tmp_path)
    (tmp_path / "westpa_scripts").mkdir()
    getattr(bare_weconvert(), writer)()
    src = os.path.join(weConvert._assets_dir(), asset)
    assert (tmp_path / dest).read_bytes() == open(src, "rb").read()
    assert os.access(tmp_path / dest, os.X_OK) == executable
    if asset.endswith(".py"):
        ast.parse((tmp_path / dest).read_text())


def test_every_python_asset_is_valid():
    """Assets are copied verbatim, so a syntax error would only show up in a
    user's run; catch it here."""
    import glob, os
    for path in glob.glob(os.path.join(weConvert._assets_dir(), "*.py")):
        ast.parse(open(path).read(), filename=path)


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
        ("_ezgillespz_westcfg", "ezgillespz"),
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


# ---------------------------------------------------------------------------
# ezgillespz — what `webng setup` writes (no ezgillespz needed)
# ---------------------------------------------------------------------------

EZG_ADAPTIVE = dict(propagator_type="ezgillespz", binning_style="adaptive", do_recycling=False,
                    traj_per_bin=10, max_centers=300, center_freq=1,
                    block_size=weConvert.EZGILLESPZ_BLOCK_SIZE)


def ezg_weconvert(**attrs):
    return make_bare_weconvert(**{**EZG_ADAPTIVE, **attrs})


class TestWriteEzgillespzPropagator:
    def test_copies_valid_python(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        ezg_weconvert()._write_ezgillespzPropagator()
        src = (tmp_path / "ezgillespz_propagator.py").read_text()
        ast.parse(src)
        assert "class ezgillespzPropagator(WESTPropagator):" in src
        assert "SECTION = (\"west\", \"ezgillespz\")" in src
        assert "bstate_lookup_debug" not in src          # the first draft's always-on debug log

    def test_setup_dispatch(self, tmp_path, monkeypatch):
        """write_static_files routes ezgillespz to its own writers, never bng's."""
        calls = []
        wc = ezg_weconvert()
        for name in ["_write_envsh", "_write_auxfuncs", "_write_runsh", "_write_restartDriver",
                     "_write_ezgillespzPropagator"]:
            monkeypatch.setattr(wc, name, lambda *a, _n=name, **k: calls.append(_n))
        for name in ["_write_bngPropagator", "_copy_bng_lib", "_write_librrPropagator"]:
            monkeypatch.setattr(wc, name, lambda *a, _n=name, **k: pytest.fail(_n + " called"))
        monkeypatch.chdir(tmp_path)
        wc.write_static_files()
        assert {"_write_restartDriver", "_write_ezgillespzPropagator"} <= set(calls)


class TestEzgillespzWestcfg:
    def _cfg(self, tmp_path, monkeypatch, **attrs):
        monkeypatch.chdir(tmp_path)
        ezg_weconvert(**attrs)._ezgillespz_westcfg()
        text = (tmp_path / "west.cfg").read_text()
        return text, yaml.safe_load(text)["west"]

    def test_propagator_and_namespace(self, tmp_path, monkeypatch):
        text, cfg = self._cfg(tmp_path, monkeypatch)
        assert cfg["propagation"]["propagator"] == "ezgillespz_propagator.ezgillespzPropagator"
        assert cfg["propagation"]["block_size"] == weConvert.EZGILLESPZ_BLOCK_SIZE
        init = cfg["ezgillespz"]["init"]
        assert init["net_file"] == "./bngl_conf/init.net"
        assert init["final_time_step"] == 50.0 and init["num_time_step"] == 3   # tau=50, plen=2
        assert (init["device"], init["kind"], init["headroom_bits"], init["thin"], init["seed"]) == \
            ("cpu", "compact", 8, False, None)
        assert cfg["ezgillespz"]["data"]["pcoords"] == ["NANOG", "GATA6"]
        assert "librr" not in text and "  bng:" not in text

    def test_options_reach_west_cfg(self, tmp_path, monkeypatch):
        _, cfg = self._cfg(tmp_path, monkeypatch, ezg_device="gpu", ezg_kind="generated",
                           ezg_headroom_bits=12, ezg_thin=True, ezg_seed=1234)
        init = cfg["ezgillespz"]["init"]
        assert (init["device"], init["kind"], init["headroom_bits"], init["thin"], init["seed"]) == \
            ("gpu", "generated", 12, True, 1234)

    def test_restart_plugin_and_adaptive_voronoi(self, tmp_path, monkeypatch):
        _, cfg = self._cfg(tmp_path, monkeypatch)
        plugins = [p["plugin"] for p in cfg["plugins"]]
        assert "restart_plugin.RestartDriver" in plugins
        assert "westpa.westext.adaptvoronoi.AdaptiveVoronoiDriver" in plugins

    def test_mabl_uses_its_driver(self, tmp_path, monkeypatch):
        _, cfg = self._cfg(tmp_path, monkeypatch, binning_style="mabl")
        assert cfg["drivers"]["we_driver"] == "MABL_driver.MABLDriver"

    def test_gen_istates_reflects_recycling(self, tmp_path, monkeypatch):
        for do_recycling in (True, False):
            _, cfg = self._cfg(tmp_path, monkeypatch, do_recycling=do_recycling)
            assert cfg["propagation"]["gen_istates"] is do_recycling


@pytest.mark.parametrize("writer, propagator", [
    ("_bng_westcfg", "bng"),
    ("_libRR_westcfg", "libRoadRunner"),
    ("_ezgillespz_westcfg", "ezgillespz"),
])
def test_seed_dataset_is_not_scaleoffset(tmp_path, monkeypatch, writer, propagator):
    """scaleoffset: 4 on an integer dataset keeps only 4 bits: every recorded seed was lost."""
    monkeypatch.chdir(tmp_path)
    getattr(ezg_weconvert(propagator_type=propagator), writer)()
    datasets = yaml.safe_load((tmp_path / "west.cfg").read_text())["west"]["data"]["datasets"]
    assert not any(d["name"] == "seed" and "scaleoffset" in d for d in datasets)


def test_runsh_work_manager(tmp_path, monkeypatch):
    """ezgillespz runs each iteration in one process; the others fan out over processes."""
    monkeypatch.chdir(tmp_path)
    ezg_weconvert()._write_runsh()
    assert 'w_run --work-manager serial "$@"' in (tmp_path / "run.sh").read_text()
    make_bare_weconvert(propagator_type="bng")._write_runsh()
    assert 'w_run --work-manager processes "$@"' in (tmp_path / "run.sh").read_text()


class TestEzgillespzOptions:
    """propagator_options parsing, through weConvert's real __init__."""

    def _parse(self, tmp_path, monkeypatch, prop=None, binning=None):
        monkeypatch.chdir(tmp_path)
        opts = {
            "propagator_options": {"propagator_type": "ezgillespz", "pcoords": ["A", "B"], **(prop or {})},
            "path_options": {"bngl_file": "x.bngl", "sim_name": "sim"},
            "sampling_options": {"tau": 2, "dimensions": 2, "pcoord_length": 10},
            "binning_options": {"style": "adaptive", **(binning or {})},
        }
        path = tmp_path / "opts.yaml"
        path.write_text(yaml.safe_dump(opts))

        class Args:
            pass
        args = Args()
        args.opts = str(path)
        return weConvert(args)

    def test_defaults(self, tmp_path, monkeypatch):
        wc = self._parse(tmp_path, monkeypatch)
        assert (wc.ezg_device, wc.ezg_kind, wc.ezg_headroom_bits, wc.ezg_thin, wc.ezg_seed) == \
            ("cpu", "compact", 8, False, None)
        assert wc.pcoord_list == ["A", "B"]
        assert wc.block_size == weConvert.EZGILLESPZ_BLOCK_SIZE    # one batch per iteration

    def test_template_block_size_does_not_cap_batches(self, tmp_path, monkeypatch):
        """Every template writes binning_options.block_size: 10; with
        ezgillespz that must not turn into 10-walker batches."""
        wc = self._parse(tmp_path, monkeypatch, binning={"block_size": 10})
        assert wc.block_size == weConvert.EZGILLESPZ_BLOCK_SIZE

    def test_batch_size_from_propagator_options(self, tmp_path, monkeypatch):
        wc = self._parse(tmp_path, monkeypatch, prop={"block_size": 50000}, binning={"block_size": 10})
        assert wc.block_size == 50000

    def test_bad_batch_size_exits(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit, match="block_size"):
            self._parse(tmp_path, monkeypatch, prop={"block_size": 0})

    def test_options(self, tmp_path, monkeypatch):
        wc = self._parse(tmp_path, monkeypatch, prop={"device": "GPU", "seed": 7, "thin": True})
        assert (wc.ezg_device, wc.ezg_seed, wc.ezg_thin) == ("gpu", 7, True)

    def test_bad_device_exits(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit, match="device"):
            self._parse(tmp_path, monkeypatch, prop={"device": "tpu"})

    def test_unknown_propagator_type_exits(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit, match="propagator_type"):
            self._parse(tmp_path, monkeypatch, prop={"propagator_type": "ezgilespz"})


# ---------------------------------------------------------------------------
# ezgillespz — the propagator itself (needs ezgillespz)
# ---------------------------------------------------------------------------

# Immigration-death in A (0 -> A at k1, A -> 0 at k2 * A); B and C never react.
# From A(0) = a, A(t) = Binomial(a, p) + Poisson(k1/k2 (1 - p)) with p = exp(-k2 t).
BD_NET = """\
begin parameters
    1 k1 50.0
    2 k2 0.5
end parameters
begin species
    1 A() {a}
    2 B() {b}
    3 C() {c}
end species
begin reactions
    1 0 1 k1
    2 1 0 k2
end reactions
begin groups
    1 Atot 1
    2 Weighted 2*2,3*3
end groups
"""
K1, K2 = 50.0, 0.5

# Yule process: A -> A + A at rate k * A. From A(0) = a, E[A(t)] = a e^{kt},
# Var = a e^{kt} (e^{kt} - 1). The total rate grows e^{kt}-fold in one launch.
YULE_NET = """\
begin parameters
    1 k 1.0
end parameters
begin species
    1 A() {a}
end species
begin reactions
    1 1 1,1 k
end reactions
begin groups
    1 Atot 1
end groups
"""

# Dimerization: two identical reactants, where propensity conventions differ
# between simulators if they are going to (BNG puts the 1/2 in the .net rate).
DIMER_NET = """\
begin parameters
    1 kf 0.01
    2 kr 1.0
end parameters
begin species
    1 A() 200
    2 B() 0
end species
begin reactions
    1 1,1 2 0.5*kf
    2 2 1,1 kr
end reactions
begin groups
    1 Atot 1
    2 Btot 2
end groups
"""


class EzgSeg:
    SEG_STATUS_COMPLETE = 2

    def __init__(self, seg_id, n_iter, parent_id, data=None):
        self.seg_id, self.n_iter, self.parent_id = seg_id, n_iter, parent_id
        self.data = data or {}


class EzgState:
    def __init__(self, auxref=None, basis_state_id=None):
        self.auxref, self.basis_state_id = auxref, basis_state_id


@pytest.fixture
def ezg_module():
    return weConvert._ezgillespz_module()


def make_ezg_propagator(mod, tmp_path, net=BD_NET, init=(1, 2, 3), num_ts=11, tau=2.0,
                        pcoords=("Atot", "Weighted"), seed=12345, headroom_bits=8, bstates=None,
                        device="cpu"):
    """An ezgillespzPropagator without WESTPA's rc: attributes set as __init__ would."""
    a, b, c = (tuple(init) + (0, 0, 0))[:3]
    (tmp_path / "init.net").write_text(net.format(a=a, b=b, c=c))
    for name, counts in (bstates or {}).items():
        (tmp_path / name).write_text(net.format(**dict(zip("abc", counts))))
    P = mod.ezgillespzPropagator.__new__(mod.ezgillespzPropagator)
    P.net_file = str(tmp_path / "init.net")
    P.t_start, P.t_end, P.num_ts = 0.0, float(tau), num_ts
    P.pcoord_keys = list(pcoords)
    P._setup(device=device, kind="compact", headroom_bits=headroom_bits, thin=False, seed=seed)
    return P


@requires_ezgillespz
class TestEzgillespzStartStates:
    """The same start-state contract as bng_propagator.py (TestBngPropagatorStartStates)."""

    def _prop(self, ezg_module, tmp_path, monkeypatch, num_ts=11):
        monkeypatch.chdir(tmp_path)
        P = make_ezg_propagator(ezg_module, tmp_path, num_ts=num_ts,
                                bstates={"SC.net": (800, 0, 0), "IM.net": (100, 5, 5)})
        P.basis_states = {0: EzgState(auxref="SC.net"), 1: EzgState(auxref="IM.net")}
        P.initial_states = {0: EzgState(basis_state_id=0), 1: EzgState(basis_state_id=0),
                            2: EzgState(basis_state_id=1)}
        return P

    @pytest.mark.parametrize("num_ts", [2, 11])
    def test_iteration1_starts_from_basis_states(self, ezg_module, tmp_path, monkeypatch, num_ts):
        P = self._prop(ezg_module, tmp_path, monkeypatch, num_ts)
        segs = [EzgSeg(0, 1, -1), EzgSeg(1, 1, -2), EzgSeg(2, 1, -3)]          # SC, SC, IM
        P.propagate(segs)
        assert [s.pcoord[0][0] for s in segs] == [800, 800, 100]
        assert segs[2].pcoord[0][1] == 2 * 5 + 3 * 5                           # weighted observable
        assert all(s.pcoord.shape == (num_ts, 2) for s in segs)
        assert all(s.status == EzgSeg.SEG_STATUS_COMPLETE for s in segs)

    def test_continuing_walker_starts_from_restart_state(self, ezg_module, tmp_path, monkeypatch):
        P = self._prop(ezg_module, tmp_path, monkeypatch)
        seg = EzgSeg(3, 2, 0, {"restart_state": np.array([42.0, 1.0, 1.0])})
        P.propagate([seg])
        assert seg.pcoord[0][0] == 42 and seg.pcoord[0][1] == 5
        assert list(seg.data["final_state"][1:]) == [1.0, 1.0]                # B, C never react
        assert seg.pcoord[-1][0] == seg.data["final_state"][0]

    def test_recycled_walkers_use_pool_then_basis_state(self, ezg_module, tmp_path, monkeypatch):
        """A recycled walker in a later iteration with no pool must start from
        its own basis state, not the model default (the first draft's bug)."""
        P = self._prop(ezg_module, tmp_path, monkeypatch)
        pooled = EzgSeg(4, 3, -3, {"restart_state": [-1, -1, -1], "pool_microstate": np.array([7.0, 0, 0])})
        unpooled = EzgSeg(5, 3, -3, {"restart_state": [-1, -1, -1], "pool_microstate": None})
        P.propagate([pooled, unpooled])
        assert pooled.pcoord[0][0] == 7
        assert unpooled.pcoord[0][0] == 100 and unpooled.pcoord[0][1] == 25   # IM, not init.net
        assert "pool_microstate" not in pooled.data

    def test_continuing_walker_without_restart_state_raises(self, ezg_module, tmp_path, monkeypatch):
        P = self._prop(ezg_module, tmp_path, monkeypatch)
        with pytest.raises(RuntimeError, match="restart_state"):
            P.propagate([EzgSeg(6, 2, 0, {})])

    def test_basis_state_pcoord_comes_from_its_net_file(self, ezg_module, tmp_path, monkeypatch):
        P = self._prop(ezg_module, tmp_path, monkeypatch)
        bstate = EzgState(auxref="SC.net")
        P.get_pcoord(bstate)
        assert list(bstate.pcoord) == [800, 0]
        assert list(P.gen_istate(EzgState(auxref="IM.net"), EzgState()).pcoord) == [100, 25]

    def test_negative_restart_counts_are_refused(self, ezg_module, tmp_path, monkeypatch):
        P = self._prop(ezg_module, tmp_path, monkeypatch)
        with pytest.raises(ValueError, match=">= 0"):
            P.propagate([EzgSeg(7, 2, 0, {"restart_state": np.array([-3.0, 0, 0])})])

    def test_unknown_pcoord_is_an_error(self, ezg_module, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="not an observable"):
            make_ezg_propagator(ezg_module, tmp_path, pcoords=("Atot", "Nope"))


@requires_ezgillespz
def test_ezgillespz_seeds_recorded_and_runs_reproducible(ezg_module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = []
    for _ in range(2):
        P = make_ezg_propagator(ezg_module, tmp_path, seed=99)
        segs = [EzgSeg(i, 2, 0, {"restart_state": np.array([10.0, 0, 0])}) for i in range(50)]
        P.propagate(segs)
        out.append(segs)
    a, b = out
    assert len({s.data["seed"] for s in a}) == 1                   # one base seed per batch
    assert [s.data["seed_row"] for s in a] == list(range(50))
    assert all(np.array_equal(x.pcoord, y.pcoord) for x, y in zip(a, b))
    assert all(x.data["events"] > 0 for x in a)
    assert len({float(s.data["final_state"][0]) for s in a}) > 1   # walkers get different streams


def _z(sample_mean, mean, var, n):
    return abs(sample_mean - mean) / math.sqrt(var / n)


@requires_ezgillespz
class TestEzgillespzAgainstExactSolutions:
    @pytest.mark.parametrize("num_ts", [2, 11])
    def test_immigration_death_distribution(self, ezg_module, tmp_path, monkeypatch, num_ts):
        """Mean and variance of A(t) against the exact law, with the segment
        split into num_ts - 1 launches (so chaining launches is tested too)."""
        monkeypatch.chdir(tmp_path)
        n, a0, t = 20000, 10, 2.0
        P = make_ezg_propagator(ezg_module, tmp_path, init=(a0, 0, 0), num_ts=num_ts, tau=t)
        segs = [EzgSeg(i, 2, 0, {"restart_state": np.array([a0, 0.0, 0.0])}) for i in range(n)]
        P.propagate(segs)
        A = np.array([s.data["final_state"][0] for s in segs])
        p = math.exp(-K2 * t)
        mean = a0 * p + K1 / K2 * (1 - p)
        var = a0 * p * (1 - p) + K1 / K2 * (1 - p)
        assert _z(A.mean(), mean, var, n) < 5
        assert abs(A.var() - var) / (var * math.sqrt(2.0 / n)) < 5
        if num_ts > 2:      # an intermediate point is a real state on the way, not a copy of the end
            mid = np.array([s.pcoord[num_ts // 2][0] for s in segs])
            pm = math.exp(-K2 * t / 2)
            assert _z(mid.mean(), a0 * pm + K1 / K2 * (1 - pm), a0 * pm * (1 - pm) + K1 / K2 * (1 - pm), n) < 5

    def test_overflow_and_near_limit_recovery(self, ezg_module, tmp_path, monkeypatch):
        """With no headroom, a Yule process outgrows the fixed-point scale within
        a launch. The runner must make the scale coarser and carry on, and the
        result must still follow the exact law."""
        monkeypatch.chdir(tmp_path)
        n, a0, t = 4000, 200, 2.0
        P = make_ezg_propagator(ezg_module, tmp_path, net=YULE_NET, init=(a0,), num_ts=3, tau=t,
                                pcoords=("Atot",), headroom_bits=0)
        segs = [EzgSeg(i, 2, 0, {"restart_state": np.array([float(a0)])}) for i in range(n)]
        P.propagate(segs)
        assert P._runner.rescales >= 1 and P._runner.headroom > 0
        A = np.array([s.data["final_state"][0] for s in segs])
        mean, var = a0 * math.exp(t), a0 * math.exp(t) * (math.exp(t) - 1)
        assert _z(A.mean(), mean, var, n) < 5


LIBSSA_RUNNER = r"""
import ctypes, json, sys
import numpy as np
lib = ctypes.CDLL(sys.argv[1])
P = ctypes.POINTER(ctypes.c_double)
lib.ssa_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
lib.ssa_set_state.argtypes = [P, ctypes.c_int]
lib.ssa_reseed.argtypes = [ctypes.c_int]
lib.ssa_run.argtypes = [ctypes.c_double, ctypes.c_double]
lib.ssa_get_state.argtypes = [P, ctypes.c_int]
lib.ssa_get_n_species.restype = ctypes.c_int
net, n, t = sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
assert lib.ssa_init(net.encode(), 1, 1) == 0
ns = lib.ssa_get_n_species()
x0 = np.array([200.0, 0.0]); buf = np.zeros(ns)
out = []
for i in range(n):
    lib.ssa_set_state(x0.ctypes.data_as(P), ns)
    lib.ssa_reseed(i + 1)
    assert lib.ssa_run(0.0, t) == 0
    lib.ssa_get_state(buf.ctypes.data_as(P), ns)
    out.append(buf[0])
with open(sys.argv[5], "w") as f:      # libssa writes its own messages to stdout
    json.dump(out, f)
"""


@requires_ezgillespz
@requires_bng_lib
def test_ezgillespz_matches_libssa_on_dimerization(ezg_module, tmp_path, monkeypatch):
    """ezgillespz and libssa on a network with two identical reactants: the
    same distribution of A at t = 1 (two-sample z-tests on mean and variance).
    A factor-of-2 disagreement in the dimerization propensity would move the
    near-equilibrium mean from about 100 to about 124, some 100 standard
    errors at this n."""
    monkeypatch.chdir(tmp_path)
    n, t = 4000, 1.0
    (tmp_path / "dimer.net").write_text(DIMER_NET)
    lib = make_bare_weconvert()._resolve_bng_lib_path()
    out_json = tmp_path / "libssa_A.json"
    res = subprocess.run([sys.executable, "-c", LIBSSA_RUNNER, lib, str(tmp_path / "dimer.net"), str(n), str(t),
                          str(out_json)], capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    A_ssa = np.array(json.loads(out_json.read_text()))

    P = make_ezg_propagator(ezg_module, tmp_path, net=DIMER_NET, num_ts=2, tau=t, pcoords=("Atot", "Btot"))
    segs = [EzgSeg(i, 2, 0, {"restart_state": np.array([200.0, 0.0])}) for i in range(n)]
    P.propagate(segs)
    A_ezg = np.array([s.data["final_state"][0] for s in segs])

    se = math.sqrt(A_ssa.var() / n + A_ezg.var() / n)
    assert abs(A_ssa.mean() - A_ezg.mean()) / se < 5, (A_ssa.mean(), A_ezg.mean())
    v1, v2 = A_ssa.var(), A_ezg.var()
    assert abs(v1 - v2) / math.sqrt(2.0 / n * (v1 ** 2 + v2 ** 2)) < 5, (v1, v2)


# ---------------------------------------------------------------------------
# ezgillespz on a GPU (skipped unless ezgillespz finds a usable CUDA or Metal GPU)
# ---------------------------------------------------------------------------

# Metal keeps its clock as an integer and divides in single precision, so
# ezgillespz holds it to a window rather than exact equality with the CPU
# (its KNOWN_ISSUES #19 measured 99.976-100 % of walkers identical). CUDA must
# match the CPU exactly: the propagator launches each batch whole, in a fixed
# walker order, so nothing machine-dependent enters (compare KNOWN_ISSUES #40).
METAL_MIN_IDENTICAL = 0.999


@pytest.fixture(scope="module")
def gpu_backend():
    """'cuda' or 'metal'. Checked only when a GPU test runs: on CUDA this
    initializes the driver, on Metal it compiles ezgillespz's runtime."""
    gg = pytest.importorskip("ezgillespz")
    if not gg.gpu_available():
        pytest.skip("ezgillespz finds no usable CUDA or Metal GPU here")
    return gg.gpu_backend()


@requires_ezgillespz
class TestEzgillespzOnGpu:
    def _batch(self, ezg_module, tmp_path, device, n, a0=10, **kw):
        P = make_ezg_propagator(ezg_module, tmp_path, init=(a0, 0, 0), device=device, seed=2024, **kw)
        segs = [EzgSeg(i, 2, 0, {"restart_state": np.array([a0, 0.0, 0.0])}) for i in range(n)]
        P.propagate(segs)
        return P, segs

    def test_gpu_reproduces_cpu(self, ezg_module, gpu_backend, tmp_path, monkeypatch):
        """The same batch from the same seed on the CPU and on the GPU, split
        into 10 launches: CUDA identical walker for walker, Metal within its window."""
        monkeypatch.chdir(tmp_path)
        n = 4096
        _, cpu = self._batch(ezg_module, tmp_path, "cpu", n)
        _, gpu = self._batch(ezg_module, tmp_path, "gpu", n)
        same = np.array([np.array_equal(c.pcoord, g.pcoord) and
                         np.array_equal(c.data["final_state"], g.data["final_state"]) for c, g in zip(cpu, gpu)])
        ev_cpu = sum(s.data["events"] for s in cpu)
        ev_gpu = sum(s.data["events"] for s in gpu)
        print("\n[{}] {} of {} walkers identical to the CPU; events {} vs {}".format(
            gpu_backend, int(same.sum()), n, ev_gpu, ev_cpu))
        if gpu_backend == "cuda":
            assert same.all() and ev_gpu == ev_cpu
        else:
            assert same.mean() >= METAL_MIN_IDENTICAL
            assert abs(ev_gpu - ev_cpu) <= 1e-3 * ev_cpu

    def test_gpu_follows_exact_law(self, ezg_module, gpu_backend, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        n, a0, t = 20000, 10, 2.0
        _, segs = self._batch(ezg_module, tmp_path, "gpu", n, a0=a0, tau=t)
        A = np.array([s.data["final_state"][0] for s in segs])
        p = math.exp(-K2 * t)
        mean = a0 * p + K1 / K2 * (1 - p)
        var = a0 * p * (1 - p) + K1 / K2 * (1 - p)
        assert _z(A.mean(), mean, var, n) < 5
        assert abs(A.var() - var) / (var * math.sqrt(2.0 / n)) < 5

    def test_gpu_overflow_recovery(self, ezg_module, gpu_backend, tmp_path, monkeypatch):
        """The overflow flag comes back from the GPU kernel and the runner recovers."""
        monkeypatch.chdir(tmp_path)
        n, a0, t = 4000, 200, 2.0
        P = make_ezg_propagator(ezg_module, tmp_path, net=YULE_NET, init=(a0,), num_ts=3, tau=t,
                                pcoords=("Atot",), headroom_bits=0, device="gpu")
        segs = [EzgSeg(i, 2, 0, {"restart_state": np.array([float(a0)])}) for i in range(n)]
        P.propagate(segs)
        assert P._runner.rescales >= 1
        A = np.array([s.data["final_state"][0] for s in segs])
        mean, var = a0 * math.exp(t), a0 * math.exp(t) * (math.exp(t) - 1)
        assert _z(A.mean(), mean, var, n) < 5
