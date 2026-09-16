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
# bng lib path resolution (bundled asset vs. explicit override)
# ---------------------------------------------------------------------------

class TestResolveBngLibPath:
    def test_explicit_override_wins(self, bare_weconvert):
        wc = bare_weconvert(libssa_path="/some/custom/path/libssa.so")
        assert wc._resolve_bng_lib_path() == "/some/custom/path/libssa.so"

    def test_falls_back_to_bundled_asset(self, bare_weconvert):
        import os
        wc = bare_weconvert(libssa_path=None)
        resolved = wc._resolve_bng_lib_path()
        assert resolved.endswith(os.path.join("assets", "libssa.so"))
        assert os.path.isfile(resolved), (
            "bundled libssa.so not found — did you forget to add it "
            "under webng/assets/?"
        )