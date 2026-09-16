"""
Shared pytest fixtures for the WEBNG test suite.
"""
import ctypes
import os

import pytest

from webng.core.weConvert import weConvert


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