"""
Two libssa.so bugs, both fixed in ssa_wrapper.cpp / network3_reset.cpp:

1. Initializing more than once in one process. Network3 (the BioNetGen engine
   inside libssa) accumulates global state across inits instead of replacing
   it, so a second ssa_init corrupted memory and segfaulted. Every ssa_init now
   resets Network3 first, and a re-init must match a fresh process exactly.

2. Species with no reactions that start at zero (e.g. a terminal product).
   Network3's on-the-fly network generation waited on stdin for BNG2.pl the
   first time such a species was populated: a hang, or a crash with stdin
   closed. libssa now turns on-the-fly generation off. BIGGER_NET below has
   such a species (C).

Each check runs in a subprocess with a timeout, so a regression fails the test
instead of crashing or hanging pytest.
"""
import json
import subprocess
import sys

from conftest import requires_bng_lib, make_bare_weconvert, SYNTHETIC_NET

pytestmark = requires_bng_lib

# A second, differently sized model (so stale array sizes would show up too),
# with a terminal product C that starts at zero.
BIGGER_NET = """\
begin parameters
  1 k1  0.5
  2 k2  0.3
end parameters
begin species
  1 A() 40
  2 B() 0
  3 C() 0
  4 D() 7
end species
begin reactions
  1 1 2 k1
  2 2 3 k2
  3 4 1 k2
end reactions
begin groups
  1 Atot 1
  2 Weighted 2*2,3*3
end groups
"""

RUNNER = r"""
import ctypes, json, sys
import numpy as np
lib = ctypes.CDLL(sys.argv[1])
P = ctypes.POINTER(ctypes.c_double)
lib.ssa_init.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
lib.ssa_run_multi.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, P, ctypes.c_int]
results = []
for spec in sys.argv[2:]:
    net, seed = spec.rsplit(":", 1)
    assert lib.ssa_init(net.encode(), int(seed), 1) == 0, "ssa_init failed for " + net
    n = lib.ssa_get_n_species()
    buf = np.zeros(11 * n)
    assert lib.ssa_run_multi(0.0, 10.0, 11, buf.ctypes.data_as(P), n) == 0
    results.append(buf.tolist())
print("RESULTS " + json.dumps(results))
"""


def _run(*specs):
    lib = make_bare_weconvert()._resolve_bng_lib_path()
    try:
        proc = subprocess.run([sys.executable, "-c", RUNNER, lib, *specs],
                              capture_output=True, text=True, timeout=120,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        # corrupted state can also loop forever instead of crashing
        raise AssertionError("libssa hung running {}".format(specs))
    assert proc.returncode == 0, (
        "libssa crashed or failed (exit code {}) running {}:\n{}".format(
            proc.returncode, specs, proc.stderr[-2000:]))
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULTS ")][-1]
    return json.loads(line[len("RESULTS "):])


def test_reinit_matches_a_fresh_process(tmp_path):
    small = tmp_path / "small.net"
    small.write_text(SYNTHETIC_NET)
    bigger = tmp_path / "bigger.net"
    bigger.write_text(BIGGER_NET)
    a, b = "{}:11".format(small), "{}:7".format(bigger)

    fresh_a, fresh_b = _run(a)[0], _run(b)[0]
    sequence = _run("{}:1".format(bigger), a, b, a)   # four inits, one process

    assert sequence[1] == fresh_a     # bit-identical to a fresh process
    assert sequence[2] == fresh_b
    assert sequence[3] == fresh_a