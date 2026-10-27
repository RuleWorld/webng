"""
Standalone check of a libssa build, needing only Python and NumPy (not WEBNG),
for testing a new build on the machine it targets -- e.g. an Intel Python on an
Apple Silicon Mac, or inside an ARM Linux container.

    python check_libssa.py libssa-<os>-<cpu>.<so|dylib>

Runs the same checks as tests/test_libssa_reinit.py, each in a subprocess with
a timeout, so a broken build fails cleanly instead of crashing or hanging:
  1. it loads, and this Python's chip matches the build's name;
  2. several inits of different models in ONE process each give exactly the
     same trajectory as a fresh process with the same model and seed;
  3. a model with a terminal product starting at zero runs (no on-the-fly
     network generation hang).
"""
import json
import os
import platform
import subprocess
import sys
import tempfile

SMALL_NET = """\
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

# differently sized, with a terminal product C that starts at zero
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


def run(lib, *specs):
    try:
        proc = subprocess.run([sys.executable, "-c", RUNNER, lib, *specs],
                              capture_output=True, text=True, timeout=120,
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        sys.exit("FAIL: libssa hung running {}".format(specs))
    if proc.returncode != 0:
        sys.exit("FAIL: libssa crashed or failed (exit code {}) running {}:\n{}".format(
            proc.returncode, specs, proc.stderr[-2000:]))
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULTS ")][-1]
    return json.loads(line[len("RESULTS "):])


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    lib = os.path.abspath(sys.argv[1])
    machine = platform.machine()
    print("This Python: {} {}".format(platform.system(), machine))
    if "-{}.".format(machine) not in os.path.basename(lib):
        print("WARNING: {} doesn't look built for this Python's chip ({}).".format(
            os.path.basename(lib), machine))

    with tempfile.TemporaryDirectory() as d:
        small, bigger = os.path.join(d, "small.net"), os.path.join(d, "bigger.net")
        open(small, "w").write(SMALL_NET)
        open(bigger, "w").write(BIGGER_NET)
        a, b = "{}:11".format(small), "{}:7".format(bigger)

        fresh_a, fresh_b = run(lib, a)[0], run(lib, b)[0]     # also: terminal product runs
        print("PASS: loads and runs, including the terminal-product model")
        sequence = run(lib, "{}:1".format(bigger), a, b, a)    # four inits, one process
        if sequence[1:] != [fresh_a, fresh_b, fresh_a]:
            sys.exit("FAIL: a re-init in the same process differed from a fresh process")
        print("PASS: four inits in one process, each identical to a fresh process")
    print("All checks passed for {}".format(os.path.basename(lib)))


if __name__ == "__main__":
    main()
