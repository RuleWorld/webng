# libssa

`libssa` is the shared library behind WEBNG's `bng` propagator: BioNetGen's
Network3 Gillespie SSA engine, wrapped so Python can call it through `ctypes`.
WEBNG ships one build per platform in `webng/assets/`, named
`libssa-<os>-<cpu>.<so|dylib>`, and picks the one matching the running Python
(`platform.system()` and `platform.machine()`). `webng setup` copies it into
each sim folder as `libssa.so`. This folder holds everything needed to rebuild
them.

| File | What it is |
|---|---|
| `build_libssa.sh` | Builds libssa for the machine it runs on, from a BioNetGen source checkout. |
| `ssa_wrapper.cpp` | The C interface Python calls (`ssa_init`, `ssa_set_state`, `ssa_run`, `ssa_run_multi`, ...). |
| `network3_reset.cpp` | Compiles Network3's `network.cpp` inside itself so libssa can reset Network3's global state at the start of every `ssa_init`, and turns off on-the-fly network generation. See the comments in the file for why both are needed. |
| `check_libssa.py` | Tests a build on the machine it targets, with only Python and NumPy. |
| `network3.patch` | WEBNG's one change to BioNetGen itself: adds `resync_gillespie_state()`, which recomputes every reaction propensity after `ssa_set_state()` changes the species counts. |

## Platforms

| Platform | Build | Notes |
|---|---|---|
| Linux, x86-64 | `libssa-linux-x86_64.so` | |
| Linux, ARM | `libssa-linux-aarch64.so` | |
| macOS, Apple Silicon | `libssa-macos-arm64.dylib` | |
| macOS, Intel | `libssa-macos-x86_64.dylib` | |
| Windows | none | Run WEBNG under WSL, where the Linux build works. WEBNG's generated scripts and WESTPA itself expect a Unix system. |

The build must match the **Python** that runs WEBNG, which can differ from the
machine: a conda environment can be Intel-based on an Apple Silicon Mac. Check
with `python -c "import platform; print(platform.machine())"`.

## Requirements

A C/C++ compiler (`gcc`/`g++` or Apple's `clang`; on macOS, the Xcode command
line tools: `xcode-select --install`), plus `ar` and `patch`, and a BioNetGen
source checkout (https://github.com/RuleWorld/bionetgen). Only its
`bng2/Network3/src` and `bng2/libsource` folders are used. No `cmake` needed:
muParser, CVODE, and Mathutils are compiled directly from BioNetGen's bundled
sources.

## Building

```bash
cd libssa
export NETWORK3_SRC=/path/to/bionetgen/bng2/Network3/src
export LIBSOURCE=/path/to/bionetgen/bng2/libsource
./build_libssa.sh
```

The script copies Network3 into `build_libssa/<platform>/` (git-ignored),
applies `network3.patch` to the copy, builds the three dependencies, and writes
`libssa-<os>-<cpu>.<so|dylib>` here. Your BioNetGen checkout is never modified;
if it already contains the patch's change, that step is skipped. Dependency
builds are reused on later runs; delete `build_libssa/` to rebuild them.

The patch has been checked against two BioNetGen versions (the one the
original library was built from, and a newer one with upstream memory-safety
fixes), which produced identical trajectories for the models tested.

## Building for another platform

`check_libssa.py` tests a build on the machine it targets, needing only Python
and NumPy (not WEBNG): `python check_libssa.py libssa-<os>-<cpu>.<so|dylib>`.
It runs the same checks as `tests/test_libssa_reinit.py`.

**Intel Mac, from an Apple Silicon Mac.** Apple's compiler can target Intel,
and Rosetta runs the result (install it with `softwareupdate --install-rosetta`
if `arch -x86_64 /usr/bin/true` fails):

```bash
cd libssa
LIBSSA_ARCH=x86_64 NETWORK3_SRC=... LIBSOURCE=... ./build_libssa.sh
CONDA_SUBDIR=osx-64 conda create -n libssa-intel python=3.10 numpy -y   # an Intel Python
conda activate libssa-intel
python check_libssa.py libssa-macos-x86_64.dylib
conda deactivate
```

(The same works in reverse, `LIBSSA_ARCH=arm64` on an Intel Mac.)

**ARM Linux, from an Apple Silicon Mac.** The Mac's chip is ARM, so Docker runs
an ARM Linux container natively. From the repository root, with your BioNetGen
checkout shared into the container:

```bash
docker run --rm -it --platform linux/arm64 \
    -v "$PWD":/webng -v /path/to/bionetgen:/bionetgen python:3.10 bash
# inside the container:
apt-get update && apt-get install -y build-essential patch
pip install numpy
cd /webng/libssa
NETWORK3_SRC=/bionetgen/bng2/Network3/src LIBSOURCE=/bionetgen/bng2/libsource ./build_libssa.sh
python check_libssa.py libssa-linux-aarch64.so
exit
```

On Linux the script only builds for the machine it runs on; for another chip,
use a container of that kind, as above.

## Shipping a new build

```bash
cp libssa/libssa-<os>-<cpu>.<so|dylib> webng/assets/
pytest tests/test_libssa_reinit.py tests/test_recycling.py tests/test_propagators.py -v
```

Commit the file in `webng/assets/` together with any source changes here, so
the shipped binaries always match what's in git.
