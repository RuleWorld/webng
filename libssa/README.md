# libssa

`libssa.so` is the shared library behind WEBNG's `bng` propagator: BioNetGen's
Network3 Gillespie SSA engine, wrapped so Python can call it through `ctypes`.
The built library ships with WEBNG at `webng/assets/libssa.so`; this folder
holds everything needed to rebuild it.

| File | What it is |
|---|---|
| `build_libssa.sh` | Builds `libssa.so` from a BioNetGen source checkout. |
| `ssa_wrapper.cpp` | The C interface Python calls (`ssa_init`, `ssa_set_state`, `ssa_run`, `ssa_run_multi`, ...). |
| `network3_reset.cpp` | Compiles Network3's `network.cpp` inside itself so libssa can reset Network3's global state at the start of every `ssa_init`, and turns off on-the-fly network generation. See the comments in the file for why both are needed. |
| `network3.patch` | WEBNG's one change to BioNetGen itself: adds `resync_gillespie_state()`, which recomputes every reaction propensity after `ssa_set_state()` changes the species counts. |

## Requirements

Linux (x86-64), `gcc`/`g++`, `make`, `cmake` (for CVODE), `patch`, and a
BioNetGen source checkout (https://github.com/RuleWorld/bionetgen). Only its
`bng2/Network3/src` and `bng2/libsource` folders are used.

## Building

```bash
cd libssa
export NETWORK3_SRC=/path/to/bionetgen/bng2/Network3/src
export LIBSOURCE=/path/to/bionetgen/bng2/libsource
./build_libssa.sh
```

The script copies Network3 into `build_libssa/` (git-ignored), applies
`network3.patch` to the copy, builds muParser, CVODE, and Mathutils from
`libsource`, and writes `libssa/libssa.so`. Your BioNetGen checkout is never
modified. If the source already contains the patch's change, the patch step is
skipped. The dependency builds in `build_libssa/` are reused on later runs;
delete that folder to rebuild them.

The patch has been checked against two BioNetGen versions (the one the
original library was built from, and a newer one with upstream memory-safety
fixes). Both builds produced identical trajectories for the models tested, so
the newer source is the better choice.

## Shipping a new build

```bash
cp libssa/libssa.so webng/assets/libssa.so
pytest tests/test_libssa_reinit.py tests/test_recycling.py -v
```

Commit `webng/assets/libssa.so` together with any source changes here, so the
shipped binary always matches what's in git.
