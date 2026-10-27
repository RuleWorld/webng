#!/bin/bash
# build_libssa.sh
#
# Builds libssa -- a Python-callable shared library wrapping Network3's
# optimized Gillespie SSA -- for the machine you run it on (Linux or macOS,
# any CPU). WEBNG ships one build per platform in webng/assets/ and picks the
# right one at run time.
#
# Usage:
#   NETWORK3_SRC=/path/to/bionetgen/bng2/Network3/src \
#   LIBSOURCE=/path/to/bionetgen/bng2/libsource ./build_libssa.sh
#
#   (or leave both unset to find them through the installed bionetgen package)
#
# Output: libssa-<os>-<cpu>.<so|dylib> next to this script, e.g.
#   libssa-linux-x86_64.so, libssa-linux-aarch64.so, libssa-macos-arm64.dylib
# The <cpu> part is what Python's platform.machine() reports on that machine.
#
# Optional: LIBSSA_EXTRA_CFLAGS / LIBSSA_EXTRA_LDFLAGS add compiler / linker
# flags (e.g. for testing a build under stricter settings).
#
# Needs only a C/C++ compiler (gcc/g++ or clang), ar, and patch: muParser,
# CVODE, and Mathutils are compiled directly from BioNetGen's bundled
# sources, without their own build systems (whose old configure scripts
# predate Apple Silicon).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CC="${CC:-cc}"
CXX="${CXX:-c++}"

# ── Platform ───────────────────────────────────────────────────────────────
case "$(uname -s)" in
    Linux)  OS=linux; EXT=so;    SHARED="-shared" ;;
    Darwin) OS=macos; EXT=dylib; SHARED="-dynamiclib" ;;
    *) echo "ERROR: unsupported OS $(uname -s) (Linux or macOS only; on Windows, use WSL)"; exit 1 ;;
esac
CPU="$(uname -m)"
PLATFORM="$OS-$CPU"
OUTPUT="$SCRIPT_DIR/libssa-$PLATFORM.$EXT"
echo "Building for: $PLATFORM  ->  $(basename "$OUTPUT")"

# ── Find Network3 source and BioNetGen's bundled libraries ─────────────────
if [ -z "${NETWORK3_SRC:-}" ]; then
    # Try the source tree the installed bionetgen package came from
    BNG_PATH=$(python -c "
import bionetgen, os
pkg = os.path.dirname(bionetgen.__file__)
for name in ('bng-linux', 'bng-mac'):
    if os.path.isdir(os.path.join(pkg, name)):
        print(os.path.join(pkg, name)); break
" 2>/dev/null || echo "")
    if [ -z "$BNG_PATH" ]; then
        echo "ERROR: set NETWORK3_SRC (and LIBSOURCE) to a BioNetGen source checkout:"
        echo "  NETWORK3_SRC=.../bng2/Network3/src  LIBSOURCE=.../bng2/libsource"
        exit 1
    fi
    NETWORK3_SRC="$BNG_PATH/../../../bng2/Network3/src"
fi
NETWORK3_SRC="$(cd "$NETWORK3_SRC" 2>/dev/null && pwd || echo "$NETWORK3_SRC")"
if [ ! -f "$NETWORK3_SRC/network.cpp" ]; then
    echo "ERROR: Network3 source not found at $NETWORK3_SRC (no network.cpp)"
    exit 1
fi
LIBSOURCE="${LIBSOURCE:-$NETWORK3_SRC/../../libsource}"
LIBSOURCE="$(cd "$LIBSOURCE" && pwd)"
echo "Network3 source: $NETWORK3_SRC"
echo "libsource:       $LIBSOURCE"

# ── Build directory (one per platform, so builds never mix) ────────────────
BUILD_DIR="$SCRIPT_DIR/build_libssa/$PLATFORM"
mkdir -p "$BUILD_DIR"

# ── Apply WEBNG's changes to Network3 (network3.patch) ────────────────────
# Works from a stock BioNetGen source: Network3 is copied into the build
# folder and the copy is patched, so the source tree itself is never
# modified. If the source already contains the change (a hand-patched copy),
# the patch step is skipped.
PATCH_FILE="$SCRIPT_DIR/network3.patch"
PATCHED_SRC="$BUILD_DIR/Network3_src"
rm -rf "$PATCHED_SRC"
cp -r "$NETWORK3_SRC" "$PATCHED_SRC"
if grep -q "resync_gillespie_state" "$PATCHED_SRC/network.cpp"; then
    echo "Network3 source already contains WEBNG's changes; not patching."
else
    echo "Applying $PATCH_FILE"
    (cd "$PATCHED_SRC" && patch -p1 --forward < "$PATCH_FILE")
fi
NETWORK3_SRC="$PATCHED_SRC"

# ── Step 1: Mathutils ──────────────────────────────────────────────────────
echo ""
echo "=== Building Mathutils ==="
MATHUTILS_DIR="$BUILD_DIR/mathutils"
if [ ! -f "$MATHUTILS_DIR/libmathutils.a" ]; then
    mkdir -p "$MATHUTILS_DIR"
    (cd "$MATHUTILS_DIR" && tar xzf "$LIBSOURCE/Mathutils.tar.gz")
    # Only the library itself: skip Mathutils' own test programs (test*.c,
    # each with its own main()) and eigensystem.c, which calls LAPACK
    # routines this build doesn't link. libssa uses neither; on macOS, clang
    # also refuses to compile them (they call functions they never declare).
    (cd "$MATHUTILS_DIR/Mathutils" && for f in *.c; do
        case "$f" in test*.c|eigensystem.c) continue ;; esac
        "$CC" -O2 -fPIC ${LIBSSA_EXTRA_CFLAGS:-} -c "$f" -o "${f%.c}.o"
    done)
    ar rcs "$MATHUTILS_DIR/libmathutils.a" "$MATHUTILS_DIR"/Mathutils/*.o
    echo "Mathutils built."
else
    echo "Mathutils already built."
fi

# ── Step 2: muParser (compiled directly; needs no configuration) ───────────
echo ""
echo "=== Building muParser ==="
MUPARSER_DIR="$BUILD_DIR/muparser"
if [ ! -f "$MUPARSER_DIR/libmuparser.a" ]; then
    mkdir -p "$MUPARSER_DIR"
    (cd "$MUPARSER_DIR" && tar xzf "$LIBSOURCE/muparser_v2_2_4.tar.gz")
    MUP_SRC="$MUPARSER_DIR/muparser_v2_2_4"
    for f in "$MUP_SRC"/src/*.cpp; do
        "$CXX" -O2 -fPIC -I"$MUP_SRC/include" -c "$f" -o "$MUPARSER_DIR/$(basename "${f%.cpp}").o"
    done
    ar rcs "$MUPARSER_DIR/libmuparser.a" "$MUPARSER_DIR"/*.o
    echo "muParser built."
else
    echo "muParser already built."
fi
MUPARSER_INCLUDE="$MUPARSER_DIR/muparser_v2_2_4/include"

# ── Step 3: CVODE (compiled directly, with the one generated header) ───────
echo ""
echo "=== Building CVODE ==="
CVODE_DIR="$BUILD_DIR/cvode"
if [ ! -f "$CVODE_DIR/libsundials_cvode.a" ]; then
    mkdir -p "$CVODE_DIR"
    (cd "$CVODE_DIR" && tar xzf "$LIBSOURCE/cvode-2.6.0.tar.gz")
    CV_SRC="$CVODE_DIR/cvode-2.6.0"
    # What CVODE's configure would generate for this build: double
    # precision, generic math, no LAPACK, no export decoration.
    cat > "$CV_SRC/include/sundials/sundials_config.h" << 'EOF'
#define SUNDIALS_PACKAGE_VERSION "2.4.0"
#define SUNDIALS_DOUBLE_PRECISION 1
#define SUNDIALS_USE_GENERIC_MATH 1
#define SUNDIALS_BLAS_LAPACK 0
#define SUNDIALS_EXPORT
EOF
    CV_INC="-I$CV_SRC/include -I$CV_SRC/src/sundials"
    mkdir -p "$CVODE_DIR/obj_cvode" "$CVODE_DIR/obj_nvec"
    # cvode_lapack.c needs LAPACK, which this build doesn't use
    for f in "$CV_SRC"/src/cvode/*.c "$CV_SRC"/src/sundials/*.c; do
        [ "$(basename "$f")" = "cvode_lapack.c" ] && continue
        "$CC" -O2 -fPIC ${LIBSSA_EXTRA_CFLAGS:-} $CV_INC -c "$f" -o "$CVODE_DIR/obj_cvode/$(basename "${f%.c}").o"
    done
    "$CC" -O2 -fPIC ${LIBSSA_EXTRA_CFLAGS:-} $CV_INC -c "$CV_SRC/src/nvec_ser/nvector_serial.c" -o "$CVODE_DIR/obj_nvec/nvector_serial.o"
    ar rcs "$CVODE_DIR/libsundials_cvode.a" "$CVODE_DIR"/obj_cvode/*.o
    ar rcs "$CVODE_DIR/libsundials_nvecserial.a" "$CVODE_DIR"/obj_nvec/*.o
    echo "CVODE built."
else
    echo "CVODE already built."
fi
CVODE_INCLUDE="$CVODE_DIR/cvode-2.6.0/include"

# ── Step 4: Collect all Network3 source files ─────────────────────────────
echo ""
echo "=== Collecting Network3 sources ==="

# All .cpp files in Network3/src, excluding:
#   run_network.cpp -- the executable's main(), not wanted in a shared library
#   network.cpp     -- compiled through network3_reset.cpp instead, which
#                      #includes it so libssa can reset Network3's file-local
#                      Gillespie state between ssa_init calls (see that file)
NETWORK3_CPP=$(find "$NETWORK3_SRC" -name "*.cpp" \
    ! -name "run_network.cpp" \
    ! -name "network.cpp" \
    | tr '\n' ' ')
echo "C++ sources: $(echo $NETWORK3_CPP | wc -w | tr -d ' ') files"

# ── Step 5: Compile libssa ─────────────────────────────────────────────────
echo ""
echo "=== Compiling $(basename "$OUTPUT") ==="

INCLUDES="\
    -I$NETWORK3_SRC \
    -I$NETWORK3_SRC/util \
    -I$NETWORK3_SRC/util/MTrand \
    -I$NETWORK3_SRC/util/rand2 \
    -I$NETWORK3_SRC/model \
    -I$NETWORK3_SRC/model/rateExpressions \
    -I$NETWORK3_SRC/model/reactions \
    -I$MUPARSER_INCLUDE \
    -I$CVODE_INCLUDE \
    -I$MATHUTILS_DIR/Mathutils"

LIBS="\
    $MUPARSER_DIR/libmuparser.a \
    $CVODE_DIR/libsundials_cvode.a \
    $CVODE_DIR/libsundials_nvecserial.a \
    $MATHUTILS_DIR/libmathutils.a"

"$CXX" -std=c++11 -O2 -fPIC $SHARED \
    -DRUN_NETWORK_VERSION=\"3.0\" \
    $INCLUDES \
    "$SCRIPT_DIR/ssa_wrapper.cpp" \
    "$SCRIPT_DIR/network3_reset.cpp" \
    $NETWORK3_CPP \
    $LIBS \
    -lm \
    ${LIBSSA_EXTRA_LDFLAGS:-} \
    -o "$OUTPUT"

echo ""
echo "=== Build complete ==="
echo "Output: $OUTPUT"
ls -lh "$OUTPUT"
echo "To ship it with WEBNG: cp $(basename "$OUTPUT") ../webng/assets/"