#!/bin/bash
# build_libssa.sh
#
# Builds libssa.so — a Python-callable shared library wrapping Network3's
# optimized Gillespie SSA. Run once after installing WEBNG.
#
# Usage:
#   ./build_libssa.sh                        # auto-detect BNG install
#   BNG_PATH=/path/to/bng-linux ./build_libssa.sh  # explicit path
#
# Output: libssa.so in the current directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Find BNG installation ──────────────────────────────────────────────────
if [ -z "${BNG_PATH:-}" ]; then
    # Try to find it via the bionetgen Python package
    BNG_PATH=$(python -c "
import bionetgen, os
pkg = os.path.dirname(bionetgen.__file__)
linux = os.path.join(pkg, 'bng-linux')
if os.path.isdir(linux):
    print(linux)
" 2>/dev/null || echo "")
fi

if [ -z "$BNG_PATH" ] || [ ! -d "$BNG_PATH" ]; then
    echo "ERROR: Could not find BNG installation."
    echo "Set BNG_PATH environment variable to the bng-linux directory."
    exit 1
fi

echo "Using BNG at: $BNG_PATH"

# ── Find Network3 source ───────────────────────────────────────────────────
# The source is distributed with the bionetgen package
NETWORK3_SRC="${NETWORK3_SRC:-${BNG_PATH}/../../../bng2/Network3/src}"
NETWORK3_SRC="$(python -c "import os; print(os.path.realpath('$NETWORK3_SRC'))")"

if [ ! -f "$NETWORK3_SRC/network.cpp" ]; then
    echo "ERROR: Network3 source not found at $NETWORK3_SRC"
    echo "Looking for network.cpp..."
    find "$(dirname $BNG_PATH)" -name "network.cpp" 2>/dev/null | head -5
    exit 1
fi

LIBSOURCE="${LIBSOURCE:-${NETWORK3_SRC}/../../libsource}"
echo "Network3 source: $NETWORK3_SRC"
echo "libsource:       $LIBSOURCE"

# ── Build directory ────────────────────────────────────────────────────────
BUILD_DIR="$SCRIPT_DIR/build_libssa"
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

# ── Step 1: Build mathutils static library ────────────────────────────────
echo ""
echo "=== Building mathutils ==="
MATHUTILS_DIR="$BUILD_DIR/mathutils"
if [ ! -f "$MATHUTILS_DIR/libmathutils.a" ]; then
    mkdir -p "$MATHUTILS_DIR"
    cd "$MATHUTILS_DIR"
    tar xzf "$LIBSOURCE/Mathutils.tar.gz"
    cd Mathutils
    for f in *.c; do
        [ -f "$f" ] || continue
        gcc -O2 -fPIC -c "$f" -o "${f%.c}.o"
    done
    ar rcs "$MATHUTILS_DIR/libmathutils.a" *.o
    echo "mathutils built."
else
    echo "mathutils already built."
fi

# ── Step 2: Build muParser static library ─────────────────────────────────
echo ""
echo "=== Building muParser ==="
MUPARSER_DIR="$BUILD_DIR/muparser"
if [ ! -f "$MUPARSER_DIR/lib/libmuparser.a" ]; then
    mkdir -p "$MUPARSER_DIR"
    cd "$BUILD_DIR"
    tar xzf "$LIBSOURCE/muparser_v2_2_4.tar.gz"
    cd muparser_v2_2_4
    ./configure --prefix="$MUPARSER_DIR" --disable-samples \
                --enable-static --disable-shared \
                CXXFLAGS="-O2 -fPIC" 2>&1 | tail -5
    make -j4 2>&1 | tail -10
    make install
    echo "muParser built."
else
    echo "muParser already built."
fi

# ── Step 3: Build CVODE static library ────────────────────────────────────
echo ""
echo "=== Building CVODE ==="
CVODE_DIR="$BUILD_DIR/cvode"
if [ ! -f "$CVODE_DIR/lib/libsundials_cvode.a" ]; then
    mkdir -p "$CVODE_DIR"
    cd "$BUILD_DIR"
    tar xzf "$LIBSOURCE/cvode-2.6.0.tar.gz"
    cd cvode-2.6.0
    mkdir -p build_static
    cd build_static
    cmake .. \
        -DCMAKE_INSTALL_PREFIX="$CVODE_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS="-O2 -fPIC" \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_STATIC_LIBS=ON \
        -DEXAMPLES_ENABLE=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        2>&1 | tail -5
    make -j4 2>&1 | tail -10
    make install
    echo "CVODE built."
else
    echo "CVODE already built."
fi

# ── Step 4: Collect all Network3 source files ─────────────────────────────
echo ""
echo "=== Collecting Network3 sources ==="

# All .cpp and .c files in Network3/src, excluding:
#   run_network.cpp -- the executable's main(), not wanted in a shared library
#   network.cpp     -- compiled through network3_reset.cpp instead, which
#                      #includes it so libssa can reset Network3's file-local
#                      Gillespie state between ssa_init calls (see that file)
NETWORK3_CPP=$(find "$NETWORK3_SRC" -name "*.cpp" \
    ! -name "run_network.cpp" \
    ! -name "network.cpp" \
    | tr '\n' ' ')

NETWORK3_C=""

echo "C++ sources: $(echo $NETWORK3_CPP | wc -w) files"
echo "C sources:   $(echo $NETWORK3_C | wc -w) files"

# ── Step 5: Compile libssa.so ─────────────────────────────────────────────
echo ""
echo "=== Compiling libssa.so ==="

INCLUDES="\
    -I$NETWORK3_SRC \
    -I$NETWORK3_SRC/util \
    -I$NETWORK3_SRC/util/MTrand \
    -I$NETWORK3_SRC/util/rand2 \
    -I$NETWORK3_SRC/model \
    -I$NETWORK3_SRC/model/rateExpressions \
    -I$NETWORK3_SRC/model/reactions \
    -I$MUPARSER_DIR/include \
    -I$CVODE_DIR/include \
    -I$MATHUTILS_DIR/Mathutils"

LIBS="\
    $MUPARSER_DIR/lib/libmuparser.a \
    $CVODE_DIR/lib/libsundials_cvode.a \
    $CVODE_DIR/lib/libsundials_nvecserial.a \
    $MATHUTILS_DIR/libmathutils.a"

g++ -std=c++11 -O2 -fPIC -shared \
    -DRUN_NETWORK_VERSION=\"3.0\" \
    $INCLUDES \
    "$SCRIPT_DIR/ssa_wrapper.cpp" \
    "$SCRIPT_DIR/network3_reset.cpp" \
    $NETWORK3_CPP \
    $NETWORK3_C \
    $LIBS \
    -lm \
    -o "$SCRIPT_DIR/libssa.so"

echo ""
echo "=== Build complete ==="
echo "Output: $SCRIPT_DIR/libssa.so"
ls -lh "$SCRIPT_DIR/libssa.so"
