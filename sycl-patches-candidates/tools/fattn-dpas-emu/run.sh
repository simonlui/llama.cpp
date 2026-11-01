#!/usr/bin/env bash
# Check candidate C60 (DPAS flash attention at sub-group 8) without an Intel GPU.
#
#   bash sycl-patches-candidates/tools/fattn-dpas-emu/run.sh [TREE] [CASE]
#
# TREE is a llama.cpp checkout with C60 applied (default: this repository). The script copies
# its fattn-dpas.cpp with the three inline-vISA DPAS helpers replaced by a sub-group emulation
# of the hardware semantics, runs the unchanged kernels at both DPAS widths on Intel's CPU
# OpenCL device and compares them with an FP64 reference: q8_0 / q4_0 KV, head sizes 64-256,
# GQA 1-32, 1-7 query tokens, masks, sinks and split-K, 112 cases, about 3 minutes.
#
# Sub-group 16 is upstream's kernel, which passes test-backend-ops on Battlemage, so a pass at 16
# says the emulated register model agrees with the layouts upstream relies on; a pass at 8 says
# the port's layouts are right under the same model. Only an Arc A-series run tests the
# hardware itself. Needs icpx (oneAPI) and the Intel CPU OpenCL runtime.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "${1:-$HERE/../../..}" && pwd)"
CASE="${2:-}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

python3 -I "$HERE/make_emu.py" "$TREE/ggml/src/ggml-sycl/fattn-dpas.cpp" "$OUT/fattn-dpas-emu.cpp"
cp "$HERE/harness.cpp" "$OUT/"
echo "building (about a minute)..."
# the harness calls the kernels directly; the backend symbols the rest of the file references are never reached
icpx -fsycl -O2 -std=gnu++17 -DGGML_SYCL_SUPPORT_XMX -DGGML_SYCL_F16 -DGGML_SYCL_WARP_SIZE=16 \
    -I"$TREE/ggml/src/ggml-sycl" -I"$TREE/ggml/src" -I"$TREE/ggml/include" \
    -Wno-deprecated-declarations -Wno-ignored-attributes -Wno-narrowing \
    "$OUT/harness.cpp" -o "$OUT/harness" -Wl,--unresolved-symbols=ignore-all
# the staged kernels use up to 63 KB of local memory
CL_CONFIG_CPU_FORCE_LOCAL_MEM_SIZE=256KB ONEAPI_DEVICE_SELECTOR=opencl:cpu "$OUT/harness" $CASE
