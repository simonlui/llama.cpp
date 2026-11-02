#!/usr/bin/env bash
# test-backend-ops on the CPU OpenCL device with hooks.patch applied (see README.md).
#   bash run.sh <build dir> [test-backend-ops args...]      e.g. -o MUL_MAT -p 'type_a=q4_0,'
#   bash run.sh <build dir> --exe <binary> [args...]        another binary from <build dir>/bin
set -euo pipefail
BUILD="$1"
shift
EXE=test-backend-ops
DEFAULT_ARGS=(-b SYCL0)
if [[ "${1:-}" == "--exe" ]]; then
    EXE="$2"
    DEFAULT_ARGS=()
    shift 2
fi
# the paths below need an Intel GPU: ESIMD, DPAS, oneDNN, and the joint_matrix dequant GEMM, which the
# CPU device lists AMX shapes for and then fails to compile
export GGML_SYCL_TEST_CPU_DEVICE=1 ONEAPI_DEVICE_SELECTOR=opencl:cpu \
       GGML_SYCL_ENABLE_ESIMD=0 GGML_SYCL_ENABLE_XMX_ESIMD=0 GGML_SYCL_MMID_ESIMD=0 GGML_SYCL_FA_GQA_ESIMD=0 \
       GGML_SYCL_FA_DPAS=0 GGML_SYCL_ENABLE_DNN=0 GGML_SYCL_XMX_GATHER_SHAPES=0 \
       CL_CONFIG_CPU_FORCE_LOCAL_MEM_SIZE=256KB SYCL_CACHE_PERSISTENT=1
exec "$BUILD/bin/$EXE" ${DEFAULT_ARGS[@]+"${DEFAULT_ARGS[@]}"} "$@"
