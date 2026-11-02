# Running the SYCL backend on the CPU OpenCL device (test only)

For checking SPMD kernel changes (MMVQ, DMMV, dequantize, GET_ROWS, the reorder paths) against the CPU
reference without an Intel GPU. ESIMD, DPAS and `joint_matrix` kernels do not run here; build them for
the GPU (`spir64_gen -device acm-g10`) and test them on the card.

Needs oneAPI (`icpx`, environment loaded with `setvars.sh`) and Intel's CPU OpenCL runtime
(`intel-opencl-rt`). In a scratch worktree:

```bash
bash sycl-patches-extracted/apply.sh .
```
```bash
git apply sycl-patches-extracted/tools/cpu-device-test/hooks.patch
```
```bash
cmake -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_SYCL=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_CXX_FLAGS=-fsycl-device-code-split=per_kernel -DCMAKE_SHARED_LINKER_FLAGS=-fsycl-device-code-split=per_kernel -DCMAKE_EXE_LINKER_FLAGS=-fsycl-device-code-split=per_kernel
```
```bash
cmake --build build-cpu -j --target ggml-sycl test-backend-ops
```
```bash
bash sycl-patches-extracted/tools/cpu-device-test/run.sh build-cpu -o MUL_MAT -p 'type_a=iq2_xxs,'
```

`-fsycl-device-code-split=per_kernel` makes the CPU compile only the kernels a test launches (about
26 minutes for the first build on 4 cores). `run.sh` turns off the ESIMD, DPAS, oneDNN and
`joint_matrix` paths, which the CPU device cannot build. Set `GGML_SYCL_TEST_FORCE_REORDER=1` to test
the reorder layouts. `hooks.patch` applies on the stack (local patches included) without the
candidates, whose CMake change it conflicts with. Revert it before unapplying the stack.

`test-backend-ops` builds fresh weights for every case, so it only sees a reorder on the call that
installs it. `reorder-reuse.cpp` covers the later readers of a reordered weight. Per type, a one-token
MUL_MAT and MUL_MAT_ID install the reorder; a 32-token MUL_MAT, a GET_ROWS (tied embeddings) and a
32-token MUL_MAT_ID then read it. All five results are checked against the CPU backend:

```bash
g++ -O2 -std=c++17 -Iggml/include sycl-patches-extracted/tools/cpu-device-test/reorder-reuse.cpp -o build-cpu/bin/reorder-reuse -Lbuild-cpu/bin -lggml -lggml-base -lggml-cpu -Wl,-rpath,$PWD/build-cpu/bin -Wl,-rpath-link,/opt/oneapi/lib
```
```bash
GGML_SYCL_TEST_FORCE_REORDER=1 bash sycl-patches-extracted/tools/cpu-device-test/run.sh build-cpu --exe reorder-reuse iq4_nl iq3_xxs
```

`run.sh --exe <name>` runs another binary from the build's `bin` with the same environment. With no
type arguments, `reorder-reuse` covers every type that has a 2D or MoE reorder.
