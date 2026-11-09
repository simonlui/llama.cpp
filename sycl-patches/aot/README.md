# ISA check for the Arc A770

`isa_check.py` compiles one `ggml/src/ggml-sycl` file ahead of time for one device and reports, per
kernel, the register file IGC picked, the bytes it spilled, and its DPAS instructions, split into real
ones and the dummies IGC inserts on Arc A-series. It needs no GPU, only the oneAPI compiler. This is the
check behind the spill and DPAS numbers in `../../sycl-patches-extracted/README.md` and
`../../sycl-patches-candidates/README.md`.

```bash
cmake -S . -B build-sycl -DGGML_SYCL=ON -DGGML_SYCL_F16=ON -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx
```
```bash
python3 sycl-patches/aot/isa_check.py build-sycl fattn-dpas.cpp --filter fattn_dpas_kernel
```

Configuring is enough: the script takes the file's compile command from `compile_commands.json`, so it
sees the same defines and includes as the real build, JIT or AOT. It then swaps the SYCL targets for
`spir64_gen` with the options `GGML_SYCL_DEVICE_ARCH` uses, and sets `GGML_SYCL_XMX_AOT_SG` to the
device's DPAS width (8 for `acm-*`, 16 otherwise). `--device bmg-g21` checks Battlemage, and
`-D NAME=VALUE` adds a define (for example `-D GGML_SYCL_FA_DPAS_QK_GUARD=1`). `--report DIR` prints
the table again for an earlier run. Each run writes to a new directory (`isa-<file>-<device>` by
default). IGC's dumps stay in its `dump/` folder for reading the assembly.

## Reading the table

| Column | Meaning | Target |
|--------|---------|--------|
| GRF | register file (128, or 256 from `grf_size<256>` / auto large GRF) | 256 halves the threads per EU (8 to 4) |
| spill | bytes spilled per thread | 0 |
| dpas | DPAS instructions in the code (static, not executions) | |
| dummy | dummy DPAS from IGC's `Wa_18023229625` (all ACM-G10 steppings) | at most one per loop body |

The dummy is an 8x8 DPAS with a null src0 whose destination and sources are one register. IGC puts one
before the first DPAS of every basic block, and before a back-to-back 8x8 DPAS that would reuse src1. It
occupies the DPAS pipe like a real one. About one dummy per real DPAS means either a branch around each
DPAS (a per-tile `if` inside the unrolled loop), or consecutive DPAS sharing src1. Order the second case
so consecutive DPAS share src2 (A, vISA `DA`) and vary src1 (B, `DB`).

Reference counts, `acm-g10`, oneAPI 2026.2 `icpx` and `ocloc`:

| Kernel set | DPAS + dummy | Note |
|------------|--------------|------|
| C60 `fattn_dpas_kernel` (42 kernels) | 473 + 87 | 490 + 292 before the Q.K guard fix; `-D GGML_SYCL_FA_DPAS_QK_GUARD=1` gives that back |
| fused-gemm, sub-group 8, 128 GRF (0010) | 4 + 1 per k-step | 0017's 256-GRF tile: 16 + 2 |
| C21 ESIMD grouped MoE GEMM | 16 + 1 | |
| C40 XMX decode (D = 256, q8_0) | 40 + 2 | |
| 0016 MMQ-XMX | 1 + 0 or 1 per k-block | one DPAS between two SLM barriers |

The counts say what the compiler produced, not how fast it runs: an instruction-count win can still
lose on the card. Decide speed with `../ab/`.
