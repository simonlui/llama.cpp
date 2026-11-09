# Opt-in SYCL candidates for the Arc A770 (DG2)

Patches that may make the A770 faster but are not proven on it yet. They are kept out of
the main stack: apply them on top of it, measure, and promote the winners into
`sycl-patches/` (upstream PRs) or `sycl-patches-extracted/` (everything else).

```bash
bash sycl-patches-extracted/apply.sh .      # main SYCL stack first
bash sycl-patches-candidates/apply.sh .     # then C10..C90
# build, test, measure
bash sycl-patches-candidates/unapply.sh .   # reverse the candidates first
bash sycl-patches-extracted/unapply.sh .
```

The candidates apply in file-name order with plain `git apply` and reverse cleanly. As with
the main stack, re-running `apply.sh` on an applied tree is not supported.

| # | Source | What | Why it may help the A770 | Off switch |
|---|--------|------|--------------------------|------------|
| C10 | PR 31251 (closed unmerged, no successor) | Reordered (SoA) IQ4_XS layout and MMVQ, multi-column and fused GLU | IQ4_XS GEMV was bound by table lookups: 261 -> 380 GB/s in the author's probe. Runs on every Intel GPU | none (reorder applies to IQ4_XS weights) |
| C20 | PR 31428 (draft, on top of carried PR 30156) | ESIMD grouped MoE GEMM that dequantizes straight into the DPAS operands | Base for C21. On its own it only replaces the sub-group-16 shape, so no change on DG2 | `GGML_SYCL_MMID_ESIMD=0` |
| C21 | cwriter `36b5038` (not in a PR) | The C20 kernel for the Xe-HPG 8x8x16 sub-group-8 DPAS shape: 8-wide DPAS, 16 rows, 1D loads | The only MoE GEMM change written for Arc A-series. 1.6-2.6x per MoE layer for C20 on Xe2; the DG2 variant was only run with software DPAS | `GGML_SYCL_MMID_ESIMD=0` |
| C30 | PR 30226 (closed unmerged) | q4_K MMVQ row pairing up to 8 columns | Speculative-decoding verify; Battlemage-gated, enable on DG2 with C90 bit 16 | C90 bit 16 off |
| C40 | ef777/a770-qwen-xe1 `da7f577`, plus a local DG2 build fix | GQA flash-decode kernel, 1-8 tokens, q8_0/f16 KV, D=64/128/256 | VEC rereads K/V per query head; this reads each KV head once for all its query heads. Author on A770: 24k n1 520 -> 410 us, 12k n4 762 -> 404 us | `GGML_SYCL_FA_GQA_DEC=0` |
| C50 | Frostbite1536/A770 `0a31dc7`, p11c F | TILE FA flushes its FP16 V accumulator to FP32 once per KV tile | Probable long-context accuracy fix: NMSE up to 2.3 at 16k keys per block, CPU OpenCL device only | none |
| C51 | Frostbite1536/A770, patch 0005 | 16 biased-V FA test cases with an FP32 CPU reference | Tests C50; CPU FA changes only under the test reference flag | none |
| C60 | local port of upstream PR 31125 | Upstream's DPAS flash attention for q4_0 / q8_0 KV (1-31 query tokens), with the DPAS width as a template parameter: sub-group 16 as upstream, sub-group 8 for Xe-HPG, picked per device at run time | Upstream measured tg128 at 16K 31.3 -> 54.2 t/s on a B580 (q4_0 KV), but gates it to Xe2. On DG2 it replaces VEC for quantized-KV decode and small batches, and it is the only candidate that covers q4_0 KV. With C40 applied, C40 still takes q8_0 decode at 1-8 tokens and D 64/128/256 | `GGML_SYCL_FA_DPAS=0` |
| C62 | PR 31574 (draft) | Lifts upstream's Alchemist (DG2) block on oneDNN fused SDPA at head_dim 64 | Head-dim-64 models without attention sinks (Llama-3.2-1B, Qwen2.5-0.5B) get oneDNN's fused attention for prompts (gpt-oss has sinks, which this path declines). The author on Windows: Qwen2.5-0.5B pp2048 +49%, pp4096 +106%, `FLASH_ATTN_EXT hsk=64` 1818/1818. The block guarded against wrong results that the PR traces to a Linux IGC regression fixed in IGC 2.40, so older IGC still computes garbage here; C63 keeps the block on Linux until you opt in. The PR's validation line reads "Arc A770 / dg2_g11", but `dg2_g11` is the ACM-G11 die of the A380/A310 and an A770 is ACM-G10, so the card name or the label is wrong; only your own run counts | `GGML_SYCL_FA_ONEDNN=0` (all oneDNN FA); with C63, off on Linux unless `GGML_SYCL_FA_ONEDNN_DG2_HD64=1` |
| C63 | local, on C62 | Keeps C62's head_dim 64 path behind `GGML_SYCL_FA_ONEDNN_DG2_HD64` outside Windows | oneDNN#5510 reproduced the wrong results on an A770 and an A750 under Ubuntu 24.04 with Level Zero 1.15.38646 (NMSE 0.17, head_dim 64 the only failing size); the fix is in IGC 2.40, which llama.cpp cannot detect. Without C63, applying the candidates on such a driver silently breaks attention for every head-dim-64 model that takes the oneDNN path | `GGML_SYCL_FA_ONEDNN_DG2_HD64=1` turns C62's path on |
| C70 | PR 31379 (draft), re-anchored on carried PR 31355 and local 0015 | The gated delta net reads its recurrent state through the GET_ROWS ids instead of from a gathered copy | Qwen3.5 / Qwen3-Next decode: one state copy fewer per delta-net layer per token. Single-sequence batches only. arthw on an A770 (upstream + 31379, Qwen3.5-0.8B Q4_K_M): tg128 +2.6% with FA off, +3.3% with FA on, pp512 unchanged | `GGML_SYCL_GDN_STATE_GATHER=0` |
| C75 | PR 31313 (draft, "WIP K1"), without its fused-gemm.cpp dedup | Vector and joint_matrix kernels for the lightning indexer | DeepSeek-V3.2-style sparse attention, decode (vector kernel) and batches of 8+ (XMX, sub-group-8 8x8x16 on DG2). Off by default | `GGML_SYCL_LIGHTNING_INDEXER=1` turns it on |
| C76 | local, on C75 | 256 registers for C75's vector kernel on ACM-G10/G11/G12 | At 128 that kernel spills 1.6-1.8 KB per thread on DG2; at 256 it does not | none (only with C75 on) |
| C80 | local port of kainlan/llama.cpp-intel-optimizations `woq_gemm_q8_0` | Q8_0 prompt batches through oneDNN's weight-only-quantized matmul: the int8 weights in the reorder layout and their block scales go to oneDNN as stored (grouped scales, f16 fpmath on integer weights) instead of a full f16 copy of the weight per batch | Skips writing and rereading 32 MB of f16 weights per 4096 x 4096 matmul in prompt processing. The fork: Mistral 7B Q8_0 pp512 2304 -> 4866 t/s on a B70, against its own MMQ. Whether Xe-HPG has a real oneDNN kernel for it is unknown: `tools/woq-q8_0-probe` checks that, its accuracy and its speed in a minute | `GGML_SYCL_WOQ_Q8_0=1` turns it on (2 also ahead of the fused XMX GEMM) |
| C90 | local | `GGML_SYCL_DG2_BMG_TUNING` bit mask that opens five Battlemage-only kernel choices to DG2 | Each choice is between kernels already built for every device, so only speed can change | `GGML_SYCL_DG2_BMG_TUNING=0` (default) |

`GGML_SYCL_DG2_BMG_TUNING` bits:

| Bit | Gate | Upstream default on DG2 |
|-----|------|-------------------------|
| 1 | quantized-KV decode (1-2 tokens) on FA TILE | VEC |
| 2 | FA VEC with 256-thread work-groups | 128 |
| 4 | q5_K multi-column MMVQ reuse and fused gate/up | per-column kernel |
| 8 | D=256 single-query decode on FA VEC (PR 30606) | dispatcher default |
| 16 | q4_K MMVQ row pairing up to 8 columns (needs C30) | pairing up to 4 |

Bits 1 and 8 choose between TILE and VEC, which C60 and C40 run ahead of for q4_0 / q8_0 KV:
measure them with `GGML_SYCL_FA_DPAS=0 GGML_SYCL_FA_GQA_DEC=0`.

## Hardware gates are different

`sycl-patches/check_arch_gates.sh` lists every architecture gate and tags the ones in files
that use DPAS, joint_matrix, ESIMD or 2D block loads as `HW`. Do not add DG2 to an `HW` gate.
The A770 runs DPAS at execution size 8; Xe2 runs it at 16. On `acm-g10` the sub-group-16
DPAS builtin compiles without an error or warning and produces no DPAS instruction
(Frostbite1536/A770 compile probes), so a widened `HW` gate can build and then compute
garbage. Those paths need a sub-group-8 port: C21 is that port for the MoE GEMM and C60 for
the DPAS flash attention. The remaining `HW` gate, upstream's XMX multi-column MMVQ, is not
worth porting: ef777 measured it on the A770 slower than dequantize + GEMM at 9-80 columns,
and their own 8-row XMX matvec slower than dp4a at 1-8. ef777 also found an XMX
flash-attention kernel of their own correct but slower (SLM and barrier stalls). That is a
different kernel from C60, so C60 has to win its benchmark like every other candidate.

## Test plan on the A770

Correctness first, with the same build as for the main stack plus the candidates:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p iq4_xs
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DG2_BMG_TUNING=31 ./build_sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_DG2_BMG_TUNING=31 ./build_sycl/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p q4_K
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_FA_GQA_DEC=0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
```

With C60 the log shows `fattn_dpas_device_sg: device 0: DPAS flash attention at sub-group 8
(Xe-HPG)` on the first quantized-KV FA call. The last command routes q8_0 decode to C60
instead of C40.

Then speed, one switch at a time against the same build (`-d` sweeps the KV depth that
the FA candidates target; use a GQA model with q8_0 KV for C40):

```bash
for t in 0 1 2 4 8 16; do GGML_SYCL_FA_DPAS=0 GGML_SYCL_FA_GQA_DEC=0 GGML_SYCL_DG2_BMG_TUNING=$t ./build_sycl/bin/llama-bench -m MODEL -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384 -r 3; done
GGML_SYCL_FA_GQA_DEC=0 ./build_sycl/bin/llama-bench -m MODEL -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384 -r 3
GGML_SYCL_MMID_ESIMD=0 ./build_sycl/bin/llama-bench -m MOE_MODEL -p 512 -n 128 -r 3
```

C60 against the kernels it replaces, for each KV type (`-p 16 -ub 16` is a 16-token batch,
the shape of a speculative-decoding verify):

```bash
for kv in q4_0 q8_0; do for dpas in 1 0; do GGML_SYCL_FA_GQA_DEC=0 GGML_SYCL_FA_DPAS=$dpas ./build_sycl/bin/llama-bench -m MODEL -fa 1 -ctk $kv -ctv $kv -p 16 -ub 16 -n 128 -d 0,4096,16384 -r 3; done; done
GGML_SYCL_FA_GQA_DEC=1 ./build_sycl/bin/llama-bench -m MODEL -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 0,4096,16384 -r 3
```

The last line is C40 on q8_0 decode, to compare with the `dpas=1 q8_0` rows.

C62, C70 and C75 / C76 (added 2026-11-03), correctness on the card:

```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_FA_ONEDNN_DG2_HD64=1 ./build_sycl/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT -p 'hsk=64'
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build_sycl/bin/test-backend-ops -b SYCL0 -o GATED_DELTA_NET_GATHER
ONEAPI_DEVICE_SELECTOR=level_zero:0 GGML_SYCL_LIGHTNING_INDEXER=1 ./build_sycl/bin/test-backend-ops -b SYCL0 -o LIGHTNING_INDEXER
```

The first command sets `GGML_SYCL_FA_ONEDNN_DG2_HD64=1` because C63 keeps C62's path off on Linux.
`ONEDNN_VERBOSE=1` shows `sdp_primitive_kernel_t` when a case takes the oneDNN path. If it fails,
your IGC predates 2.40: leave the switch at 0. `sycl-ls` prints the Level Zero driver
version in brackets (oneDNN#5510 failed on 1.15.38646 and passed on 1.14.37435), and
`dpkg -l | grep -i igc` shows the IGC package on Debian / Ubuntu. Then speed: C62 on prompts of a
head-dim-64 model, C70 on decode of a Qwen3.5 or Qwen3-Next model, C75 on its op:

```bash
for o in 0 1; do GGML_SYCL_FA_ONEDNN_DG2_HD64=$o ./build_sycl/bin/llama-bench -m HD64_MODEL -fa 1 -p 2048,4096 -n 0 -r 3; done
for g in 0 1; do GGML_SYCL_GDN_STATE_GATHER=$g ./build_sycl/bin/llama-bench -m QWEN35_MODEL -p 0 -n 128 -r 3; done
for li in 0 1; do GGML_SYCL_LIGHTNING_INDEXER=$li ./build_sycl/bin/test-backend-ops perf -b SYCL0 -o LIGHTNING_INDEXER; done
```

Promote a candidate when its tests pass and it wins at the depths you use. A tuning bit that
wins can become a plain DG2 entry in that gate.

C80 (added 2026-11-04): run the probe first. It needs oneAPI only, no backend build, and if it prints
`DECLINED`, `impl=ref...` or `FAIL`, drop C80:

```bash
icpx -fsycl -O2 sycl-patches-candidates/tools/woq-q8_0-probe/probe.cpp -o woq-probe -ldnnl
```
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./woq-probe
```

Then `GGML_SYCL_WOQ_Q8_0=2 test-backend-ops -o MUL_MAT -p type_a=q8_0`, and speed on a Q8_0 model with
`GGML_SYCL_WOQ_Q8_0=0` and `=1` at `-p 512,2048 -n 0`. The first call logs the oneDNN implementation.

`../sycl-patches/ab/` turns every command above into interleaved A/B runs with verdicts, together with the
local patches and the carried PRs (`plan.json` has a row for each switch here).

## Draft PR check (2026-11-03)

The four draft PRs behind C20, C62, C70 and C75 are still open drafts and their heads have not moved
since they were captured: 31574 `ef2a0391`, 31379 `7f72c2a5`, 31313 `5f1ba52e` (merge of master over
`11e34732`), 31428 `acfa9400`. C21 is the head of cwriter's `grouped-xmx-moe-v2-with-8x8` branch
(`36b50381`). Each PR's diff against its merge base was compared line by line with the candidate
file (plus the carried PR it sits on):

| PR | Covered by | Lines of the PR not in the candidate |
|----|------------|--------------------------------------|
| 31574 | C62 | none |
| 31379 | PR-31355 + C70 | none (`graph_optimize` and the whole-graph perf mode come from PR-31355) |
| 31313 | C75 | the `fused-gemm.cpp` dedup and its two doc lines, as the C75 header says |
| 31428 | PR-30156 + C20 | none |

Review comments on those PRs, and what was done here:

| PR | Comment | Here |
|----|---------|------|
| 31428 | arthw: upstream drops `-ze-intel-greater-than-4GB-buffer-required` from AOT builds for a known issue; can it be used with AOT, and where is `-Xs`? | C20 is right as is. With oneAPI 2026.2 for `acm-g10`, the bare `-Xs` form fails the AOT link (`Invalid option`), which is why PR 22147 skipped the flag; inside `-options`, as C20 passes it, it builds. And it matters: without it `getrows.cpp` compiles to 32-bit offsets from bindless surfaces (673 `a32` accesses), with it to 64-bit addresses (716 `a64`). So an upstream AOT build addresses at most 4 GB from the start of each buffer; a JIT build gets the flag through `-Xs`. |
| 31428 | arthw: keep an informational CMake message near `GGML_SYCL_WARP_SIZE=16` (most likely the "Skipping ..." line C20 removes, which C20 makes untrue); rename `_ggml_sycl_aot_backend`, `gg_*` / `GG_*`, `ggml_sycl_mul_mat_id_mmvq_ok`, `..._grouped_shape_ok`, `node_is_device_local` | Not changed: build messages and names only, no effect on the A770. cwriter agreed to the first rename; the next refresh picks up whatever lands. |
| 31428 | arthw: no gain on an A770 | Expected: C20 alone runs only the sub-group-16 shape. cwriter answered with the 8x8 branch that C21 carries. |
| 31379 | arthw: say in the docs which models gain; move the matcher out of `ggml-sycl.cpp` | Both done in the carried commit `7f72c2a5` (the row names Qwen3.5 / Qwen3-Next; the matcher is in `gated_delta_net.cpp`). arthw's A770 numbers are in the C70 row. |
| 31313 | arthw: no gain on an A770 | The 27B model used has no lightning indexer, so nothing was measured; cwriter's perf command is in the test plan. |
| 31574 | no review; a bot flags the PR template | The issue the PR cites, oneDNN#5510, shows the A770 itself computing head_dim 64 wrongly under Ubuntu 24.04 with Level Zero 1.15.38646: C63. |

## Build status of this group (2026-11-01)

Compiled with oneAPI 2026.2 (conda-forge) on a box without an Intel GPU. Each
risky translation unit was also built for `acm-g10` with `spir64_gen` (the SPIR-V
the A770 compiles at JIT time) and its DG2 assembly read from IGC dumps:

- C20/C21 `fused-gemm-esimd.cpp`: builds; every DPAS is `dpas.8x8 (8|M0)`. The
  default 8-wide path is spill-free at 256 GRF, which the kernel requests. A
  `GGML_SYCL_DEVICE_ARCH=acm-g10` build gets 128 GRF and spills 3.7-9 KB per
  kernel, so measure with a JIT build. (2026-11-03: oneAPI 2026.2's AOT path drops
  `grf_size` for every ESIMD kernel and keeps it for SPMD ones; a minimal ESIMD
  kernel shows the same. `ZZ-LOCAL-0017`'s `GGML_SYCL_XMX_GRF256=1` switches the
  joint_matrix tile C21 falls back to; the ESIMD remap sees the same combination
  either way.)
- C10/C30 `mmvq.cpp`: builds. Upstream's Xe2-only XMX MMVQ kernels in the same
  file also compile for DG2, as 8-wide DPAS with spills; they stay gated off.
- Since local patch `ZZ-LOCAL-0010` (2026-11-02), the stack's own `joint_matrix` grouped
  GEMM no longer spills on DG2 at 128 GRF (8 rows a sub-group, k halves not unrolled), so
  `GGML_SYCL_MMID_ESIMD=0` now compares C21 against a spill-free kernel at full occupancy
  rather than the spilling one C21 was written to replace. The same patch stops the gate + up
  MMVQ pairing rows on DG2, where every paired variant spills: that covers C10's iq4_xs pairing
  (3..4 columns, 0.7-1 KB) and C30's 5..8-column q4_K pairing under C90 bit 16 (2.3-3.6 KB),
  and C90 bit 4 now gives q5_K weight reuse without the paired gate + up (1.1-3.5 KB). C30's
  plain-MMVQ pairing does not spill on DG2 and is unaffected.
- Since local patch `ZZ-LOCAL-0012` (2026-11-03), MUL_MAT_ID batches of 2 tokens up to a per-type cap
  (4 for the k-quants, 8 for q4_0 / q8_0 / mxfp4) take the MoE MMVQ instead of the grouped GEMM, so C20 / C21
  only see wider batches. `GGML_SYCL_MMID_MMVQ_MAX_TOKENS=1` restores the old split when comparing them.
- C40 `fattn-gqa-dec.cpp`: the original failed. The q8_0 D=256 SLM ESIMD kernel's
  in-kernel `mask` null test made IGC 2.43 emit a 64-bit shift that DG2 cannot
  execute, and that failed every ESIMD kernel in the file. C40 now tests `mask`
  on the host; all 37 kernels build. At the 128 GRF of an AOT build, the D=256
  SPMD and ESIMD variants spill (0.7-12 KB). Since 2026-11-03 the D=256 SPMD
  kernels ask for `grf_size<256>` on Intel GPUs with XMX and no longer spill
  (AOT honours the request for SPMD kernels); the ESIMD ones still lose it in
  AOT (see below), so measure C40's ESIMD path with a JIT build.
- C50 TILE FA instance: builds.
- C60 `fattn-dpas.cpp`: builds for `acm-g10` (42 kernels, 560 `dpas.8x8 (8|M0)`: 473 DPAS and
  87 workaround dummies, none spill) and `bmg-g21` (the same 604 DPAS as upstream, instruction
  counts within 1%, none spill). The sub-group-8 kernels that spilled at 128 GRF (32 rows from
  D=128, staged D=256 decode) always use 256. In a JIT build each device image holds one DPAS width.
  `tools/fattn-dpas-emu/run.sh` runs the kernels on the CPU OpenCL device with emulated
  DPAS: 112 cases pass at both widths (NMSE vs FP64 <= 4.8e-5, sub-group 8 vs 16 <=
  1.6e-11), and it fails on planted sub-group-8 layout bugs. Rerun it when C60 is
  re-anchored.
- C60's Q.K guard (2026-11-09). The 782 DPAS above were 490 real and 292 dummies from IGC's
  `Wa_18023229625`: a `t < RT` test around each Q.K DPAS put every one in its own basic block, and
  IGC adds a dummy DPAS before the first DPAS of a block (8 + 8 per job in the D=128 32-row kernels).
  At sub-group 8 the test is now dropped where it always holds or only discards scores, except in the
  128-register two-tile kernels (head size 96 spills without it): 473 + 87, no spills, bmg-g21
  instruction streams identical in all 83 kernels, emulator 112/112.
  `-DGGML_SYCL_FA_DPAS_QK_GUARD=1` restores the test for the `C60-qk-guard*` A/B rows.
  `../sycl-patches/aot/isa_check.py` gives these counts from any configured build.

Added 2026-11-03, `acm-g10` AOT with the real device flags (`-fsycl-device-code-split=per_kernel`):

- C75 `lightning-indexer.cpp`: builds, 198 DPAS instructions in the XMX kernels, which do not spill. The
  vector kernel spilled 1632-1760 B per thread at 128 registers; with C76 it gets 256 on ACM and does
  not spill (`numGRF=256`).
- C70 `gated_delta_net.cpp`: 20 kernels, none spill. C62 changes only a host-side gate.
- On the CPU OpenCL device (`../sycl-patches-extracted/tools/cpu-device-test/`, every candidate but
  C20 / C21): `GATED_DELTA_NET_GATHER` 19/19 with C70's gather on and off, and C75's vector kernel
  70/70 on f32 / f16 keys. C75's XMX kernels and C62's oneDNN path cannot run there.

Added 2026-11-04:

- C80 `ggml-sycl.cpp` / `gemm.hpp`: builds with `icpx` 2026.2 (JIT) on the full stack and candidates; C80 applies
  before C90 with plain `git apply`. `tools/woq-q8_0-probe` runs its oneDNN descriptors (int8 weights
  `{k, m}` with strides `{1, k}`, f16 scales `{k / 32, m}`, mask 3, groups `{32, 1}`) on the CPU engine against
  a double-precision reference: NMSE 1.6e-13 with f32 activations. The f16-activation form, the one C80
  uses, is declined by oneDNN on a CPU without AVX512-FP16, so its kernel and accuracy are A770-only checks.

What was not checked: any number on real hardware. Apart from C60's emulation,
no kernel here ran: the backend does not run on the CPU OpenCL device (see
`SYCL_DEV_LOG.md`), and ESIMD, DPAS and joint_matrix need the GPU.
