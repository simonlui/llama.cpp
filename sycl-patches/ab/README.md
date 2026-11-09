# A/B plan for the Arc A770

Compile tests showed every patch in `sycl-patches/`, `sycl-patches-extracted/` and
`sycl-patches-candidates/` builds, and the CPU OpenCL device showed most of their SPMD kernels are
right. Neither says whether a patch makes the A770 faster. This directory holds the runs that decide
that, and the rules that turn their results into keep, promote or drop.

Three gates, in this order. A patch has to pass each one before the next counts:

1. **Correctness on the card.** `test-backend-ops` with the patch's path in use. A failure drops the patch
   or blocks it until fixed, however fast it is.
2. **Speed.** `ab.py run`: the patch off (arm A) against on (arm B) in the same binary, interleaved.
3. **Accuracy**, for switches that change the arithmetic. `ab.py kld`: both arms against CPU-backend
   logits.

| File | What |
|------|------|
| `plan.json` | 63 experiments: arms, model, `llama-bench` arguments, tier, and what each decides |
| `ab.py` | Runs them (`run`, `kld`), prints verdicts (`report`), lists what is runnable (`list`), turns `GGML_SYCL_LOCAL_OFF` names into the mask (`mask`) |
| `guard.sh` | Records the driver stack, oneDNN, GPU temperature and other GPU clients with the results (`--stamp`). Before each run it waits for a card below 60 C with no other process above 5% of an engine, then for an idle, unthrottled GPU (`--wait`). Also `--temp`, `--clients`, and GPU hangs in the kernel log (`--faults`). Reads i915 or xe sysfs |
| `drm_clients.py` | Other processes' engine use of the card from DRM fdinfo (i915 and xe), for `guard.sh` |

## How a speed verdict is made

Each experiment runs `rounds` rounds (4 by default, 6 for small-batch rows). The arm order alternates
(A B, B A, A B, B A), so clock, temperature and background drift hit both arms alike. Each invocation is
`llama-bench -r 3`. A row with a `-d` list runs one process per depth, A and B for the same depth back to
back, each after the cooldown (`"chunk_depths": false` in a row keeps one process). A round's gain is
B / A - 1 for each test (`pp512`, `tg128@d4096`, ...), with 1% as the smallest gain that counts
(`--min-gain`). The pairing is what makes 1% usable on this card: an earlier serial A770 run drifted ±3%
between steps (`SYCL_DEV_LOG.md`, "Per-step bench noise"), while repeats inside one `llama-bench` call
varied by 0.1-0.2% (pp512 1377.5 ± 2.4, tg128 29.63 ± 0.03 t/s).

A single process can still land in a slow state. An A770 flash-attention study measured sustained slow
states of 18-20% at the same reported 2000 MHz, which no sysfs file showed (84.0 against 99.1 ms on one
head size-128 shape), and about 10% from another job on the render engine. So the report gives two
estimates per test:

- **mean** of the round gains with its Student t 95% interval;
- **Hodges-Lehmann** (HL): the median of all pairwise round averages, with its exact Wilcoxon interval.
  That interval reaches 95% from 6 rounds (it reads 88% at 4: the coverage is printed next to it).

**far rounds** lists rounds more than max(2%, 3 scaled MADs) from the median round. They are kept, never
dropped: the answer to one is more rounds.

| Verdict | Condition | Action |
|---------|-----------|--------|
| WIN | 3-5 rounds: t interval above 0, mean and HL at least +1%, no far round. 6+ rounds: HL interval above 0 and HL at least +1% | keep the patch (for an opt-in: make it the default) |
| LOSS | the same, below 0 and at most -1% | drop it (for an opt-in: leave it off) |
| no effect | 3-5 rounds: t interval inside -1%..+1%, HL estimate inside too, no far round. 6+ rounds: HL interval inside -1%..+1% | drop it if it is a local perf patch or a candidate: it costs a refresh every time upstream moves |
| inconclusive | anything else, including a WIN / LOSS / no effect that a far round or the HL estimate contradicts | append rounds: `ab.py run --only NAME --rounds 4 --out SAME.jsonl` (rounds are numbered on, the order keeps alternating). Still inconclusive within ±2% at 12 rounds: treat it as no effect |
| more rounds | fewer than 3 paired rounds | a run failed; see the `.log` next to the results |
| A/A OK / A/A NOISY | `aa-*` rows: same binary and settings on both sides | NOISY: stop and find the cause before reading any other row |

With `--guard`, `ab.py` also stops when the card reaches 80 C after a run, or when `guard.sh` cannot get it
below 60 C and free of other clients within 15 minutes. Rerunning the same command appends the missing
rounds. `temp_start`, `temp_abort`, `busy_max` and `cool_wait` in the plan's `defaults` set the limits.

A patch is kept on a WIN in at least one row it targets and no LOSS in any of its rows. An accuracy
loss (below) overrides a WIN.

## What decides each patch

**Correctness only: no speed claim, keep while `test-backend-ops` passes.** Fork patches 0002 (dpct
checks), 0004 (GROUP_NORM gate), 0007 (barrier fence), 0008 (native shuffles), 0014 (GET_ROWS epsilon gate),
0015 (CONCAT gate), 0017 (ternary gate); PRs 29506 (separate compiler build), 29672 (PTQ1_0), 30206
(YaRN `truncate=false`), 30857 (f16 SWIGLU_OAI), 31109, 31138, 31460, 31465; local 0005 (ternary kernels),
0006 (F8_E4M3), 0009 (Q2_0 GET_ROWS); candidates C50 and C51 (C51's tests decide C50). The gap fills
0005, 0006 and 0009 run types that otherwise fall back to the CPU. They only earn their keep if you
run such a model. Drop them at a refresh where they conflict if you don't.

**Speed, runtime switch: decided by `plan.json` in one build.**

| Patch | Rows | Switch (A is off) |
|-------|------|-------------------|
| local 0003 | `0003-iq-ncols` | `GGML_SYCL_LOCAL_OFF` bit `iq_ncols` |
| local 0004, 0007, 0008 | `0004-moe-reorder-*`, `0008-moe-reorder-iq`, `0007-iq4_nl-reorder` | `moe_reorder`, `iq4_nl_reorder` |
| local 0010 | `0010-glu-one-row` | `dg2_glu_one_row` |
| local 0011 | `0011-rope-set-rows`, `-norm-rope`, `-moe-sum*`, `-multi-add`, `-topk-moe-ext` | one bit each |
| local 0012 | `0012-moe-glu-fuse*`, `0012-moe-tokens*` | `moe_mmvq_fuse`; `GGML_SYCL_MMID_MMVQ_MAX_TOKENS=1` |
| local 0013, 0016 | `0013-mul-mat-add-*`, `0016-mmvq-add` | `mul_mat_add`, `mmvq_add` |
| local 0014 | `0014-act-reuse-*` | `act_reuse` |
| local 0015 | `0015-softcap`, `-relu2`, `-ssm-cache`; `0011-topk-moe-ext` | `model_fusions`, `topk_moe_ext` |
| local 0017 (opt-in) | `0017-grf256-*` | `GGML_SYCL_XMX_GRF256` 0 / 1 |
| C10 | `C10-iq4_xs-reorder` | `iq4_xs_reorder` |
| C20 + C21 | `C20-C21-esimd-moe-*` | `GGML_SYCL_MMID_ESIMD=0` |
| C30 | `C30-C90-bit16` | `GGML_SYCL_DG2_BMG_TUNING` 0 / 16 |
| C40, C60 | `C40-vs-vec`, `C40-vs-C60`, `C60-*` | `GGML_SYCL_FA_GQA_DEC`, `GGML_SYCL_FA_DPAS` |
| C62 + C63 | `C62-onednn-hd64` | `GGML_SYCL_FA_ONEDNN_DG2_HD64` 0 / 1 |
| C70 | `C70-gdn-gather` | `GGML_SYCL_GDN_STATE_GATHER` |
| C80 (opt-in) | `C80-woq-q8_0`, `C80-woq-q8_0-small` | `GGML_SYCL_WOQ_Q8_0` 0 / 1 / 2 |
| C90 | `C90-bit*` | `GGML_SYCL_DG2_BMG_TUNING` bits 1, 2, 4, 8 |
| PR 30156 | `PR-30156-mmid-sched` | `GGML_SYCL_MMID_SCHED` 0 / 1 |
| PRs 31536, 31542 | `PR-31536-*` | `GGML_SYCL_ENABLE_MKL_FA=0` |

`GGML_SYCL_LOCAL_OFF` comes from `ZZ-LOCAL-0018`. Each bit restores the path the backend took before
that one patch, so the local patches can be measured one at a time without regenerating the stack.
`ab.py mask moe_sum,act_reuse` prints the number for a manual run.

**Speed, second build.** The whole stack against upstream (`stack-vs-upstream-*`), JIT against
`acm-g10` AOT (`jit-vs-aot*`), extracted 0016 (`0016x-mmq-xmx`, the `GGML_SYCL_MMQ_XMX=ON` build), C60's
Q.K guard (`C60-qk-guard*`: the build with `-DGGML_SYCL_FA_DPAS_QK_GUARD=1` is A), and oneDNN 3.14 or
later against the oneDNN `AB_BUILD` loads (`onednn-314-*`, only when that one is older).
0016 is 4,300 lines that an earlier A770 run showed no effect from on Q4_0, because the A770 rarely
reaches MMQ. No effect on q4_K as well means drop it. Extracted 0015 (softmax `reduce_over_group`) is a
cleanup. Compare `test-backend-ops perf -o SOFT_MAX` between `AB_BUILD` and `AB_BUILD_UPSTREAM`, and
drop it if it is slower.

**Not measurable on a 16 GB A770: drop unless you need the model.** C75 + C76 (lightning indexer: no
DeepSeek-V3.2-class model fits; `GGML_SYCL_LIGHTNING_INDEXER=0/1 test-backend-ops perf -o LIGHTNING_INDEXER`
is the only number available). 0015's DSV4_HC_POST gate (qwen4exp, 512 experts) and snake activation (TTS
decoders, which `llama-bench` does not run). The expert-group half of `topk_moe_ext` needs
`expert_group_count > 1` in the GGUF; I have not found a model under 16 GB that sets it.

**No-op on the A770.** PR 31285 (its check runs on BMG-G21/G31 only), PR 30606 (Battlemage route;
on DG2 only through C90 bit 8), C20 without C21 (sub-group-16 shape only), and PR 31355, which no longer
matches delta-net graphs since upstream `b74298c1`. Carry PR 31355 only until the PR follows upstream.
PRs 31126 (cached device id) and 31208 (graph check) are host-side. Their effect is inside the noise of
anything here, so keep them until upstream decides.

The Vulkan patches are outside this plan.

## Setup

Builds (the compile line in `sycl-patches/README.md`, JIT, `-DLLAMA_BUILD_TESTS=ON` for the correctness
step):

| Variable | Build | Used by |
|----------|-------|---------|
| `AB_BUILD` | full stack + candidates | everything (arm B, and arm A unless the row names another build) |
| `AB_BUILD_UPSTREAM` | same flags, no patches (a `git worktree` of `master`) | `stack-vs-upstream-*`, the SOFT_MAX check |
| `AB_BUILD_AOT` | `AB_BUILD` plus `-DGGML_SYCL_DEVICE_ARCH=acm-g10` | `jit-vs-aot*` |
| `AB_BUILD_MMQ_XMX` | `AB_BUILD` plus `-DGGML_SYCL_MMQ_XMX=ON` | `0016x-mmq-xmx` |
| `AB_BUILD_QK_GUARD` | `AB_BUILD` with `-DGGML_SYCL_FA_DPAS_QK_GUARD=1` added to `CMAKE_CXX_FLAGS` | `C60-qk-guard*` (arm A) |
| `AB_BUILD_ONEDNN314` | `AB_BUILD` configured with `-DDNNL_DIR=<oneDNN 3.14+>/lib/cmake/dnnl`; set `AB_ONEDNN314_LIB` to that oneDNN's `lib` | `onednn-314-*` (arm B) |

Models. Set the variables for the ones you have; `ab.py list` shows what runs and what is skipped:

| Variable | What it has to exercise | For example |
|----------|-------------------------|-------------|
| `AB_Q4_0` | dense q4_0 | Llama-3.1-8B Q4_0, or the Gemma-4-E2B Q4_0 from the earlier A770 numbers |
| `AB_Q4_K` | dense q4_K / q6_K with GQA, head_dim 128 | Llama-3.1-8B-Instruct Q4_K_M |
| `AB_Q5_K` | dense q5_K | Llama-3.1-8B Q5_K_M |
| `AB_Q8_0` | dense q8_0 | Llama-3.1-8B Q8_0 (8.5 GB) |
| `AB_IQ4_XS`, `AB_IQ4_NL` | dense iq4_xs, iq4_nl | Llama-3.1-8B IQ4_XS, IQ4_NL |
| `AB_IQ2` | dense iq2 / iq1 | Llama-3.1-8B IQ2_XXS or IQ2_M |
| `AB_QK_NORM` | RMS norm on Q and K before RoPE | Qwen3-8B Q4_K_M |
| `AB_HD64` | head_dim 64, no attention sinks | Llama-3.2-1B-Instruct Q8_0 |
| `AB_HD256`, `AB_GEMMA2` | head_dim 256; the logit softcap | Gemma-2-2B (serves both) |
| `AB_MOE_Q4_0` | MoE with q4_0 experts | OLMoE-1B-7B Q4_0 |
| `AB_MOE_K` | MoE with q2_K..q6_K experts | OLMoE-1B-7B Q4_K_M, or Qwen3-30B-A3B Q3_K_S if it fits |
| `AB_MOE_IQ` | MoE with iq3 / iq4_nl experts | Qwen3-30B-A3B in an IQ3 quant (about 12 GB) |
| `AB_GPT_OSS` | MXFP4 experts, ADD_ID biases, attention sinks | gpt-oss-20b MXFP4 |
| `AB_MOE_BIAS` | sigmoid router with an expert bias (DeepSeek-V3 style) | Moonlight-16B-A3B Q4_0 |
| `AB_RELU2` | squared ReLU | Nemotron-Mini-4B |
| `AB_MAMBA` | SSM_SCAN | Mamba-2 2.7B, or a hybrid such as Falcon-H1 |
| `AB_QWEN35` | gated delta net | Qwen3.5-0.8B Q4_K_M (arthw's A770 test) |
| `AB_TEXT` | text for `ab.py kld` | `wiki.test.raw` from `scripts/get-wikitext-2.sh` |

## Run order

**1. Stamp and correctness.** Keep the stamp with every result. The fork's B70 decode moved 18% with a
compute-runtime update and nothing else. The stamp also lists the GPU temperature, the other processes
using the card (run it as root to see a display server's) and the oneDNN version `AB_BUILD` loads.

```bash
bash sycl-patches/ab/guard.sh --stamp | tee ab-stamp.txt
```
```bash
export ONEAPI_DEVICE_SELECTOR=level_zero:0 SYCL_CACHE_PERSISTENT=1
```
```bash
$AB_BUILD/bin/test-backend-ops -b SYCL0 2>&1 | tee tbo-default.log
```

The default run covers every default-on patch. If it stops at an assertion in the CPU reference, run it
op by op (`-o MUL_MAT`, `-o MUL_MAT_ID`, ...). Then the paths that are off by default, or that a default
hides:

```bash
GGML_SYCL_XMX_GRF256=1 $AB_BUILD/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID
GGML_SYCL_XMX_GRF256=1 $AB_BUILD/bin/test-backend-ops -b SYCL0 -o MUL_MAT
GGML_SYCL_WOQ_Q8_0=2 $AB_BUILD/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p type_a=q8_0
GGML_SYCL_MMID_ESIMD=0 $AB_BUILD/bin/test-backend-ops -b SYCL0 -o MUL_MAT_ID
GGML_SYCL_FA_GQA_DEC=0 $AB_BUILD/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
GGML_SYCL_DG2_BMG_TUNING=31 $AB_BUILD/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT
GGML_SYCL_DG2_BMG_TUNING=31 $AB_BUILD/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p q4_K
GGML_SYCL_DG2_BMG_TUNING=31 $AB_BUILD/bin/test-backend-ops -b SYCL0 -o MUL_MAT -p q5_K
GGML_SYCL_FA_ONEDNN_DG2_HD64=1 $AB_BUILD/bin/test-backend-ops -b SYCL0 -o FLASH_ATTN_EXT -p hsk=64
GGML_SYCL_LIGHTNING_INDEXER=1 $AB_BUILD/bin/test-backend-ops -b SYCL0 -o LIGHTNING_INDEXER
GGML_SYCL_ENABLE_GRAPH=1 $AB_BUILD/bin/test-llama-archs -b SYCL0 -a '^(llama|qwen35|qwen3next|gpt-oss|gemma3)$' -s 42
```

Each `ab.py` row whose path failed here is out: do not run it.

The head_dim 64 line decides C62. oneDNN#5510's wrong results came from a Linux IGC regression fixed in
IGC 2.40. Another A770 project reports PyTorch's fused head_dim-64 SDPA correct on Linux with the `xe`
driver, compute-runtime 26.35.39758 and IGC 2.41.5. If the line passes on your stack, run `C62-onednn-hd64`.

**2. The C80 probe.** One minute, no backend build. It answers C80's three unknowns: whether Xe-HPG
has a real oneDNN kernel for the weight-only Q8_0 matmul, whether it is right in f16, and whether it
beats dequantize + f16 GEMM on Llama-3-8B projection shapes:

```bash
icpx -fsycl -O2 sycl-patches-candidates/tools/woq-q8_0-probe/probe.cpp -o woq-probe -ldnnl
```
```bash
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./woq-probe
```

C80 is dead if any of these happens: `DECLINED`, `impl=ref...`, `FAIL`, or every shape at 1.0x or below.
Drop it and skip its rows. Otherwise its `ab.py` rows measure the end-to-end effect.

**3. Tier 0, then fix the setup.** Tier 0 starts with two A/A rows (`aa-dense`, `aa-deep`). If either
reads NOISY, nothing else in the run can be trusted at 1%: stop and look at `guard.sh --clients`,
`--temp`, and the display session before going on. Tier 0 picks the build (JIT or AOT) and graph replay. It also shows
whether the stack beats upstream at all, and whether the local patches pay together on each model
family. Edit `defaults` in `plan.json` to the winners before tier 1 (for example, add
`"GGML_SYCL_ENABLE_GRAPH": "1"` to its `env`). A `local-all-*` row with no effect means the per-patch
tier 1 rows on that model can only find something below 1%. Skip them.

```bash
python3 sycl-patches/ab/ab.py list
```
```bash
python3 sycl-patches/ab/ab.py run --tier 0 --guard --out ab-t0.jsonl
```
```bash
python3 sycl-patches/ab/ab.py report ab-t0.jsonl
```

**4. Tiers 1 and 2.** Tier 1 is the local patches and candidates on the models they target. Tier 2 is
the carried PRs with a switch, and paths only a few models reach. `--only` takes a comma list of row
names:

```bash
python3 sycl-patches/ab/ab.py run --tier 1 --guard --out ab-t1.jsonl
```
```bash
python3 sycl-patches/ab/ab.py run --tier 2 --guard --out ab-t2.jsonl
```

**5. Accuracy.** The 7 rows with a `kld` entry change the arithmetic: FA kernels (C40, C60, C62), GEMM
tiles and kernels (0017, C20/C21, C80), and all local fusions together (`local-all-q4_k`). `ab.py kld`
first writes reference logits with the CPU backend (`-dev none`, once per model and context, kept in
`ab-kld-base/`). It then scores both arms against them. B fails if its mean KLD exceeds A's by more than
10% + 2e-4, or its same-top-token rate is more than 0.3 points lower:

```bash
python3 sycl-patches/ab/ab.py kld --out ab-kld.jsonl
```

The FA rows use 8192-token windows. Their CPU reference takes the longest of anything here.

**6. Host overhead, once.** Count the runtime calls per decoded token. Two runs that differ only in
`-n` give it by subtraction:

```bash
for n in 16 48; do SYCL_UR_TRACE=2 $AB_BUILD/bin/llama-bench -m $AB_Q4_K -fa 1 -p 0 -n $n -r 1 2> ur-n$n.log >/dev/null; done
```
```bash
for c in urDeviceGetInfo urKernelCreate urProgramCreate urProgramBuild urEnqueueKernelLaunch urQueueFinish; do echo "$c $(( ($(grep -c "$c" ur-n48.log) - $(grep -c "$c" ur-n16.log)) / 32 ))"; done
```

`urEnqueueKernelLaunch` per token is the kernel count; with graph replay it should fall to near zero.
Any per-token `urDeviceGetInfo`, `urKernelCreate` or `urProgram*` is host work a cache can remove. The
same A770 project removed 8 `urDeviceGetInfo` calls per forward that way. Report the numbers with the
results.

**7. After a GPU hang.** `guard.sh --faults "2 hours ago"` counts GPU hangs and resets in the kernel
log. A row that ran during one is not evidence. Rerun it.

Expect roughly an hour and a half for tier 0 and 7 to 9 hours for tiers 1 and 2 with every model set,
most of it in the `-d 0,4096,16384` FA rows, which now load the model once per depth. So run them
overnight; `--only` reruns single rows and appends rounds.

## Turning results into changes

| Result | Local patch (`ZZ-LOCAL-*`) | Candidate | Carried PR |
|--------|---------------------------|-----------|------------|
| WIN, accuracy OK | keep | promote into `sycl-patches-extracted/` (or keep the PR file) | keep |
| WIN on an opt-in (0017, C80, C62, C90 bit) | make it the default on DG2 | same | n/a |
| no effect | drop | drop | carry while the PR is open, note it in the README |
| LOSS | drop | drop | drop locally, comment on the PR with the numbers |
| accuracy loss | drop or fix | drop or fix | drop locally |
| correctness failure | fix or drop | drop | drop locally |

Bring back `ab-stamp.txt`, the `ab-*.jsonl` files and the `report` tables. Those are enough to make the
promotions and drops, and to retire the matching `GGML_SYCL_LOCAL_OFF` bits.

## Taken from kainlan/llama.cpp-intel-optimizations

- **C80 and its probe.** The fork's weight-only Q8_0 GEMM, without its primitive cache, layout tiers or
  scratch allocator. It reads the upstream reorder layout. The probe is its V4 descriptor experiment,
  turned into a check you can run on the A770.
- **`guard.sh`.** The fork's bench-guard idea (idle and unthrottled GPU, no other tenant, a stamp of the
  driver stack). The fork's version reads xe sysfs only; this one also reads i915's, in 100 lines instead
  of 290.
- **Two lessons in `plan.json`.** A driver stamp with every number. A persistent JIT cache
  (`SYCL_CACHE_PERSISTENT=1`): the fork's cold JIT took about 61 s per process on Battlemage, and each
  `ab.py` invocation is a new process.

Not taken: the fork's FA masked-cell fix (f8561ec88). It rewrites the fork's own ESIMD and XMX
kernels, while upstream's TILE / VEC kernels add the mask as CUDA's do. The NaN it guards against needs a
stale KV cell holding NaN, and the cache is cleared when it is allocated. The fork's benchmark tables
are not used either: they are for B70 / B50 / B580, several are marked invalid by the fork itself, and
its one A770 claim is unconfirmed.
