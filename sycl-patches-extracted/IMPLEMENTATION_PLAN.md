# Implementation plan: FA-XMX, oneDNN SDPA, MoE-XMX (deferred flash-attention / MoE work)

> **Numbering note.** This document predates the directory reconciliation. Where
> it says **"0013 (XMX MMQ)"** it means the XMX quantized-matmul path, which now
> ships as **`0016-sycl-mmq-xmx.patch`** (gated by `GGML_SYCL_MMQ_XMX`). The
> on-disk `0013` is the unrelated mmvq multirow-Y SLM patch. The FA-XMX / oneDNN
> SDPA / MoE-XMX items below are the **named deferrals** in the README — they are
> not yet extracted files, so treat their `0016/0017/0018` labels here as feature
> names, not fixed slots. See `README.md` for the authoritative index.

The three remaining patches from the original port shortlist are not "small
diff" mining work — each one introduces new infrastructure into mainline (new
struct types on `ggml_backend_sycl_context`, new dispatch enums, new context
fields, new build flags) and needs careful design before code is written.

This document lays out the implementation plan: phases per patch, dependencies
between them, design decisions that need to be made before starting, and the
validation strategy.

## Sequencing — recommended order

```
                ┌─────────────┐
                │ 0016 landed │   (XMX MMQ — 0016-sycl-mmq-xmx.patch)
                └──────┬──────┘
                       │ shares ggml_sycl_xmx_available() / supports_type()
                       ▼
        ┌──────────────────────────┐
        │ 0016 fattn-onednn        │
        │ - new fattn_params       │  ─────┐
        │ - new sdpa_cache field   │       │ 0017 reuses fattn_params
        │ - oneDNN graph builder   │       │ from 0016 (or both share a
        └──────────┬───────────────┘       │ new fattn-common.hpp extension)
                   │                       ▼
                   │            ┌──────────────────────────┐
                   │            │ 0017 fattn-xmx-f16       │
                   │            │ - extends BEST_FATTN_    │
                   │            │   KERNEL enum            │
                   │            │ - joint_matrix FA kernel │
                   │            └──────────────────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ 0018 moe-xmx             │     (independent — needs no shared
        │ - new MoEContext build   │      infra with 0016/0017 except XMX
        │ - moe-sort.hpp port      │      availability from 0013)
        │ - hook in mul_mat_id     │
        └──────────────────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │ Battlemage validation    │
        │ - Arc Pro B50 / B580     │
        │ - K-quant + MoE models   │
        └──────────────────────────┘
```

**Why this order:**
1. **0016 first** because it's the only one of the three that's env-gated
   (`GGML_SYCL_FA_ONEDNN_ALLOW`) and therefore can land with minimal risk to
   mainline's default behaviour. It also forces the `fattn_params` design
   decision early; 0017 then reuses that decision.
2. **0017 after 0016** so it can build on the same `fattn_params` plumbing
   without re-litigating the design.
3. **0018 last** because it's the largest and the only one that needs MoE
   hardware to validate. Locking in the FA wins from 0016/0017 first means
   any MoE issues don't block the FA improvements from shipping.

## Time estimates

| Patch | Focused session time | Cumulative |
|---|---:|---:|
| 0016 | 3–4 h | 3–4 h |
| 0017 | 2–3 h (reuses 0016 infra) | 5–7 h |
| 0018 | 4–6 h | 9–13 h |
| Battlemage validation pass | 1–2 h | 10–15 h |

Each is a separate session — bundling them produces cognitive overload and the
design decisions get rushed.

---

# 0016 — fattn-onednn

## Goal

Port oneDNN-graph-based SDPA flash attention as an env-gated alternative path
in `ggml_sycl_flash_attn_ext`. Substantial perf benefit on Intel GPUs when the
shape fits oneDNN's SDPA pattern; falls back cleanly to mainline's existing
tile/vec FA when not eligible.

## Design decisions to make first

### D1. Where does `fattn_params` live?

The terrible tree has a ~50-field `fattn_params` struct in `fattn-common.hpp`.
Mainline already has its own (smaller) `fattn-common.hpp`. Three options:

**Option A — Extend mainline's `fattn-common.hpp` in-place.**
Add the fattn_params struct and `extract_fattn_params(ggml_tensor * dst)`
helper directly into the existing file. Pro: one canonical place. Con:
changes a file that already exists in mainline.

**Option B — New `fattn-onednn-params.hpp`.**
A scoped header with just what onednn needs. Pro: isolated, doesn't touch
mainline's fattn-common. Con: duplicates fields that 0017 will want too.

**Option C — Refactor the kernel to extract params inline (no struct).**
Skip the struct entirely; the onednn function takes `ggml_tensor * dst` and
extracts everything inside. Pro: smallest diff to mainline. Con: ugly call
site, every parameter access is verbose, kernel body is ~300 lines longer.

**Recommendation: Option A.** It's the convention CUDA uses (single
`fattn-common.cuh` with all FA helpers). The added struct is small enough
that overloading mainline's file is fine. 0017 will need the same struct so
co-locating them avoids fragmentation.

### D2. Skip KV materialization (fp8) or include it?

The terrible version has ~150 lines of "materialization" kernels that unpack
fp8 KV cache into f16 buffers before feeding oneDNN. These exist because
oneDNN's SDPA doesn't natively support fp8 KV.

**Option A — Skip entirely.** Make the eligibility check reject fp8 KV.
oneDNN-fa only runs on f16 KV (the default for most models). Cleanest port
with smallest blast radius.

**Option B — Include materialization.** Requires `ggml_sycl::alloc_handle`
or equivalent USM allocator for the temporary buffers — these are
unified-cache abstractions. Would need a mainline-style alternative (raw
`sycl::malloc_device` / `sycl::free` around the call).

**Recommendation: Option A.** Mainline doesn't yet ship fp8 KV-cache as a
default config, and adding the alloc-handle equivalent is a separate
architectural discussion. Punt to a follow-up if fp8 KV traction grows.

### D3. Where does the SDPA cache live?

The terrible tree hangs an `sdpa_partition_cache *` off
`ggml_backend_sycl_context::sdpa_cache` as a `void *`. Initialised lazily;
destroyed via `ggml_sycl_sdpa_cache_destroy(void *)` on context teardown.

**Recommendation:** Match the terrible pattern exactly. Add
`void * sdpa_cache = nullptr;` to `ggml_backend_sycl_context` in
`common.hpp`. The destroy hook needs to be called from the context's
destructor (or `~ggml_backend_sycl_context`) — wherever mainline does
backend teardown.

### D4. Build flag

The terrible tree gates oneDNN-fa under `GGML_SYCL_DNNL` (existing build flag)
plus env-var `GGML_SYCL_FA_ONEDNN_ALLOW=1` at runtime. Mainline already has
`GGML_SYCL_DNNL` — reuse it for compile-time, env-var for runtime gating.

## Phases

### Phase 1 — Infrastructure (no new kernels yet)

1. Read mainline's existing `fattn-common.hpp`.
2. Add `fattn_params` struct + `extract_fattn_params(ggml_tensor * dst)`
   inline helper (per D1 Option A).
3. Add `void * sdpa_cache = nullptr;` to `ggml_backend_sycl_context`.
4. Add forward declaration `void ggml_sycl_sdpa_cache_destroy(void *);` so the
   context destructor can call it without pulling in `fattn-onednn.hpp`.
5. Wire the destroy call into the context destructor.

**Validation:** mainline builds and `llama-bench` produces same numbers as
baseline (no perf change yet, just plumbing). **Patch size so far: ~150 lines.**

### Phase 2 — Port `fattn-onednn.hpp`

1. Copy from terrible.
2. Strip materialization fields from `ggml_sycl_onednn_fa_materialized_kv`
   (D2 Option A) — the entire struct can be removed if we also strip the
   materialize_kv function from the header's public API.
3. Keep `sdpa_shape_key`, `sdpa_shape_key_hash`, `sdpa_compiled_entry`,
   `sdpa_partition_cache` as-is — these are cache types, no unified-cache deps.
4. Public API surface: `ggml_sycl_flash_attn_ext_onednn_eligible`,
   `ggml_sycl_flash_attn_ext_onednn`, `ggml_sycl_flash_attn_ext_onednn_plan`,
   `ggml_sycl_sdpa_cache_destroy`.

**Patch size: ~150 lines (after materialization strip).**

### Phase 3 — Port `fattn-onednn.cpp`

1. Copy from terrible.
2. Strip lines ~232–375 (the `materialize_one`, `release_materialized_after_event`,
   `materialize_kv` functions). Replace `materialize_kv` declaration with a stub
   that returns `false` if anything calls it.
3. Edit the eligibility check to **reject fp8 KV** at the gate, so the
   materialization path is unreachable.
4. Keep all of `build_and_compile_sdpa` (~300 lines), the SDPA cache logic,
   the eligibility check, and the main dispatch entry `ggml_sycl_flash_attn_ext_onednn`.
5. Replace any remaining `ggml_sycl::` references — but per the survey there
   are only 10 references and they're all in materialization code being stripped.

**Patch size: ~800 lines (after strip).**

### Phase 4 — Hook in mainline `fattn.cpp`

Mainline's `ggml_sycl_flash_attn_ext` is a switch on `BEST_FATTN_KERNEL_VEC`
vs `BEST_FATTN_KERNEL_TILE`. Wrap it with an env-gated oneDNN check:

```cpp
void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_set_device(ctx.device);
#if GGML_SYCL_DNNL
    if (getenv("GGML_SYCL_FA_ONEDNN_ALLOW")) {
        fattn_params params = extract_fattn_params(dst);
        if (ggml_sycl_flash_attn_ext_onednn_eligible(params, ...)) {
            if (ggml_sycl_flash_attn_ext_onednn(ctx, params)) {
                return;
            }
        }
    }
#endif
    switch (ggml_sycl_get_best_fattn_kernel(...)) { ... }
}
```

**Patch size: ~30 lines in fattn.cpp.**

### Phase 5 — Validation

1. Build with `GGML_SYCL_DNNL=ON` (mainline already finds oneDNN when present).
2. Bench without `GGML_SYCL_FA_ONEDNN_ALLOW` (verify zero regression).
3. Bench with `GGML_SYCL_FA_ONEDNN_ALLOW=1`. On A770 this likely still uses
   the BLAS path for matmul (so prefill won't change much) but the FA path
   itself should land oneDNN SDPA.
4. **Important**: CLAUDE.md notes the source-tree's onednn-fa has an
   `nc != D` contiguity bypass that produces wrong output on Mistral. Verify
   the eligibility check correctly rejects that case before bench.

## Open questions

- Should the eligibility check reject when `n_seqs > 1` (multi-sequence
  batching)? The terrible version handles it via `seq_q_offsets` /
  `seq_kv_offsets` fields. For first port, **reject** and rely on the fallback
  FA path — keeps the cache key simpler too.
- How aggressive should the cache size be? Terrible has no LRU eviction.
  For first port, accept unbounded growth — typical inference sessions only
  hit a handful of shape keys.

---

# 0017 — fattn-xmx-f16

## Goal

Port joint_matrix-based flash attention for Intel Arc Alchemist / Battlemage.
Slot into mainline's `BEST_FATTN_KERNEL` enum as a new case, chosen when
`ggml_sycl_xmx_available()` returns true and the shape fits.

## Design decisions to make first

### D1. Reuse 0016's `fattn_params` or add an FA-XMX-specific struct?

**Recommendation: reuse.** The fields needed (Q/K/V/mask/sinks pointers,
scale, max_bias, n_head_log2, head_dim, ne00–ne03, ...) are essentially
identical. Co-locating in `fattn-common.hpp` means one extraction helper
serves both paths.

### D2. How to extend `BEST_FATTN_KERNEL`?

Mainline's enum:
```cpp
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE,
    BEST_FATTN_KERNEL_TILE,
    BEST_FATTN_KERNEL_VEC,
};
```

Add:
```cpp
    BEST_FATTN_KERNEL_XMX,   // joint_matrix path, Xe-HPG / Xe2 only
```

`ggml_sycl_get_best_fattn_kernel` needs updating to prefer XMX when:
- `ggml_sycl_xmx_available()` returns true (from 0013's port)
- Q/K/V are F16
- head_dim is in the supported set (typically {64, 96, 128})
- Batch is in the supported range (XMX FA tends to win at moderate batch sizes)

### D3. Build flag

**Recommendation: reuse `GGML_SYCL_MMQ_XMX`** from 0013. Both XMX paths share
the same hardware/compiler requirement (`__has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)`).
A single flag keeps the build matrix simple. Or split into `GGML_SYCL_XMX_MMQ`
and `GGML_SYCL_XMX_FA` if reviewers prefer fine-grained gating.

## Phases

### Phase 1 — Depends on 0016 landing

The `fattn_params` struct must already exist in `fattn-common.hpp`. If 0016 is
deferred, Phase 1 of 0017 has to do that work itself.

### Phase 2 — Port `fattn-xmx-f16.hpp`

1. Copy from terrible (1460 lines, already clang-formatted).
2. Audit for terrible-only abstractions — the grep showed zero, so this should
   be a clean copy.
3. The file is header-only template-heavy code. `launch_fattn_xmx_f16` is the
   main entry; uses `DISPATCH_NCOLS` macros internally.
4. Check whether the v2 variant (`fattn-xmx-f16-v2.hpp`, 946 lines) is
   preferred over v1. CLAUDE.md mentions v2 as the current Mistral path —
   may be worth porting v2 instead and skipping v1.

### Phase 3 — Wire dispatcher

```cpp
static best_fattn_kernel ggml_sycl_get_best_fattn_kernel(int device, const ggml_tensor * dst) {
#ifdef GGML_SYCL_MMQ_XMX
    if (ggml_sycl_xmx_available() && fattn_xmx_f16_eligible(dst, device)) {
        return BEST_FATTN_KERNEL_XMX;
    }
#endif
    // ... existing logic ...
}

void ggml_sycl_flash_attn_ext(...) {
    switch (...) {
    ...
#ifdef GGML_SYCL_MMQ_XMX
    case BEST_FATTN_KERNEL_XMX:
        ggml_sycl_flash_attn_ext_xmx(ctx, dst);
        break;
#endif
    ...
    }
}
```

`fattn_xmx_f16_eligible` is a small predicate that checks head_dim, batch
range, and types.

### Phase 4 — Validation

A770 will likely show no change because mainline's dispatcher routes through
oneMKL BLAS for the test model anyway. **Document this expectation** and
defer real perf validation to Battlemage.

## Open questions

- Port v1 (`fattn-xmx-f16.hpp`) or v2 (`fattn-xmx-f16-v2.hpp`)? Per CLAUDE.md,
  v2 is the current Mistral path and gates Q-f16 cast / `afloat` accumulator
  by `GGML_SYCL_F16` build flag. Recommendation: **v2 only.**
- Does the v2 path need the `prec` field from fattn_params (for the
  PREC_F32 / PREC_DEFAULT distinction)? Yes — add to fattn_params if not
  there from 0016.

---

# 0018 — moe-xmx

## Goal

Port XMX joint_matrix kernels for MoE (Mixture of Experts) routing. The
biggest single source-tree win for GPT-OSS-class MXFP4 MoE models.

## Design decisions to make first

### D1. How to construct MoEContext?

The terrible tree uses a `MoEContext` struct with many fields
(`q_tokens`, `token_scales`, `is_coalesced`, `is_soa`, `xmx_cfg`,
`expert_scale_buf`, `q8_qs_per_expert`, `mxfp4_qs_per_expert`, etc.).
Mainline's `ggml_sycl_mul_mat_id` doesn't have this — it takes
`src0`, `src1`, `ids`, `dst` directly.

**Recommendation: build MoEContext on-demand inside the moe-xmx dispatch.**
Add a `static MoEContext build_moe_context(...)` helper that takes the
mainline params and produces the context the kernels expect. Keep MoEContext
as an internal detail of moe-xmx.hpp; don't add it to mainline-wide headers.

### D2. Which layouts to support?

Terrible supports three: AOS (default), coalesced (unified-cache reorder),
SOA (unified-cache reorder). Coalesced and SOA depend on weight reordering
that happens inside the unified-cache.

**Recommendation: AOS only for first port.** Reorder paths require either:
- Adding the reorder kernels separately (extending convert.cpp), OR
- Living with the AOS-only fast path.

Reorder kernels could be a follow-up. AOS-only still lands the MoE win.

### D3. Which quant types?

Terrible supports Q8_0 and MXFP4 in moe-xmx. Both are useful (GPT-OSS uses
MXFP4; smaller MoEs use Q8_0).

**Recommendation: both.** They share enough kernel infrastructure that doing
one means most of the work for the other is already done.

### D4. Where to hook?

Mainline has `ggml_sycl_mul_mat_id_mmvq_fused` as the existing MoE fast
path. Add a new conditional that checks XMX availability + supports_type and
routes to moe-xmx **before** the mmvq_fused check.

## Phases

### Phase 1 — Port `moe-sort.hpp`

1. Copy from terrible (507 lines). Audit for terrible-only deps.
2. Token sorting and expert assignment — likely clean (it's pure data
   manipulation). Verify with grep.
3. **Patch size: ~500 lines.**

### Phase 2 — Port `moe-xmx.hpp` (kernels)

1. Copy from terrible (2184 lines).
2. Audit for terrible-only deps. Per the earlier survey: **zero** deps in the
   kernel file itself.
3. Skip `launch_xmx_moe_gemm_q8_0_coalesced` and `_soa` variants — they need
   the reorder kernels (D2 Option). Keep the AOS variants
   (`launch_xmx_moe_gemm_q8_0`, `launch_xmx_moe_gemm_mxfp4`).
4. **Patch size: ~1500 lines after stripping the reorder-layout variants.**

### Phase 3 — Port `moe-xmx-fused.hpp` (gate+up+down)

1. Copy from terrible (982 lines).
2. Audit for deps.
3. Same coalesced/SOA strip as Phase 2.

### Phase 4 — Build MoEContext + hook

1. Write `static MoEContext build_moe_context_from_mul_mat_id(src0, src1, ids, dst)`.
2. Add the dispatch hook in `ggml_sycl_mul_mat_id`:

```cpp
static void ggml_sycl_mul_mat_id(...) {
#ifdef GGML_SYCL_MMQ_XMX
    if (ggml_sycl_xmx_available() &&
        ggml_sycl_xmx_moe_supports_type(src0->type) &&
        moe_xmx_eligible(src0, src1, ids, dst)) {
        moe_xmx::dispatch(build_moe_context_from_mul_mat_id(src0, src1, ids, dst));
        return;
    }
#endif
    if (ggml_sycl_mul_mat_id_mmvq_fused(...)) return;
    // ... existing fallback ...
}
```

3. **Patch size: ~100 lines for the hook + context builder.**

### Phase 5 — Validation

A770 has 16 GB VRAM. The smallest MoE Q4_0 model in `~/LLM/` is
`gemma-4-26B-A4B-it-heretic-ara-v2-Q4_0.gguf` (16 GB on disk). Won't fit with
framework overhead. **Validation requires Battlemage.**

If you can fit a smaller MoE somehow (Q2_K quant of a small MoE?), test
correctness via `llama-completion` deterministic gate even if perf isn't
measurable.

## Open questions

- How does `extra_gpu` layout detection translate to mainline? The terrible
  `ggml_tensor::extra` is the `ggml_tensor_extra_gpu` struct in mainline too,
  but the `optimized_feature.reorder` field handling is unified-cache-specific.
  **Recommendation:** treat all weights as AOS at the moe-xmx layer (matches D2).
- Does `ggml_sycl_mul_mat_id_mmvq_fused` already cover the common case for
  GPT-OSS? Check before adding moe-xmx — if mmvq_fused dispatches MXFP4
  efficiently, moe-xmx is only worth porting for higher-batch prefill.

---

# Cross-cutting validation strategy

## A770 (this machine) — limited measurement

- 0016: env-gated, default-off. Verify no regression at default settings.
  With `GGML_SYCL_FA_ONEDNN_ALLOW=1`, expect oneDNN to take over FA — likely
  **wash on Gemma 4.6B Q4_0** because matmul is the bottleneck and that goes
  through BLAS regardless. Need a model where FA dominates to see a signal.
- 0017: A770 has joint_matrix support but dispatcher prefers oneMKL for
  small/medium matmul shapes. Verify the BEST_FATTN_KERNEL_XMX case fires
  via `GGML_SYCL_DEBUG=1`. **Likely null measured benefit on A770.**
- 0018: cannot be benchmarked on A770 (no MoE model fits). Verify correctness
  via `llama-completion` on a small MoE if available, otherwise just verify
  build + dispatch path entry via debug logging.

## Battlemage (B50 / B580) — real signal expected

When hardware is available, run the full bench matrix:

| Model | Quant | Pattern | Expected to fire |
|---|---|---|---|
| Mistral 7B | Q4_0 | dense | 0013 (XMX MMQ), 0017 (XMX FA), 0016 (oneDNN-fa, env-gated) |
| Mistral 7B | Q4_K_M | dense | 0013, 0017, K-quant MMQ paths |
| Qwen3.5-9B | Q4_K_M | dense | same |
| GPT-OSS 20B | MXFP4 | MoE | 0018 (moe-xmx) — biggest expected win |

xmx_kernel_plan.md numbers as reference: pp16 Q4_0 3.11×, Q8_0 1.94×,
Q4_K 1.55×, Q5_K 1.66×. These were measured on B50.

## Per-patch correctness gates

Before any perf measurement:

```bash
# Deterministic completion gate (must produce expected next tokens)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/llama-completion \
    -m <model> -p '1, 2, 3, 4, 5,' -n 15 --seed 42 --temp 0

# Backend ops correctness (especially important for 0015, 0016, 0017)
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops \
    -o FLASH_ATTN_EXT          # for 0016 / 0017
ONEAPI_DEVICE_SELECTOR=level_zero:0 ./build/bin/test-backend-ops \
    -o MUL_MAT_ID              # for 0018
```

## CI considerations

Each patch should:
- Be off-by-default at runtime (env var or runtime probe).
- Compile-gate behind `GGML_SYCL_DNNL` (0016) or `GGML_SYCL_MMQ_XMX` (0017, 0018).
- Fall through cleanly when not applicable — no behavioural change vs baseline
  when the path doesn't fire.

Add `ggml-ci` to commit messages so heavy CI runs both default-off and
opt-in configurations.

---

# What this plan does not cover

- **ESIMD kernels** (`mmq-esimd.hpp` etc.). Adds a parallel third dispatch
  path; needs reviewer discussion before mainline accepts. Treat as a separate
  initiative — possibly merge with 0017 since `fattn-esimd-f16.hpp` exists,
  but most reviewers would push back on yet-another-dispatch-path.
- **Unified-cache subsystem** (tlsf-allocator, kv-tier-manager, vmem-kv,
  expert-prefetch, prefetch-scheduler, tuning-engine, layer-streaming). These
  are architectural decisions that should be discussed with upstream
  maintainers; not appropriate as drive-by ports.
- **Targeted edits in existing-file diffs** (mmvq.cpp +7394, mmq.cpp +5893,
  dmmv.cpp +3319). Most of this is unified-cache plumbing; the small perf
  tweaks would need hours of careful diff reading per file with unclear payoff.
  Park unless a specific bottleneck points back to one of these files.

---

# Summary

| Patch | Real scope | Dependencies | Risk | Battlemage win estimate |
|---|---|---|---|---|
| 0016 | ~1100 lines after strip | None (introduces fattn_params) | Low (env-gated) | Moderate — depends on model |
| 0017 | ~1700 lines | 0016's fattn_params | Low (build-flag gated) | High — directly XMX FA |
| 0018 | ~2200 lines after strip | 0013's XMX availability | Medium (MoE-specific) | Very high — GPT-OSS class models |

Total: ~5000 lines of code across three dedicated sessions, plus ~1–2 hours
of Battlemage bench validation. Each landing independently means partial
progress always sticks.
