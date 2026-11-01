# SYCL Optimization Extraction & PR Strategy

This plan outlines a phased approach to extract useful changes from the `sycl_diff.patch` (diff between `llama.cpp` and `llama.cpp-sycl-terrible`). 

To ensure the work is manageable and reviewable, we have organized the extraction into distinct, logically cohesive phases. Each phase is scoped to represent a reasonably sized Pull Request (PR) that introduces a specific class of optimization, integrates it into the backend, and includes the necessary boilerplate. 

**Workflow Change:** Following established conventions, we will *not* commit directly to the tree. Instead, the output of each phase will be a cleanly generated `.patch` file (e.g., `phase1-fusions.patch`) saved to the `sycl-patches/` directory.

## User Review Required

> [!IMPORTANT]
> The heavy lifting of dealing with the non-compliant architecture (unified caches, custom memory allocators) remains in **Phase 4 (Deferred)**. 
> 
> Phases 1 through 3 are now structured as individual deployable patch sets. 
> 
> **Does this patch-based PR grouping look manageable for your workflow?**

---

## Phase 1: Foundation & Fused Operations (Patch 1)

**Goal:** Extract pure algorithmic optimizations and standard graph fusions that do not require specialized hardware intrinsics. This establishes the baseline for further optimizations.

**Work Items:**
1.  **Extract Fusions:** Port `fused-norm-gemm.hpp`, `mmvq-rmsnorm.hpp`, and `fused-ffn.hpp`.
2.  **Core Optimizations:** Analyze and port loop unrolling and memory coalescing improvements from `mmq.cpp`, `mmvq.cpp`, and `fattn.cpp`.
3.  **Integration:** Add `#include` directives to `ggml-sycl.cpp` and implement the standard `ggml` graph rewrite rules for these fusions.
4.  **Serialization:** Generate `phase1-fusions.patch` and save it to `sycl-patches/`.
5.  **Verification:** Apply the patch using `apply.sh`, ensure `ctest -R sycl` passes, and verify standard inference latency is improved.

---

## Phase 2: ESIMD Acceleration (Patch 2)

**Goal:** Introduce Intel ESIMD (Explicit SIMD) kernels for bare-metal execution speed on Intel GPUs. ESIMD is a distinct programming model within SYCL and warrants its own patch to isolate potential compilation or driver issues.

**Work Items:**
1.  **Extract ESIMD Kernels:** Port `fused-moe-esimd.hpp`, `fattn-esimd-f16.hpp`, and `mmq-esimd.hpp`.
2.  **Strip Dependencies:** Ensure these headers rely on standard `ggml_sycl_malloc` rather than the "terrible" branch's TLSF allocators.
3.  **Integration:** Wire the ESIMD dispatch logic into `ggml-sycl.cpp`. Ensure fallback paths exist for non-Intel GPUs.
4.  **Serialization:** Generate `phase2-esimd.patch` and save it to `sycl-patches/`.
5.  **Verification:** Apply the patch, run specific ESIMD microbenchmarks (if salvaged from the diff), and verify MOE/Attention models.

---

## Phase 3: XMX Matrix Extensions (Patch 3)

**Goal:** Introduce support for Intel Xe Matrix Extensions (XMX) using DPAS (Dot Product Accumulate Systolic) instructions. This is a complex feature that heavily impacts matrix multiplication performance on Arc and Max GPUs.

**Work Items:**
1.  **Extract XMX Kernels:** Port `mmq_xmx.cpp`, `moe-xmx.hpp`, `fattn-xmx-f16.hpp`, and `xmx-esimd-common.hpp`.
2.  **Architecture Adaptation:** Remove dependencies on the `unified-kernel` monolithic dispatcher. 
3.  **Integration:** Register `mmq_xmx` within `ggml-sycl.cpp` specifically for supported GPU architectures.
4.  **Serialization:** Generate `phase3-xmx.patch` and save it to `sycl-patches/`.
5.  **Verification:** Apply the patch, compile with XMX flags enabled, and verify using hardware detection tests.

---

## Phase 4: Compliance Analysis & Subsystem Rework (Deferred Backlog)

This phase acts as a backlog for the large, non-compliant subsystems. We will only tackle these if deemed absolutely necessary, as they require significant architectural rework to match CUDA/HIP/Metal conventions.

### 1. Memory Management (`pinned-pool.*`, `tlsf-allocator.*`)
*   **Convention:** Custom allocators must implement `ggml_backend_buffer_type_i` without global singletons.
*   **Rework Framework:** Wrap the TLSF/Pinned allocators in a compliant `ggml_backend_sycl_buffer_type`.

### 2. KV-Cache Tiering & Offload (`kv-tier-manager.*`, `vmem-kv.*`)
*   **Convention:** Host-device KV splitting is handled natively by the `llama_context` scheduler.
*   **Rework Framework:** Adapt the SYCL KV cache to utilize SYCL USM for automatic paging, or implement a specific host-fallback buffer type for `ggml_backend_sched`. 

### 3. CPU Dispatch & Unified Cache (`unified-kernel.*`, `cpu-dispatch.*`)
*   **Convention:** If an op is unsupported, the scheduler automatically falls back to `ggml_backend_cpu`.
*   **Rework Framework:** Remove `cpu-dispatch.cpp` completely. Trim "Unified Cache" logic to only cache SYCL-specific kernel JIT compilations.

### 4. Layer Streaming (`layer-streaming.*`, `expert-prefetch.*`)
*   **Convention:** Weights are mapped via `mmap` or explicitly copied during initialization.
*   **Rework Framework:** Convert manual background thread prefetching into asynchronous `sycl::queue.memcpy` operations queued alongside computation.
