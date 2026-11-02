// Checks the readers of a weight the SYCL backend has already moved to its reorder (SoA) layout.
//
// test-backend-ops builds fresh weights for every case, so it only sees the reorder on the call that
// installs it (a MUL_MAT / MUL_MAT_ID of at most 8 tokens). A model reuses the weight: prompt batches
// (dequantize + GEMM, the per-expert loop, the grouped GEMM) and GET_ROWS of a tied embedding then read
// the reordered bytes. This runs, per type, one graph that installs the reorder and a second one that
// reads it the other ways, on SYCL0 and on the CPU backend, and compares the outputs (NMSE).
//
//   reorder-reuse [type ...]      default: every type with a 2D or MoE reorder
//
// Run it like test-backend-ops in run.sh (GGML_SYCL_TEST_FORCE_REORDER=1 on the CPU OpenCL device). Ops the
// SYCL backend does not support for a type are skipped on both backends.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int K = 512, M = 96, E = 4, N_USED = 2, N_PP = 32, N_ROWS = 5;

struct outputs {
    std::vector<float> mm1, mm_pp, rows, id1, id_pp;
};

static std::vector<float> read(const ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

static void set_weights(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    const int64_t n_per_row = t->ne[0], nrows = ggml_nrows(t);
    std::vector<float> f(n_per_row * nrows);
    for (auto & x : f) x = u(rng);
    std::vector<uint8_t> q(ggml_nbytes(t));
    if (t->type == GGML_TYPE_F32) {
        memcpy(q.data(), f.data(), q.size());
    } else {
        ggml_quantize_chunk(t->type, f.data(), q.data(), 0, nrows, n_per_row, nullptr);
    }
    ggml_backend_tensor_set(t, q.data(), 0, q.size());
}

static void set_f32(ggml_tensor * t, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> f(ggml_nelements(t));
    for (auto & x : f) x = u(rng);
    ggml_backend_tensor_set(t, f.data(), 0, ggml_nbytes(t));
}

static void set_i32(ggml_tensor * t, const std::vector<int32_t> & v) {
    ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
}

// the returned allocator owns the graph's outputs: free it after reading them
static ggml_gallocr_t compute(ggml_backend_t backend, ggml_cgraph * gf) {
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
    ggml_backend_synchronize(backend);
    return galloc;
}

// ops the SYCL backend cannot run are left out on both backends: `use` comes from the SYCL run
struct op_set {
    bool mm1 = true, id1 = true, mm_pp = true, rows = true, id_pp = true;
};

static void run(ggml_backend_t backend, ggml_type type, op_set & use, bool probe, outputs & out) {
    std::mt19937 rng(1234);

    ggml_init_params ip = { 64 * ggml_tensor_overhead(), nullptr, true };
    ggml_context * cw = ggml_init(ip);
    ggml_tensor * w   = ggml_new_tensor_2d(cw, type, K, M);
    ggml_tensor * we  = ggml_new_tensor_3d(cw, type, K, M, E);
    ggml_tensor * x1  = ggml_new_tensor_2d(cw, GGML_TYPE_F32, K, 1);
    ggml_tensor * xpp = ggml_new_tensor_2d(cw, GGML_TYPE_F32, K, N_PP);
    ggml_tensor * r   = ggml_new_tensor_1d(cw, GGML_TYPE_I32, N_ROWS);
    ggml_tensor * xe1 = ggml_new_tensor_3d(cw, GGML_TYPE_F32, K, 1, 1);
    ggml_tensor * xep = ggml_new_tensor_3d(cw, GGML_TYPE_F32, K, 1, N_PP);
    ggml_tensor * id1 = ggml_new_tensor_2d(cw, GGML_TYPE_I32, N_USED, 1);
    ggml_tensor * idp = ggml_new_tensor_2d(cw, GGML_TYPE_I32, N_USED, N_PP);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(cw, backend);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

    set_weights(w, rng);
    set_weights(we, rng);
    set_f32(x1, rng);
    set_f32(xpp, rng);
    set_f32(xe1, rng);
    set_f32(xep, rng);
    set_i32(r, { 3, 0, M - 1, 17, 3 });
    set_i32(id1, { 2, 0 });
    std::vector<int32_t> idv(N_USED * N_PP);
    for (int t = 0; t < N_PP; ++t) {
        idv[N_USED * t + 0] = t % E;
        idv[N_USED * t + 1] = (t * 3 + 1) % E;
    }
    set_i32(idp, idv);

    ggml_context * cg = ggml_init({ 64 * ggml_tensor_overhead() + ggml_graph_overhead() * 2, nullptr, true });

    ggml_tensor * mm1 = ggml_mul_mat(cg, w, x1);
    ggml_tensor * mi1 = ggml_mul_mat_id(cg, we, xe1, id1);
    ggml_tensor * mmp = ggml_mul_mat(cg, w, xpp);
    ggml_tensor * gr  = ggml_get_rows(cg, w, r);
    ggml_tensor * mip = ggml_mul_mat_id(cg, we, xep, idp);
    if (probe) {
        use.mm1   = ggml_backend_supports_op(backend, mm1);
        use.id1   = ggml_backend_supports_op(backend, mi1);
        use.mm_pp = ggml_backend_supports_op(backend, mmp);
        use.rows  = ggml_backend_supports_op(backend, gr);
        use.id_pp = ggml_backend_supports_op(backend, mip);
    }

    // pass 1 installs the reorder (one-token MUL_MAT and MUL_MAT_ID), pass 2 reads the installed layout
    ggml_cgraph * g1 = ggml_new_graph(cg);
    if (use.mm1) ggml_build_forward_expand(g1, mm1);
    if (use.id1) ggml_build_forward_expand(g1, mi1);
    ggml_cgraph * g2 = ggml_new_graph(cg);
    if (use.mm_pp) ggml_build_forward_expand(g2, mmp);
    if (use.rows)  ggml_build_forward_expand(g2, gr);
    if (use.id_pp) ggml_build_forward_expand(g2, mip);

    if (ggml_graph_n_nodes(g1) > 0) {
        ggml_gallocr_t ga = compute(backend, g1);
        if (use.mm1) out.mm1 = read(mm1);
        if (use.id1) out.id1 = read(mi1);
        ggml_gallocr_free(ga);
    }
    if (ggml_graph_n_nodes(g2) > 0) {
        ggml_gallocr_t ga = compute(backend, g2);
        if (use.mm_pp) out.mm_pp = read(mmp);
        if (use.rows)  out.rows  = read(gr);
        if (use.id_pp) out.id_pp = read(mip);
        ggml_gallocr_free(ga);
    }

    ggml_free(cg);
    ggml_backend_buffer_free(buf);
    ggml_free(cw);
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num += (a[i] - b[i]) * (a[i] - b[i]);
        den += b[i] * b[i];
    }
    return den > 0 ? num / den : num;
}

int main(int argc, char ** argv) {
    std::vector<ggml_type> types;
    for (int i = 1; i < argc; ++i) {
        bool found = false;
        for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
            const char * name = ggml_type_name((ggml_type) t);
            if (name && std::string(name) == argv[i]) {
                types.push_back((ggml_type) t);
                found = true;
            }
        }
        if (!found) {
            fprintf(stderr, "unknown type %s\n", argv[i]);
            return 2;
        }
    }
    if (types.empty()) {
        types = { GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
                  GGML_TYPE_Q6_K, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ4_NL };
    }

    ggml_backend_dev_t dev = ggml_backend_dev_by_name("SYCL0");
    if (!dev) {
        fprintf(stderr, "no SYCL0 device\n");
        return 2;
    }
    ggml_backend_t sycl = ggml_backend_dev_init(dev, nullptr);
    ggml_backend_t cpu  = ggml_backend_cpu_init();

    int failed = 0;
    for (ggml_type type : types) {
        outputs a, b;
        op_set  use;
        run(sycl, type, use, /*probe=*/true, a);
        run(cpu, type, use, /*probe=*/false, b);
        struct {
            const char *               name;
            const std::vector<float> & x;
            const std::vector<float> & y;
        } cmp[] = {
            { "mul_mat n=1 (installs)", a.mm1, b.mm1 },        { "mul_mat_id n=1 (installs)", a.id1, b.id1 },
            { "mul_mat n=32 (reads)", a.mm_pp, b.mm_pp },      { "get_rows (reads)", a.rows, b.rows },
            { "mul_mat_id n=32 (reads)", a.id_pp, b.id_pp },
        };
        for (const auto & c : cmp) {
            if (c.x.empty()) {
                printf("%-8s %-26s not supported on SYCL0, skipped\n", ggml_type_name(type), c.name);
                continue;
            }
            const double e    = nmse(c.x, c.y);
            const bool   pass = c.x.size() == c.y.size() && e < 5e-4;
            failed += !pass;
            printf("%-8s %-26s nmse %.3e  %s\n", ggml_type_name(type), c.name, e, pass ? "OK" : "FAIL");
        }
    }

    ggml_backend_free(sycl);
    ggml_backend_free(cpu);
    printf("%s\n", failed ? "FAILED" : "all passed");
    return failed != 0;
}
