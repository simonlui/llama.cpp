// Probe for candidate C80 (GGML_SYCL_WOQ_Q8_0): does oneDNN run the weight-only-quantized Q8_0 matmul on this
// device with a real kernel, is it right, and is it faster than dequantize-to-f16 + f16 matmul (the path it
// replaces)? Uses the exact descriptors of DnnlGemmWrapper::woq_q8_0_gemm on the Q8_0 reorder layout.
//
//   icpx -fsycl -O2 probe.cpp -o woq-probe -ldnnl
//   ONEAPI_DEVICE_SELECTOR=level_zero:0 ./woq-probe            # default shapes, GPU engine
//   ./woq-probe --cpu                                          # CPU engine: the weight and scale layout only,
//                                                              # with f32 activations (f16 needs AVX512-FP16)
//   ./woq-probe 4096 4096 512                                  # rows k tokens
#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <unordered_map>
#include <vector>

#include "dnnl.hpp"
#include "dnnl_sycl.hpp"

using dt = dnnl::memory::data_type;

struct shape {
    int64_t m, k, n;  // weight rows, reduction, tokens
};

static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int run(sycl::queue & q, dnnl::engine & eng, dnnl::stream & strm, const shape & s, int iters, bool f32_src) {
    const int64_t m = s.m, k = s.k, n = s.n, groups = k / 32;
    std::mt19937  rng(1234 + (unsigned) (m + k + n));
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Q8_0 weights in the reorder layout: qs [m][k] int8, d [m][k/32] f16
    std::vector<int8_t>     qs(m * k);
    std::vector<sycl::half> d(m * groups);
    for (int64_t r = 0; r < m; ++r) {
        for (int64_t g = 0; g < groups; ++g) {
            float x[32], amax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                x[i] = dist(rng) * 0.05f;
                amax = std::max(amax, std::fabs(x[i]));
            }
            const float dd = amax / 127.0f;
            d[r * groups + g] = (sycl::half) dd;
            for (int i = 0; i < 32; ++i) {
                qs[r * k + g * 32 + i] = (int8_t) std::lround(dd ? x[i] / dd : 0.0f);
            }
        }
    }
    std::vector<sycl::half> act(n * k);
    for (auto & v : act) {
        v = (sycl::half) dist(rng);
    }
    // the scales the candidate stages: [k/32][m]
    std::vector<sycl::half> d_kn(groups * m);
    for (int64_t g = 0; g < groups; ++g) {
        for (int64_t r = 0; r < m; ++r) {
            d_kn[g * m + r] = d[r * groups + g];
        }
    }

    auto * qs_d   = sycl::malloc_device<int8_t>(m * k, q);
    auto * dkn_d  = sycl::malloc_device<sycl::half>(groups * m, q);
    auto * act_d  = sycl::malloc_device<sycl::half>(n * k, q);
    auto * act32_d = sycl::malloc_device<float>(n * k, q);
    auto * dst_d  = sycl::malloc_device<float>(n * m, q);
    auto * w16_d  = sycl::malloc_device<sycl::half>(m * k, q);
    auto * dref_d = sycl::malloc_device<sycl::half>(m * groups, q);
    q.memcpy(qs_d, qs.data(), qs.size()).wait();
    q.memcpy(dkn_d, d_kn.data(), d_kn.size() * 2).wait();
    q.memcpy(act_d, act.data(), act.size() * 2).wait();
    {
        std::vector<float> act32(act.begin(), act.end());
        q.memcpy(act32_d, act32.data(), act32.size() * 4).wait();
    }
    q.memcpy(dref_d, d.data(), d.size() * 2).wait();

    // weight-only path: the descriptors of DnnlGemmWrapper::woq_q8_0_gemm
    const auto src_md = dnnl::memory::desc({ n, k }, f32_src ? dt::f32 : dt::f16, { k, 1 });
    const auto wei_md = dnnl::memory::desc({ k, m }, dt::s8, { 1, k });
    const auto dst_md = dnnl::memory::desc({ n, m }, dt::f32, { m, 1 });
    const auto scl_md = dnnl::memory::desc({ groups, m }, dt::f16, { m, 1 });
    dnnl::primitive_attr attr;
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), { 32, 1 }, dt::f16);
    attr.set_fpmath_mode(f32_src ? dnnl::fpmath_mode::strict : dnnl::fpmath_mode::f16, true);
    dnnl::matmul::primitive_desc pd;
    try {
        pd = dnnl::matmul::primitive_desc(eng, src_md, wei_md, dst_md, attr);
    } catch (const dnnl::error & e) {
        std::printf("m=%lld k=%lld n=%lld  woq: DECLINED by oneDNN (%s)\n", (long long) m, (long long) k,
                    (long long) n, e.what());
        return 1;
    }
    const char * impl = pd.impl_info_str();
    dnnl::matmul woq(pd);
    std::unordered_map<int, dnnl::memory> args = {
        { DNNL_ARG_SRC, dnnl::memory(src_md, eng, f32_src ? (void *) act32_d : (void *) act_d) },
        { DNNL_ARG_WEIGHTS, dnnl::memory(wei_md, eng, qs_d) },
        { DNNL_ARG_DST, dnnl::memory(dst_md, eng, dst_d) },
        { DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(scl_md, eng, dkn_d) },
    };
    woq.execute(strm, args);
    strm.wait();

    std::vector<float> out(n * m);
    q.memcpy(out.data(), dst_d, out.size() * 4).wait();
    double err2 = 0.0, ref2 = 0.0, maxabs = 0.0;
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t r = 0; r < m; ++r) {
            double acc = 0.0;
            for (int64_t i = 0; i < k; ++i) {
                acc += (double) (float) act[t * k + i] * (double) qs[r * k + i] * (double) (float) d[r * groups + i / 32];
            }
            const double e = out[t * m + r] - acc;
            err2 += e * e;
            ref2 += acc * acc;
            maxabs = std::max(maxabs, std::fabs(e));
        }
    }
    const double nmse = ref2 > 0 ? err2 / ref2 : err2;

    // the path it replaces: dequantize to f16, then an f16 x f16 matmul (row_gemm's descriptors, transposed)
    if (f32_src) {
        std::printf("m=%lld k=%lld n=%lld  impl=%s  nmse=%.3g max_abs=%.3g  (layout check, f32 activations)\n",
                    (long long) m, (long long) k, (long long) n, impl, nmse, maxabs);
        for (void * p : { (void *) qs_d, (void *) dkn_d, (void *) act_d, (void *) act32_d, (void *) dst_d,
                          (void *) w16_d, (void *) dref_d }) {
            sycl::free(p, q);
        }
        return nmse < 1e-10 ? 0 : 2;
    }
    const auto src16_md = dnnl::memory::desc({ n, k }, dt::f16, { k, 1 });
    const auto w16_md = dnnl::memory::desc({ k, m }, dt::f16, { 1, k });
    dnnl::primitive_attr attr16;
    attr16.set_fpmath_mode(dnnl::fpmath_mode::f16);
    dnnl::matmul gemm16(dnnl::matmul::primitive_desc(eng, src16_md, w16_md, dst_md, attr16));
    std::unordered_map<int, dnnl::memory> args16 = {
        { DNNL_ARG_SRC, dnnl::memory(src16_md, eng, act_d) },
        { DNNL_ARG_WEIGHTS, dnnl::memory(w16_md, eng, w16_d) },
        { DNNL_ARG_DST, dnnl::memory(dst_md, eng, dst_d) },
    };
    auto dequant = [&]() {
        q.parallel_for(sycl::range<1>(m * k), [=](sycl::id<1> i) {
            w16_d[i] = (sycl::half) ((float) qs_d[i] * (float) dref_d[i / 32]);
        });
    };
    auto stage = [&]() {
        q.parallel_for(sycl::range<2>(groups, m), [=](sycl::id<2> id) {
            dkn_d[id[0] * m + id[1]] = dref_d[id[1] * groups + id[0]];
        });
    };

    auto time_it = [&](auto && body) {
        body();
        q.wait();
        strm.wait();
        const double t0 = now_ms();
        for (int i = 0; i < iters; ++i) {
            body();
        }
        q.wait();
        strm.wait();
        return (now_ms() - t0) / iters;
    };
    const double t_woq = time_it([&] { stage(); woq.execute(strm, args); });
    const double t_f16 = time_it([&] { dequant(); gemm16.execute(strm, args16); });

    std::printf("m=%lld k=%lld n=%lld  impl=%s  nmse=%.3g max_abs=%.3g  woq %.3f ms  dequant+f16 %.3f ms  (%.2fx)%s\n",
                (long long) m, (long long) k, (long long) n, impl, nmse, maxabs, t_woq, t_f16, t_f16 / t_woq,
                std::strncmp(impl, "ref", 3) == 0 ? "  REFERENCE KERNEL: the candidate declines this" : "");
    for (void * p : { (void *) qs_d, (void *) dkn_d, (void *) act_d, (void *) act32_d, (void *) dst_d, (void *) w16_d,
                      (void *) dref_d }) {
        sycl::free(p, q);
    }
    return nmse < 1e-4 ? 0 : 2;
}

int main(int argc, char ** argv) {
    bool cpu = false;
    std::vector<int64_t> dims;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--cpu")) {
            cpu = true;
        } else {
            dims.push_back(std::atoll(argv[i]));
        }
    }
    std::vector<shape> shapes;
    if (dims.size() == 3) {
        shapes.push_back({ dims[0], dims[1], dims[2] });
    } else if (cpu) {
        shapes = { { 256, 512, 16 }, { 512, 1024, 72 } };
    } else {
        // Llama-3-8B projections at a 512-token batch, and the narrowest batch C80 mode 1 takes
        shapes = { { 4096, 4096, 512 }, { 14336, 4096, 512 }, { 4096, 14336, 512 }, { 1024, 4096, 512 },
                   { 4096, 4096, 65 }, { 14336, 4096, 128 } };
    }
    sycl::queue q(cpu ? sycl::device(sycl::cpu_selector_v) : sycl::device(sycl::gpu_selector_v),
                  sycl::property::queue::in_order());
    std::printf("device: %s, driver %s\n", q.get_device().get_info<sycl::info::device::name>().c_str(),
                q.get_device().get_info<sycl::info::device::driver_version>().c_str());
    auto eng  = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    auto strm = dnnl::sycl_interop::make_stream(eng, q);
    int  rc   = 0;
    for (const auto & s : shapes) {
        rc |= run(q, eng, strm, s, cpu ? 2 : 20, cpu);
    }
    std::printf(rc == 0 ? "PASS\n" : "FAIL (1 = declined, 2 = wrong results)\n");
    return rc;
}
