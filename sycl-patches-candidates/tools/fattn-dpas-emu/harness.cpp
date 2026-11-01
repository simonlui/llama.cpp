// CPU check of fattn-dpas (candidate C60) at sub-group 16 and 8 with emulated DPAS against an FP64 reference.
// Built and run by run.sh; argument: one case index to run only that case.
#include "fattn-dpas-emu.cpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int g_ggml_sycl_fa_dpas = 1;

struct tcase {
    bool  q8;
    int   D, G, ne01, ne11, nkvh, ne03;
    bool  mask, sinks;
    int   nsplit;  // requested key splits
};

static size_t row_bytes(bool q8, int D) { return (size_t) (D / 32) * (q8 ? 34 : 18); }

static void quant_row(bool q8, const float * x, uint8_t * out, float * deq, int D) {
    for (int b = 0; b < D / 32; ++b) {
        const float * xb = x + 32 * b;
        uint8_t *     ob = out + b * (q8 ? 34 : 18);
        if (q8) {
            float amax = 0.0f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(xb[i]));
            const float      d  = amax / 127.0f;
            const float      id = d ? 1.0f / d : 0.0f;
            const sycl::half dh = d;
            memcpy(ob, &dh, 2);
            const float dd = (float) dh;
            for (int i = 0; i < 32; ++i) {
                const int8_t q = (int8_t) std::lround(xb[i] * id);
                ob[2 + i]      = (uint8_t) q;
                deq[32 * b + i] = q * dd;
            }
        } else {
            float amax = 0.0f, mx = 0.0f;
            for (int i = 0; i < 32; ++i) if (std::fabs(xb[i]) > amax) { amax = std::fabs(xb[i]); mx = xb[i]; }
            const float      d  = mx / -8.0f;
            const float      id = d ? 1.0f / d : 0.0f;
            const sycl::half dh = d;
            memcpy(ob, &dh, 2);
            const float dd = (float) dh;
            for (int j = 0; j < 16; ++j) {
                const int q0 = std::min(15, (int) (xb[j] * id + 8.5f));
                const int q1 = std::min(15, (int) (xb[j + 16] * id + 8.5f));
                ob[2 + j]         = (uint8_t) (q0 | (q1 << 4));
                deq[32 * b + j]      = (q0 - 8) * dd;
                deq[32 * b + j + 16] = (q1 - 8) * dd;
            }
        }
    }
}

template <int SG, int D, int NR16, int KG, int TC, int ST, bool Q8>
static void launch(sycl::queue & q, fa_dpas_params p, int nkvh, int ne03, int nsplit_req) {
    constexpr int TK = 16 * KG;
    p.nq  = std::max(1, std::min(p.ne01, 16 * NR16 / p.G));
    p.nqc = (p.ne01 + p.nq - 1) / p.nq;
    const int ntiles = (p.ne11 + TK - 1) / TK;
    int nsplit = std::max(1, std::min(nsplit_req, ntiles));
    p.chunk    = ((ntiles + nsplit - 1) / nsplit) * TK;
    nsplit     = (p.ne11 + p.chunk - 1) / p.chunk;
    p.nsplit   = nsplit;
    if (nsplit > 1) {
        p.parts = sycl::malloc_shared<float>((size_t) nsplit * p.ne01 * p.ne02 * ne03 * D, q);
        p.meta  = sycl::malloc_shared<sycl::float2>((size_t) nsplit * p.ne01 * p.ne02 * ne03, q);
    }
    fattn_dpas_kernel<SG, D, NR16, KG, TC, ST, false, Q8>(p, nkvh, ne03, &q);
    q.wait_and_throw();
    if (nsplit > 1) {  // host combine
        const int64_t nrows = (int64_t) p.ne01 * p.ne02 * ne03;
        for (int64_t jdu = 0; jdu < nrows; ++jdu) {
            float M = -INFINITY;
            for (int s = 0; s < nsplit; ++s) M = std::max(M, p.meta[jdu * nsplit + s].x());
            double den = 0.0;
            std::vector<double> num(D, 0.0);
            for (int s = 0; s < nsplit; ++s) {
                const double f = std::exp((double) p.meta[jdu * nsplit + s].x() - M);
                den += f * p.meta[jdu * nsplit + s].y();
                for (int d = 0; d < D; ++d) num[d] += f * p.parts[(jdu * nsplit + s) * D + d];
            }
            for (int d = 0; d < D; ++d) p.dst[jdu * D + d] = den > 0 ? (float) (num[d] / den) : 0.0f;
        }
        sycl::free(p.parts, q);
        sycl::free(p.meta, q);
    }
}

// the kernel choice of fattn_dpas_dispatch
template <int SG, bool Q8, int D>
static void dispatch(sycl::queue & q, const fa_dpas_params & p, int nkvh, int ne03, int nsplit, bool aligned16) {
    const bool rows16 = p.ne01 * p.G <= 16;
    if constexpr (D == 256) {
        if (p.ne01 == 1 && p.G <= 16 && aligned16) {
            launch<SG, 256, 1, 8, 1, 1, Q8>(q, p, nkvh, ne03, nsplit);
            return;
        }
    }
    if constexpr (Q8 && D == 128) {
        if (rows16) launch<SG, 128, 1, 16, 1, 2, Q8>(q, p, nkvh, ne03, nsplit);
        else        launch<SG, 128, 2, 8, 2, 2, Q8>(q, p, nkvh, ne03, nsplit);
        return;
    }
    constexpr int KG16 = D == 512 ? 8 : 16;
    if (rows16) launch<SG, D, 1, KG16, 1, 0, Q8>(q, p, nkvh, ne03, nsplit);
    else        launch<SG, D, 2, 8, 2, 0, Q8>(q, p, nkvh, ne03, nsplit);
}

template <int SG, bool Q8>
static void run_d(sycl::queue & q, int D, const fa_dpas_params & p, int nkvh, int ne03, int nsplit, bool a16) {
    switch (D) {
        case 64:  dispatch<SG, Q8, 64>(q, p, nkvh, ne03, nsplit, a16); break;
        case 96:  dispatch<SG, Q8, 96>(q, p, nkvh, ne03, nsplit, a16); break;
        case 128: dispatch<SG, Q8, 128>(q, p, nkvh, ne03, nsplit, a16); break;
        case 256: dispatch<SG, Q8, 256>(q, p, nkvh, ne03, nsplit, a16); break;
        default: abort();
    }
}

int main(int argc, char ** argv) {
    sycl::queue q{ sycl::cpu_selector_v };
    std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    std::vector<tcase> cases;
    for (bool q8 : { true, false })
        for (int D : { 64, 96, 128, 256 })
            for (auto ng : std::vector<std::pair<int, int>>{ { 1, 1 }, { 1, 4 }, { 2, 8 }, { 3, 8 }, { 5, 4 }, { 1, 32 }, { 7, 2 } })
                for (int ne11 : { 77, 300 }) {
                    const int  i = (int) cases.size();
                    cases.push_back({ q8, D, ng.second, ng.first, ne11, 2, 1 + (i % 3 == 0), i % 2 == 0, i % 5 == 0,
                                      (i % 4 == 1) ? 3 : 1 });
                }
    const int only = argc > 1 ? atoi(argv[1]) : -1;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int fails = 0;
    double worst_ref = 0.0, worst_x = 0.0;
    for (size_t ci = 0; ci < cases.size(); ++ci) {
        if (only >= 0 && (int) ci != only) continue;
        const tcase & c = cases[ci];
        const int D = c.D, G = c.G, ne01 = c.ne01, ne11 = c.ne11, nkvh = c.nkvh, ne03 = c.ne03, ne02 = nkvh * G;
        const size_t rb = row_bytes(c.q8, D);
        // Q f32 [D, ne01, ne02, ne03], K/V [rb, ne11, nkvh, ne03]
        std::vector<float> Qh((size_t) D * ne01 * ne02 * ne03), Kd((size_t) D * ne11 * nkvh * ne03), Vd(Kd.size());
        for (auto & v : Qh) v = nd(rng);
        std::vector<float> tmp(D);
        float * Q  = sycl::malloc_shared<float>(Qh.size(), q);
        uint8_t * K = sycl::malloc_shared<uint8_t>(rb * ne11 * nkvh * ne03 + 16, q);
        uint8_t * V = sycl::malloc_shared<uint8_t>(rb * ne11 * nkvh * ne03 + 16, q);
        memcpy(Q, Qh.data(), Qh.size() * 4);
        for (int64_t r = 0; r < (int64_t) ne11 * nkvh * ne03; ++r) {
            for (int d = 0; d < D; ++d) tmp[d] = nd(rng);
            quant_row(c.q8, tmp.data(), K + r * rb, &Kd[r * D], D);
            for (int d = 0; d < D; ++d) tmp[d] = nd(rng);
            quant_row(c.q8, tmp.data(), V + r * rb, &Vd[r * D], D);
        }
        const int ne11p = (ne11 + 63) / 64 * 64, ne01p = (ne01 + 7) / 8 * 8;
        sycl::half * M = c.mask ? sycl::malloc_shared<sycl::half>((size_t) ne11p * ne01p, q) : nullptr;
        if (M) {
            for (int i = 0; i < ne01p; ++i)
                for (int j = 0; j < ne11p; ++j) {
                    const bool masked = j > ne11 - ne01 + i;  // causal
                    M[(size_t) i * ne11p + j] = masked ? sycl::half(-INFINITY) : sycl::half(-0.25f * (j % 3));
                }
        }
        float * S = c.sinks ? sycl::malloc_shared<float>(ne02, q) : nullptr;
        if (S) for (int h = 0; h < ne02; ++h) S[h] = nd(rng);
        const float scale = 1.0f / std::sqrt((float) D);
        // FP64 reference
        std::vector<double> ref((size_t) ne03 * ne01 * ne02 * D);
        for (int s = 0; s < ne03; ++s)
            for (int ci2 = 0; ci2 < ne01; ++ci2)
                for (int h = 0; h < ne02; ++h) {
                    const int kvh = h / G;
                    const float * qr = &Qh[(((size_t) s * ne02 + h) * ne01 + ci2) * D];
                    std::vector<double> sc(ne11);
                    double mx = S ? S[h] : -INFINITY;
                    for (int j = 0; j < ne11; ++j) {
                        const float * kr = &Kd[(((size_t) s * nkvh + kvh) * ne11 + j) * D];
                        double acc = 0;
                        for (int d = 0; d < D; ++d) acc += (double) qr[d] * scale * kr[d];
                        if (M) acc += (double) (float) M[(size_t) ci2 * ne11p + j];
                        sc[j] = acc;
                        mx = std::max(mx, acc);
                    }
                    double den = S ? std::exp(S[h] - mx) : 0.0;
                    std::vector<double> o(D, 0.0);
                    for (int j = 0; j < ne11; ++j) {
                        const double w = std::exp(sc[j] - mx);
                        den += w;
                        const float * vr = &Vd[(((size_t) s * nkvh + kvh) * ne11 + j) * D];
                        for (int d = 0; d < D; ++d) o[d] += w * vr[d];
                    }
                    const size_t jdu = ((size_t) s * ne01 + ci2) * ne02 + h;
                    for (int d = 0; d < D; ++d) ref[jdu * D + d] = o[d] / den;
                }
        fa_dpas_params p{};
        p.Q = (const char *) Q; p.K = (const char *) K; p.V = (const char *) V; p.mask = (const char *) M;
        p.sinks = S; p.scale = scale; p.G = G; p.ne01 = ne01; p.ne02 = ne02; p.ne11 = ne11; p.ne33 = 1;
        p.nb01 = (int64_t) D * 4; p.nb02 = p.nb01 * ne01; p.nb03 = p.nb02 * ne02;
        p.nb11 = rb; p.nb12 = rb * ne11; p.nb13 = p.nb12 * nkvh;
        p.nb21 = p.nb11; p.nb22 = p.nb12; p.nb23 = p.nb13;
        p.nb31 = (int64_t) ne11p * 2; p.nb33 = p.nb31 * ne01p;
        const bool a16 = rb % 16 == 0;
        std::vector<float> out[2];
        for (int w = 0; w < 2; ++w) {
            float * dst = sycl::malloc_shared<float>(ref.size(), q);
            for (size_t i = 0; i < ref.size(); ++i) dst[i] = NAN;
            p.dst = dst;
            if (w == 0) { c.q8 ? run_d<16, true>(q, D, p, nkvh, ne03, c.nsplit, a16) : run_d<16, false>(q, D, p, nkvh, ne03, c.nsplit, a16); }
            else        { c.q8 ? run_d<8, true>(q, D, p, nkvh, ne03, c.nsplit, a16)  : run_d<8, false>(q, D, p, nkvh, ne03, c.nsplit, a16); }
            out[w].assign(dst, dst + ref.size());
            sycl::free(dst, q);
        }
        double e16 = 0, e8 = 0, ex = 0, nr = 0, nx = 0;
        bool nan = false;
        for (size_t i = 0; i < ref.size(); ++i) {
            nan = nan || !std::isfinite(out[0][i]) || !std::isfinite(out[1][i]);
            e16 += (out[0][i] - ref[i]) * (out[0][i] - ref[i]);
            e8  += (out[1][i] - ref[i]) * (out[1][i] - ref[i]);
            ex  += ((double) out[1][i] - out[0][i]) * ((double) out[1][i] - out[0][i]);
            nr  += ref[i] * ref[i];
            nx  += (double) out[0][i] * out[0][i];
        }
        const double n16 = e16 / nr, n8 = e8 / nr, nx8 = ex / nx;
        const bool ok = !nan && n16 < 5e-4 && n8 < 5e-4 && nx8 < 1e-10;
        worst_ref = std::max({ worst_ref, n16, n8 });
        worst_x   = std::max(worst_x, nx8);
        fails += !ok;
        std::printf("%-4s %3zu %s D=%3d G=%2d ne01=%d ne11=%3d nseq=%d mask=%d sinks=%d split=%d  nmse sg16=%.2e sg8=%.2e  sg8-vs-sg16=%.2e%s\n",
                    ok ? "ok" : "FAIL", ci, c.q8 ? "q8_0" : "q4_0", D, G, ne01, ne11, ne03, c.mask, c.sinks, c.nsplit,
                    n16, n8, nx8, nan ? " NaN" : "");
        std::fflush(stdout);
        sycl::free(Q, q); sycl::free(K, q); sycl::free(V, q);
        if (M) sycl::free(M, q);
        if (S) sycl::free(S, q);
    }
    std::printf("%s: %d failures, worst NMSE vs FP64 %.2e, worst sg8 vs sg16 %.2e\n", fails ? "FAILED" : "PASSED", fails, worst_ref, worst_x);
    return fails != 0;
}
