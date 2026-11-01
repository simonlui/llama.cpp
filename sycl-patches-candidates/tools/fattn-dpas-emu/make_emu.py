# Copy fattn-dpas.cpp (with C60 applied) and swap its inline-vISA DPAS helpers for a sub-group emulation of the
# hardware semantics, so the kernel runs on the CPU OpenCL device. The helpers stay out of line: inlined into every
# unrolled call site they make the CPU JIT compile take over 15 minutes.
#   python3 make_emu.py <tree>/ggml/src/ggml-sycl/fattn-dpas.cpp <out.cpp>
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
a = s.index('// d = a x b. a: 8 rows x 32 s8. b: 32 x SG s4')
b = s.index('// Block b of a row = f16 scale')
emu = r'''// ---- emulated DPAS, CPU test only (sycl-patches-candidates/tools/fattn-dpas-emu) -------------------------------------------------------------------------------
// Register model, the same for both widths: A (src2) is the 256-byte register region of the per-lane vector a, element
// e of lane l at byte (e SG + l) sizeof(element), row r = bytes 32 r .. 32 r + 31, K index k = byte k of the row (f16:
// half k). B (src1) dword d of column n is element d of lane n: K 4d..4d+3 (s8), 8d..8d+7 (s4, low nibble first),
// 2d, 2d+1 (f16, low half first). D[r][n] is element r of lane n.
template <int SG, typename TA> __attribute__((noinline)) static void fa_emu_flat_a(const TA & a, uint8_t * A) {
    auto          sg = sycl::ext::oneapi::this_work_item::get_sub_group();
    constexpr int ES = 256 / (8 * SG);
#pragma nounroll
    for (int e = 0; e < 8; ++e) {
        for (int l = 0; l < SG; ++l) {
            uint32_t v;
            if constexpr (ES == 2) {
                v = (uint16_t) sycl::select_from_group(sg, (short) a[e], l);
            } else {
                v = (uint32_t) sycl::select_from_group(sg, (int) a[e], l);
            }
            for (int bi = 0; bi < ES; ++bi) {
                A[(e * SG + l) * ES + bi] = (uint8_t) (v >> (8 * bi));
            }
        }
    }
}
template <int SG> __attribute__((noinline)) static void fa_dpas_s8_s4(const fa_dpas_a<SG> & a, const fa_dpas_int4 & b, fa_dpas_int8 & d) {
    uint8_t A[256];
    fa_emu_flat_a<SG>(a, A);
#pragma nounroll
    for (int r = 0; r < 8; ++r) {
        int acc = 0;
        for (int k = 0; k < 32; ++k) {
            const int nib = (int) (((uint32_t) b[k / 8] >> (4 * (k % 8))) & 0xF);
            acc += (int) (int8_t) A[r * 32 + k] * (nib >= 8 ? nib - 16 : nib);
        }
        d[r] = acc;
    }
}
template <int SG> __attribute__((noinline)) static void fa_dpas_s8_s8(const fa_dpas_a<SG> & a, const fa_dpas_int8 & b, fa_dpas_int8 & d) {
    uint8_t A[256];
    fa_emu_flat_a<SG>(a, A);
#pragma nounroll
    for (int r = 0; r < 8; ++r) {
        int acc = 0;
        for (int k = 0; k < 32; ++k) {
            acc += (int) (int8_t) A[r * 32 + k] * (int) (int8_t) (((uint32_t) b[k / 4] >> (8 * (k % 4))) & 0xFF);
        }
        d[r] = acc;
    }
}
template <int SG> __attribute__((noinline)) static void fa_dpas_f16(const fa_dpas_a<SG> & a, const fa_dpas_int8 & b, fa_dpas_float8 & acc) {
    uint8_t A[256];
    fa_emu_flat_a<SG>(a, A);
#pragma nounroll
    for (int r = 0; r < 8; ++r) {
        float s = acc[r];
        for (int k = 0; k < 16; ++k) {
            const uint16_t ah = (uint16_t) (A[r * 32 + 2 * k] | (A[r * 32 + 2 * k + 1] << 8));
            const uint16_t bh = (uint16_t) (((uint32_t) b[k / 2] >> (16 * (k % 2))) & 0xFFFF);
            s += (float) sycl::bit_cast<sycl::half>(ah) * (float) sycl::bit_cast<sycl::half>(bh);
        }
        acc[r] = s;
    }
}

'''
s = s[:a] + emu + s[b:]
assert '__asm__' not in s
open(dst, 'w').write(s)
print('emu written')
