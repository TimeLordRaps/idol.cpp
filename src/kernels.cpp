#include "kernels.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace idol::detail {

namespace {

// IDOL_PROFILE=1 makes the process print, at exit, how long the GEMMs, the attention and the layer norms took in total.
// Off by default; the check costs one branch per kernel call.
struct Profile {
    double gemm = 0, attention = 0, norm = 0;
    long gemm_calls = 0;
    bool on = std::getenv("IDOL_PROFILE") != nullptr;
    ~Profile() {
        if (on)
            std::fprintf(stderr, "profile: gemm %.3f ms (%ld calls), attention %.3f ms, layer_norm %.3f ms\n", gemm,
                         gemm_calls, attention, norm);
    }
};
Profile g_profile;

struct Scope {
    double* slot;
    std::chrono::steady_clock::time_point start;
    explicit Scope(double* s) : slot(g_profile.on ? s : nullptr) {
        if (slot) start = std::chrono::steady_clock::now();
    }
    ~Scope() {
        if (slot) *slot += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }
};

WeightT to_weight(float v) {
#ifdef IDOL_HALF_WEIGHTS
    return _cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT);
#else
    return v;
#endif
}

// Eight packed weights as fp32.
__m256 load8(const float* p) { return _mm256_loadu_ps(p); }
[[maybe_unused]] __m256 load8(const std::uint16_t* p) {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

float dot(const float* a, const float* b, int n) {
    __m256 acc = _mm256_setzero_ps();
    int k = 0;
    for (; k + 8 <= n; k += 8) acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc);
    float sum = hsum(acc);
    for (; k < n; ++k) sum += a[k] * b[k];
    return sum;
}

constexpr int kKChunk = 32;  // depth summed per microkernel call; see gemm_packed

// Six rows of A (row stride lda) against one 16-column panel of W over the full K. Twelve accumulators stay in
// registers; the six-by-sixteen tile is written out once at the end.
template <typename W>
void microkernel_6x16(const float* a, int lda, const W* b, int K, float (*tile)[kCols]) {
    const float* a1 = a + lda;
    const float* a2 = a1 + lda;
    const float* a3 = a2 + lda;
    const float* a4 = a3 + lda;
    const float* a5 = a4 + lda;
    __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00;
    __m256 c30 = c00, c31 = c00, c40 = c00, c41 = c00, c50 = c00, c51 = c00;
    for (int k = 0; k < K; ++k) {
        const __m256 b0 = load8(b + (std::size_t)k * kCols);
        const __m256 b1 = load8(b + (std::size_t)k * kCols + 8);
        __m256 av = _mm256_broadcast_ss(a + k);
        c00 = _mm256_fmadd_ps(av, b0, c00);
        c01 = _mm256_fmadd_ps(av, b1, c01);
        av = _mm256_broadcast_ss(a1 + k);
        c10 = _mm256_fmadd_ps(av, b0, c10);
        c11 = _mm256_fmadd_ps(av, b1, c11);
        av = _mm256_broadcast_ss(a2 + k);
        c20 = _mm256_fmadd_ps(av, b0, c20);
        c21 = _mm256_fmadd_ps(av, b1, c21);
        av = _mm256_broadcast_ss(a3 + k);
        c30 = _mm256_fmadd_ps(av, b0, c30);
        c31 = _mm256_fmadd_ps(av, b1, c31);
        av = _mm256_broadcast_ss(a4 + k);
        c40 = _mm256_fmadd_ps(av, b0, c40);
        c41 = _mm256_fmadd_ps(av, b1, c41);
        av = _mm256_broadcast_ss(a5 + k);
        c50 = _mm256_fmadd_ps(av, b0, c50);
        c51 = _mm256_fmadd_ps(av, b1, c51);
    }
    _mm256_store_ps(tile[0], c00);
    _mm256_store_ps(tile[0] + 8, c01);
    _mm256_store_ps(tile[1], c10);
    _mm256_store_ps(tile[1] + 8, c11);
    _mm256_store_ps(tile[2], c20);
    _mm256_store_ps(tile[2] + 8, c21);
    _mm256_store_ps(tile[3], c30);
    _mm256_store_ps(tile[3] + 8, c31);
    _mm256_store_ps(tile[4], c40);
    _mm256_store_ps(tile[4] + 8, c41);
    _mm256_store_ps(tile[5], c50);
    _mm256_store_ps(tile[5] + 8, c51);
}

// Serial tile of C = A[M x K] * W[N x K]^T + bias for the attention scores, which are activations and not packed.
void gemm_block(const float* a, int K, const float* w, int N, float* c, int m0, int m1, int n0, int n1) {
    for (int n = n0; n < n1; ++n) {
        const float* row = w + (std::size_t)n * K;
        for (int m = m0; m < m1; ++m) c[(std::size_t)m * N + n] = dot(a + (std::size_t)m * K, row, K);
    }
}

}  // namespace

const WeightT* PackedLinear::panel(int n0) const {
    return panels.data() + (std::size_t)(n0 / kCols) * in * kCols;
}

const float* PackedLinear::bias_at(int n0) const { return bias.data() + n0; }

std::size_t PackedLinear::bytes() const {
    return panels.size() * sizeof(WeightT) + qdata.size() + bias.size() * sizeof(float);
}

PackedLinear pack_linear(const std::vector<float>& weight, const std::vector<float>& bias, int out, int in) {
    PackedLinear p;
    p.out = out;
    p.in = in;
    const int npanels = (out + kCols - 1) / kCols;
    p.panels.assign((std::size_t)npanels * in * kCols, to_weight(0.0f));
    for (int q = 0; q < npanels; ++q)
        for (int k = 0; k < in; ++k)
            for (int c = 0; c < kCols; ++c) {
                const int n = q * kCols + c;
                if (n < out) p.panels[((std::size_t)q * in + k) * kCols + c] = to_weight(weight[(std::size_t)n * in + k]);
            }
    p.bias.assign((std::size_t)npanels * kCols, 0.0f);
    for (int n = 0; n < out; ++n) p.bias[n] = bias[n];
    return p;
}

namespace {

constexpr int kQuantHeader = 64;  // 16 half scales and 16 half minimums

bool quant_bits_ok(int bits) { return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8; }

// Eight rows of k from bit planes of widths W0 and W1 (W1 = 0 for a single plane), as 8 x 16 fp32 values.
template <int W0, int W1>
inline void dequant8(const std::uint8_t* p, __m256 s0, __m256 s1, __m256 m0, __m256 m1, float* out) {
    const __m128i* v = reinterpret_cast<const __m128i*>(p);
    constexpr int R0 = 8 / W0;
    constexpr int R1 = W1 ? 8 / W1 : 1;
    const __m128i mask0 = _mm_set1_epi8(static_cast<char>((1 << W0) - 1));
    const __m128i mask1 = _mm_set1_epi8(static_cast<char>(W1 ? (1 << W1) - 1 : 0));
#pragma GCC unroll 8
    for (int r = 0; r < 8; ++r) {
        __m128i x = _mm_and_si128(_mm_srl_epi16(_mm_loadu_si128(v + r / R0), _mm_cvtsi32_si128(W0 * (r % R0))), mask0);
        if (W1) {
            const __m128i y = _mm_and_si128(
                _mm_srl_epi16(_mm_loadu_si128(v + W0 + r / R1), _mm_cvtsi32_si128(W1 * (r % R1))), mask1);
            x = _mm_or_si128(x, _mm_sll_epi16(y, _mm_cvtsi32_si128(W0)));
        }
        const __m256 lo = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(x));
        const __m256 hi = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(x, 8)));
        _mm256_storeu_ps(out + r * kCols, _mm256_fmadd_ps(lo, s0, m0));
        _mm256_storeu_ps(out + r * kCols + 8, _mm256_fmadd_ps(hi, s1, m1));
    }
}

// Thirty-two rows of k, starting at k0 (a multiple of 32), of one quantized panel as 32 x 16 fp32 values.
void dequant_chunk(const std::uint8_t* panel, int bits, int group, int k0, float* out) {
    const std::uint8_t* g = panel + (std::size_t)(k0 / group) * (kQuantHeader + 2 * (std::size_t)group * bits);
    const __m256 s0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(g)));
    const __m256 s1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(g + 16)));
    const __m256 m0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(g + 32)));
    const __m256 m1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(g + 48)));
    const std::uint8_t* codes = g + kQuantHeader + (std::size_t)((k0 % group) / 8) * 16 * bits;
    for (int c = 0; c < 4; ++c) {
        const std::uint8_t* p = codes + (std::size_t)c * 16 * bits;
        float* o = out + c * 8 * kCols;
        switch (bits) {
            case 2: dequant8<2, 0>(p, s0, s1, m0, m1, o); break;
            case 3: dequant8<2, 1>(p, s0, s1, m0, m1, o); break;
            case 4: dequant8<4, 0>(p, s0, s1, m0, m1, o); break;
            case 5: dequant8<4, 1>(p, s0, s1, m0, m1, o); break;
            case 6: dequant8<4, 2>(p, s0, s1, m0, m1, o); break;
            default: dequant8<8, 0>(p, s0, s1, m0, m1, o); break;
        }
    }
}

// Bias, residual, GELU and the store for one 6 x 16 tile (tile rows kCols apart) whose top-left output is (m0, n0).
void finish_tile(const float* tile, int m0, int n0, int mr, int nc, int n, const float* bias, const float* residual,
                 bool gelu, float* c) {
    for (int i = 0; i < mr; ++i) {
        const float* t = tile + (std::size_t)i * kCols;
        float* crow = c + (std::size_t)(m0 + i) * n + n0;
        const float* rrow = residual ? residual + (std::size_t)(m0 + i) * n + n0 : nullptr;
        if (nc == kCols) {
            __m256 v0 = _mm256_add_ps(_mm256_load_ps(t), _mm256_loadu_ps(bias));
            __m256 v1 = _mm256_add_ps(_mm256_load_ps(t + 8), _mm256_loadu_ps(bias + 8));
            if (rrow) {
                v0 = _mm256_add_ps(v0, _mm256_loadu_ps(rrow));
                v1 = _mm256_add_ps(v1, _mm256_loadu_ps(rrow + 8));
            }
            if (gelu) {
                v0 = gelu256(v0);
                v1 = gelu256(v1);
            }
            _mm256_storeu_ps(crow, v0);
            _mm256_storeu_ps(crow + 8, v1);
        } else {
            // Ragged right edge (the 14-wide output head): scalar, with the same epilogue.
            for (int j = 0; j < nc; ++j) {
                float v = t[j] + bias[j];
                if (rrow) v += rrow[j];
                if (gelu) v = gelu_scalar(v);
                crow[j] = v;
            }
        }
    }
}

// Rows of A handled per task by the quantized GEMM, so one dequantized chunk of the panel serves kQuantBlock tiles.
constexpr int kQuantBlock = 8;

void gemm_quant(const float* a, int m, const PackedLinear& w, int n_offset, int n, const float* residual, bool gelu,
                float* c) {
    const int K = w.in;
    const int mpanels = (m + kRows - 1) / kRows, npanels = (n + kCols - 1) / kCols;
    const int mtasks = (mpanels + kQuantBlock - 1) / kQuantBlock;
    const std::size_t panel_bytes = quant_panel_bytes(K, w.qbits, w.qgroup);
#pragma omp parallel for collapse(2) schedule(static)
    for (int np = 0; np < npanels; ++np)
        for (int mt = 0; mt < mtasks; ++mt) {
            alignas(32) float tiles[kQuantBlock][kRows][kCols];
            alignas(32) float chunk[kKChunk * kCols];
            const int mp0 = mt * kQuantBlock, count = std::min(kQuantBlock, mpanels - mp0);
            const int n0 = np * kCols, nc = std::min(kCols, n - n0);
            const std::uint8_t* panel = w.qdata.data() + (std::size_t)(n_offset / kCols + np) * panel_bytes;
            const float* rows[kQuantBlock];
            static thread_local std::vector<float> pad;
            for (int j = 0; j < count; ++j) {
                const int m0 = (mp0 + j) * kRows, mr = std::min(kRows, m - m0);
                rows[j] = a + (std::size_t)m0 * K;
                if (mr < kRows) {
                    // Only the last tile of A is ragged; its missing rows are zero-padded in a per-thread copy.
                    pad.assign((std::size_t)kRows * K, 0.0f);
                    std::memcpy(pad.data(), rows[j], sizeof(float) * (std::size_t)mr * K);
                    rows[j] = pad.data();
                }
            }
            // The same chunked summation as gemm_packed, so a quantized layer and a dense layer holding the same
            // dequantized weights give the same result.
            for (int k0 = 0; k0 < K; k0 += kKChunk) {
                dequant_chunk(panel, w.qbits, w.qgroup, k0, chunk);
                for (int j = 0; j < count; ++j) {
                    if (k0 == 0) {
                        microkernel_6x16(rows[j], K, chunk, kKChunk, tiles[j]);
                        continue;
                    }
                    alignas(32) float part[kRows][kCols];
                    microkernel_6x16(rows[j] + k0, K, chunk, kKChunk, part);
                    for (int i = 0; i < kRows; ++i) {
                        _mm256_store_ps(tiles[j][i], _mm256_add_ps(_mm256_load_ps(tiles[j][i]), _mm256_load_ps(part[i])));
                        _mm256_store_ps(tiles[j][i] + 8,
                                        _mm256_add_ps(_mm256_load_ps(tiles[j][i] + 8), _mm256_load_ps(part[i] + 8)));
                    }
                }
            }
            const float* bias = w.bias_at(n_offset + n0);
            for (int j = 0; j < count; ++j) {
                const int m0 = (mp0 + j) * kRows;
                finish_tile(&tiles[j][0][0], m0, n0, std::min(kRows, m - m0), nc, n, bias, residual, gelu, c);
            }
        }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------
// Integer kernel for quantized layers, on CPUs with AVX-VNNI (vpdpbusd: 32 multiply-adds per instruction).
//
// The activations are quantized to int8 per row and per weight group, the weight codes are unpacked to bytes and
// regrouped four rows of k per 32-bit lane, and each group's products are summed in int32. With w = min + code * scale,
// a group contributes  scale_w * scale_a * sum(qa * code) + min_w * scale_a * sum(qa)  to the output. The activation
// rounding is the one error this path adds on top of the weight quantization.
// ---------------------------------------------------------------------------------------------------------------------
#if defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12) && !defined(IDOL_NO_VNNI_PATH)
#define IDOL_VNNI_PATH 1
#endif

namespace {

#ifdef IDOL_VNNI_PATH

constexpr int kMaxGroup = 128;

// Eight rows of k from bit planes, as eight vectors of 16 unsigned codes (one per column).
template <int W0, int W1>
inline void decode8(const std::uint8_t* p, __m128i* rows) {
    const __m128i* v = reinterpret_cast<const __m128i*>(p);
    constexpr int R0 = 8 / W0;
    constexpr int R1 = W1 ? 8 / W1 : 1;
    const __m128i mask0 = _mm_set1_epi8(static_cast<char>((1 << W0) - 1));
    const __m128i mask1 = _mm_set1_epi8(static_cast<char>(W1 ? (1 << W1) - 1 : 0));
#pragma GCC unroll 8
    for (int r = 0; r < 8; ++r) {
        __m128i x = _mm_and_si128(_mm_srl_epi16(_mm_loadu_si128(v + r / R0), _mm_cvtsi32_si128(W0 * (r % R0))), mask0);
        if (W1) {
            const __m128i y = _mm_and_si128(
                _mm_srl_epi16(_mm_loadu_si128(v + W0 + r / R1), _mm_cvtsi32_si128(W1 * (r % R1))), mask1);
            x = _mm_or_si128(x, _mm_sll_epi16(y, _mm_cvtsi32_si128(W0)));
        }
        rows[r] = x;
    }
}

// Four rows of 16 codes to 64 bytes: lane (column c, bytes k0..k3) at byte 4 * c.
inline void transpose4(const __m128i* r, std::uint8_t* out) {
    const __m128i a = _mm_unpacklo_epi8(r[0], r[1]), b = _mm_unpackhi_epi8(r[0], r[1]);
    const __m128i c = _mm_unpacklo_epi8(r[2], r[3]), d = _mm_unpackhi_epi8(r[2], r[3]);
    __m128i* o = reinterpret_cast<__m128i*>(out);
    _mm_store_si128(o, _mm_unpacklo_epi16(a, c));
    _mm_store_si128(o + 1, _mm_unpackhi_epi16(a, c));
    _mm_store_si128(o + 2, _mm_unpacklo_epi16(b, d));
    _mm_store_si128(o + 3, _mm_unpackhi_epi16(b, d));
}

// Unpacks one group of a panel into group / 4 blocks of 64 bytes.
void decode_group(const std::uint8_t* codes, int bits, int group, std::uint8_t* out) {
    for (int c = 0; c < group / 8; ++c) {
        const std::uint8_t* p = codes + (std::size_t)c * 16 * bits;
        __m128i rows[8];
        switch (bits) {
            case 2: decode8<2, 0>(p, rows); break;
            case 3: decode8<2, 1>(p, rows); break;
            case 4: decode8<4, 0>(p, rows); break;
            case 5: decode8<4, 1>(p, rows); break;
            case 6: decode8<4, 2>(p, rows); break;
            default: decode8<8, 0>(p, rows); break;
        }
        transpose4(rows, out + (std::size_t)c * 128);
        transpose4(rows + 4, out + (std::size_t)c * 128 + 64);
    }
}

// qa[m][k] int8, scale[m][g] and qsum[m][g] for rows [0, mpad); rows past m are zero.
void quantize_activations(const float* a, int m, int mpad, int K, int group, std::int8_t* qa, float* scale,
                          std::int32_t* qsum) {
    const int ng = K / group;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < mpad; ++r) {
        for (int g = 0; g < ng; ++g) {
            std::int8_t* q = qa + (std::size_t)r * K + (std::size_t)g * group;
            if (r >= m) {
                std::memset(q, 0, (std::size_t)group);
                scale[(std::size_t)r * ng + g] = 0.0f;
                qsum[(std::size_t)r * ng + g] = 0;
                continue;
            }
            const float* x = a + (std::size_t)r * K + (std::size_t)g * group;
            __m256 mx = _mm256_setzero_ps();
            const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
            for (int k = 0; k < group; k += 8) mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(x + k), absmask));
            __m128 h = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
            h = _mm_max_ps(h, _mm_movehl_ps(h, h));
            h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
            const float amax = _mm_cvtss_f32(h);
            const float s = amax / 127.0f, inv = amax > 0.0f ? 127.0f / amax : 0.0f;
            std::int32_t sum = 0;
            for (int k = 0; k < group; k += 8) {
                const __m256i qi = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(x + k), _mm256_set1_ps(inv)));
                alignas(32) std::int32_t t[8];
                _mm256_store_si256(reinterpret_cast<__m256i*>(t), qi);
                for (int j = 0; j < 8; ++j) {
                    q[k + j] = static_cast<std::int8_t>(t[j]);
                    sum += t[j];
                }
            }
            scale[(std::size_t)r * ng + g] = s;
            qsum[(std::size_t)r * ng + g] = sum;
        }
    }
}

// One task: output panel np, row-tiles [mp0, mp0 + count). Float tiles are accumulated group by group.
__attribute__((target("avx2,fma,avxvnni"))) void vnni_task(const PackedLinear& w, const std::uint8_t* panel, int K,
                                                           const std::int8_t* qa, const float* scale_a,
                                                           const std::int32_t* qsum, int mp0, int count,
                                                           float (*tiles)[kRows][kCols]) {
    const int group = w.qgroup, ng = K / group;
    alignas(64) std::uint8_t codes[kMaxGroup * kCols];
    for (int j = 0; j < count; ++j)
        for (int i = 0; i < kRows; ++i) {
            _mm256_store_ps(tiles[j][i], _mm256_setzero_ps());
            _mm256_store_ps(tiles[j][i] + 8, _mm256_setzero_ps());
        }
    for (int g = 0; g < ng; ++g) {
        const std::uint8_t* gp = panel + (std::size_t)g * (kQuantHeader + 2 * (std::size_t)group * w.qbits);
        const __m256 sw0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gp)));
        const __m256 sw1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gp + 16)));
        const __m256 mw0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gp + 32)));
        const __m256 mw1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(gp + 48)));
        decode_group(gp + kQuantHeader, w.qbits, group, codes);
        for (int j = 0; j < count; ++j) {
            const int r0 = (mp0 + j) * kRows;
            __m256i acc0[kRows], acc1[kRows];
            for (int i = 0; i < kRows; ++i) acc0[i] = acc1[i] = _mm256_setzero_si256();
            for (int kb = 0; kb < group / 4; ++kb) {
                const __m256i b0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(codes + (std::size_t)kb * 64));
                const __m256i b1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(codes + (std::size_t)kb * 64 + 32));
                for (int i = 0; i < kRows; ++i) {
                    std::int32_t four;
                    std::memcpy(&four, qa + (std::size_t)(r0 + i) * K + (std::size_t)g * group + (std::size_t)kb * 4, 4);
                    const __m256i av = _mm256_set1_epi32(four);
                    acc0[i] = _mm256_dpbusd_avx_epi32(acc0[i], b0, av);
                    acc1[i] = _mm256_dpbusd_avx_epi32(acc1[i], b1, av);
                }
            }
            for (int i = 0; i < kRows; ++i) {
                const float sa = scale_a[(std::size_t)(r0 + i) * ng + g];
                const float sq = sa * static_cast<float>(qsum[(std::size_t)(r0 + i) * ng + g]);
                const __m256 vsa = _mm256_set1_ps(sa), vsq = _mm256_set1_ps(sq);
                __m256 t0 = _mm256_load_ps(tiles[j][i]), t1 = _mm256_load_ps(tiles[j][i] + 8);
                t0 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc0[i]), _mm256_mul_ps(sw0, vsa), _mm256_fmadd_ps(mw0, vsq, t0));
                t1 = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc1[i]), _mm256_mul_ps(sw1, vsa), _mm256_fmadd_ps(mw1, vsq, t1));
                _mm256_store_ps(tiles[j][i], t0);
                _mm256_store_ps(tiles[j][i] + 8, t1);
            }
        }
    }
}

void gemm_quant_int8(const float* a, int m, const PackedLinear& w, int n_offset, int n, const float* residual,
                     bool gelu, float* c) {
    const int K = w.in, group = w.qgroup, ng = K / group;
    const int mpanels = (m + kRows - 1) / kRows, npanels = (n + kCols - 1) / kCols;
    const int mpad = mpanels * kRows;
    const int mtasks = (mpanels + kQuantBlock - 1) / kQuantBlock;
    const std::size_t panel_bytes = quant_panel_bytes(K, w.qbits, group);
    std::vector<std::int8_t> qa((std::size_t)mpad * K);
    std::vector<float> scale_a((std::size_t)mpad * ng);
    std::vector<std::int32_t> qsum((std::size_t)mpad * ng);
    quantize_activations(a, m, mpad, K, group, qa.data(), scale_a.data(), qsum.data());
#pragma omp parallel for collapse(2) schedule(static)
    for (int np = 0; np < npanels; ++np)
        for (int mt = 0; mt < mtasks; ++mt) {
            alignas(32) float tiles[kQuantBlock][kRows][kCols];
            const int mp0 = mt * kQuantBlock, count = std::min(kQuantBlock, mpanels - mp0);
            const int n0 = np * kCols, nc = std::min(kCols, n - n0);
            const std::uint8_t* panel = w.qdata.data() + (std::size_t)(n_offset / kCols + np) * panel_bytes;
            vnni_task(w, panel, K, qa.data(), scale_a.data(), qsum.data(), mp0, count, tiles);
            const float* bias = w.bias_at(n_offset + n0);
            for (int j = 0; j < count; ++j) {
                const int m0 = (mp0 + j) * kRows;
                finish_tile(&tiles[j][0][0], m0, n0, std::min(kRows, m - m0), nc, n, bias, residual, gelu, c);
            }
        }
}

bool cpu_has_vnni() { return __builtin_cpu_supports("avxvnni") != 0; }

#endif  // IDOL_VNNI_PATH

std::atomic<int> g_quant_kernel{-1};  // -1: decide from IDOL_QUANT_KERNEL and the CPU; 0: float; 1: integer

bool use_int8_kernel() {
#ifdef IDOL_VNNI_PATH
    int mode = g_quant_kernel.load();
    if (mode < 0) {
        const char* env = std::getenv("IDOL_QUANT_KERNEL");
        mode = env && std::string(env) == "float" ? 0 : 1;
    }
    return mode == 1 && cpu_has_vnni();
#else
    return false;
#endif
}

}  // namespace

void set_quant_kernel(int mode) { g_quant_kernel.store(mode); }

bool quant_int8_supported() {
#ifdef IDOL_VNNI_PATH
    return cpu_has_vnni();
#else
    return false;
#endif
}

std::size_t quant_panel_bytes(int in, int bits, int group) {
    return (std::size_t)(in / group) * (kQuantHeader + 2 * (std::size_t)group * bits);
}

PackedLinear quantized_linear(std::vector<std::uint8_t> blob, const std::vector<float>& bias, int out, int in, int bits,
                              int group) {
    if (!quant_bits_ok(bits)) throw std::invalid_argument("quantized layers use 2, 3, 4, 5, 6 or 8 bits");
    if (group <= 0 || group % 32 != 0 || in % group != 0)
        throw std::invalid_argument("the quantization group must be a multiple of 32 that divides the input size");
    const int npanels = (out + kCols - 1) / kCols;
    if (blob.size() != (std::size_t)npanels * quant_panel_bytes(in, bits, group))
        throw std::invalid_argument("a quantized blob has the wrong size for its shape");
    PackedLinear p;
    p.out = out;
    p.in = in;
    p.qbits = bits;
    p.qgroup = group;
    p.qdata = std::move(blob);
    p.bias.assign((std::size_t)npanels * kCols, 0.0f);
    for (int n = 0; n < out; ++n) p.bias[n] = bias[n];
    return p;
}

std::vector<float> dequantize_linear(const PackedLinear& w) {
    std::vector<float> weight((std::size_t)w.out * w.in);
    const int npanels = (w.out + kCols - 1) / kCols;
    const std::size_t panel_bytes = quant_panel_bytes(w.in, w.qbits, w.qgroup);
    alignas(32) float chunk[kKChunk * kCols];
    for (int q = 0; q < npanels; ++q)
        for (int k0 = 0; k0 < w.in; k0 += kKChunk) {
            dequant_chunk(w.qdata.data() + (std::size_t)q * panel_bytes, w.qbits, w.qgroup, k0, chunk);
            for (int k = 0; k < kKChunk; ++k)
                for (int col = 0; col < kCols; ++col) {
                    const int n = q * kCols + col;
                    if (n < w.out) weight[(std::size_t)n * w.in + k0 + k] = chunk[k * kCols + col];
                }
        }
    return weight;
}

void gemm_packed(const float* a, int m, const PackedLinear& w, int n_offset, int n, const float* residual, bool gelu,
                 float* c) {
    const Scope timing(&g_profile.gemm);
    ++g_profile.gemm_calls;
    if (w.qbits > 0) {
#ifdef IDOL_VNNI_PATH
        if (w.qgroup <= kMaxGroup && use_int8_kernel()) {
            gemm_quant_int8(a, m, w, n_offset, n, residual, gelu, c);
            return;
        }
#endif
        gemm_quant(a, m, w, n_offset, n, residual, gelu, c);
        return;
    }
    const int K = w.in;
    const int mpanels = (m + kRows - 1) / kRows, npanels = (n + kCols - 1) / kCols;
#pragma omp parallel for collapse(2) schedule(static)
    for (int np = 0; np < npanels; ++np)
        for (int mp = 0; mp < mpanels; ++mp) {
            alignas(32) float tile[kRows][kCols];
            const int m0 = mp * kRows, n0 = np * kCols;
            const int mr = std::min(kRows, m - m0), nc = std::min(kCols, n - n0);
            const float* rows = a + (std::size_t)m0 * K;
            if (mr < kRows) {
                // The last tile's missing rows are zero-padded in a per-thread copy, so the kernel can read six rows.
                static thread_local std::vector<float> pad;
                pad.assign((std::size_t)kRows * K, 0.0f);
                std::memcpy(pad.data(), rows, sizeof(float) * (std::size_t)mr * K);
                rows = pad.data();
            }
            // The depth is summed in chunks of kKChunk and the partial tiles are added together. A single running sum
            // over K = 2048 loses about half the accuracy to rounding; chunking cuts the largest error against an
            // fp64 reference by roughly 3.5x on the dim-512 test model, at no measurable cost in speed.
            const WeightT* panel = w.panel(n_offset) + (std::size_t)np * K * kCols;
            microkernel_6x16(rows, K, panel, std::min(kKChunk, K), tile);
            for (int k0 = kKChunk; k0 < K; k0 += kKChunk) {
                alignas(32) float part[kRows][kCols];
                microkernel_6x16(rows + k0, K, panel + (std::size_t)k0 * kCols, std::min(kKChunk, K - k0), part);
                for (int i = 0; i < kRows; ++i) {
                    _mm256_store_ps(tile[i], _mm256_add_ps(_mm256_load_ps(tile[i]), _mm256_load_ps(part[i])));
                    _mm256_store_ps(tile[i] + 8, _mm256_add_ps(_mm256_load_ps(tile[i] + 8), _mm256_load_ps(part[i] + 8)));
                }
            }
            finish_tile(&tile[0][0], m0, n0, mr, nc, n, w.bias_at(n_offset + n0), residual, gelu, c);
        }
}

void layer_norm(const float* x, int rows, int dim, const float* gamma, const float* beta, float* y) {
    const Scope timing(&g_profile.norm);
    const float eps = 1e-5f;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float* in = x + (std::size_t)r * dim;
        float* out = y + (std::size_t)r * dim;
        float mean = 0.0f;
        for (int d = 0; d < dim; ++d) mean += in[d];
        mean /= dim;
        float var = 0.0f;
        for (int d = 0; d < dim; ++d) {
            const float c = in[d] - mean;
            var += c * c;
        }
        var /= dim;
        const float inv = 1.0f / std::sqrt(var + eps);
        for (int d = 0; d < dim; ++d) out[d] = (in[d] - mean) * inv * gamma[d] + beta[d];
    }
}

namespace {

void attention_dot(const float* q, int lq, int ldq, const float* k, const float* v, int lk, int ldkv, int heads,
                   int head_dim, float* out, int dim);

// Attention for head sizes that are a multiple of 16. K is transposed once per head so that a tile of 4 queries by 16
// keys of scores is head_dim rank-1 updates in registers, V is copied once per head so the P times V tile of 4 queries
// by 16 dims streams it contiguously, and each tile keeps its accumulators in registers instead of re-reading V for
// every query row. Scores of a 32-query block live in a per-thread buffer, as in attention_dot.
void attention_blocked(const float* q, int lq, int ldq, const float* k, const float* v, int lk, int ldkv, int heads,
                       int head_dim, float* out, int dim) {
    const float scale = 1.0f / std::sqrt((float)head_dim);
    constexpr int kQueryBlock = 32;
    const int lkp = (lk + 15) & ~15;
    const std::size_t hd = (std::size_t)head_dim;
    std::vector<float> kt((std::size_t)heads * hd * lkp, 0.0f), vh((std::size_t)heads * lk * hd);
    constexpr int kPackBlock = 128;
    const int pack_blocks = (lk + kPackBlock - 1) / kPackBlock;
#pragma omp parallel for collapse(2) schedule(static)
    for (int h = 0; h < heads; ++h)
        for (int pb = 0; pb < pack_blocks; ++pb) {
            float* ktp = kt.data() + (std::size_t)h * hd * lkp;
            float* vhp = vh.data() + (std::size_t)h * lk * hd;
            const int j1 = std::min(lk, (pb + 1) * kPackBlock);
            for (int j = pb * kPackBlock; j < j1; ++j) {
                const float* kr = k + (std::size_t)j * ldkv + (std::size_t)h * hd;
                for (int d = 0; d < head_dim; ++d) ktp[(std::size_t)d * lkp + j] = kr[d];
                std::memcpy(vhp + (std::size_t)j * hd, v + (std::size_t)j * ldkv + (std::size_t)h * hd, sizeof(float) * hd);
            }
        }

    const int qblocks = (lq + kQueryBlock - 1) / kQueryBlock;
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int h = 0; h < heads; ++h)
        for (int qb = 0; qb < qblocks; ++qb) {
            static thread_local std::vector<float> s;
            const int i0 = qb * kQueryBlock, bm = std::min(lq, i0 + kQueryBlock) - i0;
            s.resize((std::size_t)bm * lkp);
            const float* ktp = kt.data() + (std::size_t)h * hd * lkp;
            const float* vhp = vh.data() + (std::size_t)h * lk * hd;

            // Scores: 4 queries by 16 keys at a time.
            for (int r0 = 0; r0 < bm; r0 += 4) {
                const float* qrow[4];
                for (int r = 0; r < 4; ++r)
                    qrow[r] = q + (std::size_t)(i0 + std::min(r0 + r, bm - 1)) * ldq + (std::size_t)h * hd;
                for (int jb = 0; jb < lkp; jb += 16) {
                    __m256 acc[4][2];
                    for (int r = 0; r < 4; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_ps();
                    const float* kp = ktp + jb;
                    for (int d = 0; d < head_dim; ++d, kp += lkp) {
                        const __m256 k0 = _mm256_loadu_ps(kp), k1 = _mm256_loadu_ps(kp + 8);
                        for (int r = 0; r < 4; ++r) {
                            const __m256 qd = _mm256_broadcast_ss(qrow[r] + d);
                            acc[r][0] = _mm256_fmadd_ps(qd, k0, acc[r][0]);
                            acc[r][1] = _mm256_fmadd_ps(qd, k1, acc[r][1]);
                        }
                    }
                    for (int r = 0; r < 4 && r0 + r < bm; ++r) {
                        _mm256_storeu_ps(&s[(std::size_t)(r0 + r) * lkp + jb], acc[r][0]);
                        _mm256_storeu_ps(&s[(std::size_t)(r0 + r) * lkp + jb + 8], acc[r][1]);
                    }
                }
            }

            // Softmax over the lk real keys of each row (the padding columns are never read again).
            for (int i = 0; i < bm; ++i) {
                float* row = s.data() + (std::size_t)i * lkp;
                float mx = -INFINITY;
                for (int j = 0; j < lk; ++j) mx = std::max(mx, row[j]);
                const __m256 vmax = _mm256_set1_ps(mx), vscale = _mm256_set1_ps(scale);
                __m256 acc = _mm256_setzero_ps();
                int j = 0;
                for (; j + 8 <= lk; j += 8) {
                    const __m256 e = exp256(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + j), vmax), vscale));
                    _mm256_storeu_ps(row + j, e);
                    acc = _mm256_add_ps(acc, e);
                }
                float total = hsum(acc);
                for (; j < lk; ++j) {
                    row[j] = std::exp((row[j] - mx) * scale);
                    total += row[j];
                }
                const float inv = 1.0f / total;
                for (j = 0; j < lk; ++j) row[j] *= inv;
            }

            // P times V: 4 queries by 16 dims at a time.
            for (int r0 = 0; r0 < bm; r0 += 4) {
                const float* prow[4];
                for (int r = 0; r < 4; ++r) prow[r] = s.data() + (std::size_t)std::min(r0 + r, bm - 1) * lkp;
                for (int c = 0; c < head_dim; c += 16) {
                    __m256 acc[4][2];
                    for (int r = 0; r < 4; ++r) acc[r][0] = acc[r][1] = _mm256_setzero_ps();
                    const float* vp = vhp + c;
                    for (int j = 0; j < lk; ++j, vp += hd) {
                        const __m256 v0 = _mm256_loadu_ps(vp), v1 = _mm256_loadu_ps(vp + 8);
                        for (int r = 0; r < 4; ++r) {
                            const __m256 p = _mm256_broadcast_ss(prow[r] + j);
                            acc[r][0] = _mm256_fmadd_ps(p, v0, acc[r][0]);
                            acc[r][1] = _mm256_fmadd_ps(p, v1, acc[r][1]);
                        }
                    }
                    for (int r = 0; r < 4 && r0 + r < bm; ++r) {
                        float* o = out + (std::size_t)(i0 + r0 + r) * dim + (std::size_t)h * hd + c;
                        _mm256_storeu_ps(o, acc[r][0]);
                        _mm256_storeu_ps(o + 8, acc[r][1]);
                    }
                }
            }
        }
}

}  // namespace

void attention(const float* q, int lq, int ldq, const float* k, const float* v, int lk, int ldkv, int heads,
               int head_dim, float* out, int dim) {
    const Scope timing(&g_profile.attention);
    static const bool force_dot = [] {
        const char* e = std::getenv("IDOL_ATTENTION");
        return e && std::strcmp(e, "dot") == 0;
    }();
    if (head_dim % 16 == 0 && !force_dot)
        attention_blocked(q, lq, ldq, k, v, lk, ldkv, heads, head_dim, out, dim);
    else
        attention_dot(q, lq, ldq, k, v, lk, ldkv, heads, head_dim, out, dim);
}

namespace {

void attention_dot(const float* q, int lq, int ldq, const float* k, const float* v, int lk, int ldkv, int heads,
                   int head_dim, float* out, int dim) {
    const float scale = 1.0f / std::sqrt((float)head_dim);
    constexpr int kQueryBlock = 32;
    const int qblocks = (lq + kQueryBlock - 1) / kQueryBlock;
#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int h = 0; h < heads; ++h)
        for (int qb = 0; qb < qblocks; ++qb) {
            // Scratch is per thread and reused across calls, so the hot loop does not allocate.
            static thread_local std::vector<float> qh, kh, s;
            const int i0 = qb * kQueryBlock, i1 = std::min(lq, i0 + kQueryBlock), bm = i1 - i0;
            qh.resize((std::size_t)bm * head_dim);
            kh.resize((std::size_t)lk * head_dim);
            s.resize((std::size_t)bm * lk);
            for (int i = 0; i < bm; ++i)
                std::memcpy(&qh[(std::size_t)i * head_dim], q + (std::size_t)(i0 + i) * ldq + (std::size_t)h * head_dim,
                            sizeof(float) * head_dim);
            for (int j = 0; j < lk; ++j)
                std::memcpy(&kh[(std::size_t)j * head_dim], k + (std::size_t)j * ldkv + (std::size_t)h * head_dim,
                            sizeof(float) * head_dim);
            gemm_block(qh.data(), head_dim, kh.data(), lk, s.data(), 0, bm, 0, lk);
            for (int i = 0; i < bm; ++i) {
                float* row = s.data() + (std::size_t)i * lk;
                float mx = -INFINITY;
                for (int j = 0; j < lk; ++j) mx = std::max(mx, row[j]);
                const __m256 vmax = _mm256_set1_ps(mx), vscale = _mm256_set1_ps(scale);
                __m256 acc = _mm256_setzero_ps();
                int j = 0;
                for (; j + 8 <= lk; j += 8) {
                    const __m256 e = exp256(_mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + j), vmax), vscale));
                    _mm256_storeu_ps(row + j, e);
                    acc = _mm256_add_ps(acc, e);
                }
                float total = hsum(acc);
                for (; j < lk; ++j) {
                    row[j] = std::exp((row[j] - mx) * scale);
                    total += row[j];
                }
                const float inv = 1.0f / total;
                for (j = 0; j < lk; ++j) row[j] *= inv;
            }
            for (int i = 0; i < bm; ++i) {
                float* __restrict o = out + (std::size_t)(i0 + i) * dim + (std::size_t)h * head_dim;
                const float* row = s.data() + (std::size_t)i * lk;
                for (int d = 0; d < head_dim; ++d) o[d] = 0.0f;
                for (int j = 0; j < lk; ++j) {
                    const float p = row[j];
                    const float* __restrict vrow = v + (std::size_t)j * ldkv + (std::size_t)h * head_dim;
                    for (int d = 0; d < head_dim; ++d) o[d] += p * vrow[d];
                }
            }
        }
}

}  // namespace

__m256 exp256(__m256 x) {
    // Range reduction to 2^n * e^r, then a degree-5 polynomial for e^r (Cephes constants, as in avx_mathfun).
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f)), _mm256_set1_ps(88.3762626647949f));
    const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f)),
                                      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(0.693359375f)));
    x = _mm256_sub_ps(x, _mm256_mul_ps(fx, _mm256_set1_ps(-2.12194440e-4f)));
    const __m256 z = _mm256_mul_ps(x, x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, z, x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    const __m256i n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

__m256 gelu256(__m256 x) {
    const __m256 sign_bit = _mm256_set1_ps(-0.0f);
    const __m256 z = _mm256_mul_ps(x, _mm256_set1_ps(0.70710678118654752f));
    const __m256 az = _mm256_andnot_ps(sign_bit, z);
    const __m256 t = _mm256_div_ps(_mm256_set1_ps(1.0f), _mm256_fmadd_ps(_mm256_set1_ps(0.3275911f), az, _mm256_set1_ps(1.0f)));
    __m256 p = _mm256_set1_ps(1.061405429f);
    p = _mm256_fmadd_ps(t, p, _mm256_set1_ps(-1.453152027f));
    p = _mm256_fmadd_ps(t, p, _mm256_set1_ps(1.421413741f));
    p = _mm256_fmadd_ps(t, p, _mm256_set1_ps(-0.284496736f));
    p = _mm256_fmadd_ps(t, p, _mm256_set1_ps(0.254829592f));
    p = _mm256_mul_ps(t, p);
    const __m256 decay = exp256(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(az, az)));
    const __m256 erf_abs = _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(p, decay));
    const __m256 erf = _mm256_or_ps(erf_abs, _mm256_and_ps(z, sign_bit));
    return _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(0.5f), x), _mm256_add_ps(_mm256_set1_ps(1.0f), erf));
}

float gelu_scalar(float x) {
    const float z = x * 0.70710678118654752f;
    const float az = std::fabs(z);
    const float t = 1.0f / (1.0f + 0.3275911f * az);
    const float p = t * (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f));
    const float erf_abs = 1.0f - p * std::exp(-az * az);
    return 0.5f * x * (1.0f + (z < 0.0f ? -erf_abs : erf_abs));
}

}  // namespace idol::detail
