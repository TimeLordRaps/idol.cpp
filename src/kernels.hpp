// Internal kernels for decoder inference: a packed GEMM with a fused epilogue, layer norm, multi-head
// attention, and the exp and GELU approximations. Not part of the public API.
#pragma once

#if (defined(__GNUC__) || defined(__clang__)) && \
    (!defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__))
#error "idol needs AVX2, FMA and F16C. Build with -mavx2 -mfma -mf16c on x86-64 (the CMake project sets these)."
#endif

#include <immintrin.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace idol::detail {

inline constexpr int kRows = 6;   // rows of A per microkernel call
inline constexpr int kCols = 16;  // columns of W per panel (two AVX2 vectors)

#ifdef IDOL_HALF_WEIGHTS
using WeightT = std::uint16_t;  // IEEE half precision, converted to fp32 inside the microkernel
#else
using WeightT = float;
#endif

// A linear layer's weight packed into panels of kCols output columns, for gemm_packed.
struct PackedLinear {
    int out = 0;                  // N: output features
    int in = 0;                   // K: input features
    std::vector<WeightT> panels;  // ceil(N / kCols) panels, each K x kCols, output columns contiguous per k
    std::vector<float> bias;      // padded to a multiple of kCols so full-width vector loads stay in bounds

    // Quantized weights (qbits > 0) replace panels. Each panel is in / qgroup groups: a 64-byte header with 16 half
    // scales and 16 half minimums (one per column), then the codes as bit planes. A plane of width w stores 8 / w rows
    // of k in each 16-byte vector, one byte per column; a code is the sum of its planes, the first plane in the low
    // bits. Eight rows of k hold 16 * qbits bytes. A weight is min + code * scale. See docs/format.html.
    int qbits = 0;                    // 2, 3, 4, 5, 6 or 8; 0 means the dense panels are used
    int qgroup = 0;                   // rows of k per scale and minimum: a multiple of 32 that divides in
    std::vector<std::uint8_t> qdata;

    const WeightT* panel(int n0) const;
    const float* bias_at(int n0) const;
    std::size_t bytes() const;
};

// Packs weight (out x in, row-major) and bias (out).
PackedLinear pack_linear(const std::vector<float>& weight, const std::vector<float>& bias, int out, int in);

// Bytes of one quantized panel (16 columns) over `in` rows, for the given width and group.
std::size_t quant_panel_bytes(int in, int bits, int group);

// Wraps an already-quantized blob (ceil(out / 16) panels, laid out as described in PackedLinear). Throws
// std::invalid_argument when bits, group, in or the blob size do not fit the format.
PackedLinear quantized_linear(std::vector<std::uint8_t> blob, const std::vector<float>& bias, int out, int in, int bits,
                              int group);

// Which kernel quantized layers use. -1 decides from the IDOL_QUANT_KERNEL environment variable ("float" forces the
// float kernel) and the CPU; 0 forces the float kernel; 1 asks for the integer kernel, which needs AVX-VNNI and falls
// back to the float one without it. quant_int8_supported() says whether the integer kernel can run here.
void set_quant_kernel(int mode);
bool quant_int8_supported();

// Expands a quantized layer back to its out x in row-major weights, as the GEMM sees them. For tests and tools.
std::vector<float> dequantize_linear(const PackedLinear& w);

// C[m x n] = A[m x in] * W^T + bias for the output columns [n_offset, n_offset + n) of w. If residual is given (laid
// out like C, and it may alias C), it is added before the optional GELU.
void gemm_packed(const float* a, int m, const PackedLinear& w, int n_offset, int n, const float* residual, bool gelu,
                 float* c);

// Layer norm over each of rows vectors of length dim. y may alias x.
void layer_norm(const float* x, int rows, int dim, const float* gamma, const float* beta, float* y);

// Scaled dot-product attention over projected rows. Head h uses columns [h * head_dim, (h + 1) * head_dim).
// q has lq rows with stride ldq; k and v have lk rows with stride ldkv. out has lq rows of stride dim.
void attention(const float* q, int lq, int ldq, const float* k, const float* v, int lk, int ldkv, int heads,
               int head_dim, float* out, int dim);

// exp on eight floats (relative error about 1e-7 over the softmax range) and exact GELU on eight floats, with erf
// from Abramowitz and Stegun 7.1.26 (absolute error below 1.5e-7 before rounding).
__m256 exp256(__m256 x);
__m256 gelu256(__m256 x);
float gelu_scalar(float x);

}  // namespace idol::detail
