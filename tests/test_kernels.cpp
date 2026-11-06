// Kernel tests. Each kernel is checked against a plain double-precision reference written here, so a wrong loop bound,
// stride or epilogue shows up as a numeric mismatch.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "kernels.hpp"
#include "test.hpp"

namespace {

using idol::detail::pack_linear;

// Float16 weights carry about three decimal digits, so the GEMM comparisons allow a wider error in that build.
#ifdef IDOL_HALF_WEIGHTS
constexpr double kGemmTolerance = 2e-2;
#else
constexpr double kGemmTolerance = 1e-4;
#endif

std::vector<float> random_values(std::size_t count, unsigned seed, float bound) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-bound, bound);
    std::vector<float> values(count);
    for (float& v : values) v = dist(rng);
    return values;
}

// out[m x n] = a[m x in] * weight[n x in]^T + bias[n]
std::vector<float> naive_linear(const std::vector<float>& a, int m, const std::vector<float>& weight,
                                const std::vector<float>& bias, int n, int in) {
    std::vector<float> out(static_cast<std::size_t>(m) * n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = bias[j];
            for (int k = 0; k < in; ++k) sum += static_cast<double>(a[i * in + k]) * weight[j * in + k];
            out[i * n + j] = static_cast<float>(sum);
        }
    }
    return out;
}

double exact_gelu(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

// A reference packer for the quantized layout, written from the format description and not from the kernel: per
// 16-column panel and group of rows, a min/max scale and minimum rounded to half, then the codes as bit planes.
// `expected` receives the weights a correct kernel must reproduce (row-major, out x in).
std::vector<std::uint8_t> reference_quantize(const std::vector<float>& weight, int out, int in, int bits, int group,
                                             std::vector<float>& expected) {
    const std::vector<std::vector<int>> planes = {{}, {}, {2}, {2, 1}, {4}, {4, 1}, {4, 2}, {}, {8}};
    const int levels = (1 << bits) - 1;
    auto to_half = [](float v) { return _cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT); };
    auto from_half = [](std::uint16_t h) { return _cvtsh_ss(h); };
    const int npanels = (out + 15) / 16;
    expected.assign(static_cast<std::size_t>(out) * in, 0.0f);
    std::vector<std::uint8_t> blob;
    for (int q = 0; q < npanels; ++q) {
        for (int g = 0; g < in / group; ++g) {
            std::vector<std::uint8_t> header(64, 0);
            std::vector<int> codes(static_cast<std::size_t>(group) * 16, 0);  // [row][column]
            for (int c = 0; c < 16; ++c) {
                const int n = q * 16 + c;
                if (n >= out) continue;
                float lo = weight[static_cast<std::size_t>(n) * in + g * group], hi = lo;
                for (int r = 0; r < group; ++r) {
                    const float v = weight[static_cast<std::size_t>(n) * in + g * group + r];
                    lo = std::min(lo, v);
                    hi = std::max(hi, v);
                }
                const std::uint16_t sh = to_half((hi - lo) / static_cast<float>(levels)), mh = to_half(lo);
                const float scale = from_half(sh), minimum = from_half(mh);
                std::memcpy(&header[2 * c], &sh, 2);
                std::memcpy(&header[32 + 2 * c], &mh, 2);
                for (int r = 0; r < group; ++r) {
                    const float v = weight[static_cast<std::size_t>(n) * in + g * group + r];
                    const int code = scale > 0.0f ? std::clamp(static_cast<int>(std::lround((v - minimum) / scale)), 0, levels) : 0;
                    codes[static_cast<std::size_t>(r) * 16 + c] = code;
                    expected[static_cast<std::size_t>(n) * in + g * group + r] =
                        std::fmaf(static_cast<float>(code), scale, minimum);
                }
            }
            blob.insert(blob.end(), header.begin(), header.end());
            for (int r0 = 0; r0 < group; r0 += 8) {
                int shift = 0;
                for (int width : planes[bits]) {
                    const int per_vector = 8 / width;
                    for (int v = 0; v < width; ++v)
                        for (int c = 0; c < 16; ++c) {
                            int byte = 0;
                            for (int s = 0; s < per_vector; ++s) {
                                const int code = codes[static_cast<std::size_t>(r0 + v * per_vector + s) * 16 + c];
                                byte |= ((code >> shift) & ((1 << width) - 1)) << (s * width);
                            }
                            blob.push_back(static_cast<std::uint8_t>(byte));
                        }
                    shift += width;
                }
            }
        }
    }
    return blob;
}

}  // namespace

TEST(kernels, gemm_matches_naive_with_bias) {
    const int m = 7, in = 50, out = 37;
    const auto weight = random_values(static_cast<std::size_t>(out) * in, 1, 0.5f);
    const auto bias = random_values(out, 2, 0.5f);
    const auto a = random_values(static_cast<std::size_t>(m) * in, 3, 1.0f);
    const auto packed = pack_linear(weight, bias, out, in);
    const auto expected = naive_linear(a, m, weight, bias, out, in);

    std::vector<float> c(expected.size());
    idol::detail::gemm_packed(a.data(), m, packed, 0, out, nullptr, false, c.data());
    for (std::size_t i = 0; i < c.size(); ++i) CHECK_NEAR(c[i], expected[i], kGemmTolerance);
}

TEST(kernels, gemm_handles_depths_around_the_chunk_size) {
    for (int in : {1, 7, 31, 32, 33, 64, 65, 200}) {
        const int m = 5, out = 20;
        const auto weight = random_values(static_cast<std::size_t>(out) * in, 40 + in, 0.5f);
        const auto bias = random_values(out, 41, 0.5f);
        const auto a = random_values(static_cast<std::size_t>(m) * in, 42, 1.0f);
        const auto packed = pack_linear(weight, bias, out, in);
        const auto expected = naive_linear(a, m, weight, bias, out, in);
        std::vector<float> c(expected.size());
        idol::detail::gemm_packed(a.data(), m, packed, 0, out, nullptr, false, c.data());
        for (std::size_t i = 0; i < c.size(); ++i) CHECK_NEAR(c[i], expected[i], kGemmTolerance);
    }
}

TEST(kernels, gemm_column_window_adds_residual_before_gelu) {
    const int m = 6, in = 40, out = 37, first = 16, n = 21;
    const auto weight = random_values(static_cast<std::size_t>(out) * in, 4, 0.5f);
    const auto bias = random_values(out, 5, 0.5f);
    const auto a = random_values(static_cast<std::size_t>(m) * in, 6, 1.0f);
    const auto residual = random_values(static_cast<std::size_t>(m) * n, 7, 0.5f);
    const auto packed = pack_linear(weight, bias, out, in);
    const auto full = naive_linear(a, m, weight, bias, out, in);

    std::vector<float> c(static_cast<std::size_t>(m) * n);
    idol::detail::gemm_packed(a.data(), m, packed, first, n, residual.data(), true, c.data());
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            const double pre = full[i * out + first + j] + residual[i * n + j];
            CHECK_NEAR(c[i * n + j], exact_gelu(pre), kGemmTolerance);
        }
    }
}

TEST(kernels, layer_norm_matches_naive_and_runs_in_place) {
    const int rows = 5, dim = 24;
    const auto x = random_values(static_cast<std::size_t>(rows) * dim, 8, 2.0f);
    const auto gamma = random_values(dim, 9, 1.0f);
    const auto beta = random_values(dim, 10, 0.5f);

    std::vector<double> expected(x.size());
    for (int r = 0; r < rows; ++r) {
        double mean = 0.0;
        for (int d = 0; d < dim; ++d) mean += x[r * dim + d];
        mean /= dim;
        double var = 0.0;
        for (int d = 0; d < dim; ++d) var += (x[r * dim + d] - mean) * (x[r * dim + d] - mean);
        var /= dim;
        const double inv = 1.0 / std::sqrt(var + 1e-5);
        for (int d = 0; d < dim; ++d) expected[r * dim + d] = (x[r * dim + d] - mean) * inv * gamma[d] + beta[d];
    }

    std::vector<float> y(x.size());
    idol::detail::layer_norm(x.data(), rows, dim, gamma.data(), beta.data(), y.data());
    for (std::size_t i = 0; i < y.size(); ++i) CHECK_NEAR(y[i], expected[i], 1e-4);

    std::vector<float> in_place = x;
    idol::detail::layer_norm(in_place.data(), rows, dim, gamma.data(), beta.data(), in_place.data());
    for (std::size_t i = 0; i < in_place.size(); ++i) CHECK_NEAR(in_place[i], expected[i], 1e-4);
}

static void check_attention(int lq, int lk, int heads, int head_dim) {
    const int dim = heads * head_dim;
    const int ldq = dim + 3;  // rows wider than dim, as they are in a projection buffer
    const int ldkv = dim + 1;
    const auto q = random_values(static_cast<std::size_t>(lq) * ldq, 11, 1.0f);
    const auto k = random_values(static_cast<std::size_t>(lk) * ldkv, 12, 1.0f);
    const auto v = random_values(static_cast<std::size_t>(lk) * ldkv, 13, 1.0f);

    std::vector<float> out(static_cast<std::size_t>(lq) * dim);
    idol::detail::attention(q.data(), lq, ldq, k.data(), v.data(), lk, ldkv, heads, head_dim, out.data(), dim);

    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    for (int i = 0; i < lq; ++i) {
        for (int h = 0; h < heads; ++h) {
            std::vector<double> score(lk);
            double top = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < lk; ++j) {
                double dot = 0.0;
                for (int d = 0; d < head_dim; ++d)
                    dot += static_cast<double>(q[i * ldq + h * head_dim + d]) * k[j * ldkv + h * head_dim + d];
                score[j] = dot * scale;
                if (score[j] > top) top = score[j];
            }
            double total = 0.0;
            for (int j = 0; j < lk; ++j) {
                score[j] = std::exp(score[j] - top);
                total += score[j];
            }
            for (int d = 0; d < head_dim; ++d) {
                double acc = 0.0;
                for (int j = 0; j < lk; ++j) acc += score[j] / total * v[j * ldkv + h * head_dim + d];
                CHECK_NEAR(out[i * dim + h * head_dim + d], acc, 1e-4);
            }
        }
    }
}

TEST(kernels, attention_matches_naive_softmax_with_strided_rows) { check_attention(5, 9, 3, 4); }

// Head sizes that are a multiple of 16 take the register-tiled path; the counts below leave tails in every direction.
TEST(kernels, blocked_attention_matches_naive_softmax_with_ragged_sizes) {
    check_attention(37, 53, 2, 32);
    check_attention(3, 16, 1, 16);
    check_attention(70, 130, 2, 64);
}

TEST(kernels, quantized_layers_dequantize_and_multiply_like_their_dense_equivalent) {
    idol::detail::set_quant_kernel(0);  // the float kernel sums in the same order as the dense one, so it is exact
    for (const int bits : {2, 3, 4, 5, 6, 8}) {
        for (const int group : {32, 64}) {
            const int m = 19, in = 128, out = 37;
            const auto weight = random_values(static_cast<std::size_t>(out) * in, 10 + bits, 0.5f);
            const auto bias = random_values(out, 20, 0.5f);
            const auto a = random_values(static_cast<std::size_t>(m) * in, 30, 1.0f);
            std::vector<float> expected;
            auto blob = reference_quantize(weight, out, in, bits, group, expected);
            const auto quantized = idol::detail::quantized_linear(std::move(blob), bias, out, in, bits, group);

            const auto restored = idol::detail::dequantize_linear(quantized);
            CHECK(restored.size() == expected.size());
            for (std::size_t i = 0; i < restored.size(); ++i) CHECK(restored[i] == expected[i]);

            // The same weights held as a dense layer must give the same product.
            const auto dense = pack_linear(expected, bias, out, in);
            std::vector<float> from_quantized(static_cast<std::size_t>(m) * out), from_dense(from_quantized.size());
            idol::detail::gemm_packed(a.data(), m, quantized, 0, out, nullptr, false, from_quantized.data());
            idol::detail::gemm_packed(a.data(), m, dense, 0, out, nullptr, false, from_dense.data());
            for (std::size_t i = 0; i < from_dense.size(); ++i) CHECK_NEAR(from_quantized[i], from_dense[i], kGemmTolerance);
#ifndef IDOL_HALF_WEIGHTS
            for (std::size_t i = 0; i < from_dense.size(); ++i) CHECK(from_quantized[i] == from_dense[i]);
#endif

            // A column offset, a residual and the GELU go through the same epilogue.
            std::vector<float> residual(static_cast<std::size_t>(m) * 16, 0.25f), part_q(residual.size()), part_d(residual.size());
            idol::detail::gemm_packed(a.data(), m, quantized, 16, 16, residual.data(), true, part_q.data());
            idol::detail::gemm_packed(a.data(), m, dense, 16, 16, residual.data(), true, part_d.data());
            for (std::size_t i = 0; i < part_d.size(); ++i) CHECK_NEAR(part_q[i], part_d[i], kGemmTolerance);
        }
    }
}

TEST(kernels, integer_kernel_stays_close_to_the_float_kernel) {
    if (!idol::detail::quant_int8_supported()) return;  // no AVX-VNNI on this CPU: nothing to test
    for (const int bits : {2, 3, 4, 5, 6, 8}) {
        for (const int group : {32, 64, 128}) {
            const int m = 19, in = 256, out = 37;
            const auto weight = random_values(static_cast<std::size_t>(out) * in, 40 + bits, 0.5f);
            const auto bias = random_values(out, 50, 0.5f);
            const auto a = random_values(static_cast<std::size_t>(m) * in, 60, 1.0f);
            std::vector<float> expected;
            auto blob = reference_quantize(weight, out, in, bits, group, expected);
            const auto quantized = idol::detail::quantized_linear(std::move(blob), bias, out, in, bits, group);
            std::vector<float> by_float(static_cast<std::size_t>(m) * out), by_int(by_float.size());
            idol::detail::set_quant_kernel(0);
            idol::detail::gemm_packed(a.data(), m, quantized, 0, out, nullptr, false, by_float.data());
            idol::detail::set_quant_kernel(1);
            idol::detail::gemm_packed(a.data(), m, quantized, 0, out, nullptr, false, by_int.data());
            // Rounding the activations to 8 bits moves each product by about 0.4% of its row's largest value.
            double scale = 0.0;
            for (float v : by_float) scale = std::max(scale, std::fabs(static_cast<double>(v)));
            for (std::size_t i = 0; i < by_int.size(); ++i) CHECK_NEAR(by_int[i], by_float[i], 0.02 * scale);
        }
    }
    idol::detail::set_quant_kernel(-1);
}

TEST(kernels, quantized_layers_reject_shapes_the_format_cannot_hold) {
    const std::vector<float> bias(16, 0.0f);
    CHECK_THROWS(idol::detail::quantized_linear(std::vector<std::uint8_t>(10), bias, 16, 64, 4, 32));
    CHECK_THROWS(idol::detail::quantized_linear(std::vector<std::uint8_t>(10), bias, 16, 64, 7, 32));
    CHECK_THROWS(idol::detail::quantized_linear(std::vector<std::uint8_t>(10), bias, 16, 64, 4, 48));
    CHECK_THROWS(idol::detail::quantized_linear(std::vector<std::uint8_t>(10), bias, 16, 96, 4, 64));
}

TEST(kernels, exp256_has_relative_error_below_1e6_on_the_softmax_range) {
    alignas(32) float x[8];
    alignas(32) float y[8];
    for (int base = 0; base < 8000; base += 8) {
        for (int lane = 0; lane < 8; ++lane) x[lane] = -20.0f + 0.005f * static_cast<float>(base + lane);
        _mm256_store_ps(y, idol::detail::exp256(_mm256_load_ps(x)));
        for (int lane = 0; lane < 8; ++lane) {
            const double exact = std::exp(static_cast<double>(x[lane]));
            CHECK_NEAR(y[lane] / exact, 1.0, 1e-6);
        }
    }
}

TEST(kernels, gelu_approximations_match_the_exact_gelu) {
    alignas(32) float x[8];
    alignas(32) float y[8];
    for (int base = 0; base < 1200; base += 8) {
        for (int lane = 0; lane < 8; ++lane) x[lane] = -6.0f + 0.01f * static_cast<float>(base + lane);
        _mm256_store_ps(y, idol::detail::gelu256(_mm256_load_ps(x)));
        for (int lane = 0; lane < 8; ++lane) {
            const double exact = exact_gelu(x[lane]);
            CHECK_NEAR(y[lane], exact, 2e-6);
            CHECK_NEAR(idol::detail::gelu_scalar(x[lane]), exact, 2e-6);
        }
    }
}
