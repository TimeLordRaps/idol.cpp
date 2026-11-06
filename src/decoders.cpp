#include "idol/decoders.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <utility>

#include "kernels.hpp"
#include "weights.hpp"

namespace idol {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// y = layer_norm(x) over each row of length dim (eps as given), then optionally * gamma + beta, then optionally
// * (1 + scale) + shift (the adaLN modulation). y may alias x.
void norm_rows(const float* x, float* y, int rows, int dim, float eps, const float* gamma = nullptr,
               const float* beta = nullptr, const float* shift = nullptr, const float* scale = nullptr) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const float* xr = x + (std::size_t)r * dim;
        float* yr = y + (std::size_t)r * dim;
        double sum = 0.0;
        for (int c = 0; c < dim; ++c) sum += xr[c];
        const float mean = (float)(sum / dim);
        double sq = 0.0;
        for (int c = 0; c < dim; ++c) sq += (double)(xr[c] - mean) * (xr[c] - mean);
        const float inv = 1.0f / std::sqrt((float)(sq / dim) + eps);
        for (int c = 0; c < dim; ++c) {
            float v = (xr[c] - mean) * inv;
            if (gamma) v = v * gamma[c] + beta[c];
            if (scale) v = v * (1.0f + scale[c]) + shift[c];
            yr[c] = v;
        }
    }
}

// x += h * gate (per column) over rows of length dim.
void add_gated(float* x, const float* h, int rows, int dim, const float* gate) {
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        float* xr = x + (std::size_t)r * dim;
        const float* hr = h + (std::size_t)r * dim;
        for (int c = 0; c < dim; ++c) xr[c] += hr[c] * gate[c];
    }
}

// Gaussian error linear unit (GELU) with the tanh approximation, in place.
void gelu_tanh(float* x, std::size_t count) {
    const __m256 k0 = _mm256_set1_ps(0.7978845608028654f), k1 = _mm256_set1_ps(0.044715f), half = _mm256_set1_ps(0.5f),
                 one = _mm256_set1_ps(1.0f), two = _mm256_set1_ps(2.0f);
    const std::ptrdiff_t blocks = (std::ptrdiff_t)(count / 8);
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t b = 0; b < blocks; ++b) {
        const __m256 v = _mm256_loadu_ps(x + b * 8);
        const __m256 u = _mm256_mul_ps(k0, _mm256_fmadd_ps(_mm256_mul_ps(k1, _mm256_mul_ps(v, v)), v, v));
        const __m256 e = detail::exp256(_mm256_mul_ps(two, u));
        const __m256 t = _mm256_sub_ps(one, _mm256_div_ps(two, _mm256_add_ps(e, one)));
        _mm256_storeu_ps(x + b * 8, _mm256_mul_ps(_mm256_mul_ps(half, v), _mm256_add_ps(one, t)));
    }
    for (std::size_t i = (std::size_t)blocks * 8; i < count; ++i) {
        const float v = x[i], u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(u));
    }
}

// RMS-normalise each head of the rows (ld floats apart) to unit length, then * gamma[head, :] * sqrt(head_dim).
void rms_heads(float* x, int rows, int ld, int heads, int hd, const float* gamma) {
    const float root = std::sqrt((float)hd);
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r)
        for (int h = 0; h < heads; ++h) {
            float* p = x + (std::size_t)r * ld + (std::size_t)h * hd;
            const float* g = gamma + (std::size_t)h * hd;
            float ss = 0.0f;
            for (int d = 0; d < hd; ++d) ss += p[d] * p[d];
            const float mult = root / std::max(std::sqrt(ss), 1e-12f);
            for (int d = 0; d < hd; ++d) p[d] = p[d] * mult * g[d];
        }
}

float silu(float x) { return x / (1.0f + std::exp(-x)); }

float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }

}  // namespace

struct Decoders::Impl {
    struct Attn {
        detail::PackedLinear q, kv, qkv, out;  // cross attention uses q and kv, self attention qkv
        std::vector<float> q_gamma, k_gamma;
    };
    struct OctBlock {
        Attn cross;
        detail::PackedLinear ff1, ff2;
    };
    struct GsBlock {
        Attn self, cross;
        std::vector<float> norm2_gamma, norm2_beta;
        detail::PackedLinear ff1, ff2;
    };

    DecoderConfig cfg;
    detail::Representation rep;
    detail::PackedLinear oct_in, oct_input, oct_l1, oct_l2, oct_ada, oct_out;
    detail::PackedLinear gs_in, gs_input, gs_out;
    std::vector<OctBlock> oct;
    std::vector<GsBlock> gs;
    std::vector<float> perturbation;  // per_anchor x 3
    float base_offset_scale = 0.0f;
    std::vector<float> oct_pos_freqs, gs_pos_freqs;  // the position embedders' frequencies, as PyTorch computed them
    std::vector<float> level_freqs;                  // the level embedder's frequencies
    std::size_t weight_bytes = 0;

    explicit Impl(detail::Export ex);
    void pos_embed(const std::vector<float>& freqs, const float* x, int rows, float* out) const;
    void cross_attention(const Attn& a, const float* h, int rows, const float* cond, int cond_rows, float* q, float* kv,
                         float* out) const;
    void octree_logits(const float* x, int rows, float resolution, const float* cond, int cond_rows, float* out) const;
    void gaussian_features(const float* points, int rows, const float* cond, int cond_rows, float* out) const;
};

Decoders::Impl::Impl(detail::Export ex) : cfg(ex.decoder_config), rep(ex.rep) {
    if (!ex.decoders) throw std::runtime_error("this export is not a decoders export (tools/export_idol.py)");
    const int D = cfg.dim, H4 = 4 * cfg.dim;
    if (D <= 0 || cfg.heads <= 0 || D % cfg.heads != 0 || cfg.cond <= 0 || cfg.per_anchor <= 0 || cfg.oct_blocks < 0 ||
        cfg.gs_blocks < 0)
        throw std::runtime_error("manifest.txt describes an invalid decoder configuration");

    using detail::take_tensor;
    auto linear = [&](const std::string& prefix, int out, int in) {
        detail::QTensor q;
        if (detail::take_qtensor(ex, prefix + ".weight", q)) {
            if (q.out != out || q.in != in)
                throw std::runtime_error("quantized tensor '" + prefix + ".weight' has the wrong shape");
            detail::Tensor b = take_tensor(ex, prefix + ".bias", out);
            return detail::quantized_linear(std::move(q.blob), b.data, out, in, q.bits, q.group);
        }
        detail::Tensor w = take_tensor(ex, prefix + ".weight", (std::int64_t)out * in);
        detail::Tensor b = take_tensor(ex, prefix + ".bias", out);
        return detail::pack_linear(w.data, b.data, out, in);
    };
    auto vec = [&](const std::string& name, std::int64_t count) { return take_tensor(ex, name, count).data; };
    auto attn = [&](const std::string& p, bool cross) {
        Attn a;
        if (cross) {
            a.q = linear(p + ".to_q", D, D);
            a.kv = linear(p + ".to_kv", 2 * D, cfg.cond);
        } else {
            a.qkv = linear(p + ".to_qkv", 3 * D, D);
        }
        a.out = linear(p + ".to_out", D, D);
        a.q_gamma = vec(p + ".q_rms_norm.gamma", D);
        a.k_gamma = vec(p + ".k_rms_norm.gamma", D);
        return a;
    };

    oct_in = linear("oct.in_proj", D, 3);
    oct_input = linear("oct.input_layer", D, D);
    oct_l1 = linear("oct.l_embedder.mlp.0", D, 256);
    oct_l2 = linear("oct.l_embedder.mlp.2", D, D);
    oct_ada = linear("oct.adaLN_modulation.1", 6 * D, D);
    oct_out = linear("oct.out_proj", 8, D);
    for (int i = 0; i < cfg.oct_blocks; ++i) {
        const std::string p = "oct.blocks." + std::to_string(i) + ".";
        OctBlock b;
        b.cross = attn(p + "cross_attn", true);
        b.ff1 = linear(p + "mlp.mlp.0", H4, D);
        b.ff2 = linear(p + "mlp.mlp.2", D, H4);
        oct.push_back(std::move(b));
    }

    const int outs = cfg.per_anchor * 15;
    gs_in = linear("gs.in_proj", D, 3);
    gs_input = linear("gs.input_layer", D, D);
    gs_out = linear("gs.out_proj", outs, D);
    perturbation = vec("gs.points_offset_perturbation", (std::int64_t)cfg.per_anchor * 3);
    base_offset_scale = vec("gs.base_offset_scale", 1)[0];
    for (int i = 0; i < cfg.gs_blocks; ++i) {
        const std::string p = "gs.blocks." + std::to_string(i) + ".";
        GsBlock b;
        b.self = attn(p + "self_attn", false);
        b.cross = attn(p + "cross_attn", true);
        b.norm2_gamma = vec(p + "norm2.weight", D);
        b.norm2_beta = vec(p + "norm2.bias", D);
        b.ff1 = linear(p + "mlp.mlp.0", H4, D);
        b.ff2 = linear(p + "mlp.mlp.2", D, H4);
        gs.push_back(std::move(b));
    }

    // The embedders' frequency tables come from the export: PyTorch builds them in float32 (2^linspace(0, 10) for the
    // positions, exp(-log(1024) i / 128) for the level), and the angles reach ~3000 rad, so recomputing them here
    // would move the features by ~1e-3.
    oct_pos_freqs = vec("oct.pos_embedder.freqs", D / 3 / 2);
    gs_pos_freqs = vec("gs.pos_embedder.freqs", D / 3 / 2);
    level_freqs = vec("oct.l_embedder.freqs", 128);

    for (const detail::PackedLinear* l : {&oct_in, &oct_input, &oct_l1, &oct_l2, &oct_ada, &oct_out, &gs_in, &gs_input,
                                          &gs_out})
        weight_bytes += l->bytes();
    for (const OctBlock& b : oct)
        weight_bytes += b.cross.q.bytes() + b.cross.kv.bytes() + b.cross.out.bytes() + b.ff1.bytes() + b.ff2.bytes();
    for (const GsBlock& b : gs)
        weight_bytes += b.self.qkv.bytes() + b.self.out.bytes() + b.cross.q.bytes() + b.cross.kv.bytes() +
                        b.cross.out.bytes() + b.ff1.bytes() + b.ff2.bytes();
}

// The sine and cosine of each coordinate at each frequency, coordinate by coordinate, zero-padded to the model width.
void Decoders::Impl::pos_embed(const std::vector<float>& freqs, const float* x, int rows, float* out) const {
    const int D = cfg.dim, fd = (int)freqs.size();
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        float* o = out + (std::size_t)r * D;
        for (int c = 0; c < 3; ++c) {
            const float v = x[(std::size_t)r * 3 + c];
            for (int j = 0; j < fd; ++j) {
                const float ang = (v * freqs[j]) * kPi;
                o[c * 2 * fd + j] = std::sin(ang);
                o[c * 2 * fd + fd + j] = std::cos(ang);
            }
        }
        for (int j = 6 * fd; j < D; ++j) o[j] = 0.0f;
    }
}

void Decoders::Impl::cross_attention(const Attn& a, const float* h, int rows, const float* cond, int cond_rows, float* q,
                                     float* kv, float* out) const {
    const int D = cfg.dim, hd = D / cfg.heads;
    detail::gemm_packed(h, rows, a.q, 0, D, nullptr, false, q);
    rms_heads(q, rows, D, cfg.heads, hd, a.q_gamma.data());
    detail::gemm_packed(cond, cond_rows, a.kv, 0, 2 * D, nullptr, false, kv);
    rms_heads(kv, cond_rows, 2 * D, cfg.heads, hd, a.k_gamma.data());
    detail::attention(q, rows, D, kv, kv + D, cond_rows, 2 * D, cfg.heads, hd, out, D);
}

void Decoders::Impl::octree_logits(const float* x, int rows, float resolution, const float* cond, int cond_rows,
                                   float* out) const {
    const int D = cfg.dim, H4 = 4 * cfg.dim;
    const std::size_t n = (std::size_t)rows * D;
    std::vector<float> X(n), H(n), Q(n), A(n), F((std::size_t)rows * H4), KV((std::size_t)cond_rows * 2 * D);

    pos_embed(oct_pos_freqs, x, rows, H.data());
    detail::gemm_packed(x, rows, oct_in, 0, D, H.data(), false, X.data());
    detail::gemm_packed(X.data(), rows, oct_input, 0, D, nullptr, false, H.data());
    std::swap(X, H);

    // The octree level, embedded and turned into the shared adaLN modulation (shift, scale, gate for each half).
    std::vector<float> emb(256), l1(D), l2(D), mod(6 * D);
    for (int i = 0; i < 128; ++i) {
        const float ang = ((resolution * level_freqs[i]) * 2.0f) * kPi;
        emb[i] = std::cos(ang);
        emb[128 + i] = std::sin(ang);
    }
    detail::gemm_packed(emb.data(), 1, oct_l1, 0, D, nullptr, false, l1.data());
    for (float& v : l1) v = silu(v);
    detail::gemm_packed(l1.data(), 1, oct_l2, 0, D, nullptr, false, l2.data());
    for (float& v : l2) v = silu(v);
    detail::gemm_packed(l2.data(), 1, oct_ada, 0, 6 * D, nullptr, false, mod.data());
    const float *shift_a = mod.data(), *scale_a = mod.data() + D, *gate_a = mod.data() + 2 * D,
                *shift_m = mod.data() + 3 * D, *scale_m = mod.data() + 4 * D, *gate_m = mod.data() + 5 * D;

    for (const OctBlock& b : oct) {
        norm_rows(X.data(), H.data(), rows, D, 1e-6f, nullptr, nullptr, shift_a, scale_a);
        cross_attention(b.cross, H.data(), rows, cond, cond_rows, Q.data(), KV.data(), A.data());
        detail::gemm_packed(A.data(), rows, b.cross.out, 0, D, nullptr, false, H.data());
        add_gated(X.data(), H.data(), rows, D, gate_a);

        norm_rows(X.data(), H.data(), rows, D, 1e-6f, nullptr, nullptr, shift_m, scale_m);
        detail::gemm_packed(H.data(), rows, b.ff1, 0, H4, nullptr, false, F.data());
        gelu_tanh(F.data(), (std::size_t)rows * H4);
        detail::gemm_packed(F.data(), rows, b.ff2, 0, D, nullptr, false, H.data());
        add_gated(X.data(), H.data(), rows, D, gate_m);
    }
    norm_rows(X.data(), H.data(), rows, D, 1e-5f);
    detail::gemm_packed(H.data(), rows, oct_out, 0, 8, nullptr, false, out);
}

void Decoders::Impl::gaussian_features(const float* points, int rows, const float* cond, int cond_rows,
                                       float* out) const {
    const int D = cfg.dim, H4 = 4 * cfg.dim, hd = D / cfg.heads;
    const std::size_t n = (std::size_t)rows * D;
    std::vector<float> X(n), H(n), Q(n), A(n), F((std::size_t)rows * H4), QKV((std::size_t)rows * 3 * D),
        KV((std::size_t)cond_rows * 2 * D);

    pos_embed(gs_pos_freqs, points, rows, H.data());
    detail::gemm_packed(points, rows, gs_in, 0, D, H.data(), false, X.data());
    detail::gemm_packed(X.data(), rows, gs_input, 0, D, nullptr, false, H.data());
    std::swap(X, H);

    for (const GsBlock& b : gs) {
        norm_rows(X.data(), H.data(), rows, D, 1e-6f);
        detail::gemm_packed(H.data(), rows, b.self.qkv, 0, 3 * D, nullptr, false, QKV.data());
        rms_heads(QKV.data(), rows, 3 * D, cfg.heads, hd, b.self.q_gamma.data());
        rms_heads(QKV.data() + D, rows, 3 * D, cfg.heads, hd, b.self.k_gamma.data());
        detail::attention(QKV.data(), rows, 3 * D, QKV.data() + D, QKV.data() + 2 * D, rows, 3 * D, cfg.heads, hd,
                          A.data(), D);
        detail::gemm_packed(A.data(), rows, b.self.out, 0, D, X.data(), false, X.data());

        norm_rows(X.data(), H.data(), rows, D, 1e-6f, b.norm2_gamma.data(), b.norm2_beta.data());
        cross_attention(b.cross, H.data(), rows, cond, cond_rows, Q.data(), KV.data(), A.data());
        detail::gemm_packed(A.data(), rows, b.cross.out, 0, D, X.data(), false, X.data());

        norm_rows(X.data(), H.data(), rows, D, 1e-6f);
        detail::gemm_packed(H.data(), rows, b.ff1, 0, H4, nullptr, false, F.data());
        gelu_tanh(F.data(), (std::size_t)rows * H4);
        detail::gemm_packed(F.data(), rows, b.ff2, 0, D, X.data(), false, X.data());
    }
    norm_rows(X.data(), H.data(), rows, D, 1e-5f);
    detail::gemm_packed(H.data(), rows, gs_out, 0, cfg.per_anchor * 15, nullptr, false, out);
}

Decoders::Decoders(const std::string& dir) : impl_(std::make_unique<Impl>(detail::read_export(dir))) {}
Decoders::~Decoders() = default;
Decoders::Decoders(Decoders&&) noexcept = default;
Decoders& Decoders::operator=(Decoders&&) noexcept = default;

const DecoderConfig& Decoders::config() const { return impl_->cfg; }
std::size_t Decoders::weight_bytes() const { return impl_->weight_bytes; }

void Decoders::octree_logits(const float* x, int rows, float resolution, const float* cond, int cond_rows,
                             float* out) const {
    impl_->octree_logits(x, rows, resolution, cond, cond_rows, out);
}

void Decoders::gaussian_features(const float* points, int rows, const float* cond, int cond_rows, float* out) const {
    impl_->gaussian_features(points, rows, cond, cond_rows, out);
}

std::vector<float> Decoders::sample_anchors(const float* cond, int cond_rows, int num_points, int level,
                                            std::uint64_t seed) const {
    if (num_points < 1 || level < 1 || level > 20) throw std::invalid_argument("sample_anchors: bad num_points or level");
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);

    // Cells are (x, y, z) integer coordinates with a count of the points that landed in them.
    std::vector<std::uint32_t> coords = {0, 0, 0};
    std::vector<std::int64_t> counts = {num_points};
    std::vector<float> centres, logits;
    for (int l = 1; l <= level; ++l) {
        const int parents = (int)counts.size();
        const float res_p = (float)(1u << (l - 1));
        centres.resize((std::size_t)parents * 3);
        for (std::size_t i = 0; i < centres.size(); ++i) centres[i] = ((float)coords[i] + 0.5f) / res_p;
        logits.resize((std::size_t)parents * 8);
        impl_->octree_logits(centres.data(), parents, (float)(1u << l), cond, cond_rows, logits.data());

        std::vector<std::uint32_t> next_coords;
        std::vector<std::int64_t> next_counts;
        for (int p = 0; p < parents; ++p) {
            // softmax over the eight children, then systematic resampling of this cell's points among them
            float* row = logits.data() + (std::size_t)p * 8;
            float mx = row[0];
            for (int c = 1; c < 8; ++c) mx = std::max(mx, row[c]);
            float total = 0.0f;
            for (int c = 0; c < 8; ++c) total += (row[c] = std::exp(row[c] - mx));
            float cdf[8], acc = 0.0f;
            for (int c = 0; c < 8; ++c) cdf[c] = std::min(acc += row[c] / total, 1.0f - 1e-7f);
            const std::int64_t n = counts[p];
            std::int64_t got[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            const float u0 = uniform(rng) / (float)n;
            for (std::int64_t k = 0; k < n; ++k) {
                const float u = std::min(u0 + (float)k / (float)n, 1.0f - 1e-7f);
                got[std::min<int>(7, (int)(std::lower_bound(cdf, cdf + 8, u) - cdf))]++;
            }
            for (int c = 0; c < 8; ++c)
                if (got[c] > 0) {
                    next_coords.push_back(coords[(std::size_t)p * 3 + 0] * 2 + (c & 1));
                    next_coords.push_back(coords[(std::size_t)p * 3 + 1] * 2 + ((c >> 1) & 1));
                    next_coords.push_back(coords[(std::size_t)p * 3 + 2] * 2 + ((c >> 2) & 1));
                    next_counts.push_back(got[c]);
                }
        }
        coords = std::move(next_coords);
        counts = std::move(next_counts);
    }

    // Each point lands at a uniformly random place inside its leaf cell.
    std::vector<float> points;
    points.reserve((std::size_t)num_points * 3);
    const float cells = (float)(1u << level);
    for (std::size_t cell = 0; cell < counts.size(); ++cell)
        for (std::int64_t k = 0; k < counts[cell]; ++k)
            for (int c = 0; c < 3; ++c) points.push_back(((float)coords[cell * 3 + c] + uniform(rng)) / cells);
    return points;
}

GaussianCloud Decoders::to_gaussians(const float* anchors, int rows, const float* features) const {
    const detail::Representation& rp = impl_->rep;
    const int G = impl_->cfg.per_anchor, stride = G * 15;
    const int xyz0 = 0, dc0 = G * 3, sc0 = G * 6, rot0 = G * 9, op0 = G * 13, os0 = G * 14;
    const double sb = rp.scaling_bias, ob = rp.opacity_bias;
    const float scale_offset = (float)(sb + std::log(-std::expm1(-sb)));  // inverse softplus
    const float opacity_offset = (float)std::log(ob / (1.0 - ob));         // inverse sigmoid

    GaussianCloud cloud;
    const std::size_t total = (std::size_t)rows * G;
    cloud.xyz.resize(total * 3);
    cloud.f_dc.resize(total * 3);
    cloud.scale.resize(total * 3);
    cloud.rotation.resize(total * 4);
    cloud.opacity.resize(total);
#pragma omp parallel for schedule(static)
    for (int a = 0; a < rows; ++a) {
        const float* f = features + (std::size_t)a * stride;
        for (int g = 0; g < G; ++g) {
            const std::size_t i = (std::size_t)a * G + g;
            const float offset_scale = softplus(f[os0 + g] + impl_->base_offset_scale);
            for (int c = 0; c < 3; ++c) {
                const float o = std::tanh(f[xyz0 + g * 3 + c] * rp.lr_xyz + impl_->perturbation[g * 3 + c]) * 0.5f *
                                rp.perturbe * offset_scale;
                cloud.xyz[i * 3 + c] = (o + anchors[(std::size_t)a * 3 + c]) - 0.5f;
                cloud.f_dc[i * 3 + c] = f[dc0 + g * 3 + c] * rp.lr_dc;
                const float s = softplus(f[sc0 + g * 3 + c] * rp.lr_scaling + scale_offset);
                cloud.scale[i * 3 + c] = std::sqrt(s * s + rp.kernel * rp.kernel);
            }
            float q[4], norm = 0.0f;
            for (int c = 0; c < 4; ++c) {
                q[c] = f[rot0 + g * 4 + c] * rp.lr_rotation + (c == 0 ? 1.0f : 0.0f);
                norm += q[c] * q[c];
            }
            norm = std::max(std::sqrt(norm), 1e-12f);
            for (int c = 0; c < 4; ++c) cloud.rotation[i * 4 + c] = q[c] / norm;
            cloud.opacity[i] = 1.0f / (1.0f + std::exp(-(f[op0 + g] * rp.lr_opacity + opacity_offset)));
        }
    }
    return cloud;
}

GaussianCloud Decoders::decode(const float* cond, int cond_rows, int num_gaussians, int level,
                               std::uint64_t seed) const {
    const int G = impl_->cfg.per_anchor;
    const int anchors = std::max(1, num_gaussians / G);
    const std::vector<float> points = sample_anchors(cond, cond_rows, anchors, level, seed);
    std::vector<float> features((std::size_t)anchors * G * 15);
    impl_->gaussian_features(points.data(), anchors, cond, cond_rows, features.data());
    return to_gaussians(points.data(), anchors, features.data());
}

void write_gaussian_ply(const GaussianCloud& cloud, const std::string& path) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    static const char* names[] = {"x",       "y",       "z",       "nx",      "ny",      "nz",
                                  "f_dc_0",  "f_dc_1",  "f_dc_2",  "opacity", "scale_0", "scale_1",
                                  "scale_2", "rot_0",   "rot_1",   "rot_2",   "rot_3"};
    out << "ply\nformat binary_little_endian 1.0\nelement vertex " << cloud.size() << "\n";
    for (const char* name : names) out << "property float " << name << "\n";
    out << "end_header\n";
    const float h = std::sqrt(0.5f);
    std::vector<float> row(17);
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        const float *p = &cloud.xyz[i * 3], *r = &cloud.rotation[i * 4];
        // z-up -> y-up: (x, y, z) -> (x, -z, y), and the rotation is premultiplied by +90 degrees about x
        row[0] = p[0], row[1] = -p[2], row[2] = p[1];
        row[3] = row[4] = row[5] = 0.0f;
        for (int c = 0; c < 3; ++c) row[6 + c] = cloud.f_dc[i * 3 + c];
        const float op = std::min(std::max(cloud.opacity[i], 1e-6f), 1.0f - 1e-6f);
        row[9] = std::log(op / (1.0f - op));
        for (int c = 0; c < 3; ++c) row[10 + c] = std::log(std::max(cloud.scale[i * 3 + c], 1e-6f));
        row[13] = h * r[0] - h * r[1];
        row[14] = h * r[1] + h * r[0];
        row[15] = h * r[2] - h * r[3];
        row[16] = h * r[3] + h * r[2];
        out.write(reinterpret_cast<const char*>(row.data()), sizeof(float) * row.size());
    }
    if (!out) throw std::runtime_error("short write to " + path);
}

}  // namespace idol
