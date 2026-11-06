// idol::Decoders: the octree decoder and the Gaussian decoder of the real Idol model (speridlabs/idol) on the CPU.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace idol {

// The shape of an exported pair of decoders, as recorded in manifest.txt.
struct DecoderConfig {
    int dim = 0;         // model width of both decoders (1024)
    int heads = 0;       // attention heads (16)
    int oct_blocks = 0;  // blocks in the octree decoder (4)
    int gs_blocks = 0;   // blocks in the Gaussian decoder (16)
    int cond = 0;        // channels of the latent the decoders attend to (16)
    int per_anchor = 0;  // Gaussians the Gaussian decoder emits per anchor point (32)
};

// The Gaussians as renderers use them: activated values, the object z-up inside [-0.5, 0.5]^3.
struct GaussianCloud {
    std::vector<float> xyz;       // 3 per Gaussian
    std::vector<float> f_dc;      // 3: degree-0 spherical-harmonic colour
    std::vector<float> scale;     // 3: activated scales (not log)
    std::vector<float> rotation;  // 4: unit quaternion, w first
    std::vector<float> opacity;   // 1: activated (0..1)
    std::size_t size() const { return opacity.size(); }
};

// Both decoders loaded from an export directory written by tools/export_idol.py (optionally with weight matrices
// quantized by tools/idolq.py). Const methods are safe to call concurrently only if you do not nest them: each call
// already uses every thread.
class Decoders {
public:
    explicit Decoders(const std::string& dir);
    ~Decoders();
    Decoders(Decoders&&) noexcept;
    Decoders& operator=(Decoders&&) noexcept;
    Decoders(const Decoders&) = delete;
    Decoders& operator=(const Decoders&) = delete;

    const DecoderConfig& config() const;
    std::size_t weight_bytes() const;  // bytes held by the packed weights

    // Octree decoder. x: rows cell centres in [0, 1]^3 (rows x 3); resolution: the cells' octree resolution (2^level);
    // cond: the latent, cond_rows x config().cond. Writes rows x 8 logits: how a cell's points split among its children.
    void octree_logits(const float* x, int rows, float resolution, const float* cond, int cond_rows, float* out) const;

    // Gaussian decoder. points: rows anchors in [0, 1]^3 (rows x 3). Writes rows x (per_anchor * 15) raw values.
    void gaussian_features(const float* points, int rows, const float* cond, int cond_rows, float* out) const;

    // Descends the octree from the root to `level`, placing num_points anchors in [0, 1]^3 (num_points x 3). The
    // sampling uses its own generator seeded with `seed`, so it does not reproduce PyTorch's draws.
    std::vector<float> sample_anchors(const float* cond, int cond_rows, int num_points, int level,
                                      std::uint64_t seed) const;

    // Turns anchors and their raw features into activated Gaussians (rows * per_anchor of them).
    GaussianCloud to_gaussians(const float* anchors, int rows, const float* features) const;

    // Latent -> Gaussians: sample_anchors, gaussian_features, to_gaussians. num_gaussians is rounded down to a
    // multiple of per_anchor.
    GaussianCloud decode(const float* cond, int cond_rows, int num_gaussians, int level, std::uint64_t seed) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Writes the cloud as a binary 3DGS PLY, turned from z-up to the y-up convention viewers use, as the SDK does.
void write_gaussian_ply(const GaussianCloud& cloud, const std::string& path);

}  // namespace idol
