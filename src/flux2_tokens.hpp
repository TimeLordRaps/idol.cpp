// Postprocessing for the FLUX.2 image autoencoder used by Idol. This is an
// internal float32 reference; the native encoder and mixed-precision parity
// are not yet established. The caller supplies the Gaussian noise explicitly.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace idol::detail {

inline std::size_t checked_product(std::size_t a, std::size_t b) {
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        throw std::invalid_argument("FLUX.2 tensor dimensions overflow");
    return a * b;
}

// moments: [batch, 2*channels, height, width], mean followed by log variance.
// noise: [batch, channels, height, width]. All input arrays are contiguous.
// Result: [batch, 5 + height/2*width/2, 4*channels], with five zero tokens.
inline std::vector<float> flux2_conditioning_tokens(
        const std::vector<float>& moments, const std::vector<float>& noise,
        const std::vector<float>& running_mean, const std::vector<float>& running_variance,
        std::size_t batch, std::size_t channels, std::size_t height, std::size_t width,
        float epsilon) {
    if (!batch || !channels || !height || !width || height % 2 || width % 2 ||
        !std::isfinite(epsilon) || epsilon <= 0)
        throw std::invalid_argument("invalid FLUX.2 conditioning dimensions or epsilon");
    const auto pixels = checked_product(height, width);
    const auto per_image = checked_product(channels, pixels);
    const auto samples = checked_product(batch, per_image);
    const auto features = checked_product(channels, 4);
    const auto spatial_tokens = pixels / 4;
    if (spatial_tokens > std::numeric_limits<std::size_t>::max() - 5 ||
        moments.size() != checked_product(samples, 2) || noise.size() != samples ||
        running_mean.size() != features || running_variance.size() != features)
        throw std::invalid_argument("invalid FLUX.2 conditioning tensor sizes");
    std::vector<float> inverse_std(features);
    for (std::size_t f = 0; f < features; ++f) {
        if (!std::isfinite(running_mean[f]) || !std::isfinite(running_variance[f]) ||
            running_variance[f] < 0)
            throw std::invalid_argument("invalid FLUX.2 normalization statistics");
        inverse_std[f] = 1.0f / std::sqrt(running_variance[f] + epsilon);
        if (!std::isfinite(inverse_std[f]))
            throw std::invalid_argument("invalid FLUX.2 normalization scale");
    }
    std::vector<float> tokens(checked_product(checked_product(batch, spatial_tokens + 5), features), 0);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t c = 0; c < channels; ++c) {
            for (std::size_t y = 0; y < height; ++y) {
                for (std::size_t x = 0; x < width; ++x) {
                    const auto index = c * pixels + y * width + x;
                    const float mean = moments[b * 2 * per_image + index];
                    const float logvar = moments[b * 2 * per_image + per_image + index];
                    const float z = noise[b * per_image + index];
                    if (!std::isfinite(mean) || !std::isfinite(logvar) || !std::isfinite(z))
                        throw std::invalid_argument("non-finite FLUX.2 latent distribution");
                    const float sample = mean + std::exp(0.5f * std::clamp(logvar, -30.0f, 20.0f)) * z;
                    const auto feature = 4 * c + 2 * (y % 2) + x % 2;
                    const auto token = 5 + (y / 2) * (width / 2) + x / 2;
                    const float value = (sample - running_mean[feature]) * inverse_std[feature];
                    if (!std::isfinite(value))
                        throw std::invalid_argument("non-finite FLUX.2 conditioning token");
                    tokens[(b * (spatial_tokens + 5) + token) * features + feature] = value;
                }
            }
        }
    }
    return tokens;
}

}  // namespace idol::detail
