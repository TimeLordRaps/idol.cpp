#include <limits>
#include <vector>

#include "flux2_tokens.hpp"
#include "test.hpp"

TEST(flux2_tokens, samples_patchifies_normalizes_and_pads_each_batch) {
    // Two batches, two channels, 2x2 pixels: both the batch and channel axes
    // matter. Each channel's four pixels become four adjacent features.
    const std::vector<float> moments = {
        0,1,2,3, 10,11,12,13, 0,0,0,0, 0,0,0,0,
        20,21,22,23, 30,31,32,33, 0,0,0,0, 0,0,0,0};
    const std::vector<float> noise(16, 1.0f);
    const std::vector<float> means = {1,2,3,4, 11,12,13,14};
    const std::vector<float> variance(8, 3.0f);
    const auto result = idol::detail::flux2_conditioning_tokens(moments, noise, means, variance, 2, 2, 2, 2, 1.0f);
    CHECK(result.size() == 2 * 6 * 8);
    for (int batch = 0; batch < 2; ++batch) {
        for (int i = 0; i < 40; ++i) CHECK_NEAR(result[batch * 48 + i], 0, 0);
        for (int i = 0; i < 8; ++i) CHECK_NEAR(result[batch * 48 + 40 + i], batch * 10.0f, 1e-6);
    }
}

TEST(flux2_tokens, keeps_spatial_token_order_and_clamps_log_variance) {
    const std::vector<float> moments = {0,1,2,3,4,5,6,7, -30,-30,20,20,0,0,0,0};
    const auto result = idol::detail::flux2_conditioning_tokens(
        moments, std::vector<float>(8, 0), std::vector<float>(4, 0),
        std::vector<float>(4, 0), 1, 1, 2, 4, 1.0f);
    CHECK(result.size() == 28);
    const std::vector<float> expected = {0,1,4,5, 2,3,6,7};
    for (int i = 0; i < 8; ++i) CHECK_NEAR(result[20 + i], expected[i], 0);
    auto noisy = moments;
    noisy[8] = 30;
    const auto output = idol::detail::flux2_conditioning_tokens(
        noisy, std::vector<float>(8, 1), std::vector<float>(4, 0),
        std::vector<float>(4, 0), 1, 1, 2, 4, 1.0f);
    CHECK_NEAR(output[20], std::exp(10.0f), 0.01);
}

TEST(flux2_tokens, rejects_invalid_shapes_and_statistics) {
    const std::vector<float> moments(8, 0), noise(4, 0), stats(4, 0);
    CHECK_THROWS(idol::detail::flux2_conditioning_tokens(moments, noise, stats, stats, 1, 1, 1, 4, 1));
    CHECK_THROWS(idol::detail::flux2_conditioning_tokens(moments, noise, stats, stats, 1, 1, 2, 2, 0));
    CHECK_THROWS(idol::detail::flux2_conditioning_tokens(moments, noise, stats, {0,0,-1,0}, 1, 1, 2, 2, 1));
    auto invalid = moments;
    invalid[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS(idol::detail::flux2_conditioning_tokens(invalid, noise, stats, stats, 1, 1, 2, 2, 1));
}
