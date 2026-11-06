#include <fstream>
#include <vector>
#include "flux2_tokens.hpp"

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    constexpr std::size_t batch = 2, channels = 3, height = 4, width = 6;
    constexpr std::size_t per_image = channels * height * width;
    std::vector<float> moments(batch * 2 * per_image), noise(batch * per_image);
    std::vector<float> mean(channels * 4), variance(channels * 4);
    for (std::size_t b = 0; b < batch; ++b) {
        for (std::size_t i = 0; i < per_image; ++i) {
            const auto index = b * per_image + i;
            moments[b * 2 * per_image + i] = (static_cast<int>(index % 17) - 8) * 0.125f;
            moments[b * 2 * per_image + per_image + i] = (static_cast<int>((index * 7) % 19) - 9) * 0.125f;
            noise[index] = (static_cast<int>((index * 3) % 13) - 6) * 0.25f;
        }
    }
    for (std::size_t i = 0; i < mean.size(); ++i) {
        mean[i] = (static_cast<int>(i) - 6) * 0.1f;
        variance[i] = static_cast<float>(i + 1) * 0.25f;
    }
    const auto tokens = idol::detail::flux2_conditioning_tokens(
        moments, noise, mean, variance, batch, channels, height, width, 1e-5f);
    std::ofstream output(argv[1], std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(tokens.data()),
                 static_cast<std::streamsize>(tokens.size() * sizeof(float)));
    return output ? 0 : 1;
}
