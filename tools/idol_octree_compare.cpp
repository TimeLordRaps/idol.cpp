// Compare sampled octree leaf-cell distributions on the same latent and seed.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "idol/decoders.hpp"
#include "octree_compare.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
int parse_integer(const std::string& value) {
    std::size_t used = 0;
    const int result = std::stoi(value, &used);
    if (used != value.size()) throw std::invalid_argument("invalid integer: " + value);
    return result;
}

std::uint64_t parse_seed(const std::string& value) {
    if (value.empty() || value[0] == '-') throw std::invalid_argument("invalid seed");
    std::size_t used = 0;
    const auto result = std::stoull(value, &used);
    if (used != value.size()) throw std::invalid_argument("invalid seed");
    return static_cast<std::uint64_t>(result);
}

std::vector<float> read_latent(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    const auto bytes = in.tellg();
    if (bytes <= 0 || bytes % sizeof(float)) throw std::runtime_error("invalid latent file size");
    std::vector<float> data(static_cast<std::size_t>(bytes) / sizeof(float));
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(data.data()), bytes)) throw std::runtime_error("cannot read latent");
    for (float v : data) if (!std::isfinite(v)) throw std::runtime_error("non-finite latent");
    return data;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4) throw std::invalid_argument("usage: idol_octree_compare REFERENCE CANDIDATE LATENT "
                                                  "[--candidate EXPORT]... [--anchors N] [--level N] [--seed N] "
                                                  "[--candidate-seed N] [--noise-seed N] [--compare-levels N,...] "
                                                  "[--probability-levels N,...] [--probability-only] [--threads N]");
        int anchors = 8192, level = 8, threads = 0;
        std::uint64_t seed = 0, candidate_seed = 0, noise_seed = 0;
        bool has_candidate_seed = false, has_noise_seed = false, probability_only = false;
        std::vector<std::string> candidates = {argv[2]};
        std::vector<int> compare_levels, probability_levels;
        for (int i = 4; i < argc; ++i) {
            const std::string flag = argv[i];
            if (flag == "--probability-only") { probability_only = true; continue; }
            if (++i >= argc) throw std::invalid_argument(flag + " needs a value");
            const std::string value = argv[i];
            if (flag == "--anchors") anchors = parse_integer(value);
            else if (flag == "--level") level = parse_integer(value);
            else if (flag == "--threads") threads = parse_integer(value);
            else if (flag == "--seed") seed = parse_seed(value);
            else if (flag == "--candidate-seed") { candidate_seed = parse_seed(value); has_candidate_seed = true; }
            else if (flag == "--noise-seed") { noise_seed = parse_seed(value); has_noise_seed = true; }
            else if (flag == "--candidate") candidates.push_back(value);
            else if (flag == "--compare-levels" || flag == "--probability-levels") {
                auto& levels = flag == "--compare-levels" ? compare_levels : probability_levels;
                std::size_t start = 0;
                while (true) {
                    const std::size_t end = value.find(',', start);
                    const int selected = parse_integer(value.substr(start, end - start));
                    if (std::find(levels.begin(), levels.end(), selected) != levels.end())
                        throw std::invalid_argument("duplicate comparison level");
                    levels.push_back(selected);
                    if (end == std::string::npos) break;
                    start = end + 1;
                }
            }
            else throw std::invalid_argument("unknown option " + flag);
        }
        if (anchors < 1 || level < 1 || level > 20 || threads < 0) throw std::invalid_argument("invalid parameters");
        if (compare_levels.empty()) compare_levels.push_back(level);
        for (int selected : compare_levels)
            if (selected < 1 || selected > level) throw std::invalid_argument("comparison level must be within sampled depth");
        if (probability_only && has_noise_seed) throw std::invalid_argument("--noise-seed is not used with --probability-only");
        if (probability_only && probability_levels.empty())
            for (int selected = 1; selected <= level; ++selected) probability_levels.push_back(selected);
        for (int selected : probability_levels)
            if (selected < 1 || selected > level) throw std::invalid_argument("probability level must be within sampled depth");
#ifdef _OPENMP
        if (threads > 0) omp_set_num_threads(threads);
#endif
        idol::Decoders reference(argv[1]);
        if (reference.config().cond <= 0) throw std::runtime_error("invalid conditioning dimension");
        const auto latent = read_latent(argv[3]);
        if (latent.size() % reference.config().cond ||
            latent.size() / reference.config().cond > static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::runtime_error("invalid latent token count");
        const int tokens = static_cast<int>(latent.size() / reference.config().cond);
        const auto ref = reference.sample_anchors(latent.data(), tokens, anchors, level, seed);
        if (!has_candidate_seed) candidate_seed = seed;
        if (ref.size() != static_cast<std::size_t>(anchors) * 3) throw std::runtime_error("unexpected reference anchor count");
        struct ParentReference { int level; idol::detail::OctreeVisitedParents parents; std::vector<float> logits; };
        std::vector<ParentReference> parent_references;
        for (int selected : probability_levels) {
            auto parents = idol::detail::reference_parents_from_samples(ref, selected);
            std::vector<float> logits(parents.counts.size() * 8);
            reference.octree_logits(parents.centers.data(), static_cast<int>(parents.counts.size()),
                                    static_cast<float>(std::uint64_t{1} << selected), latent.data(), tokens, logits.data());
            parent_references.push_back({selected, std::move(parents), std::move(logits)});
        }
        auto report = [&](long long index, std::uint64_t compared_seed, const std::vector<float>& got) {
            if (got.size() != ref.size()) throw std::runtime_error("unexpected candidate anchor count");
            for (int selected : compare_levels) {
                const auto comparison = idol::detail::compare_octree_cells(ref, got, selected);
                std::printf("RESULT candidate=%lld anchors=%d tokens=%d level=%d cell_level=%d seed=%llu candidate_seed=%llu "
                            "cell_total_variation=%.9f reference_occupied=%zu candidate_occupied=%zu shared_occupied=%llu\n",
                            index, anchors, tokens, level, selected, static_cast<unsigned long long>(seed),
                            static_cast<unsigned long long>(compared_seed), comparison.total_variation,
                            comparison.reference_occupied, comparison.candidate_occupied,
                            static_cast<unsigned long long>(comparison.shared_occupied));
                std::fflush(stdout);
            }
        };
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            idol::Decoders candidate(candidates[index]);
            if (reference.config().cond != candidate.config().cond) throw std::runtime_error("conditioning dimensions differ");
            if (!probability_only)
                report(static_cast<long long>(index), candidate_seed,
                       candidate.sample_anchors(latent.data(), tokens, anchors, level, candidate_seed));
            for (const auto& entry : parent_references) {
                std::vector<float> logits(entry.logits.size());
                candidate.octree_logits(entry.parents.centers.data(), static_cast<int>(entry.parents.counts.size()),
                                        static_cast<float>(std::uint64_t{1} << entry.level), latent.data(), tokens, logits.data());
                const auto comparison = idol::detail::compare_octree_probabilities(entry.logits, logits, entry.parents.counts);
                std::printf("PROB_RESULT candidate=%zu anchors=%d tokens=%d sample_level=%d level=%d parents=%zu "
                            "weighted_mean_tv=%.9f max_tv=%.9f\n", index, anchors, tokens, level, entry.level,
                            entry.parents.counts.size(), comparison.mean_total_variation, comparison.max_total_variation);
                std::fflush(stdout);
            }
        }
        if (has_noise_seed) report(-1, noise_seed, reference.sample_anchors(latent.data(), tokens, anchors, level, noise_seed));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "idol_octree_compare: %s\n", e.what());
        return 2;
    }
}
