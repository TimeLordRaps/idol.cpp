#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace idol::detail {

struct OctreeCellComparison {
    double total_variation = 0;
    std::size_t reference_occupied = 0, candidate_occupied = 0, shared_occupied = 0;
};

struct OctreeProbabilityComparison {
    double mean_total_variation = 0, max_total_variation = 0;
};

// Each row contains eight child logits. Total variation compares the resulting categorical probabilities.
inline OctreeProbabilityComparison compare_octree_probabilities(const std::vector<float>& reference,
                                                                 const std::vector<float>& candidate,
                                                                 const std::vector<std::uint64_t>& weights = {}) {
    if (reference.empty() || reference.size() % 8 || reference.size() != candidate.size())
        throw std::invalid_argument("invalid octree logit arrays");
    if (!weights.empty() && weights.size() != reference.size() / 8)
        throw std::invalid_argument("invalid octree probability weights");
    OctreeProbabilityComparison result;
    double total_weight = 0;
    for (std::size_t i = 0; i < reference.size(); i += 8) {
        double max_ref = -std::numeric_limits<double>::infinity(), max_got = max_ref;
        for (int j = 0; j < 8; ++j) {
            if (!std::isfinite(reference[i + j]) || !std::isfinite(candidate[i + j]))
                throw std::invalid_argument("non-finite octree logits");
            max_ref = std::max(max_ref, static_cast<double>(reference[i + j]));
            max_got = std::max(max_got, static_cast<double>(candidate[i + j]));
        }
        double ref[8], got[8], sum_ref = 0, sum_got = 0;
        for (int j = 0; j < 8; ++j) {
            sum_ref += (ref[j] = std::exp(static_cast<double>(reference[i + j]) - max_ref));
            sum_got += (got[j] = std::exp(static_cast<double>(candidate[i + j]) - max_got));
        }
        double variation = 0;
        for (int j = 0; j < 8; ++j) variation += std::abs(ref[j] / sum_ref - got[j] / sum_got);
        variation = std::min(1.0, variation * 0.5);
        const double weight = weights.empty() ? 1.0 : static_cast<double>(weights[i / 8]);
        result.mean_total_variation += variation * weight;
        total_weight += weight;
        if (weight > 0) result.max_total_variation = std::max(result.max_total_variation, variation);
    }
    if (total_weight == 0) throw std::invalid_argument("zero octree probability weight");
    result.mean_total_variation /= total_weight;
    return result;
}

struct OctreeVisitedParents {
    std::vector<float> centers;
    std::vector<std::uint64_t> counts;
};

// A sample exactly on an interior cell boundary may have rounded there from the neighboring leaf. Reject it instead
// of silently claiming the wrong reference path. Zero and one have only one valid neighboring cell in the domain.
inline OctreeVisitedParents reference_parents_from_samples(const std::vector<float>& points, int child_level) {
    if (child_level < 1 || child_level > 20 || points.empty() || points.size() % 3)
        throw std::invalid_argument("invalid sampled anchors or child level");
    const std::uint64_t resolution = std::uint64_t{1} << (child_level - 1);
    std::unordered_map<std::uint64_t, std::size_t> indices;
    OctreeVisitedParents result;
    for (std::size_t i = 0; i < points.size(); i += 3) {
        std::uint64_t coord[3];
        for (int j = 0; j < 3; ++j) {
            const float value = points[i + j];
            if (!std::isfinite(value) || value < 0 || value > 1) throw std::invalid_argument("invalid sampled anchor");
            const double scaled = static_cast<double>(value) * resolution;
            if (scaled > 0 && scaled < resolution && std::floor(scaled) == scaled)
                throw std::invalid_argument("sample on interior parent boundary; reference path is ambiguous");
            coord[j] = std::min(static_cast<std::uint64_t>(scaled), resolution - 1);
        }
        const std::uint64_t key = (coord[0] * resolution + coord[1]) * resolution + coord[2];
        const auto inserted = indices.emplace(key, result.counts.size());
        if (inserted.second) {
            for (std::uint64_t c : coord) result.centers.push_back(static_cast<float>((c + 0.5) / resolution));
            result.counts.push_back(0);
        }
        ++result.counts[inserted.first->second];
    }
    return result;
}

inline OctreeCellComparison compare_octree_cells(const std::vector<float>& reference,
                                                  const std::vector<float>& candidate, int level) {
    if (level < 1 || level > 20 || reference.empty() || reference.size() % 3 || reference.size() != candidate.size())
        throw std::invalid_argument("invalid anchor arrays or level");
    const std::uint64_t resolution = std::uint64_t{1} << level;
    auto counts = [resolution](const std::vector<float>& points) {
        std::unordered_map<std::uint64_t, std::uint64_t> result;
        for (std::size_t i = 0; i < points.size(); i += 3) {
            std::uint64_t coord[3];
            for (int j = 0; j < 3; ++j) {
                const float x = points[i + j];
                if (!std::isfinite(x) || x < 0 || x > 1) throw std::invalid_argument("invalid sampled anchor");
                coord[j] = std::min(static_cast<std::uint64_t>(x * resolution), resolution - 1);
            }
            ++result[(coord[0] * resolution + coord[1]) * resolution + coord[2]];
        }
        return result;
    };
    const auto left = counts(reference), right = counts(candidate);
    std::uint64_t difference = 0;
    std::size_t shared = 0;
    for (const auto& item : left) {
        const auto found = right.find(item.first);
        const std::uint64_t other = found == right.end() ? 0 : found->second;
        difference += item.second > other ? item.second - other : other - item.second;
        if (other) ++shared;
    }
    for (const auto& item : right) if (left.find(item.first) == left.end()) difference += item.second;
    return {static_cast<double>(difference) / (2.0 * (reference.size() / 3)), left.size(), right.size(), shared};
}

}  // namespace idol::detail
