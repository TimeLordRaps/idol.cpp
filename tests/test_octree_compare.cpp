#include <limits>
#include <vector>

#include "octree_compare.hpp"
#include "test.hpp"

TEST(octree_compare, compares_count_distributions_and_boundaries) {
    const std::vector<float> left = {0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
                                     0.1f, 0.1f, 0.1f, 0.9f, 0.9f, 0.9f};
    const std::vector<float> right = {0.1f, 0.1f, 0.1f, 0.9f, 0.9f, 0.9f,
                                      0.9f, 0.9f, 0.9f, 1.0f, 1.0f, 1.0f};
    const auto same = idol::detail::compare_octree_cells(left, left, 1);
    CHECK_NEAR(same.total_variation, 0, 0);
    const auto changed = idol::detail::compare_octree_cells(left, right, 1);
    CHECK_NEAR(changed.total_variation, 0.5, 0);
    CHECK(changed.reference_occupied == 2 && changed.candidate_occupied == 2 && changed.shared_occupied == 2);
    const auto disjoint = idol::detail::compare_octree_cells({0, 0, 0}, {1, 1, 1}, 1);
    CHECK_NEAR(disjoint.total_variation, 1, 0);
    CHECK(disjoint.shared_occupied == 0);
}

TEST(octree_compare, rejects_invalid_inputs) {
    const std::vector<float> one = {0.1f, 0.1f, 0.1f};
    CHECK_THROWS(idol::detail::compare_octree_cells({}, {}, 1));
    CHECK_THROWS(idol::detail::compare_octree_cells(one, {0.1f}, 1));
    CHECK_THROWS(idol::detail::compare_octree_cells(one, one, 21));
    CHECK_THROWS(idol::detail::compare_octree_cells(one, {1.1f, 0, 0}, 1));
    CHECK_THROWS(idol::detail::compare_octree_cells(one, {std::numeric_limits<float>::quiet_NaN(), 0, 0}, 1));
}

TEST(octree_compare, child_probability_distance_ignores_logit_offsets) {
    const std::vector<float> flat(8, 0), shifted(8, 100);
    const auto same = idol::detail::compare_octree_probabilities(flat, shifted);
    CHECK_NEAR(same.mean_total_variation, 0, 0);
    CHECK_NEAR(same.max_total_variation, 0, 0);
    const std::vector<float> left = {100, 0, 0, 0, 0, 0, 0, 0};
    const std::vector<float> right = {0, 100, 0, 0, 0, 0, 0, 0};
    const auto changed = idol::detail::compare_octree_probabilities(left, right);
    CHECK(changed.mean_total_variation > 0.999 && changed.mean_total_variation <= 1.0);
    std::vector<float> reference = flat, candidate = shifted;
    reference.insert(reference.end(), left.begin(), left.end());
    candidate.insert(candidate.end(), right.begin(), right.end());
    const auto weighted = idol::detail::compare_octree_probabilities(reference, candidate, {3, 1});
    CHECK_NEAR(weighted.mean_total_variation, 0.25, 1e-8);
    CHECK_THROWS(idol::detail::compare_octree_probabilities(reference, candidate, {0, 0}));
    CHECK_THROWS(idol::detail::compare_octree_probabilities(reference, candidate, {1}));
    CHECK_THROWS(idol::detail::compare_octree_probabilities(flat, {0}));
    auto invalid = flat;
    invalid[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS(idol::detail::compare_octree_probabilities(flat, invalid));
}

TEST(octree_compare, derives_only_unambiguous_reference_parent_cells) {
    const auto parents = idol::detail::reference_parents_from_samples(
        {0.1f, 0.1f, 0.1f, 0.9f, 0.9f, 0.9f, 0.1f, 0.1f, 0.1f}, 2);
    CHECK(parents.counts.size() == 2 && parents.counts[0] == 2 && parents.counts[1] == 1);
    CHECK_NEAR(parents.centers[0], 0.25, 0);
    CHECK_NEAR(parents.centers[3], 0.75, 0);
    const auto endpoints = idol::detail::reference_parents_from_samples({0, 0, 0, 1, 1, 1}, 2);
    CHECK(endpoints.counts.size() == 2);
    CHECK_THROWS(idol::detail::reference_parents_from_samples({0.5f, 0.1f, 0.1f}, 2));
    CHECK_THROWS(idol::detail::reference_parents_from_samples({0.1f, 0.1f, 0.1f}, 0));
}
