// Mesh tests: the blur against its closed form, the surface's topology, the piece filter, the OBJ writer, and the input
// checks.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "idol/mesh.hpp"
#include "mesh_internal.hpp"
#include "test.hpp"

namespace {

// Every grid point of a ball of the given radius, one splat per point, each with a high opacity.
idol::SplatCloud ball_of_splats(float radius, float spacing) {
    idol::SplatCloud cloud;
    const int reach = static_cast<int>(std::ceil(radius / spacing));
    for (int i = -reach; i <= reach; ++i) {
        for (int j = -reach; j <= reach; ++j) {
            for (int k = -reach; k <= reach; ++k) {
                const float x = i * spacing;
                const float y = j * spacing;
                const float z = k * spacing;
                if (x * x + y * y + z * z > radius * radius) continue;
                cloud.x.push_back(x);
                cloud.y.push_back(y);
                cloud.z.push_back(z);
                cloud.opacity.push_back(8.0f);
            }
        }
    }
    return cloud;
}

// The number of directed edges that break a closed, consistently oriented surface. Each directed edge must occur once,
// and its reverse must occur once too.
int count_broken_edges(const idol::TriangleMesh& mesh) {
    std::map<std::pair<std::uint32_t, std::uint32_t>, int> directed;
    for (std::size_t t = 0; t < mesh.triangle_count(); ++t) {
        const std::uint32_t* f = &mesh.faces[3 * t];
        for (int e = 0; e < 3; ++e) directed[std::make_pair(f[e], f[(e + 1) % 3])] += 1;
    }
    int broken = 0;
    for (const auto& [edge, count] : directed) {
        const auto reverse = directed.find(std::make_pair(edge.second, edge.first));
        if (count != 1 || reverse == directed.end() || reverse->second != 1) ++broken;
    }
    return broken;
}

}  // namespace

TEST(mesh, blur_of_an_impulse_is_the_separable_truncated_gaussian) {
    const int dims[3] = {9, 9, 9};
    std::vector<float> in(9 * 9 * 9, 0.0f);
    std::vector<float> out(in.size());
    in[(4 * 9 + 4) * 9 + 4] = 1.0f;
    idol::detail::gaussian_blur(in.data(), out.data(), dims, 1.5f, 6);

    // Weights exp(-x^2 / 2 sigma^2) over |x| <= 6, normalised over that window, as scipy's gaussian_filter1d does.
    std::vector<double> weight(13);
    double total = 0.0;
    for (int x = -6; x <= 6; ++x) {
        weight[x + 6] = std::exp(-(x * x) / (2.0 * 1.5 * 1.5));
        total += weight[x + 6];
    }
    const double w0 = weight[6] / total;
    const double w1 = weight[7] / total;
    CHECK_NEAR(out[(4 * 9 + 4) * 9 + 4], w0 * w0 * w0, 1e-7);
    CHECK_NEAR(out[(4 * 9 + 4) * 9 + 5], w0 * w0 * w1, 1e-7);
    CHECK_NEAR(out[(4 * 9 + 5) * 9 + 5], w0 * w1 * w1, 1e-7);
}

TEST(mesh, filled_ball_becomes_a_closed_oriented_surface) {
    const idol::TriangleMesh mesh = idol::splats_to_mesh(ball_of_splats(0.3f, 0.006f));
    CHECK(mesh.triangle_count() > 1000);
    CHECK(count_broken_edges(mesh) == 0);

    double radius_sum = 0.0;
    for (std::size_t v = 0; v < mesh.vertex_count(); ++v) {
        const float* p = &mesh.vertices[3 * v];
        radius_sum += std::sqrt(static_cast<double>(p[0]) * p[0] + static_cast<double>(p[1]) * p[1] +
                                static_cast<double>(p[2]) * p[2]);
    }
    const double mean_radius = radius_sum / static_cast<double>(mesh.vertex_count());
    CHECK(mean_radius > 0.27 && mean_radius < 0.36);
}

TEST(mesh, rejects_bad_options_and_mismatched_columns) {
    idol::SplatCloud cloud;
    cloud.x = {0.0f};
    cloud.y = {0.0f};
    cloud.z = {0.0f};
    cloud.opacity = {1.0f};

    idol::MeshOptions bad;
    bad.resolution = 1;
    CHECK_THROWS(idol::splats_to_mesh(cloud, bad));

    bad = idol::MeshOptions{};
    bad.hi = bad.lo;
    CHECK_THROWS(idol::splats_to_mesh(cloud, bad));

    bad = idol::MeshOptions{};
    bad.blur_sigma = 0.0f;
    CHECK_THROWS(idol::splats_to_mesh(cloud, bad));

    bad = idol::MeshOptions{};
    bad.min_piece = -1.0f;
    CHECK_THROWS(idol::splats_to_mesh(cloud, bad));

    idol::SplatCloud ragged = cloud;
    ragged.y.push_back(0.0f);
    CHECK_THROWS(idol::splats_to_mesh(ragged));
}

TEST(mesh, splats_outside_the_grid_are_an_error) {
    idol::SplatCloud far;
    far.x = {5.0f};
    far.y = {5.0f};
    far.z = {5.0f};
    far.opacity = {1.0f};
    CHECK_THROWS(idol::splats_to_mesh(far));
}

TEST(mesh, keep_large_pieces_drops_small_components_and_compacts) {
    idol::TriangleMesh mesh;
    mesh.vertices.assign(15 * 3, 0.0f);  // positions do not matter here; only the connectivity does
    // A strip of ten triangles on vertices 0..11, and one separate triangle on vertices 12..14.
    for (std::uint32_t i = 0; i < 10; ++i) {
        mesh.faces.push_back(i);
        mesh.faces.push_back(i + 1);
        mesh.faces.push_back(i + 2);
    }
    mesh.faces.push_back(12);
    mesh.faces.push_back(13);
    mesh.faces.push_back(14);

    idol::detail::keep_large_pieces(mesh, 0.5f);
    CHECK(mesh.triangle_count() == 10);
    CHECK(mesh.vertex_count() == 12);
    for (std::uint32_t index : mesh.faces) CHECK(index < mesh.vertex_count());
}

TEST(mesh, percentile99_ignores_non_positive_values) {
    std::vector<float> values = {-5.0f, 0.0f};
    for (int v = 1; v <= 101; ++v) values.push_back(static_cast<float>(v));
    CHECK_NEAR(idol::detail::positive_percentile99(values.data(), values.size()), 100.0, 1e-4);

    const std::vector<float> none = {0.0f, -1.0f};
    CHECK_NEAR(idol::detail::positive_percentile99(none.data(), none.size()), 0.0, 0.0);
}

TEST(mesh, write_obj_writes_one_based_faces) {
    idol::TriangleMesh mesh;
    mesh.vertices = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
    mesh.faces = {0, 1, 2};
    const auto path = std::filesystem::temp_directory_path() / "idol_mesh_test.obj";
    idol::write_obj(path.string(), mesh);

    std::stringstream text;
    {
        std::ifstream in(path);
        text << in.rdbuf();
    }
    const std::string content = text.str();
    CHECK(content.find("f 1 2 3") != std::string::npos);
    std::filesystem::remove(path);
}
