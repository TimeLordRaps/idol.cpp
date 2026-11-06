// Splat-to-mesh conversion: opacity-weighted density, Gaussian blur, marching cubes, and a large-piece filter.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace idol {

// Splat centres and opacity logits, one entry per Gaussian: the PLY columns x, y, z and opacity.
struct SplatCloud {
    std::vector<float> x, y, z, opacity;

    std::size_t size() const { return x.size(); }
};

// Settings for the conversion. The defaults are the ones the idol sprite pipeline uses.
struct MeshOptions {
    double lo = -0.55;          // the grid covers [lo, hi] on each axis; splat coordinates sit inside about +/-0.5
    double hi = 0.55;
    int resolution = 176;       // voxels per axis
    float blur_sigma = 1.5f;    // Gaussian sigma in voxels; the kernel is truncated at round(4 * sigma)
    float iso_fraction = 0.2f;  // the surface sits at this fraction of the 99th-percentile density
    float min_piece = 0.05f;    // keep connected pieces with at least this fraction of the largest piece's faces
};

struct TriangleMesh {
    std::vector<float> vertices;       // x, y, z per vertex
    std::vector<std::uint32_t> faces;  // three vertex indices per triangle

    std::size_t vertex_count() const { return vertices.size() / 3; }
    std::size_t triangle_count() const { return faces.size() / 3; }
};

// Converts a splat cloud to a triangle mesh in the same world units as the splats. Throws std::invalid_argument for
// bad options or columns of different lengths, and when no splat falls inside the grid.
TriangleMesh splats_to_mesh(const SplatCloud& splats, const MeshOptions& options = {});

// Writes a Wavefront OBJ file. Throws std::runtime_error if the file cannot be written.
void write_obj(const std::string& path, const TriangleMesh& mesh);

}  // namespace idol
