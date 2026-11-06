// Mesh kernels shared by splats_to_mesh and the tests. Internal; not part of the public API.
#pragma once

#include <cstddef>

#include "idol/mesh.hpp"

namespace idol::detail {

// Gaussian blur of a row-major dims[0] x dims[1] x dims[2] array, truncated at radius voxels. Past the array's edge
// the index reflects back in, as scipy's gaussian_filter does by default (d c b a | a b c d | d c b a).
void gaussian_blur(const float* in, float* out, const int dims[3], float sigma, int radius);

// Marching cubes over density at level. A corner is inside where density >= level. Each grid edge that crosses the
// level gets one vertex, shared by the cubes around it. Positions are index * step + origin.
TriangleMesh march_cubes(const float* density, const int dims[3], float level, float step, const float origin[3]);

// Keeps the connected pieces whose face count is at least min_piece times the largest piece's, and compacts the
// vertices. Only the faces of kept pieces survive, in their original order.
void keep_large_pieces(TriangleMesh& mesh, float min_piece);

// numpy's 99th percentile (linear interpolation) of the positive entries. Returns 0 when there are none.
float positive_percentile99(const float* values, std::size_t n);

}  // namespace idol::detail
