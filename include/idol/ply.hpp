// Reads the splat columns of a PLY file.
#pragma once

#include <string>

#include "idol/mesh.hpp"

namespace idol {

// Reads x, y, z and opacity from the vertex element of a binary little-endian PLY file. Other vertex properties are
// skipped. Throws std::runtime_error for a file that is not PLY, is not binary little-endian, lacks one of the four
// columns, or is truncated.
SplatCloud read_splat_ply(const std::string& path);

}  // namespace idol
