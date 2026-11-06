#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "idol/mesh.hpp"
#include "mc_tables.h"
#include "mesh_internal.hpp"

namespace idol {

namespace {

// Cube corners in the classic order. Bit c of a case index is set where the density at corner c is inside.
constexpr int kCorner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};

std::vector<double> gaussian_weights(float sigma, int radius) {
    std::vector<double> w(2 * radius + 1);
    double total = 0.0;
    for (int k = -radius; k <= radius; ++k) {
        w[k + radius] = std::exp(-0.5 * k * k / ((double)sigma * sigma));
        total += w[k + radius];
    }
    for (double& weight : w) weight /= total;
    return w;
}

// Blur along axis 2 (the contiguous one), one line at a time.
void blur_lines(const float* in, float* out, const int dims[3], const double* w, int radius) {
    const int n = dims[2];
#pragma omp parallel for collapse(2) schedule(static)
    for (int x = 0; x < dims[0]; ++x)
        for (int y = 0; y < dims[1]; ++y) {
            const std::size_t base = ((std::size_t)x * dims[1] + y) * n;
            for (int t = 0; t < n; ++t) {
                double acc = 0.0;
                if (t >= radius && t + radius < n) {
                    const float* p = in + base + (t - radius);
                    for (int k = 0; k <= 2 * radius; ++k) acc += w[k] * p[k];
                } else {
                    for (int k = -radius; k <= radius; ++k) {
                        int s = t + k;
                        while (s < 0 || s >= n) s = s < 0 ? -s - 1 : 2 * n - 1 - s;
                        acc += w[k + radius] * in[base + s];
                    }
                }
                out[base + t] = (float)acc;
            }
        }
}

// Blur along axis 0 or 1, written plane by plane: each output plane is a weighted sum of whole input planes, so the
// inner loop runs over contiguous memory. The sums run in the same order as blur_lines.
void blur_planes(const float* in, float* out, const int dims[3], int axis, const double* w, int radius) {
    const int n = dims[axis];
    const int outer = axis == 0 ? 1 : dims[0];
    const std::size_t inner = axis == 0 ? (std::size_t)dims[1] * dims[2] : (std::size_t)dims[2];
#pragma omp parallel for collapse(2) schedule(static)
    for (int o = 0; o < outer; ++o)
        for (int t = 0; t < n; ++t) {
            std::vector<double> acc(inner, 0.0);
            for (int k = -radius; k <= radius; ++k) {
                int s = t + k;
                while (s < 0 || s >= n) s = s < 0 ? -s - 1 : 2 * n - 1 - s;
                const float* src = in + ((std::size_t)o * n + s) * inner;
                const double wk = w[k + radius];
                for (std::size_t j = 0; j < inner; ++j) acc[j] += wk * src[j];
            }
            float* dst = out + ((std::size_t)o * n + t) * inner;
            for (std::size_t j = 0; j < inner; ++j) dst[j] = (float)acc[j];
        }
}

std::int32_t find_root(std::vector<std::int32_t>& parent, std::int32_t v) {
    while (parent[v] != v) {
        parent[v] = parent[parent[v]];
        v = parent[v];
    }
    return v;
}

void unite(std::vector<std::int32_t>& parent, std::int32_t a, std::int32_t b) {
    a = find_root(parent, a);
    b = find_root(parent, b);
    if (a != b) parent[std::max(a, b)] = std::min(a, b);
}

}  // namespace

namespace detail {

void gaussian_blur(const float* in, float* out, const int dims[3], float sigma, int radius) {
    const std::vector<double> w = gaussian_weights(sigma, radius);
    const std::size_t count = (std::size_t)dims[0] * dims[1] * dims[2];
    std::vector<float> t1(count), t2(count);
    blur_planes(in, t1.data(), dims, 0, w.data(), radius);
    blur_planes(t1.data(), t2.data(), dims, 1, w.data(), radius);
    blur_lines(t2.data(), out, dims, w.data(), radius);
}

TriangleMesh march_cubes(const float* density, const int dims[3], float level, float step, const float origin[3]) {
    const int nx = dims[0], ny = dims[1], nz = dims[2];
    auto point = [&](int x, int y, int z) { return ((std::size_t)x * ny + y) * nz + z; };
    auto inside = [&](std::size_t p) { return density[p] >= level; };

    // Per edge: the axis it runs along, and the lower endpoint's offset from the cube's corner.
    int edge_axis[12], edge_min[12][3];
    for (int e = 0; e < 12; ++e) {
        edge_axis[e] = 0;
        for (int k = 0; k < 3; ++k) {
            edge_min[e][k] = std::min(kEdgeOffsets[e][0][k], kEdgeOffsets[e][1][k]);
            if (kEdgeOffsets[e][0][k] != kEdgeOffsets[e][1][k]) edge_axis[e] = k;
        }
    }
    int triangles[256];
    for (int kase = 0; kase < 256; ++kase) {
        int len = 0;
        while (len < 16 && kCases[kase][len] >= 0) ++len;
        triangles[kase] = len / 3;
    }

    const int cx = nx - 1, cy = ny - 1, cz = nz - 1;
    auto cell = [&](int x, int y, int z) { return ((std::size_t)x * cy + y) * cz + z; };
    std::vector<std::uint8_t> kase_of((std::size_t)cx * cy * cz);
#pragma omp parallel for schedule(static)
    for (int x = 0; x < cx; ++x)
        for (int y = 0; y < cy; ++y)
            for (int z = 0; z < cz; ++z) {
                int kase = 0;
                for (int c = 0; c < 8; ++c)
                    if (inside(point(x + kCorner[c][0], y + kCorner[c][1], z + kCorner[c][2]))) kase |= 1 << c;
                kase_of[cell(x, y, z)] = (std::uint8_t)kase;
            }

    // Vertices. Each grid point owns the edges that leave it along +x, +y and +z. The vertices of one x plane are
    // numbered together, so a count pass gives every plane its first index and the planes can be filled in parallel.
    const int dimv[3] = {nx, ny, nz};
    std::vector<std::size_t> vertex_base(nx + 1, 0);
    {
        std::vector<std::size_t> per_x(nx, 0);
#pragma omp parallel for schedule(static)
        for (int x = 0; x < nx; ++x) {
            std::size_t count = 0;
            for (int y = 0; y < ny; ++y)
                for (int z = 0; z < nz; ++z) {
                    const bool in = inside(point(x, y, z));
                    if (x + 1 < nx && in != inside(point(x + 1, y, z))) ++count;
                    if (y + 1 < ny && in != inside(point(x, y + 1, z))) ++count;
                    if (z + 1 < nz && in != inside(point(x, y, z + 1))) ++count;
                }
            per_x[x] = count;
        }
        for (int x = 0; x < nx; ++x) vertex_base[x + 1] = vertex_base[x] + per_x[x];
    }
    TriangleMesh mesh;
    mesh.vertices.resize(3 * vertex_base[nx]);
    std::unique_ptr<std::uint32_t[]> edge_vertex(new std::uint32_t[(std::size_t)nx * ny * nz * 3]);
#pragma omp parallel for schedule(static)
    for (int x = 0; x < nx; ++x) {
        std::size_t id = vertex_base[x];
        for (int y = 0; y < ny; ++y)
            for (int z = 0; z < nz; ++z) {
                const std::size_t p = point(x, y, z);
                const int at[3] = {x, y, z};
                for (int k = 0; k < 3; ++k) {
                    if (at[k] + 1 >= dimv[k]) continue;
                    int to[3] = {x, y, z};
                    to[k] += 1;
                    const std::size_t q = point(to[0], to[1], to[2]);
                    if (inside(p) == inside(q)) continue;
                    const float t = (level - density[p]) / (density[q] - density[p]);
                    edge_vertex[p * 3 + k] = (std::uint32_t)id;
                    for (int j = 0; j < 3; ++j) {
                        const float along = j == k ? t : 0.0f;
                        mesh.vertices[3 * id + j] = (at[j] + along) * step + origin[j];
                    }
                    ++id;
                }
            }
    }

    // Triangles. Each cube's table entry names its edges, whose shared vertices were numbered above.
    std::vector<std::size_t> face_base(nx + 1, 0);
    {
        std::vector<std::size_t> per_x(nx, 0);
#pragma omp parallel for schedule(static)
        for (int x = 0; x < cx; ++x) {
            std::size_t count = 0;
            for (int y = 0; y < cy; ++y)
                for (int z = 0; z < cz; ++z) count += triangles[kase_of[cell(x, y, z)]];
            per_x[x] = count;
        }
        for (int x = 0; x < nx; ++x) face_base[x + 1] = face_base[x] + per_x[x];
    }
    mesh.faces.resize(3 * face_base[nx]);
#pragma omp parallel for schedule(static)
    for (int x = 0; x < cx; ++x) {
        std::size_t f = face_base[x];
        for (int y = 0; y < cy; ++y)
            for (int z = 0; z < cz; ++z) {
                const int kase = kase_of[cell(x, y, z)];
                for (int i = 0; kCases[kase][i] >= 0; i += 3, ++f)
                    for (int j = 0; j < 3; ++j) {
                        const int e = kCases[kase][i + j];
                        const std::size_t corner = point(x + edge_min[e][0], y + edge_min[e][1], z + edge_min[e][2]);
                        mesh.faces[3 * f + j] = edge_vertex[corner * 3 + edge_axis[e]];
                    }
            }
    }
    return mesh;
}

void keep_large_pieces(TriangleMesh& mesh, float min_piece) {
    const std::size_t nv = mesh.vertex_count(), nf = mesh.triangle_count();
    std::vector<std::int32_t> parent(nv);
    std::iota(parent.begin(), parent.end(), 0);
    for (std::size_t f = 0; f < nf; ++f) {
        const std::uint32_t* t = &mesh.faces[3 * f];
        unite(parent, (std::int32_t)t[0], (std::int32_t)t[1]);
        unite(parent, (std::int32_t)t[1], (std::int32_t)t[2]);
    }
    std::vector<std::int32_t> piece(nf), faces_in_piece(nv, 0);
    std::int32_t largest = 0;
    for (std::size_t f = 0; f < nf; ++f) {
        piece[f] = find_root(parent, (std::int32_t)mesh.faces[3 * f]);
        largest = std::max(largest, ++faces_in_piece[piece[f]]);
    }
    const double threshold = (double)min_piece * largest;
    auto kept = [&](std::size_t f) { return faces_in_piece[piece[f]] >= threshold; };

    std::vector<std::uint8_t> used(nv, 0);
    for (std::size_t f = 0; f < nf; ++f)
        if (kept(f))
            for (int j = 0; j < 3; ++j) used[mesh.faces[3 * f + j]] = 1;

    std::vector<std::uint32_t> remap(nv, 0);
    std::vector<float> vertices;
    std::uint32_t count = 0;
    for (std::size_t v = 0; v < nv; ++v)
        if (used[v]) {
            remap[v] = count++;
            vertices.insert(vertices.end(), &mesh.vertices[3 * v], &mesh.vertices[3 * v] + 3);
        }
    std::vector<std::uint32_t> faces;
    for (std::size_t f = 0; f < nf; ++f)
        if (kept(f))
            for (int j = 0; j < 3; ++j) faces.push_back(remap[mesh.faces[3 * f + j]]);
    mesh.vertices.swap(vertices);
    mesh.faces.swap(faces);
}

float positive_percentile99(const float* values, std::size_t n) {
    std::vector<float> pos;
    pos.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
        if (values[i] > 0) pos.push_back(values[i]);
    if (pos.empty()) return 0.0f;
    const double at = 0.99 * (double)(pos.size() - 1);
    const std::size_t lo = (std::size_t)std::floor(at);
    std::nth_element(pos.begin(), pos.begin() + lo, pos.end());
    const float low = pos[lo];
    if (lo + 1 >= pos.size()) return low;
    const float high = *std::min_element(pos.begin() + lo + 1, pos.end());
    return low + (float)(at - (double)lo) * (high - low);
}

}  // namespace detail

TriangleMesh splats_to_mesh(const SplatCloud& splats, const MeshOptions& o) {
    if (o.resolution < 2) throw std::invalid_argument("MeshOptions::resolution must be at least 2");
    if (!(o.hi > o.lo)) throw std::invalid_argument("MeshOptions::hi must be greater than lo");
    if (!(o.blur_sigma > 0.0f)) throw std::invalid_argument("MeshOptions::blur_sigma must be positive");
    if (!(o.iso_fraction > 0.0f)) throw std::invalid_argument("MeshOptions::iso_fraction must be positive");
    if (!(o.min_piece >= 0.0f)) throw std::invalid_argument("MeshOptions::min_piece must not be negative");
    const std::size_t n = splats.size();
    if (splats.y.size() != n || splats.z.size() != n || splats.opacity.size() != n)
        throw std::invalid_argument("SplatCloud columns must have equal lengths");

    const int res = o.resolution;
    const float lo = (float)o.lo;
    const float step = (float)((o.hi - o.lo) / res);
    const int radius = (int)std::floor(4.0 * o.blur_sigma + 0.5);  // the kernel's truncation: scipy's rule

    // Scatter each splat's opacity weight into its voxel. Splats outside the grid are dropped.
    std::vector<std::array<int, 3>> cells;
    std::vector<float> weights;
    cells.reserve(n);
    weights.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const float xyz[3] = {splats.x[i], splats.y[i], splats.z[i]};
        std::array<int, 3> cell{};
        bool inside = true;
        for (int k = 0; k < 3 && inside; ++k) {
            const float q = std::floor((xyz[k] - lo) / step);
            inside = q >= 0.0f && q < (float)res;
            cell[k] = (int)q;
        }
        if (!inside) continue;
        cells.push_back(cell);
        weights.push_back(1.0f / (1.0f + std::exp(-splats.opacity[i])));
    }
    if (cells.empty()) throw std::invalid_argument("no splat falls inside the grid");

    // Crop to the occupied box plus the kernel radius. Outside that margin the blurred density is exactly zero, and
    // inside it the reflect boundary reads only zeros, so the crop gives the same density as the full grid.
    std::array<int, 3> low{}, shape{};
    for (int k = 0; k < 3; ++k) {
        int mn = res, mx = -1;
        for (const auto& c : cells) {
            mn = std::min(mn, c[k]);
            mx = std::max(mx, c[k]);
        }
        low[k] = std::max(mn - radius, 0);
        shape[k] = std::min(mx + radius + 1, res) - low[k];
    }
    const std::size_t count = (std::size_t)shape[0] * shape[1] * shape[2];
    std::vector<float> grid(count, 0.0f);
    for (std::size_t i = 0; i < cells.size(); ++i) {
        const auto& c = cells[i];
        const std::size_t flat =
            ((std::size_t)(c[0] - low[0]) * shape[1] + (c[1] - low[1])) * shape[2] + (c[2] - low[2]);
        grid[flat] += weights[i];
    }

    const int dims[3] = {shape[0], shape[1], shape[2]};
    std::vector<float> density(count);
    detail::gaussian_blur(grid.data(), density.data(), dims, o.blur_sigma, radius);

    const float p99 = detail::positive_percentile99(density.data(), count);
    if (!(p99 > 0.0f)) throw std::invalid_argument("splat density is empty");
    const float level = o.iso_fraction * p99;
    const float origin[3] = {(float)low[0] * step + lo, (float)low[1] * step + lo, (float)low[2] * step + lo};

    TriangleMesh mesh = detail::march_cubes(density.data(), dims, level, step, origin);
    detail::keep_large_pieces(mesh, o.min_piece);
    return mesh;
}

void write_obj(const std::string& path, const TriangleMesh& mesh) {
    std::ofstream out(path, std::ios::binary);
    if (!out) throw std::runtime_error("cannot write " + path);
    out << "# Written by idol.cpp\n";
    char line[128];
    for (std::size_t v = 0; v < mesh.vertex_count(); ++v) {
        std::snprintf(line, sizeof line, "v %.6f %.6f %.6f\n", mesh.vertices[3 * v], mesh.vertices[3 * v + 1],
                      mesh.vertices[3 * v + 2]);
        out << line;
    }
    for (std::size_t f = 0; f < mesh.triangle_count(); ++f)
        out << "f " << mesh.faces[3 * f] + 1 << ' ' << mesh.faces[3 * f + 1] + 1 << ' ' << mesh.faces[3 * f + 2] + 1
            << '\n';
    if (!out) throw std::runtime_error("write failed for " + path);
}

}  // namespace idol
