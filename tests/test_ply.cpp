// PLY reader tests: column offsets across mixed types, skipped properties and elements, and the refusals.
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "idol/ply.hpp"
#include "test.hpp"

namespace {

namespace fs = std::filesystem;

fs::path scratch_file(const std::string& name) {
    return fs::temp_directory_path() / ("idol_ply_test_" + name + ".ply");
}

void write_text(const fs::path& path, const std::string& text) {
    std::ofstream out(path, std::ios::binary);
    out << text;
}

// The header, with the vertex properties in order and any extra element lines inserted before end_header.
std::string header(const std::string& format, std::size_t count, const std::vector<std::string>& properties,
                   const std::string& extra = "") {
    std::string text = "ply\nformat " + format + " 1.0\nelement vertex " + std::to_string(count) + "\n";
    for (const std::string& property : properties) text += "property " + property + "\n";
    return text + extra + "end_header\n";
}

template <typename T>
void append(std::string& bytes, T value) {
    char raw[sizeof(T)];
    std::memcpy(raw, &value, sizeof(T));
    bytes.append(raw, sizeof(T));
}

}  // namespace

TEST(ply, reads_columns_at_their_offsets_among_other_types) {
    const std::vector<std::string> properties = {"float x",       "float y",     "float z",
                                                 "uchar red",     "float opacity", "float f_dc_0"};
    std::string body;
    append(body, 1.5f);
    append(body, -2.25f);
    append(body, 3.0f);
    append(body, static_cast<std::uint8_t>(200));
    append(body, 0.5f);
    append(body, 9.0f);
    append(body, 4.0f);
    append(body, 5.0f);
    append(body, 6.0f);
    append(body, static_cast<std::uint8_t>(7));
    append(body, -1.0f);
    append(body, 8.0f);

    const fs::path path = scratch_file("offsets");
    write_text(path, header("binary_little_endian", 2, properties) + body);
    const idol::SplatCloud cloud = idol::read_splat_ply(path.string());
    fs::remove(path);

    CHECK(cloud.size() == 2);
    CHECK_NEAR(cloud.x[0], 1.5, 0.0);
    CHECK_NEAR(cloud.y[0], -2.25, 0.0);
    CHECK_NEAR(cloud.z[0], 3.0, 0.0);
    CHECK_NEAR(cloud.opacity[0], 0.5, 0.0);
    CHECK_NEAR(cloud.x[1], 4.0, 0.0);
    CHECK_NEAR(cloud.opacity[1], -1.0, 0.0);
}

TEST(ply, skips_elements_that_follow_the_vertices) {
    const std::vector<std::string> properties = {"float x", "float y", "float z", "float opacity"};
    std::string body;
    for (float v : {1.0f, 2.0f, 3.0f, 4.0f}) append(body, v);
    const std::string extra = "element face 1\nproperty list uchar int vertex_indices\n";
    const fs::path path = scratch_file("trailing_elements");
    write_text(path, header("binary_little_endian", 1, properties, extra) + body);
    const idol::SplatCloud cloud = idol::read_splat_ply(path.string());
    fs::remove(path);

    CHECK(cloud.size() == 1);
    CHECK_NEAR(cloud.opacity[0], 4.0, 0.0);
}

TEST(ply, rejects_ascii_files) {
    const std::vector<std::string> properties = {"float x", "float y", "float z", "float opacity"};
    const fs::path path = scratch_file("ascii");
    write_text(path, header("ascii", 1, properties) + "1 2 3 4\n");
    CHECK_THROWS(idol::read_splat_ply(path.string()));
    fs::remove(path);
}

TEST(ply, rejects_a_missing_opacity_column) {
    const std::vector<std::string> properties = {"float x", "float y", "float z"};
    std::string body;
    for (float v : {1.0f, 2.0f, 3.0f}) append(body, v);
    const fs::path path = scratch_file("no_opacity");
    write_text(path, header("binary_little_endian", 1, properties) + body);
    CHECK_THROWS(idol::read_splat_ply(path.string()));
    fs::remove(path);
}

TEST(ply, rejects_truncated_vertex_data) {
    const std::vector<std::string> properties = {"float x", "float y", "float z", "float opacity"};
    std::string body;
    for (float v : {1.0f, 2.0f, 3.0f, 4.0f}) append(body, v);
    const fs::path path = scratch_file("truncated");
    write_text(path, header("binary_little_endian", 2, properties) + body);
    CHECK_THROWS(idol::read_splat_ply(path.string()));
    fs::remove(path);
}

TEST(ply, rejects_files_that_are_not_ply) {
    const fs::path path = scratch_file("not_ply");
    write_text(path, "hello world\n");
    CHECK_THROWS(idol::read_splat_ply(path.string()));
    fs::remove(path);
}

TEST(ply, rejects_a_missing_file) {
    CHECK_THROWS(idol::read_splat_ply(scratch_file("does_not_exist").string()));
}
