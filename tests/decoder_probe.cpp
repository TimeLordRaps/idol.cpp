// Test helper: dump both raw decoder outputs for a supplied export and reference case.
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "idol/decoders.hpp"

static std::vector<float> read_floats(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path);
    const auto size = in.tellg();
    if (size < 0 || size % sizeof(float)) throw std::runtime_error("invalid float file: " + path);
    std::vector<float> result(static_cast<std::size_t>(size) / sizeof(float));
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(result.data()), size)) throw std::runtime_error("cannot read " + path);
    return result;
}

static void write_floats(const std::string& path, const std::vector<float>& values) {
    std::ofstream out(path, std::ios::binary);
    if (!out.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float)))
        throw std::runtime_error("cannot write " + path);
}

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    const std::string dir = argv[1], case_dir = argv[2], output = argv[3];
    idol::Decoders decoder(dir);
    const auto anchors = read_floats(case_dir + "/anchors.bin"), cond = read_floats(case_dir + "/cond.bin");
    const int rows = static_cast<int>(anchors.size() / 3), tokens = static_cast<int>(cond.size() / decoder.config().cond);
    std::vector<float> oct(static_cast<std::size_t>(rows) * 8);
    std::vector<float> gs(static_cast<std::size_t>(rows) * decoder.config().per_anchor * 15);
    decoder.octree_logits(anchors.data(), rows, 32, cond.data(), tokens, oct.data());
    decoder.gaussian_features(anchors.data(), rows, cond.data(), tokens, gs.data());
    write_floats(output + ".oct", oct);
    write_floats(output + ".gs", gs);
}
