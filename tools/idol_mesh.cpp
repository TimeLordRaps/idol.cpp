// idol_mesh: converts a splat PLY to a Wavefront OBJ mesh with idol::splats_to_mesh.
//
// Usage: idol_mesh <input.ply> <output.obj> [--resolution N] [--sigma S] [--iso F] [--min-piece F] [--lo X] [--hi X]
//
// The input is a binary little-endian PLY with x, y, z and opacity on its vertex element. The defaults are the sprite
// pipeline's; see idol::MeshOptions for what each option means.

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "idol/idol.hpp"

namespace {

struct Args {
    std::string input;
    std::string output;
    idol::MeshOptions options;
    bool help = false;
};

void print_usage(std::FILE* out) {
    std::fprintf(out,
                 "usage: idol_mesh <input.ply> <output.obj> [--resolution N] [--sigma S] [--iso F] [--min-piece F]"
                 " [--lo X] [--hi X]\n");
}

int parse_integer(const std::string& text, const std::string& flag) {
    std::size_t used = 0;
    int value = 0;
    try {
        value = std::stoi(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != text.size()) throw std::invalid_argument(flag + " needs a whole number, got '" + text + "'");
    return value;
}

double parse_number(const std::string& text, const std::string& flag) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        used = 0;
    }
    if (used != text.size()) throw std::invalid_argument(flag + " needs a number, got '" + text + "'");
    return value;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            args.help = true;
        } else if (arg.size() > 1 && arg[0] == '-') {
            if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
            const std::string value = argv[++i];
            if (arg == "--resolution") {
                args.options.resolution = parse_integer(value, arg);
            } else if (arg == "--sigma") {
                args.options.blur_sigma = static_cast<float>(parse_number(value, arg));
            } else if (arg == "--iso") {
                args.options.iso_fraction = static_cast<float>(parse_number(value, arg));
            } else if (arg == "--min-piece") {
                args.options.min_piece = static_cast<float>(parse_number(value, arg));
            } else if (arg == "--lo") {
                args.options.lo = parse_number(value, arg);
            } else if (arg == "--hi") {
                args.options.hi = parse_number(value, arg);
            } else {
                throw std::invalid_argument("unknown option " + arg);
            }
        } else if (args.input.empty()) {
            args.input = arg;
        } else if (args.output.empty()) {
            args.output = arg;
        } else {
            throw std::invalid_argument("unexpected argument " + arg);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        if (args.help) {
            print_usage(stdout);
            return 0;
        }
        if (args.input.empty() || args.output.empty()) {
            print_usage(stderr);
            return 2;
        }

        const idol::SplatCloud splats = idol::read_splat_ply(args.input);
        const auto t0 = std::chrono::steady_clock::now();
        const idol::TriangleMesh mesh = idol::splats_to_mesh(splats, args.options);
        const auto t1 = std::chrono::steady_clock::now();
        idol::write_obj(args.output, mesh);
        std::printf("%zu splats -> %zu vertices, %zu triangles in %.1f ms\n", splats.size(), mesh.vertex_count(),
                    mesh.triangle_count(), std::chrono::duration<double, std::milli>(t1 - t0).count());
        return 0;
    } catch (const std::invalid_argument& error) {
        std::fprintf(stderr, "idol_mesh: %s\n", error.what());
        print_usage(stderr);
        return 2;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "idol_mesh: %s\n", error.what());
        return 1;
    }
}
