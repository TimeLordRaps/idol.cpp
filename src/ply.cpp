#include "idol/ply.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace idol {

namespace {

enum class Scalar { I8, U8, I16, U16, I32, U32, F32, F64 };

struct Property {
    std::string name;
    Scalar type;
    std::size_t offset;
};

Scalar parse_scalar(const std::string& name, const std::string& path) {
    if (name == "char" || name == "int8") return Scalar::I8;
    if (name == "uchar" || name == "uint8") return Scalar::U8;
    if (name == "short" || name == "int16") return Scalar::I16;
    if (name == "ushort" || name == "uint16") return Scalar::U16;
    if (name == "int" || name == "int32") return Scalar::I32;
    if (name == "uint" || name == "uint32") return Scalar::U32;
    if (name == "float" || name == "float32") return Scalar::F32;
    if (name == "double" || name == "float64") return Scalar::F64;
    throw std::runtime_error(path + ": unsupported PLY property type '" + name + "'");
}

std::size_t scalar_size(Scalar s) {
    switch (s) {
        case Scalar::I8:
        case Scalar::U8: return 1;
        case Scalar::I16:
        case Scalar::U16: return 2;
        case Scalar::I32:
        case Scalar::U32:
        case Scalar::F32: return 4;
        case Scalar::F64: return 8;
    }
    return 0;
}

double read_scalar(const char* p, Scalar s) {
    switch (s) {
        case Scalar::I8: { std::int8_t v; std::memcpy(&v, p, sizeof v); return v; }
        case Scalar::U8: { std::uint8_t v; std::memcpy(&v, p, sizeof v); return v; }
        case Scalar::I16: { std::int16_t v; std::memcpy(&v, p, sizeof v); return v; }
        case Scalar::U16: { std::uint16_t v; std::memcpy(&v, p, sizeof v); return v; }
        case Scalar::I32: { std::int32_t v; std::memcpy(&v, p, sizeof v); return v; }
        case Scalar::U32: { std::uint32_t v; std::memcpy(&v, p, sizeof v); return v; }
        case Scalar::F32: { float v; std::memcpy(&v, p, sizeof v); return v; }
        case Scalar::F64: { double v; std::memcpy(&v, p, sizeof v); return v; }
    }
    return 0.0;
}

bool host_is_little_endian() {
    const std::uint16_t probe = 1;
    unsigned char first = 0;
    std::memcpy(&first, &probe, 1);
    return first == 1;
}

std::string trimmed(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) s.pop_back();
    return s;
}

const Property& require(const std::vector<Property>& props, const char* name, const std::string& path) {
    for (const Property& p : props)
        if (p.name == name) return p;
    throw std::runtime_error(path + ": the vertex element has no '" + name + "' property");
}

}  // namespace

SplatCloud read_splat_ply(const std::string& path) {
    if (!host_is_little_endian()) throw std::runtime_error("read_splat_ply needs a little-endian host");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);

    std::string line;
    if (!std::getline(in, line) || trimmed(line) != "ply") throw std::runtime_error(path + " is not a PLY file");

    bool binary = false, ended = false, in_vertex = false, saw_vertex = false;
    std::size_t count = 0, stride = 0;
    std::vector<Property> props;
    while (std::getline(in, line)) {
        std::istringstream row(trimmed(line));
        std::string word;
        row >> word;
        if (word == "end_header") {
            ended = true;
            break;
        }
        if (word == "format") {
            std::string format;
            row >> format;
            if (format != "binary_little_endian")
                throw std::runtime_error(path + ": only binary_little_endian PLY is supported, found '" + format + "'");
            binary = true;
        } else if (word == "element") {
            std::string name;
            std::size_t n = 0;
            row >> name >> n;
            in_vertex = name == "vertex";
            if (in_vertex) {
                count = n;
                saw_vertex = true;
            } else if (!saw_vertex) {
                throw std::runtime_error(path + ": the vertex element must come before '" + name + "'");
            }
        } else if (word == "property" && in_vertex) {
            std::string type, name;
            row >> type;
            if (type == "list") throw std::runtime_error(path + ": list properties are not supported on vertices");
            row >> name;
            const Scalar scalar = parse_scalar(type, path);
            props.push_back({name, scalar, stride});
            stride += scalar_size(scalar);
        }
    }
    if (!ended) throw std::runtime_error(path + " has no end_header");
    if (!binary) throw std::runtime_error(path + " has no format line");
    if (!saw_vertex) throw std::runtime_error(path + " has no vertex element");

    const Property& px = require(props, "x", path);
    const Property& py = require(props, "y", path);
    const Property& pz = require(props, "z", path);
    const Property& po = require(props, "opacity", path);

    std::vector<char> data(count * stride);
    in.read(data.data(), (std::streamsize)data.size());
    if ((std::size_t)in.gcount() != data.size()) throw std::runtime_error(path + " is truncated");

    SplatCloud cloud;
    cloud.x.reserve(count);
    cloud.y.reserve(count);
    cloud.z.reserve(count);
    cloud.opacity.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const char* row = data.data() + i * stride;
        cloud.x.push_back((float)read_scalar(row + px.offset, px.type));
        cloud.y.push_back((float)read_scalar(row + py.offset, py.type));
        cloud.z.push_back((float)read_scalar(row + pz.offset, pz.type));
        cloud.opacity.push_back((float)read_scalar(row + po.offset, po.type));
    }
    return cloud;
}

}  // namespace idol
