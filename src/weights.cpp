#include "weights.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace idol::detail {

namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

std::int64_t element_count(const std::vector<std::int64_t>& shape) {
    std::int64_t count = 1;
    for (std::int64_t d : shape) {
        if (d < 0) fail("negative dimension in manifest.txt");
        count *= d;
    }
    return count;
}

}  // namespace

Export read_export(const std::string& dir) {
    std::ifstream manifest(dir + "/manifest.txt");
    if (!manifest) fail("cannot open " + dir + "/manifest.txt");

    Export ex;
    std::string line;
    if (!std::getline(manifest, line)) fail(dir + "/manifest.txt is empty");
    {
        std::istringstream head(line);
        std::string tag;
        head >> tag;
        DecoderConfig& d = ex.decoder_config;
        head >> d.dim >> d.heads >> d.oct_blocks >> d.gs_blocks >> d.cond >> d.per_anchor;
        if (!head || tag != "decoders")
            fail(dir + "/manifest.txt: the first line must be 'decoders dim heads oct gs cond per_anchor'");
        ex.decoders = true;
    }

    std::ifstream bin(dir + "/weights.bin", std::ios::binary);
    if (!bin) fail("cannot open " + dir + "/weights.bin");
    bin.seekg(0, std::ios::end);
    const std::streamoff file_bytes = bin.tellg();

    while (std::getline(manifest, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        std::istringstream row(line);
        std::string name;
        int ndim = 0;
        row >> name;
        if (name == "rep") {
            Representation& r = ex.rep;
            row >> r.perturbe >> r.kernel >> r.scaling_bias >> r.opacity_bias >> r.lr_xyz >> r.lr_dc >> r.lr_opacity >>
                r.lr_scaling >> r.lr_rotation;
            if (!row) fail("malformed rep line in manifest.txt");
            continue;
        }
        if (name == "case") {
            row >> ex.case_anchors >> ex.case_latent >> ex.case_level;
            if (!row) fail("malformed case line in manifest.txt");
            continue;
        }
        if (name == "qtensor") {
            // qtensor <name> <out> <in> <bits> <group> <offset> <nbytes>: a quantized weight matrix.
            QTensor q;
            std::int64_t offset = 0, nbytes = 0;
            row >> name >> q.out >> q.in >> q.bits >> q.group >> offset >> nbytes;
            if (!row || name.empty()) fail("malformed qtensor line in manifest.txt");
            if (offset < 0 || nbytes < 0 || offset + nbytes > file_bytes) fail("qtensor '" + name + "' lies outside weights.bin");
            q.blob.resize((std::size_t)nbytes);
            bin.seekg(offset);
            bin.read(reinterpret_cast<char*>(q.blob.data()), nbytes);
            if (bin.gcount() != nbytes) fail("short read for qtensor '" + name + "'");
            ex.qtensors[name] = std::move(q);
            continue;
        }
        row >> ndim;
        Tensor t;
        for (int i = 0; i < ndim; ++i) {
            std::int64_t d = 0;
            row >> d;
            t.shape.push_back(d);
        }
        std::int64_t offset = 0, nbytes = 0;
        row >> offset >> nbytes;
        if (!row || name.empty() || ndim < 0) fail("malformed manifest.txt line for '" + name + "'");
        if (nbytes != element_count(t.shape) * (std::int64_t)sizeof(float))
            fail("tensor '" + name + "' has a byte count that does not match its shape");
        if (offset < 0 || offset + nbytes > file_bytes) fail("tensor '" + name + "' lies outside weights.bin");

        t.data.resize((std::size_t)(nbytes / sizeof(float)));
        bin.seekg(offset);
        bin.read(reinterpret_cast<char*>(t.data.data()), nbytes);
        if (bin.gcount() != nbytes) fail("short read for tensor '" + name + "'");
        ex.tensors[name] = std::move(t);
    }
    return ex;
}

bool take_qtensor(Export& ex, const std::string& name, QTensor& out) {
    auto it = ex.qtensors.find(name);
    if (it == ex.qtensors.end()) return false;
    out = std::move(it->second);
    ex.qtensors.erase(it);
    return true;
}

Tensor take_tensor(Export& ex, const std::string& name, std::int64_t expected_count) {
    auto it = ex.tensors.find(name);
    if (it == ex.tensors.end()) fail("missing tensor '" + name + "'");
    if ((std::int64_t)it->second.data.size() != expected_count)
        fail("tensor '" + name + "' has " + std::to_string(it->second.data.size()) + " values, expected " +
             std::to_string(expected_count));
    Tensor t = std::move(it->second);
    ex.tensors.erase(it);
    return t;
}

}  // namespace idol::detail
