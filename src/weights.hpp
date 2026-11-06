// Reads a decoder export directory. Internal; the format is documented in docs/format.html.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "idol/decoders.hpp"

namespace idol::detail {

struct Tensor {
    std::vector<std::int64_t> shape;
    std::vector<float> data;
};

// A weight matrix stored quantized: the blob is already in the panel layout the GEMM reads (see PackedLinear).
struct QTensor {
    int out = 0, in = 0, bits = 0, group = 0;
    std::vector<std::uint8_t> blob;
};

// How the Gaussian decoder turns raw outputs into Gaussians (the "rep" line of a decoders export).
struct Representation {
    float perturbe = 1.5f, kernel = 0.0009f, scaling_bias = 0.004f, opacity_bias = 0.1f;
    float lr_xyz = 1.0f, lr_dc = 1.0f, lr_opacity = 1.0f, lr_scaling = 1.0f, lr_rotation = 0.1f;
};

// An export directory read into memory: the config from manifest.txt and every tensor from weights.bin.
struct Export {
    bool decoders = false;
    DecoderConfig decoder_config;
    Representation rep;
    int case_anchors = 0, case_latent = 0, case_level = 0;  // the reference case's "case" line, if any
    std::map<std::string, Tensor> tensors;
    std::map<std::string, QTensor> qtensors;  // weight matrices written by tools/quant_search.py
};

// Reads dir/manifest.txt and dir/weights.bin. Throws std::runtime_error on a missing or malformed file.
Export read_export(const std::string& dir);

// Moves the named quantized matrix out of ex, or returns false if the export holds that matrix as float32.
bool take_qtensor(Export& ex, const std::string& name, QTensor& out);

// Moves the named tensor out of ex. Throws std::runtime_error if it is missing or does not hold expected_count values.
Tensor take_tensor(Export& ex, const std::string& name, std::int64_t expected_count);

}  // namespace idol::detail
