"""Materialize Idol's octree and Gaussian decoders from GGUF for the native decoder runtime.

GGUF Q8_0 and Q4_0 blocks are translated to the native packed layout without re-quantizing their values. The denoiser
and preprocessing weights remain in the original GGUF file for the full-pipeline bridge.
"""
import argparse
from pathlib import Path

import gguf
import numpy as np
import torch
import idolq
from idol_gguf import IdolGGUF


def native_blob(tensor):
    shape = tuple(reversed(tensor.shape.tolist()))
    if len(shape) != 2 or shape[1] % 32:
        raise ValueError("quantized native decoder tensors must be matrices with 32-element groups")
    out, width = shape
    if tensor.tensor_type == gguf.GGMLQuantizationType.Q8_0:
        bits, block_bytes, bias = 8, 34, 128
    elif tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0:
        bits, block_bytes, bias = 4, 18, 8
    else:
        raise ValueError("unsupported quantized decoder tensor")
    blocks = np.asarray(tensor.data).reshape(out, width // 32, block_bytes)
    scales = np.ascontiguousarray(blocks[:, :, :2]).view("<f2").reshape(out, -1).astype(np.float32)
    if not np.isfinite(scales).all():
        raise ValueError("non-finite GGUF scale")
    payload = np.ascontiguousarray(blocks[:, :, 2:])
    if bits == 8:
        codes = (payload.view(np.int8).astype(np.int16) + bias).astype(np.uint8).reshape(out, width)
    else:
        codes = np.concatenate((payload & 15, payload >> 4), axis=-1).reshape(out, width)
    minimum = (-bias * scales).astype(np.float16).astype(np.float32)
    if not np.isfinite(minimum).all() or not np.array_equal(minimum, -bias * scales):
        raise ValueError("GGUF scale cannot be represented exactly in native layout")
    blob = idolq.pack_blob(torch.from_numpy(codes), torch.from_numpy(scales), torch.from_numpy(minimum), bits, 32)
    return blob, bits, shape


def materialize(model, output):
    output = Path(output)
    config = model.config
    oct_args, gs_args = config["decoder"]["args"], config["decoder_gs"]["args"]
    if oct_args["model_channels"] != gs_args["model_channels"] or oct_args["num_heads"] != gs_args["num_heads"]:
        raise ValueError("native decoder requires matching octree and Gaussian dimensions and heads")
    dim, heads = oct_args["model_channels"], oct_args["num_heads"]
    rep = gs_args["representation_config"]
    header = f"decoders {dim} {heads} {oct_args['num_blocks']} {gs_args['num_blocks']} {oct_args['cond_channels']} {rep['num_gaussians']}"
    manifest = [header]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "weights.bin").open("xb") as stream:
        def place(data):
            padding = (-stream.tell()) % idolq.ALIGN
            stream.write(b"\0" * padding)
            offset = stream.tell()
            stream.write(data)
            return offset

        def add_dense(name, value):
            array = np.ascontiguousarray(value, dtype="<f4")
            offset = place(array.tobytes())
            dims = " ".join(str(d) for d in value.shape)
            manifest.append(f"{name} {value.ndim} {dims} {offset} {array.nbytes}")

        for component, prefix in (("octree", "oct."), ("gaussian", "gs.")):
            for name, tensor in sorted(model.tensors[component].items()):
                shape = tuple(reversed(tensor.shape.tolist()))
                if tensor.tensor_type in (gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0):
                    blob, bits, shape = native_blob(tensor)
                    offset = place(blob)
                    manifest.append(f"qtensor {prefix}{name} {shape[0]} {shape[1]} {bits} 32 {offset} {len(blob)}")
                else:
                    values = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(shape)
                    if not np.isfinite(values).all():
                        raise ValueError("non-finite decoder tensor: " + name)
                    add_dense(prefix + name, values)
        # These nonpersistent SDK frequency tables are derived from the architecture, not stored in its checkpoint.
        for prefix in ("oct.", "gs."):
            freq_dim = dim // 3 // 2
            frequencies = torch.pow(2.0, torch.linspace(0.0, 10.0, steps=freq_dim, dtype=torch.float32)).numpy()
            add_dense(prefix + "pos_embedder.freqs", frequencies)
        half = 128
        levels = torch.exp(-np.log(1024) * torch.arange(0, half, dtype=torch.float32) / half).numpy()
        add_dense("oct.l_embedder.freqs", levels)
    manifest.append("rep " + " ".join(str(v) for v in (
        rep["perturbe_size"], rep["3d_filter_kernel_size"], rep["scaling_bias"], rep["opacity_bias"],
        rep["lr"]["_xyz"], rep["lr"]["_features_dc"], rep["lr"]["_scaling"], rep["lr"]["_opacity"],
        rep["lr"]["_rotation"])))
    (output / "manifest.txt").write_text("\n".join(manifest) + "\n", encoding="ascii")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    materialize(IdolGGUF(args.model), args.out)
    print("wrote native decoder export:", args.out)


if __name__ == "__main__":
    main()
