"""Keep quantized linear weights packed; optionally fuse supported CUDA calls.

The block-decoding formulas follow gguf-py; its license appears in
THIRD_PARTY_NOTICES.md. Unsupported calls use the existing expansion path.
"""
import gguf
import numpy as np
import torch
import torch.nn.functional as F


class PackedLinear(torch.nn.Module):
    def __init__(self, source, bias):
        super().__init__()
        self.shape = tuple(reversed(source.shape.tolist()))
        self.kind = source.tensor_type
        if len(self.shape) != 2 or self.kind not in (gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0,
                gguf.GGMLQuantizationType.Q2_K, gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K):
            raise ValueError("unsupported packed matrix")
        self.register_buffer("packed", torch.from_numpy(np.array(source.data, copy=True)).reshape(-1))
        self.bias = torch.nn.Parameter(bias.detach(), requires_grad=False) if bias is not None else None
        self.fused = False

    def dequantize(self, dtype):
        if self.kind == gguf.GGMLQuantizationType.F16:
            return self.packed.reshape(self.shape).to(dtype)
        if self.kind == gguf.GGMLQuantizationType.Q8_0:
            blocks = self.packed.reshape(-1, 34)
            scale = blocks[:, :2].contiguous().view(torch.float16).float()
            codes = blocks[:, 2:].contiguous().view(torch.int8).float()
        elif self.kind == gguf.GGMLQuantizationType.Q4_0:
            blocks = self.packed.reshape(-1, 18)
            scale = blocks[:, :2].contiguous().view(torch.float16).float()
            packed = blocks[:, 2:]
            codes = torch.stack((packed & 15, packed >> 4), dim=1).reshape(-1, 32).to(torch.int16).sub_(8).float()
        elif self.kind == gguf.GGMLQuantizationType.Q2_K:
            blocks = self.packed.reshape(-1, 84)
            scales, packed = blocks[:, :16], blocks[:, 16:80]
            d = blocks[:, 80:82].contiguous().view(torch.float16).float()
            dmin = blocks[:, 82:84].contiguous().view(torch.float16).float()
            scales_d = (d * (scales & 15).float()).reshape(-1, 16, 1)
            scales_m = (dmin * (scales >> 4).float()).reshape(-1, 16, 1)
            shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=packed.device).reshape(1, 1, 4, 1)
            codes = ((packed.reshape(-1, 2, 1, 32) >> shifts) & 3).reshape(-1, 16, 16).float()
            return (scales_d * codes - scales_m).reshape(self.shape).to(dtype)
        elif self.kind == gguf.GGMLQuantizationType.Q4_K:
            blocks = self.packed.reshape(-1, 144)
            d = blocks[:, :2].contiguous().view(torch.float16).float()
            dmin = blocks[:, 2:4].contiguous().view(torch.float16).float()
            s = blocks[:, 4:16].reshape(-1, 3, 4)
            a, b, c = s[:, 0:1], s[:, 1:2], s[:, 2:3]
            scales = torch.cat((a & 63, (c & 15) | ((a >> 2) & 48)), dim=-1).reshape(-1, 8)
            mins = torch.cat((b & 63, (c >> 4) | ((b >> 2) & 48)), dim=-1).reshape(-1, 8)
            packed = blocks[:, 16:]
            shifts = torch.tensor([0, 4], dtype=torch.uint8, device=packed.device).reshape(1, 1, 2, 1)
            codes = ((packed.reshape(-1, 4, 1, 32) >> shifts) & 15).reshape(-1, 8, 32).float()
            return ((d * scales.float()).reshape(-1, 8, 1) * codes -
                    (dmin * mins.float()).reshape(-1, 8, 1)).reshape(self.shape).to(dtype)
        else:
            blocks = self.packed.reshape(-1, 210)
            low, high = blocks[:, :128], blocks[:, 128:192]
            scales = blocks[:, 192:208].contiguous().view(torch.int8).float()
            d = blocks[:, 208:210].contiguous().view(torch.float16).float()
            shifts_low = torch.tensor([0, 4], dtype=torch.uint8, device=low.device).reshape(1, 1, 2, 1)
            shifts_high = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=high.device).reshape(1, 1, 4, 1)
            low = ((low.reshape(-1, 2, 1, 64) >> shifts_low) & 15).reshape(-1, 8, 32)
            high = ((high.reshape(-1, 2, 1, 32) >> shifts_high) & 3).reshape(-1, 8, 32)
            codes = (low | (high << 4)).to(torch.int16).sub_(32).reshape(-1, 16, 16).float()
            return ((d * scales).reshape(-1, 16, 1) * codes).reshape(self.shape).to(dtype)
        return (codes * scale).reshape(self.shape).to(dtype)

    def forward(self, value):
        kinds = {gguf.GGMLQuantizationType.Q4_0: 4, gguf.GGMLQuantizationType.Q8_0: 8,
                 gguf.GGMLQuantizationType.Q4_K: 44}
        needs_grad = torch.is_grad_enabled() and (value.requires_grad or
                     (self.bias is not None and self.bias.requires_grad))
        if (self.fused and value.is_cuda and value.dtype in (torch.float16, torch.bfloat16)
                and self.kind in kinds and not needs_grad):
            from idol_triton import fused_linear
            return fused_linear(value, self.packed, self.shape, kinds[self.kind], self.bias)
        return F.linear(value, self.dequantize(value.dtype), self.bias)


def enable_fused_packed(model):
    """Enable supported packed layers; return the number configured for CUDA."""
    supported = {gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q8_0,
                 gguf.GGMLQuantizationType.Q4_K}
    count = 0
    for layer in model.modules():
        if isinstance(layer, PackedLinear) and layer.kind in supported:
            layer.fused = True
            count += 1
    return count


def replace_quantized_linear_layers(model, source, include_half=False):
    """Replace all quantized linear weights after checking the complete mapping.

    IdolGGUF validates tensor values before calling this function. A quantized
    non-linear tensor is an error, rather than an unnoticed dense allocation.
    """
    supported = {gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0,
                 gguf.GGMLQuantizationType.Q2_K, gguf.GGMLQuantizationType.Q4_K,
                 gguf.GGMLQuantizationType.Q6_K}
    selected = {name: tensor for name, tensor in source.items()
                if tensor.tensor_type in supported or
                (include_half and tensor.tensor_type == gguf.GGMLQuantizationType.F16
                 and len(tensor.shape) == 2 and name.endswith('.weight'))}
    names = []
    for key, tensor in selected.items():
        if not key.endswith('.weight'):
            raise ValueError('quantized tensor is not a linear weight: ' + key)
        name = key[:-len('.weight')]
        try:
            module = model.get_submodule(name)
        except AttributeError as exc:
            raise ValueError('quantized linear layer is missing: ' + key) from exc
        if not isinstance(module, torch.nn.Linear) or tuple(reversed(tensor.shape.tolist())) != tuple(module.weight.shape):
            raise ValueError('quantized tensor does not match a linear layer: ' + key)
        names.append(name)
    for name in names:
        key = name + '.weight'
        tensor = selected[key]
        module = model.get_submodule(name)
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, PackedLinear(tensor, module.bias).to(device=module.weight.device))
    return [name + '.weight' for name in names]
