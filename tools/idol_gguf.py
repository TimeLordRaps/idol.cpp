"""Read the Idol GGUF container and load its tensors into the upstream Idol modules.

Quantized tensors are dequantized when loading PyTorch modules. This is a correctness-first full-pipeline bridge;
it does not claim quantized GPU execution or compatibility with any other loader.
"""
import json
from pathlib import Path

import gguf
import numpy as np
import torch

COMPONENTS = ("denoiser", "octree", "gaussian", "background")
SUPPORTED = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16,
             gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0,
             gguf.GGMLQuantizationType.Q2_K, gguf.GGMLQuantizationType.Q4_K,
             gguf.GGMLQuantizationType.Q6_K,
             gguf.GGMLQuantizationType.I8, gguf.GGMLQuantizationType.I16,
             gguf.GGMLQuantizationType.I32, gguf.GGMLQuantizationType.I64}


def validate_tensor_values(tensor):
    """Check numeric validity without retaining a dequantized copy of the tensor."""
    k_formats = {gguf.GGMLQuantizationType.Q2_K: (84, (80, 82)),
                 gguf.GGMLQuantizationType.Q4_K: (144, (0, 2)),
                 gguf.GGMLQuantizationType.Q6_K: (210, (208,))}
    if tensor.tensor_type in k_formats:
        block_bytes, offsets = k_formats[tensor.tensor_type]
        raw = np.asarray(tensor.data).reshape(-1, block_bytes)
        return all(np.isfinite(np.ascontiguousarray(raw[:, offset:offset + 2]).view('<f2')).all()
                   for offset in offsets)
    if tensor.tensor_type in (gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q8_0):
        # Both formats store one float16 scale followed by integer codes in each
        # 32-element block. Integer codes are finite; a finite scale guarantees
        # finite float32 values when multiplied by their bounded codes.
        block_bytes = 18 if tensor.tensor_type == gguf.GGMLQuantizationType.Q4_0 else 34
        raw = np.asarray(tensor.data).reshape(-1, block_bytes)
        scales = np.ascontiguousarray(raw[:, :2]).view("<f2")
        return bool(np.isfinite(scales).all())
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        # The all-ones bfloat16 exponent represents infinities and NaNs.
        values = np.ascontiguousarray(tensor.data).view('<u2').reshape(-1)
        return all(not np.any((values[i:i + 1_000_000] & 0x7f80) == 0x7f80)
                   for i in range(0, values.size, 1_000_000))
    if tensor.tensor_type in (gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32):
        values = np.asarray(tensor.data).reshape(-1)
        # Bound temporary boolean allocations for large matrices.
        return all(np.isfinite(values[i:i + 1_000_000]).all() for i in range(0, values.size, 1_000_000))
    return True


def validate_runtime_half_storage(model, module):
    """Fail if a runtime-aware file would downcast a currently float32 matrix."""
    field = model.reader.get_field('idol.quantization.tier')
    if field is None:
        return
    tier = field.contents()
    denoiser_kind = None
    mixed_denoiser = False
    suffixes = {'_DENOISER_Q2K_COV': (gguf.GGMLQuantizationType.Q2_K, False),
                '_DENOISER_Q4K_COV': (gguf.GGMLQuantizationType.Q4_K, False),
                '_DENOISER_Q6K_COV': (gguf.GGMLQuantizationType.Q6_K, False),
                '_DENOISER_Q80_COV': (gguf.GGMLQuantizationType.Q8_0, False),
                '_DENOISER_Q2KF16_COV': (gguf.GGMLQuantizationType.Q2_K, True),
                '_DENOISER_Q4KF16_COV': (gguf.GGMLQuantizationType.Q4_K, True),
                '_DENOISER_Q6KF16_COV': (gguf.GGMLQuantizationType.Q6_K, True),
                '_DENOISER_Q80F16_COV': (gguf.GGMLQuantizationType.Q8_0, True)}
    for suffix, (kind, mixed) in suffixes.items():
        if tier.endswith(suffix):
            tier = tier[:-len(suffix)]
            denoiser_kind = kind
            mixed_denoiser = mixed
            break
    if tier not in ('RUNTIME_F16_Q8', 'RUNTIME_F16_GAUSSIAN_Q2K', 'RUNTIME_F16_GAUSSIAN_Q4K',
                    'RUNTIME_F16_GAUSSIAN_Q6K', 'RUNTIME_F16_GAUSSIAN_Q2K_COV',
                    'RUNTIME_F16_GAUSSIAN_Q4K_COV', 'RUNTIME_F16_GAUSSIAN_Q6K_COV',
                    'RUNTIME_F16_GAUSSIAN_Q2Q4_COV'):
        return
    expected = module.state_dict()
    checked_blocks = 0
    preserved_blocks = set()
    seen_blocks = set()
    if mixed_denoiser:
        allocation = model.reader.get_field('idol.quantization.denoiser_preserved_blocks')
        if allocation is None:
            raise ValueError('mixed denoiser tier is missing its block allocation')
        values = json.loads(allocation.contents())
        if (not isinstance(values, list) or any(type(value) is not int or value < 0 for value in values)
                or values != sorted(set(values))):
            raise ValueError('invalid mixed denoiser block allocation')
        preserved_blocks = set(values)
    for name, tensor in model.tensors['denoiser'].items():
        if (denoiser_kind is not None and name.startswith('blocks.') and name.endswith('weight')
                and len(tensor.shape) == 2 and int(tensor.shape[0]) % 256 == 0):
            checked_blocks += 1
            parts = name.split('.')
            if mixed_denoiser and (len(parts) < 3 or not parts[1].isdigit()):
                raise ValueError('invalid mixed denoiser block name: ' + name)
            block = int(parts[1]) if mixed_denoiser else None
            if mixed_denoiser:
                seen_blocks.add(block)
            required_kind = gguf.GGMLQuantizationType.F16 if block in preserved_blocks else denoiser_kind
            if tensor.tensor_type != required_kind or name not in expected or expected[name].dtype != torch.float16:
                raise ValueError('encoded denoiser block is incompatible with the runtime: ' + name)
        if (tensor.tensor_type == gguf.GGMLQuantizationType.F16 and len(tensor.shape) == 2
                and (name not in expected or expected[name].dtype != torch.float16)):
            raise ValueError('runtime-aware half-precision matrix is not half precision in this SDK: ' + name)
    if denoiser_kind is not None and checked_blocks == 0:
        raise ValueError('encoded denoiser tier has no eligible runtime blocks')
    if mixed_denoiser and (not preserved_blocks <= seen_blocks or preserved_blocks == seen_blocks):
        raise ValueError('mixed denoiser block allocation does not match the runtime')


class IdolGGUF:
    def __init__(self, path):
        self.path = Path(path)
        self.reader = gguf.GGUFReader(self.path)
        arch = self.reader.get_field("general.architecture")
        config = self.reader.get_field("idol.config.json")
        if arch is None or arch.contents() != "idol" or config is None:
            raise ValueError("not an Idol GGUF file with an embedded configuration")
        self.config = json.loads(config.contents())
        if not all(k in self.config for k in ("denoiser", "decoder", "decoder_gs", "image_encoders")):
            raise ValueError("incomplete Idol configuration")
        self.tensors = {name: {} for name in COMPONENTS}
        for tensor in self.reader.tensors:
            parts = tensor.name.split(".", 2)
            if len(parts) != 3 or parts[0] != "idol" or parts[1] not in self.tensors or not parts[2]:
                raise ValueError("unexpected tensor in Idol GGUF: " + tensor.name)
            group = self.tensors[parts[1]]
            if parts[2] in group or tensor.tensor_type not in SUPPORTED:
                raise ValueError("duplicate tensor or unsupported GGUF encoding: " + tensor.name)
            group[parts[2]] = tensor
        if any(not values for values in self.tensors.values()):
            raise ValueError("Idol GGUF is missing a component")

    def load_module(self, component, module):
        if component not in self.tensors:
            raise ValueError("unknown component: " + component)
        expected = module.state_dict()
        source = self.tensors[component]
        if set(expected) != set(source):
            missing, extra = sorted(set(expected) - set(source)), sorted(set(source) - set(expected))
            raise ValueError(f"{component} tensor names do not match; missing={missing}, extra={extra}")
        # Validate all names, shapes, and numeric values before modifying any
        # module state. Then materialize one tensor at a time to avoid keeping
        # a second full component-sized state dictionary resident in memory.
        for name, target in expected.items():
            tensor = source[name]
            shape = tuple(reversed(tensor.shape.tolist()))
            if shape != tuple(target.shape):
                raise ValueError(f"{component}.{name} shape {shape} does not match {tuple(target.shape)}")
            if not validate_tensor_values(tensor):
                raise ValueError(f"non-finite value in {component}.{name}")
        with torch.no_grad():
            for name, target in expected.items():
                tensor = source[name]
                shape = tuple(target.shape)
                if tensor.tensor_type in {gguf.GGMLQuantizationType.I8, gguf.GGMLQuantizationType.I16,
                                          gguf.GGMLQuantizationType.I32, gguf.GGMLQuantizationType.I64}:
                    values = np.asarray(tensor.data).reshape(shape)
                else:
                    values = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(shape)
                target.copy_(torch.from_numpy(np.array(values, copy=True)))
        return module

    def load_module_packed(self, component, module, *, include_half=False):
        """Load a meta-initialized module without first expanding packed matrices."""
        from accelerate.utils import set_module_tensor_to_device
        from idol_packed import PackedLinear

        if component not in self.tensors:
            raise ValueError('unknown component: ' + component)
        expected, source = module.state_dict(), self.tensors[component]
        if set(expected) != set(source):
            raise ValueError(f'{component} tensor names do not match')
        supported = {gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0,
                     gguf.GGMLQuantizationType.Q2_K, gguf.GGMLQuantizationType.Q4_K,
                     gguf.GGMLQuantizationType.Q6_K}
        selected = {}
        for name, target in expected.items():
            tensor = source[name]
            if tuple(reversed(tensor.shape.tolist())) != tuple(target.shape):
                raise ValueError(f'{component}.{name} shape does not match')
            if not validate_tensor_values(tensor):
                raise ValueError(f'non-finite value in {component}.{name}')
            packed = tensor.tensor_type in supported or (include_half and
                     tensor.tensor_type == gguf.GGMLQuantizationType.F16 and
                     len(tensor.shape) == 2 and name.endswith('.weight'))
            if packed:
                if not name.endswith('.weight'):
                    raise ValueError('quantized tensor is not a linear weight: ' + name)
                layer_name = name[:-7]
                try:
                    layer = module.get_submodule(layer_name)
                except AttributeError as exc:
                    raise ValueError('quantized linear layer is missing: ' + name) from exc
                if not isinstance(layer, torch.nn.Linear):
                    raise ValueError('quantized tensor is not a linear weight: ' + name)
                selected[layer_name] = layer

        skipped = {name + suffix for name in selected for suffix in ('.weight', '.bias')}
        integer_types = {gguf.GGMLQuantizationType.I8, gguf.GGMLQuantizationType.I16,
                         gguf.GGMLQuantizationType.I32, gguf.GGMLQuantizationType.I64}
        for name, target in expected.items():
            if name in skipped:
                continue
            tensor = source[name]
            values = (np.asarray(tensor.data) if tensor.tensor_type in integer_types
                      else gguf.dequantize(tensor.data, tensor.tensor_type))
            values = torch.from_numpy(np.array(values, copy=True).reshape(target.shape))
            set_module_tensor_to_device(module, name, 'cpu', value=values, dtype=target.dtype)
        for name, original in selected.items():
            bias = None
            if original.bias is not None:
                tensor = source[name + '.bias']
                values = gguf.dequantize(tensor.data, tensor.tensor_type)
                bias = torch.from_numpy(np.array(values, copy=True)).reshape(original.bias.shape).to(original.bias.dtype)
            parent_name, _, leaf = name.rpartition('.')
            parent = module.get_submodule(parent_name) if parent_name else module
            setattr(parent, leaf, PackedLinear(source[name + '.weight'], bias))
        if any(value.is_meta for value in list(module.parameters()) + list(module.buffers())):
            raise ValueError('packed module retains uninitialized tensors')
        return module
