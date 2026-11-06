"""Experimental adapters for separately obtained image-encoder GGUF weights.

The direct adapters construct their architectures without first loading dense
weights. They do not change the original license and access requirements.
"""
import hashlib
import json
from pathlib import Path

import gguf
import numpy as np
import torch

from idol_gguf import validate_tensor_values
from idol_packed import replace_quantized_linear_layers
from idol_packed import PackedLinear


class ExternalReader(gguf.GGUFReader):
    def _push_field(self, field, skip_sum=False):
        # Some DINO conversions store metadata under gguf-py's synthetic header
        # names. Preserve those entries under a separate namespace, while still
        # rejecting genuine duplicate on-disk fields.
        if field.name in self.fields and field.name in {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}:
            field = field._replace(name='source.' + field.name)
        return super()._push_field(field, skip_sum=skip_sum)


def dino_config_from_metadata(reader):
    """Read the exact Idol DINOv3 architecture from a third-party file."""
    from transformers import AutoConfig

    keys = ('model_type', 'image_size', 'patch_size', 'num_channels', 'hidden_size',
            'intermediate_size', 'num_hidden_layers', 'num_attention_heads', 'hidden_act',
            'attention_dropout', 'initializer_range', 'layer_norm_eps', 'layerscale_value',
            'drop_path_rate', 'use_gated_mlp', 'rope_theta', 'query_bias', 'key_bias',
            'value_bias', 'proj_bias', 'mlp_bias', 'num_register_tokens', 'pos_embed_rescale')
    fields = {key: reader.get_field(key) for key in keys}
    if any(field is None for field in fields.values()):
        raise ValueError('DINO encoder GGUF lacks required architecture metadata')
    values = {key: field.contents() for key, field in fields.items()}
    fixed = {'model_type': 'dinov3_vit', 'image_size': 224, 'patch_size': 16,
             'num_channels': 3, 'hidden_size': 1280, 'intermediate_size': 5120,
             'num_hidden_layers': 32, 'num_attention_heads': 20, 'hidden_act': 'silu',
             'use_gated_mlp': True, 'query_bias': True, 'key_bias': False,
             'value_bias': True, 'proj_bias': True, 'mlp_bias': True,
             'num_register_tokens': 4}
    if any(type(values[key]) is not type(expected) or values[key] != expected
           for key, expected in fixed.items()):
        raise ValueError('DINO encoder GGUF does not describe the Idol DINOv3 architecture')
    floats = {'attention_dropout': 0.0, 'initializer_range': 0.02, 'layer_norm_eps': 1e-5,
              'layerscale_value': 1.0, 'drop_path_rate': 0.0, 'rope_theta': 100.0,
              'pos_embed_rescale': 2.0}
    if any(type(values[key]) not in (int, float) or not np.isclose(values[key], expected, rtol=1e-6, atol=1e-12)
           for key, expected in floats.items()):
        raise ValueError('DINO encoder GGUF has unexpected architecture parameters')
    # GGUF stores these scalars as float32. Use the known configuration's
    # original decimal values after checking that the file describes them.
    values.update(floats)
    model_type = values.pop('model_type')
    return AutoConfig.for_model(model_type, **values)


def public_flux_config():
    """Load the independently hosted, pinned copy of Idol's FLUX.2 VAE config."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download('kaiyuyue/FLUX.2-dev-vae', 'config.json',
                           revision='9301428b4c77ccf59b23d319cc5aa8bc077afb57', token=False)
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != '7948580b2d53da578d49b9fed61b45d94a4c037ded5057ebd3adb4040bb5aef5':
        raise ValueError('public FLUX.2 autoencoder configuration has an unexpected digest')
    return json.loads(data)


def build_flux_vae(repository, path):
    """Construct a FLUX.2 autoencoder from its configuration and GGUF tensors."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from diffusers.models import AutoencoderKLFlux2

    config = (AutoencoderKLFlux2.load_config(repository, subfolder='vae') if repository
              else public_flux_config())
    with init_empty_weights():
        module = AutoencoderKLFlux2.from_config(config).eval().requires_grad_(False)
    reader = ExternalReader(path)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    expected = module.state_dict()
    missing = set(expected) - set(tensors)
    allowed_missing = {'bn.num_batches_tracked'}
    if len(tensors) != len(reader.tensors) or missing - allowed_missing or set(tensors) - set(expected):
        raise ValueError('FLUX.2 autoencoder tensor names do not match its configuration')
    if missing and (expected['bn.num_batches_tracked'].is_meta or expected['bn.num_batches_tracked'].item() != 0):
        raise ValueError('omitted encoder batch counter is not zero')
    allowed = {gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32}
    for name, tensor in tensors.items():
        if (tensor.tensor_type not in allowed or tuple(reversed(tensor.shape.tolist())) != tuple(expected[name].shape)
                or not expected[name].is_floating_point() or not validate_tensor_values(tensor)):
            raise ValueError('invalid FLUX.2 autoencoder tensor: ' + name)
    for name, tensor in tensors.items():
        values = torch.from_numpy(np.array(tensor.data, copy=True).reshape(expected[name].shape))
        set_module_tensor_to_device(module, name, 'cpu', value=values, dtype=torch.bfloat16)
    return module


def build_dino(repository, path):
    """Construct a DINOv3 module on CPU without loading original dense weights."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers import AutoConfig, AutoModel

    reader = ExternalReader(path)
    architecture = reader.get_field('general.architecture')
    if architecture is None or architecture.contents() != 'dinov3':
        raise ValueError('DINO encoder GGUF has unexpected architecture')
    config = AutoConfig.from_pretrained(repository) if repository else dino_config_from_metadata(reader)
    with init_empty_weights():
        module = AutoModel.from_config(config).eval().requires_grad_(False)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    expected = module.state_dict()
    if len(tensors) != len(reader.tensors) or set(tensors) != set(expected):
        raise ValueError('DINO encoder tensor names do not match its configuration')
    packed = {gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q8_0,
              gguf.GGMLQuantizationType.Q2_K, gguf.GGMLQuantizationType.Q4_K,
              gguf.GGMLQuantizationType.Q6_K}
    linear_names = []
    for name, tensor in tensors.items():
        if (tensor.tensor_type not in packed | {gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32,
                                                gguf.GGMLQuantizationType.BF16}
                or tuple(reversed(tensor.shape.tolist())) != tuple(expected[name].shape)
                or not validate_tensor_values(tensor)):
            raise ValueError('invalid DINO encoder tensor: ' + name)
        if tensor.tensor_type in packed:
            if not name.endswith('.weight'):
                raise ValueError('packed DINO tensor is not a linear weight: ' + name)
            try:
                layer = module.get_submodule(name[:-7])
            except AttributeError as exc:
                raise ValueError('missing DINO linear layer: ' + name) from exc
            if not isinstance(layer, torch.nn.Linear):
                raise ValueError('packed DINO tensor is not a linear weight: ' + name)
            linear_names.append(name[:-7])
    skipped = {name + '.weight' for name in linear_names} | {name + '.bias' for name in linear_names}
    for name, tensor in tensors.items():
        if name not in skipped:
            values = torch.from_numpy(np.array(gguf.dequantize(tensor.data, tensor.tensor_type), copy=True).reshape(expected[name].shape))
            set_module_tensor_to_device(module, name, 'cpu', value=values, dtype=torch.bfloat16)
    for name in linear_names:
        original = module.get_submodule(name)
        bias = None
        if original.bias is not None:
            bias_tensor = tensors[name + '.bias']
            bias = torch.from_numpy(np.array(gguf.dequantize(bias_tensor.data, bias_tensor.tensor_type), copy=True)).to(torch.bfloat16)
        parent_name, _, leaf = name.rpartition('.')
        setattr(module.get_submodule(parent_name), leaf, PackedLinear(tensors[name + '.weight'], bias))
    return module


def load_encoder(module, path, *, dino):
    reader = ExternalReader(path)
    architecture = reader.get_field('general.architecture')
    if dino and (architecture is None or architecture.contents() != 'dinov3'):
        raise ValueError('DINO encoder GGUF has unexpected architecture')
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    if len(tensors) != len(reader.tensors):
        raise ValueError('duplicate encoder tensor name')
    expected = module.state_dict()
    missing = set(expected) - set(tensors)
    allowed_missing = {'bn.num_batches_tracked'} if not dino else set()
    if missing - allowed_missing or set(tensors) - set(expected):
        raise ValueError(f'encoder tensor names differ: missing={sorted(missing)}, extra={sorted(set(tensors)-set(expected))}')
    if missing and expected['bn.num_batches_tracked'].item() != 0:
        raise ValueError('omitted encoder batch counter is not zero')
    packed = {gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q8_0,
              gguf.GGMLQuantizationType.Q2_K, gguf.GGMLQuantizationType.Q4_K,
              gguf.GGMLQuantizationType.Q6_K}
    allowed = packed | {gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32}
    if dino:
        allowed.add(gguf.GGMLQuantizationType.BF16)
    for name, tensor in tensors.items():
        if (tensor.tensor_type not in allowed or tuple(reversed(tensor.shape.tolist())) != tuple(expected[name].shape)
                or not validate_tensor_values(tensor)):
            raise ValueError('invalid encoder tensor: ' + name)
        if tensor.tensor_type in packed:
            if not dino or not name.endswith('.weight'):
                raise ValueError('unsupported packed encoder tensor: ' + name)
            try:
                layer = module.get_submodule(name[:-len('.weight')])
            except AttributeError as exc:
                raise ValueError('missing encoder linear layer: ' + name) from exc
            if not isinstance(layer, torch.nn.Linear):
                raise ValueError('packed encoder tensor is not linear: ' + name)
    with torch.no_grad():
        for name, tensor in tensors.items():
            if tensor.tensor_type in packed:
                continue
            values = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(expected[name].shape)
            expected[name].copy_(torch.from_numpy(np.array(values, copy=True)))
    del expected
    if dino:
        return replace_quantized_linear_layers(module, tensors)
    return []
