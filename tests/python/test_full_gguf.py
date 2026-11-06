"""Round-trip the complete four-component GGUF export and reject missing input."""
import importlib.util
import io
import json
import os
import sys
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

if any(importlib.util.find_spec(name) is None for name in ("numpy", "gguf", "safetensors", "torch")):
    print("skipped: needs numpy, gguf, safetensors and torch")
    sys.exit(77)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import gguf
import numpy as np
import torch
from safetensors.numpy import save_file
from export_full_gguf import COMPONENTS, GGMLQuantizer, encoding, export
from idol_gguf import IdolGGUF, validate_runtime_half_storage
from gguf_to_decoder import native_blob
import idol_generate
from idol_packed import PackedLinear, replace_quantized_linear_layers, enable_fused_packed
from idol_encoder_gguf import build_dino, build_flux_vae, load_encoder, dino_config_from_metadata, public_flux_config
from export_dinov3_gguf import LINEAR


encoders = {'dinov3': 'gated/dino', 'flux2_vae': 'gated/flux'}
assert idol_generate.encoder_configs_requiring_access(encoders, None, None, 'dino.gguf', 'flux.gguf') == []
assert idol_generate.encoder_configs_requiring_access(encoders, None, None, 'dino.gguf', None) == ['gated/flux']
assert idol_generate.encoder_configs_requiring_access(encoders, 'override/dino', None, 'dino.gguf', 'flux.gguf') == ['override/dino']

class FakeModule:
    def __init__(self, *args, **kwargs):
        pass
    def eval(self):
        return self
    def requires_grad_(self, flag):
        return self
    def to(self, device):
        return self
    def register_forward_pre_hook(self, hook):
        pass

class FakePipeline:
    pass

class FakeImageEncoder:
    pass

fake_model = SimpleNamespace(config={'image_encoders': encoders, 'denoiser': {'args': {}}},
                             load_module=lambda component, module: module)
fake_sdk = {'idol': SimpleNamespace(),
            'idol.conditioning': SimpleNamespace(ImageEncoder=FakeImageEncoder,
                require_encoder_access=lambda repositories: (_ for _ in ()).throw(AssertionError('unexpected gated access'))),
            'idol.models': SimpleNamespace(BiRefNet=FakeModule, LatentSeqMMFlowModel=FakeModule),
            'idol.pipeline': SimpleNamespace(IdolPipeline=FakePipeline),
            'diffusers.models': SimpleNamespace(AutoencoderKLFlux2=FakeModule)}
with patch.dict(sys.modules, fake_sdk), patch.object(idol_generate, 'validate_runtime_half_storage'), \
        patch('idol_encoder_gguf.build_dino', return_value=FakeModule()) as make_dino, \
        patch('idol_encoder_gguf.build_flux_vae', return_value=FakeModule()) as make_flux:
    idol_generate.make_pipeline(fake_model, 'cpu', None, None, dino_gguf='dino.gguf', flux_vae_gguf='flux.gguf')
    make_dino.assert_called_once_with(None, 'dino.gguf')
    make_flux.assert_called_once_with(None, 'flux.gguf')
    try:
        idol_generate.make_pipeline(fake_model, 'cpu', None, None, fused_kernels=True)
    except ValueError as exc:
        assert 'CUDA' in str(exc)
    else:
        raise AssertionError('fused kernels were accepted without CUDA')

if importlib.util.find_spec('transformers') is not None:
    metadata = {'model_type': 'dinov3_vit', 'image_size': 224, 'patch_size': 16,
                'num_channels': 3, 'hidden_size': 1280, 'intermediate_size': 5120,
                'num_hidden_layers': 32, 'num_attention_heads': 20, 'hidden_act': 'silu',
                'use_gated_mlp': True, 'query_bias': True, 'key_bias': False,
                'value_bias': True, 'proj_bias': True, 'mlp_bias': True,
                'num_register_tokens': 4, 'attention_dropout': 0.0, 'initializer_range': 0.02,
                'layer_norm_eps': 1e-5, 'layerscale_value': 1.0, 'drop_path_rate': 0.0,
                'rope_theta': 100.0, 'pos_embed_rescale': 2.0}
    class MetadataReader:
        def get_field(self, key):
            return SimpleNamespace(contents=lambda: metadata[key]) if key in metadata else None
    config = dino_config_from_metadata(MetadataReader())
    assert config.hidden_size == 1280 and config.use_gated_mlp
    metadata['num_hidden_layers'] = 33
    try:
        dino_config_from_metadata(MetadataReader())
    except ValueError:
        pass
    else:
        raise AssertionError('unexpected DINO architecture was accepted')
    del metadata['hidden_size']
    try:
        dino_config_from_metadata(MetadataReader())
    except ValueError:
        pass
    else:
        raise AssertionError('incomplete DINO architecture metadata was accepted')

if importlib.util.find_spec('huggingface_hub') is not None:
    with tempfile.TemporaryDirectory() as temporary:
        invalid = Path(temporary) / 'config.json'
        invalid.write_text('{}')
        with patch('huggingface_hub.hf_hub_download', return_value=str(invalid)) as download:
            try:
                public_flux_config()
            except ValueError as exc:
                assert 'digest' in str(exc)
            else:
                raise AssertionError('untrusted public configuration was accepted')
            assert download.call_args.kwargs['token'] is False


def unpack_native(blob, bits, out, width):
    groups = width // 32
    panels = (out + 15) // 16
    data = np.frombuffer(blob, dtype=np.uint8).reshape(panels, groups, 64 + 64 * bits)
    scale = np.ascontiguousarray(data[:, :, :32]).view("<f2").reshape(panels, groups, 16).astype(np.float32)
    minimum = np.ascontiguousarray(data[:, :, 32:64]).view("<f2").reshape(panels, groups, 16).astype(np.float32)
    body = data[:, :, 64:].reshape(panels, groups, 4, bits, 16)
    if bits == 8:
        codes = body.reshape(panels, groups, 32, 16)
    else:
        codes = np.stack((body & 15, body >> 4), axis=4).reshape(panels, groups, 32, 16)
    values = codes.astype(np.float32) * scale[:, :, None, :] + minimum[:, :, None, :]
    return values.transpose(0, 3, 1, 2).reshape(panels * 16, width)[:out]


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root / "config.json").write_text(json.dumps({"denoiser": {}, "decoder": {}, "decoder_gs": {}, "image_encoders": {}}))
    source = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)
    for file in COMPONENTS.values():
        save_file({"linear.weight": source, "linear.bias": np.array([0.2, 0.5], dtype=np.float32)}, str(root / file))
    for tier in ("F32", "Q8_0", "Q4_0", "Q8_SAFE", "Q4_SAFE"):
        target = root / (tier + ".gguf")
        result = export(root, target, tier, "test-revision")
        reader = gguf.GGUFReader(target)
        assert len(reader.tensors) == 8 and result["tensor_counts"] == {key: 2 for key in COMPONENTS}
        assert reader.get_field("general.architecture").contents() == "idol"
        for tensor in reader.tensors:
            if tensor.name.endswith("linear.weight"):
                actual = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(source.shape)
                tolerance = 0.15 if tier in ("Q4_0", "Q4_SAFE") else 0.01 if tier in ("Q8_0", "Q8_SAFE") else 0.0
                np.testing.assert_allclose(actual, source, rtol=0, atol=tolerance)
                if tensor.name.startswith("idol.octree."):
                    assert tensor.tensor_type == gguf.GGMLQuantizationType.F32
                elif tier in ("Q8_SAFE", "Q4_SAFE") and not tensor.name.startswith("idol.gaussian."):
                    assert tensor.tensor_type == gguf.GGMLQuantizationType.F32
                elif tier in ("Q8_0", "Q4_0", "Q8_SAFE", "Q4_SAFE"):
                    blob, bits, shape = native_blob(tensor)
                    assert bits == (8 if tier in ("Q8_0", "Q8_SAFE") else 4) and shape == source.shape
                    np.testing.assert_allclose(unpack_native(blob, bits, *shape), actual, rtol=0, atol=1e-7)
        del tensor, actual, reader
        model = IdolGGUF(target)
        for component in COMPONENTS:
            module = torch.nn.Module()
            module.linear = torch.nn.Linear(32, 2)
            model.load_module(component, module)
            assert torch.isfinite(module.linear.weight).all()
            tensor = model.tensors[component]["linear.weight"]
            expected = gguf.dequantize(tensor.data, tensor.tensor_type).reshape(source.shape)
            np.testing.assert_array_equal(module.linear.weight.detach().numpy(), expected)
        if tier in ("Q8_0", "Q4_0"):
            module = torch.nn.Module()
            module.linear = torch.nn.Linear(32, 2)
            model.load_module("denoiser", module)
            sample = torch.linspace(-1, 1, 96).reshape(3, 32)
            expected_output = module.linear(sample).detach()
            if importlib.util.find_spec('accelerate') is not None:
                from accelerate import init_empty_weights
                with init_empty_weights():
                    direct = torch.nn.Module()
                    direct.linear = torch.nn.Linear(32, 2)
                model.load_module_packed('denoiser', direct)
                assert isinstance(direct.linear, PackedLinear)
                torch.testing.assert_close(direct.linear(sample), expected_output, rtol=0, atol=0)
                assert not any(value.is_meta for value in direct.parameters())
            assert replace_quantized_linear_layers(module, model.tensors["denoiser"]) == ["linear.weight"]
            torch.testing.assert_close(module.linear(sample), expected_output, rtol=0, atol=0)
        # Malformed values must be rejected before any module tensor changes.
        module = torch.nn.Module()
        module.linear = torch.nn.Linear(32, 2)
        before = {name: value.clone() for name, value in module.state_dict().items()}
        bad = model.tensors["denoiser"]["linear.bias"]
        values = np.array(bad.data, copy=True)
        values.reshape(-1)[0] = np.nan
        model.tensors["denoiser"]["linear.bias"] = bad._replace(data=values)
        try:
            model.load_module("denoiser", module)
        except ValueError as exc:
            assert "non-finite" in str(exc)
        else:
            raise AssertionError("non-finite tensor accepted")
        assert all(torch.equal(value, before[name]) for name, value in module.state_dict().items())
        if tier in ("Q8_0", "Q4_0"):
            bad = model.tensors["gaussian"]["linear.weight"]
            values = np.array(bad.data, copy=True)
            values.reshape(-1)[:2] = [0, 124]  # positive infinity in the block's float16 scale
            model.tensors["gaussian"]["linear.weight"] = bad._replace(data=values)
            try:
                model.load_module("gaussian", module)
            except ValueError as exc:
                assert "non-finite" in str(exc)
            else:
                raise AssertionError("non-finite quantization scale accepted")
        del model, module, tensor, bad, values, expected, before
        try:
            export(root, target, tier, "test-revision")
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing GGUF was overwritten")
    for kind in ('Q4_0', 'Q8_0'):
        target = root / ('octree-' + kind + '.gguf')
        result = export(root, target, 'Q4_0', 'test-revision', octree_quant=kind)
        reader = gguf.GGUFReader(target)
        octree = next(t for t in reader.tensors if t.name == 'idol.octree.linear.weight')
        assert octree.tensor_type == getattr(gguf.GGMLQuantizationType, kind)
        assert result['octree_quant'] == kind and result['octree_quantized_count'] == 1
        assert reader.get_field('idol.quantization.octree').contents() == kind
        assert 'eligible octree matrices ' + kind in reader.get_field('idol.quantization.policy').contents()
        expected = gguf.dequantize(octree.data, octree.tensor_type).reshape(source.shape)
        blob, bits, shape = native_blob(octree)
        np.testing.assert_allclose(unpack_native(blob, bits, *shape), expected, rtol=0, atol=1e-7)
        del expected, octree, reader
    try:
        export(root, root / 'bad-octree.gguf', 'F32', 'test-revision', octree_quant='Q4_0')
    except ValueError:
        pass
    else:
        raise AssertionError('F32 export accepted quantized octree weights')
    save_file({'linear.weight': np.ones((2, 31), dtype=np.float32)}, str(root / COMPONENTS['octree']))
    try:
        export(root, root / 'no-octree-matrices.gguf', 'Q4_0', 'test-revision', octree_quant='Q4_0')
    except ValueError as exc:
        assert 'no eligible octree' in str(exc)
    else:
        raise AssertionError('export claimed octree quantization without quantized matrices')
# Guard callbacks can return status values; a forward pre-hook must not pass
# that value back to PyTorch as replacement model arguments.
with patch.object(sys, 'argv', ['idol_generate', '--repo', 'example/model', '--image', 'input.png', '--out', 'out.ply']), \
     patch.object(sys, 'stderr', new_callable=io.StringIO) as error:
    try:
        idol_generate.main()
    except SystemExit as exc:
        assert exc.code == 2 and '--file is required with --repo' in error.getvalue()
    else:
        raise AssertionError('repository download without a filename was accepted')
calls = []
idol_generate.checkpoint = lambda: calls.append(True) or 123
layer = torch.nn.Linear(2, 1)
layer.register_forward_pre_hook(idol_generate.checkpoint_before_forward)
assert layer(torch.ones(1, 2)).shape == (1, 1) and calls == [True]
cloud = SimpleNamespace(xyz=torch.zeros(2, 3), features_dc=torch.zeros(2, 1, 3), scaling=torch.ones(2, 3),
                        rotation=torch.zeros(2, 4), opacity=torch.ones(2, 1))
idol_generate.validate_gaussian_cloud(cloud, 2)
cloud.opacity[0, 0] = float('nan')
try:
    idol_generate.validate_gaussian_cloud(cloud, 2)
except ValueError as exc:
    assert 'non-finite opacity' in str(exc)
else:
    raise AssertionError('non-finite Gaussian output accepted')
if importlib.util.find_spec('tensordict') is not None:
    from PIL import Image
    from tensordict import TensorDict
    fake_pipeline = SimpleNamespace(
        device=torch.device('cpu'), optimize=False,
        denoiser=SimpleNamespace(q_token_length=2, in_channels=16, cam_channels=4),
        preprocess=lambda image, mode: image,
        image_encoder=lambda pixels: TensorDict({'feature': torch.zeros(1, 2)}, batch_size=1),
        _sampling_settings=lambda: nullcontext())
    fake_camera = SimpleNamespace(decode_camera=lambda value: {'yaw': 0.0})
    fake_sampler = SimpleNamespace(sample_flow=lambda model, noise, **kwargs: noise)
    with patch.dict(sys.modules, {'idol': SimpleNamespace(), 'idol.camera': fake_camera, 'idol.sampler': fake_sampler}):
        image = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
        ordinary, _, camera = idol_generate.image_to_latent(fake_pipeline, image, 'synthetic', 42, 1, 3.0, 3.0)
        staged, _, staged_camera = idol_generate.image_to_latent(fake_pipeline, image, 'synthetic', 42, 1, 3.0, 3.0, True)
        np.testing.assert_array_equal(ordinary, staged)
        assert camera == staged_camera
# Compare the device-side block decoders to the installed GGUF reference using
# independent, fixed byte patterns, including the K-block formats not yet
# accepted by the production Idol container reader.
rng = np.random.default_rng(42)
for kind, block_bytes, values_per_block, offsets in (
    (gguf.GGMLQuantizationType.Q2_K, 84, 256, (80, 82)),
    (gguf.GGMLQuantizationType.Q4_K, 144, 256, (0, 2)),
    (gguf.GGMLQuantizationType.Q6_K, 210, 256, (208,)),
):
    raw = rng.integers(0, 256, size=(4, block_bytes), dtype=np.uint8)
    for offset in offsets:
        raw[:, offset:offset + 2] = [0, 60]  # finite float16 scale 1
    pseudo = SimpleNamespace(shape=np.array([512, 2]), tensor_type=kind, data=raw)
    layer = PackedLinear(pseudo, None)
    expected = gguf.dequantize(raw.reshape(2, -1), kind).reshape(2, 512)
    np.testing.assert_array_equal(layer.dequantize(torch.float32).numpy(), expected)
ggml_lib = os.environ.get('IDOL_GGML_LIB')
if ggml_lib:
    quantizer = GGMLQuantizer(ggml_lib)
    sample_weights = np.random.default_rng(42).normal(size=(4, 256)).astype(np.float32)
    importance = np.ones(256, dtype=np.float32)
    importance[::3] = 1000
    for kind in (gguf.GGMLQuantizationType.Q2_K, gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K):
        plain = gguf.dequantize(quantizer.quantize(sample_weights, kind), kind).reshape(sample_weights.shape)
        weighted = gguf.dequantize(quantizer.quantize(sample_weights, kind, importance), kind).reshape(sample_weights.shape)
        assert np.sum((weighted - sample_weights) ** 2 * importance) < np.sum((plain - sample_weights) ** 2 * importance)
    for invalid in (importance[:-1], np.full(256, np.nan), np.zeros(256)):
        try:
            quantizer.quantize(sample_weights, gguf.GGMLQuantizationType.Q4_K, invalid)
        except ValueError as exc:
            assert 'importance' in str(exc)
        else:
            raise AssertionError('invalid GGML importance vector accepted')
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 'config.json').write_text(json.dumps({'denoiser': {}, 'decoder': {}, 'decoder_gs': {}, 'image_encoders': {}}))
        weights = np.linspace(-1, 1, 512, dtype=np.float32).reshape(2, 256)
        for file in COMPONENTS.values():
            save_file({'linear.weight': weights, 'linear.bias': np.array([0.2, 0.5], dtype=np.float32)}, str(root / file))
        for tier in ('Q2_K', 'Q4_K', 'Q6_K'):
            target = root / (tier + '.gguf')
            export(root, target, tier, 'test-revision', ggml_lib)
            model = IdolGGUF(target)
            assert model.tensors['denoiser']['linear.weight'].tensor_type == getattr(gguf.GGMLQuantizationType, tier)
            module = torch.nn.Module()
            module.linear = torch.nn.Linear(256, 2)
            model.load_module('denoiser', module)
            sample = torch.linspace(-1, 1, 512).reshape(2, 256)
            baseline = module.linear(sample).detach()
            assert replace_quantized_linear_layers(module, model.tensors['denoiser']) == ['linear.weight']
            torch.testing.assert_close(module.linear(sample), baseline, rtol=0, atol=0)
            del model, module
else:
    print('K-block export round trip skipped: IDOL_GGML_LIB is unset')
with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / 'encoder.gguf'
    writer = gguf.GGUFWriter(path, arch='dinov3')
    original = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)
    quantized = gguf.quantize(original, gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor('linear.weight', quantized, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.add_tensor('linear.bias', np.array([0.2, 0.5], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    module = torch.nn.Module()
    module.linear = torch.nn.Linear(32, 2)
    assert load_encoder(module, path, dino=True) == ['linear.weight']
    np.testing.assert_array_equal(module.linear.dequantize(torch.float32).numpy(),
                                  gguf.dequantize(quantized, gguf.GGMLQuantizationType.Q8_0))
    assert module.linear(torch.zeros(1, 32)).shape == (1, 2)
    before = module.linear(torch.ones(1, 32)).detach()
    assert enable_fused_packed(module) == 1 and module.linear.fused
    torch.testing.assert_close(module.linear(torch.ones(1, 32)), before, rtol=0, atol=0)
    if all(importlib.util.find_spec(name) is not None for name in ('accelerate', 'transformers')):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 2)
        with patch('transformers.AutoConfig.from_pretrained', return_value=object()), \
                patch('transformers.AutoModel.from_config', side_effect=lambda config: TinyEncoder()):
            direct = build_dino('test-configuration', path)
        sample = torch.linspace(-1, 1, 32).reshape(1, 32).to(torch.bfloat16)
        module = module.to(torch.bfloat16)
        torch.testing.assert_close(direct.linear(sample), module.linear(sample), rtol=0, atol=0)
    else:
        print('direct encoder loading test skipped: needs accelerate and transformers')
with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / 'encoder-bf16.gguf'
    original = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)
    original[0, 0] = 1e-30
    encoded = gguf.quantize(original, gguf.GGMLQuantizationType.BF16)
    writer = gguf.GGUFWriter(path, arch='dinov3')
    writer.add_tensor('linear.weight', encoded, raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.add_tensor('linear.bias', np.array([0.2, 0.5], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    module = torch.nn.Module()
    module.linear = torch.nn.Linear(32, 2)
    assert load_encoder(module, path, dino=True) == []
    expected = torch.from_numpy(original).to(torch.bfloat16).float()
    torch.testing.assert_close(module.linear.weight, expected, rtol=0, atol=0)
    if all(importlib.util.find_spec(name) is not None for name in ('accelerate', 'transformers')):
        class TinyBFloatEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(32, 2)
        with patch('transformers.AutoConfig.from_pretrained', return_value=object()), \
                patch('transformers.AutoModel.from_config', side_effect=lambda config: TinyBFloatEncoder()):
            direct = build_dino('test-configuration', path)
        torch.testing.assert_close(direct.linear.weight.float(), expected, rtol=0, atol=0)
    bad = encoded.copy()
    bad.reshape(-1)[:2] = np.array([255, 127], dtype=np.uint8)
    bad_path = Path(temporary) / 'encoder-bf16-nan.gguf'
    writer = gguf.GGUFWriter(bad_path, arch='dinov3')
    writer.add_tensor('linear.weight', bad, raw_dtype=gguf.GGMLQuantizationType.BF16)
    writer.add_tensor('linear.bias', np.array([0.2, 0.5], dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    try:
        load_encoder(module, bad_path, dino=True)
    except ValueError:
        pass
    else:
        raise AssertionError('non-finite bfloat16 encoder tensor accepted')
if all(importlib.util.find_spec(name) is not None for name in ('accelerate', 'diffusers')):
    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)
            self.bn = torch.nn.BatchNorm1d(2)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / 'vae.gguf'
        original = TinyVAE().eval().to(torch.bfloat16)
        writer = gguf.GGUFWriter(path, arch='test-vae')
        for name, value in original.state_dict().items():
            if name != 'bn.num_batches_tracked':
                writer.add_tensor(name, value.float().numpy().astype(np.float16))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        load_encoder(original, path, dino=False)
        fake_vae = SimpleNamespace(load_config=lambda repository, subfolder: {}, from_config=lambda config: TinyVAE())
        with patch.dict(sys.modules, {'diffusers.models': SimpleNamespace(AutoencoderKLFlux2=fake_vae)}):
            direct = build_flux_vae('test-configuration', path)
        for name, value in original.state_dict().items():
            torch.testing.assert_close(direct.state_dict()[name], value, rtol=0, atol=0)
        bad_path = Path(temporary) / 'invalid-vae.gguf'
        writer = gguf.GGUFWriter(bad_path, arch='test-vae')
        writer.add_tensor('linear.weight', np.zeros((2, 4), dtype=np.float16))
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        with patch.dict(sys.modules, {'diffusers.models': SimpleNamespace(AutoencoderKLFlux2=fake_vae)}):
            try:
                build_flux_vae('test-configuration', bad_path)
            except ValueError as exc:
                assert 'tensor names' in str(exc)
            else:
                raise AssertionError('incomplete autoencoder file was accepted')
else:
    print('direct autoencoder loading test skipped: needs accelerate and diffusers')
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root / 'config.json').write_text(json.dumps({'denoiser': {}, 'decoder': {}, 'decoder_gs': {}, 'image_encoders': {}}))
    weights = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)
    bias = np.array([0.2, 0.5], dtype=np.float32)
    save_file({'blocks.12.attn.weight': weights, 'blocks.12.attn.bias': bias}, str(root / COMPONENTS['denoiser']))
    for component in ('octree', 'gaussian', 'background'):
        save_file({'linear.weight': weights, 'linear.bias': bias}, str(root / COMPONENTS[component]))
    target = root / 'runtime-half.gguf'
    export(root, target, 'RUNTIME_F16_Q8', 'test-revision')
    model = IdolGGUF(target)
    assert model.tensors['denoiser']['blocks.12.attn.weight'].tensor_type == gguf.GGMLQuantizationType.F16
    assert model.tensors['gaussian']['linear.weight'].tensor_type == gguf.GGMLQuantizationType.Q8_0
    module = torch.nn.Module()
    module.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(13)])
    module.blocks[12].attn = torch.nn.Linear(32, 2).half()
    validate_runtime_half_storage(model, module)
    model.load_module('denoiser', module)
    np.testing.assert_array_equal(module.blocks[12].attn.weight.detach().numpy(), weights.astype(np.float16))
    expected = module.blocks[12].attn(torch.ones(1, 32, dtype=torch.float16)).detach()
    if importlib.util.find_spec('accelerate') is not None:
        from accelerate import init_empty_weights
        with init_empty_weights():
            direct = torch.nn.Module()
            direct.blocks = torch.nn.ModuleList([torch.nn.Module() for _ in range(13)])
            direct.blocks[12].attn = torch.nn.Linear(32, 2).half()
        validate_runtime_half_storage(model, direct)
        model.load_module_packed('denoiser', direct, include_half=True)
        assert direct.blocks[12].attn.kind == gguf.GGMLQuantizationType.F16
        torch.testing.assert_close(direct.blocks[12].attn(torch.ones(1, 32, dtype=torch.float16)), expected, rtol=0, atol=0)
    assert replace_quantized_linear_layers(module, model.tensors['denoiser'], include_half=True) == ['blocks.12.attn.weight']
    torch.testing.assert_close(module.blocks[12].attn(torch.ones(1, 32, dtype=torch.float16)), expected, rtol=0, atol=0)
    del model
assert LINEAR.fullmatch('layer.31.mlp.gate_proj.weight')
assert not LINEAR.fullmatch('layer.32.mlp.gate_proj.weight')
assert not LINEAR.fullmatch('layer.0.norm1.weight')
example = np.linspace(-1, 1, 64, dtype=np.float32).reshape(2, 32)
assert encoding('denoiser', 'blocks.12.attn.weight', example, 'Q8_MIXED')[1] == gguf.GGMLQuantizationType.Q8_0
assert encoding('denoiser', 'blocks.11.attn.weight', example, 'Q8_MIXED')[1] is None
assert encoding('background', 'blocks.12.attn.weight', example, 'Q8_MIXED')[1] is None
half, half_type = encoding('denoiser', 'blocks.12.attn.weight', example, 'RUNTIME_F16_Q8')
assert half.dtype == np.float16 and half_type is None
assert encoding('denoiser', 'blocks.12.attn.weight', example, 'RUNTIME_F16_Q8')[0].dtype == np.float16
assert encoding('denoiser', 'repo_layers.12.gate_map.weight', example, 'RUNTIME_F16_Q8')[0].dtype == np.float32
half_source = SimpleNamespace(shape=np.array([32, 2]), tensor_type=gguf.GGMLQuantizationType.F16, data=half)
half_model = torch.nn.Module()
half_model.linear = torch.nn.Linear(32, 2)
field = SimpleNamespace(contents=lambda: 'RUNTIME_F16_Q8')
fake_model = SimpleNamespace(reader=SimpleNamespace(get_field=lambda name: field),
                             tensors={'denoiser': {'linear.weight': half_source}})
try:
    validate_runtime_half_storage(fake_model, half_model)
except ValueError as exc:
    assert 'not half precision' in str(exc)
else:
    raise AssertionError('runtime precision mismatch accepted')
half_model.half()
validate_runtime_half_storage(fake_model, half_model)
half_model.float()
with torch.no_grad():
    half_model.linear.weight.copy_(torch.from_numpy(half.astype(np.float32)))
sample = torch.linspace(-1, 1, 96).reshape(3, 32)
before = half_model.linear(sample).detach()
assert replace_quantized_linear_layers(half_model, {'linear.weight': half_source}, include_half=True) == ['linear.weight']
torch.testing.assert_close(half_model.linear(sample), before, rtol=0, atol=0)
print("four-component GGUF round trips and guard hook passed")
