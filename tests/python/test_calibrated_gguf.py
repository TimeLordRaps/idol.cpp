"""Exercise calibration provenance checks and complete mixed-precision exports."""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

if any(importlib.util.find_spec(name) is None for name in ('numpy', 'gguf', 'safetensors', 'torch')):
    print('skipped: needs numpy, gguf, safetensors and torch')
    sys.exit(77)
library = os.environ.get('IDOL_GGML_LIB')
if not library:
    print('skipped: needs IDOL_GGML_LIB for K-block quantization')
    sys.exit(77)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
import gguf
import numpy as np
import torch
from safetensors.numpy import save_file
from create_gaussian_calibration import create
from calibrate_gaussian_covariance import create as create_covariance
from calibrate_denoiser_covariance import create as create_denoiser_covariance
from calibrate_denoiser import verify_original_denoiser
from create_mixed_gaussian_codes import create as create_mixed_gaussian
from create_mixed_denoiser_codes import create as create_mixed_denoiser
from select_gaussian_mixed import run as select_mixed_gaussian
from export_full_gguf import COMPONENTS, GGMLQuantizer, digest, export
from idol_gguf import IdolGGUF, validate_runtime_half_storage
from select_gaussian_calibration import run as select_calibration


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root / 'config.json').write_text(json.dumps({'denoiser': {}, 'decoder': {}, 'decoder_gs': {}, 'image_encoders': {}}))
    source = np.random.default_rng(42).normal(size=(2, 256)).astype(np.float32)
    for filename in COMPONENTS.values():
        save_file({'blocks.0.linear.weight': source, 'blocks.0.linear.bias': np.array([0.2, 0.5], np.float32)}, str(root / filename))
    moments = root / 'moments.pt'
    diagonal = torch.ones(256)
    diagonal[::3] = 1000
    torch.save({'gs.blocks.0.linear.weight': torch.diag(diagonal)}, moments)
    report = root / 'moments.json'
    report.write_text('{}')
    sdk = root / 'sdk'
    (sdk / 'idol').mkdir(parents=True)
    (sdk / 'idol' / '__init__.py').write_text('')
    latents = root / 'latents'
    latents.mkdir()
    (latents / '0.f32').write_bytes(b'first')
    (latents / '1.f32').write_bytes(b'first')
    args = SimpleNamespace(weights=root, sdk=sdk, latents=latents, moments=moments, moments_report=report,
                           ggml_lib=Path(library), out=root / 'selection.json', type='Q4_K', holdout=1,
                           calib=1, anchors=1, threads=1, min_available_gb=0, seed=42)
    try:
        select_calibration(args)
    except ValueError as exc:
        assert 'byte-disjoint' in str(exc)
    else:
        raise AssertionError('duplicate held-out and calibration latents accepted')
    assert not args.out.exists()
    (latents / '1.f32').write_bytes(b'second')
    try:
        select_calibration(args)
    except ValueError as exc:
        assert 'moments report' in str(exc)
    else:
        raise AssertionError('unbound selection inputs accepted')
    assert not args.out.exists()
    report.write_text(json.dumps({'status': 'complete', 'hashes_unchanged_after_run': True,
                                  'hessians_sha256': digest(moments),
                                  'inputs': {'gaussian_weights': digest(root / COMPONENTS['gaussian'])}}))
    quantizer = GGMLQuantizer(library)
    for kind in ('Q2_K', 'Q4_K', 'Q6_K'):
        selection = root / (kind + '-selection.json')
        selection.write_text(json.dumps({'scope': 'experiment ' + kind, 'status': 'complete',
                                         'hashes_unchanged_after_run': True,
                                         'inputs_sha256': {'gaussian': digest(root / COMPONENTS['gaussian']),
                                                           'moments': digest(moments), 'moments_report': digest(report)},
                                         'selection': {'choices': {'blocks.0': {'weighted': True}}}}))
        plan = root / (kind + '-plan.npz')
        created = create(root / COMPONENTS['gaussian'], moments, report, selection, kind, plan)
        assert created['eligible_matrices'] == created['weighted_matrices'] == 1
        tier = 'RUNTIME_F16_GAUSSIAN_' + kind.replace('_', '')
        target = root / (kind + '.gguf')
        result = export(root, target, tier, 'test-revision', library, plan)
        model = IdolGGUF(target)
        assert model.reader.get_field('idol.quantization.calibration_sha256').contents() == digest(plan)
        assert result['source_sha256']['gaussian_calibration'] == digest(plan)
        tensor = model.tensors['gaussian']['blocks.0.linear.weight']
        quant_type = getattr(gguf.GGMLQuantizationType, kind)
        assert tensor.tensor_type == quant_type
        imatrix = np.maximum(diagonal.numpy() / diagonal.numpy().mean(), 1e-6)
        expected = gguf.dequantize(quantizer.quantize(source, quant_type, imatrix), quant_type).reshape(source.shape)
        np.testing.assert_array_equal(gguf.dequantize(tensor.data, quant_type).reshape(source.shape), expected)
        module = torch.nn.Module()
        module.blocks = torch.nn.ModuleList([torch.nn.Module()])
        module.blocks[0].linear = torch.nn.Linear(256, 2).half()
        validate_runtime_half_storage(model, module)
        module.float()
        try:
            validate_runtime_half_storage(model, module)
        except ValueError:
            pass
        else:
            raise AssertionError('incompatible runtime accepted for calibrated tier')
        del model, tensor
        with np.load(plan, allow_pickle=False) as archive:
            corrupt = {name: archive[name].copy() for name in archive.files}
        corrupt['blocks.0.linear.weight'] = corrupt['blocks.0.linear.weight'][:-1]
        bad_plan = root / (kind + '-bad.npz')
        with bad_plan.open('xb') as stream:
            np.savez(stream, **corrupt)
        bad_target = root / (kind + '-bad.gguf')
        try:
            export(root, bad_target, tier, 'test-revision', library, bad_plan)
        except ValueError as exc:
            assert 'calibration vector' in str(exc)
        else:
            raise AssertionError('malformed calibration accepted')
        assert not bad_target.exists() and not bad_target.with_name(bad_target.name + '.partial').exists()
    covariance_codes = root / 'covariance-q2.npz'
    covariance = create_covariance(root / COMPONENTS['gaussian'], moments, report, library, covariance_codes)
    assert covariance['eligible_matrices'] == 1
    weighted_codes = root / 'weighted-covariance-q2.npz'
    weighted = create_covariance(root / COMPONENTS['gaussian'], moments, report, library, weighted_codes, importance=True)
    assert weighted['importance'] == 'covariance_diagonal'
    with np.load(covariance_codes, allow_pickle=False) as plain_archive, np.load(weighted_codes, allow_pickle=False) as weighted_archive:
        plain_matrix = gguf.dequantize(plain_archive['blocks.0.linear.weight'], gguf.GGMLQuantizationType.Q2_K).reshape(source.shape)
        weighted_matrix = gguf.dequantize(weighted_archive['blocks.0.linear.weight'], gguf.GGMLQuantizationType.Q2_K).reshape(source.shape)
    assert np.sum((source-weighted_matrix)**2 * diagonal.numpy()) < np.sum((source-plain_matrix)**2 * diagonal.numpy())
    wrong_covariance_report = root / 'wrong-covariance-report.json'
    wrong_covariance_report.write_text(json.dumps({'status': 'complete', 'hashes_unchanged_after_run': True,
                                                    'hessians_sha256': '0' * 64,
                                                    'inputs': {'gaussian_weights': digest(root / COMPONENTS['gaussian'])}}))
    wrong_covariance_output = root / 'wrong-covariance-output.npz'
    try:
        create_covariance(root / COMPONENTS['gaussian'], moments, wrong_covariance_report, library, wrong_covariance_output)
    except ValueError as exc:
        assert 'does not bind' in str(exc)
    else:
        raise AssertionError('unbound covariance accepted')
    assert not wrong_covariance_output.exists() and not wrong_covariance_output.with_name(wrong_covariance_output.name + '.partial').exists()
    wide_source = root / 'wide-gaussian.safetensors'
    wide = np.random.default_rng(7).normal(size=(8, 512)).astype(np.float32)
    save_file({'matrix.weight': wide}, str(wide_source))
    wide_moments = root / 'wide-moments.pt'
    eye = torch.eye(256)
    wide_hessian = torch.cat((torch.cat((eye, 0.8 * eye), dim=1),
                              torch.cat((0.8 * eye, eye), dim=1)), dim=0)
    torch.save({'gs.matrix.weight': wide_hessian}, wide_moments)
    wide_report = root / 'wide-moments.json'
    wide_report.write_text(json.dumps({'status': 'complete', 'hashes_unchanged_after_run': True,
                                       'hessians_sha256': digest(wide_moments),
                                       'inputs': {'gaussian_weights': digest(wide_source)}}))
    wide_codes = root / 'wide-covariance.npz'
    create_covariance(wide_source, wide_moments, wide_report, library, wide_codes)
    with np.load(wide_codes, allow_pickle=False) as archive:
        compensated = gguf.dequantize(archive['matrix.weight'], gguf.GGMLQuantizationType.Q2_K).reshape(wide.shape)
    plain = gguf.dequantize(quantizer.quantize(wide, gguf.GGMLQuantizationType.Q2_K),
                            gguf.GGMLQuantizationType.Q2_K).reshape(wide.shape)
    hessian = wide_hessian.numpy()
    def covariance_error(candidate):
        difference = wide - candidate
        return float(((difference @ hessian) * difference).sum())
    assert covariance_error(compensated) < covariance_error(plain)
    covariance_target = root / 'covariance-q2.gguf'
    result = export(root, covariance_target, 'RUNTIME_F16_GAUSSIAN_Q2K_COV', 'test-revision',
                    gaussian_encoded=covariance_codes)
    model = IdolGGUF(covariance_target)
    assert result['source_sha256']['gaussian_encoded'] == digest(covariance_codes)
    assert model.reader.get_field('idol.quantization.encoded_sha256').contents() == digest(covariance_codes)
    tensor = model.tensors['gaussian']['blocks.0.linear.weight']
    assert tensor.tensor_type == gguf.GGMLQuantizationType.Q2_K
    with np.load(covariance_codes, allow_pickle=False) as archive:
        encoded = {name: archive[name].copy() for name in archive.files}
    np.testing.assert_array_equal(tensor.data, encoded['blocks.0.linear.weight'])
    module = torch.nn.Module()
    module.blocks = torch.nn.ModuleList([torch.nn.Module()])
    module.blocks[0].linear = torch.nn.Linear(256, 2).half()
    validate_runtime_half_storage(model, module)
    module.float()
    try:
        validate_runtime_half_storage(model, module)
    except ValueError:
        pass
    else:
        raise AssertionError('incompatible runtime accepted for covariance tier')
    del model, tensor
    for label, bad_values in (
            ('shape', {**encoded, 'blocks.0.linear.weight': encoded['blocks.0.linear.weight'][:, :-1]}),
            ('nonfinite', {**encoded, 'blocks.0.linear.weight': encoded['blocks.0.linear.weight'].copy()}),
            ('source', {**encoded, '__metadata__': np.frombuffer(json.dumps({**covariance, 'source_sha256': '0' * 64}).encode(), dtype=np.uint8)}),
            ('quantizer', {**encoded, '__metadata__': np.frombuffer(json.dumps({**covariance, 'quantizer_sha256': 'invalid'}).encode(), dtype=np.uint8)})):
        if label == 'nonfinite':
            bad_values['blocks.0.linear.weight'][0, 80:82] = np.array([255, 127], dtype=np.uint8)
        bad_codes = root / ('covariance-bad-' + label + '.npz')
        with bad_codes.open('xb') as stream:
            np.savez(stream, **bad_values)
        bad_target = root / ('covariance-bad-' + label + '.gguf')
        try:
            export(root, bad_target, 'RUNTIME_F16_GAUSSIAN_Q2K_COV', 'test-revision', gaussian_encoded=bad_codes)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid covariance archive accepted: ' + label)
        assert not bad_target.exists() and not bad_target.with_name(bad_target.name + '.partial').exists()
    duplicate_codes = root / 'covariance-duplicate.npz'
    shutil.copyfile(covariance_codes, duplicate_codes)
    with zipfile.ZipFile(duplicate_codes, 'a') as archive:
        metadata_entry = archive.read('__metadata__.npy')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            archive.writestr('__metadata__.npy', metadata_entry)
    try:
        export(root, root / 'covariance-duplicate.gguf', 'RUNTIME_F16_GAUSSIAN_Q2K_COV', 'test-revision',
               gaussian_encoded=duplicate_codes)
    except ValueError as exc:
        assert 'duplicated' in str(exc)
    else:
        raise AssertionError('duplicate encoded tensor accepted')
    combined_source = root / 'combined-source'
    combined_source.mkdir()
    shutil.copyfile(root / 'config.json', combined_source / 'config.json')
    for filename in COMPONENTS.values():
        shutil.copyfile(root / filename, combined_source / filename)
    save_file({'blocks.0.linear.weight': source, 'blocks.0.linear.bias': np.array([0.2, 0.5], np.float32),
               'noise_refiner.linear.weight': source, 'noise_refiner.linear.bias': np.array([0.2, 0.5], np.float32)},
              str(combined_source / COMPONENTS['denoiser']))
    original_module = torch.nn.Module()
    original_module.blocks = torch.nn.ModuleList([torch.nn.Module()])
    original_module.blocks[0].linear = torch.nn.Linear(256, 2).half()
    original_module.noise_refiner = torch.nn.Module()
    original_module.noise_refiner.linear = torch.nn.Linear(256, 2).half()
    with torch.no_grad():
        for layer in (original_module.blocks[0].linear, original_module.noise_refiner.linear):
            layer.weight.copy_(torch.from_numpy(source))
            layer.bias.copy_(torch.tensor([0.2, 0.5]))
    verify_original_denoiser(original_module, combined_source / COMPONENTS['denoiser'])
    with torch.no_grad():
        original_module.blocks[0].linear.weight[0, 0] += 1
    try:
        verify_original_denoiser(original_module, combined_source / COMPONENTS['denoiser'])
    except ValueError as exc:
        assert 'differs from the original runtime' in str(exc)
    else:
        raise AssertionError('modified denoiser was accepted for calibration')
    denoiser_moments = root / 'denoiser-moments.pt'
    torch.save({'blocks.0.linear.weight': torch.diag(diagonal)}, denoiser_moments)
    denoiser_report = root / 'denoiser-moments.json'
    denoiser_report.write_text(json.dumps({'hessians_sha256': digest(denoiser_moments),
                                           'inputs_sha256': {'source_sha256': digest(combined_source / COMPONENTS['denoiser']),
                                                             'model_sha256': 'a' * 64},
                                           'variants': {'plain': {}, 'damp_0.3': {}}}))
    denoiser_codes = root / 'denoiser-covariance.npz'
    denoiser_result = create_denoiser_covariance(combined_source / COMPONENTS['denoiser'], denoiser_moments,
                                                denoiser_report, library, denoiser_codes)
    assert denoiser_result['eligible_matrices'] == 1
    combined_target = root / 'combined.gguf'
    combined = export(combined_source, combined_target, 'RUNTIME_F16_GAUSSIAN_Q2K_COV', 'test-revision',
                      gaussian_encoded=covariance_codes, denoiser_encoded=denoiser_codes)
    assert combined['tier'] == 'RUNTIME_F16_GAUSSIAN_Q2K_COV_DENOISER_Q80_COV'
    model = IdolGGUF(combined_target)
    assert model.reader.get_field('idol.quantization.denoiser_encoded_sha256').contents() == digest(denoiser_codes)
    assert model.tensors['denoiser']['blocks.0.linear.weight'].tensor_type == gguf.GGMLQuantizationType.Q8_0
    assert model.tensors['gaussian']['blocks.0.linear.weight'].tensor_type == gguf.GGMLQuantizationType.Q2_K
    all_q8_target = root / 'calibrated-all-q8.gguf'
    all_q8 = export(combined_source, all_q8_target, 'Q8_0', 'test-revision',
                    denoiser_encoded=denoiser_codes, octree_quant='Q8_0')
    all_q8_model = IdolGGUF(all_q8_target)
    assert all_q8['tier'] == 'Q8_0_DENOISER_Q80_COV'
    assert all_q8_model.reader.get_field('idol.quantization.denoiser_encoded_sha256').contents() == digest(denoiser_codes)
    assert all_q8_model.tensors['denoiser']['blocks.0.linear.weight'].tensor_type == gguf.GGMLQuantizationType.Q8_0
    assert all_q8_model.tensors['gaussian']['blocks.0.linear.weight'].tensor_type == gguf.GGMLQuantizationType.Q8_0
    del all_q8_model
    module = torch.nn.Module()
    module.blocks = torch.nn.ModuleList([torch.nn.Module()])
    module.blocks[0].linear = torch.nn.Linear(256, 2).half()
    module.noise_refiner = torch.nn.Module()
    module.noise_refiner.linear = torch.nn.Linear(256, 2).half()
    validate_runtime_half_storage(model, module)
    module.blocks.float()
    try:
        validate_runtime_half_storage(model, module)
    except ValueError as exc:
        assert 'encoded denoiser block' in str(exc)
    else:
        raise AssertionError('changed encoded-block runtime precision accepted')
    module.blocks.half()
    saved_denoiser = model.tensors['denoiser']
    model.tensors['denoiser'] = {name: tensor for name, tensor in saved_denoiser.items() if not name.startswith('blocks.')}
    try:
        validate_runtime_half_storage(model, module)
    except ValueError as exc:
        assert 'no eligible runtime blocks' in str(exc)
    else:
        raise AssertionError('encoded tier without denoiser blocks accepted')
    model.tensors['denoiser'] = saved_denoiser
    del saved_denoiser
    module.float()
    try:
        validate_runtime_half_storage(model, module)
    except ValueError:
        pass
    else:
        raise AssertionError('incompatible runtime accepted for combined tier')
    del model
    with np.load(denoiser_codes, allow_pickle=False) as archive:
        denoiser_values = {name: archive[name].copy() for name in archive.files}
    denoiser_values['blocks.0.linear.weight'] = denoiser_values['blocks.0.linear.weight'][:, :-1]
    bad_denoiser_codes = root / 'bad-denoiser-covariance.npz'
    with bad_denoiser_codes.open('xb') as stream:
        np.savez(stream, **denoiser_values)
    bad_combined = root / 'bad-combined.gguf'
    try:
        export(combined_source, bad_combined, 'RUNTIME_F16_GAUSSIAN_Q2K_COV', 'test-revision',
               gaussian_encoded=covariance_codes, denoiser_encoded=bad_denoiser_codes)
    except ValueError as exc:
        assert 'encoded denoiser matrix' in str(exc)
    else:
        raise AssertionError('malformed denoiser archive accepted')
    assert not bad_combined.exists() and not bad_combined.with_name(bad_combined.name + '.partial').exists()
    mixed_denoiser_source = root / 'mixed-denoiser-source'
    mixed_denoiser_source.mkdir()
    shutil.copyfile(root / 'config.json', mixed_denoiser_source / 'config.json')
    for filename in COMPONENTS.values():
        shutil.copyfile(root / filename, mixed_denoiser_source / filename)
    save_file({'blocks.0.linear.weight': source, 'blocks.1.linear.weight': source * 0.7,
               'noise_refiner.linear.weight': source}, str(mixed_denoiser_source / COMPONENTS['denoiser']))
    mixed_denoiser_moments = root / 'mixed-denoiser-moments.pt'
    torch.save({'blocks.0.linear.weight': torch.diag(diagonal), 'blocks.1.linear.weight': torch.diag(diagonal)},
               mixed_denoiser_moments)
    mixed_denoiser_report = root / 'mixed-denoiser-moments.json'
    mixed_denoiser_report.write_text(json.dumps({'hessians_sha256': digest(mixed_denoiser_moments),
                                                 'inputs_sha256': {'source_sha256': digest(mixed_denoiser_source / COMPONENTS['denoiser']),
                                                                   'model_sha256': 'a' * 64},
                                                 'variants': {'plain': {}, 'damp_0.3': {}}}))
    all_denoiser_codes = root / 'all-denoiser-codes.npz'
    create_denoiser_covariance(mixed_denoiser_source / COMPONENTS['denoiser'], mixed_denoiser_moments,
                               mixed_denoiser_report, library, all_denoiser_codes)
    denoiser_ranking = root / 'denoiser-ranking.json'
    denoiser_ranking.write_text(json.dumps({'source_sha256': digest(mixed_denoiser_source / COMPONENTS['denoiser']),
                                            'model_sha256': 'a' * 64, 'blocks': {'0': {'case': 0.1}, '1': {'case': 0.2}}}))
    mixed_denoiser_codes = root / 'mixed-denoiser-codes.npz'
    mixed_denoiser_result = create_mixed_denoiser(mixed_denoiser_source / COMPONENTS['denoiser'], all_denoiser_codes,
                                                   denoiser_ranking, mixed_denoiser_codes, blocks=1)
    assert mixed_denoiser_result['preserved_blocks'] == [1] and mixed_denoiser_result['encoded_matrix_count'] == 1
    incompatible_q8 = root / 'incompatible-mixed-all-q8.gguf'
    try:
        export(mixed_denoiser_source, incompatible_q8, 'Q8_0', 'test-revision',
               denoiser_encoded=mixed_denoiser_codes)
    except ValueError as exc:
        assert 'complete Q8_0 denoiser archive' in str(exc)
    else:
        raise AssertionError('mixed-preservation archive accepted for Q8_0 base')
    assert not incompatible_q8.exists()
    wrong_encoded_ranking = root / 'wrong-encoded-ranking.json'
    ranking_data = json.loads(denoiser_ranking.read_text())
    ranking_data.update({'quantization_source': 'provided_encoded_archive', 'inputs_sha256': {'encoded': '0' * 64}})
    wrong_encoded_ranking.write_text(json.dumps(ranking_data))
    bad_mixed_denoiser = root / 'bad-mixed-denoiser.npz'
    try:
        create_mixed_denoiser(mixed_denoiser_source / COMPONENTS['denoiser'], all_denoiser_codes,
                               wrong_encoded_ranking, bad_mixed_denoiser, blocks=1)
    except ValueError as exc:
        assert 'ranking does not bind' in str(exc)
    else:
        raise AssertionError('ranking from a different encoded archive accepted')
    assert not bad_mixed_denoiser.exists()
    mixed_denoiser_target = root / 'mixed-denoiser.gguf'
    exported = export(mixed_denoiser_source, mixed_denoiser_target, 'RUNTIME_F16_GAUSSIAN_Q2K_COV', 'test-revision',
                      gaussian_encoded=covariance_codes, denoiser_encoded=mixed_denoiser_codes)
    assert exported['tier'] == 'RUNTIME_F16_GAUSSIAN_Q2K_COV_DENOISER_Q80F16_COV'
    model = IdolGGUF(mixed_denoiser_target)
    assert model.tensors['denoiser']['blocks.0.linear.weight'].tensor_type == gguf.GGMLQuantizationType.Q8_0
    assert model.tensors['denoiser']['blocks.1.linear.weight'].tensor_type == gguf.GGMLQuantizationType.F16
    assert model.reader.get_field('idol.quantization.denoiser_preserved_blocks').contents() == '[1]'
    module = torch.nn.Module()
    module.blocks = torch.nn.ModuleList([torch.nn.Module(), torch.nn.Module()])
    for item in module.blocks:
        item.linear = torch.nn.Linear(256, 2).half()
    module.noise_refiner = torch.nn.Module()
    module.noise_refiner.linear = torch.nn.Linear(256, 2).half()
    validate_runtime_half_storage(model, module)
    module.blocks[1].float()
    try:
        validate_runtime_half_storage(model, module)
    except ValueError:
        pass
    else:
        raise AssertionError('incompatible preserved denoiser runtime accepted')
    del model
    mixed_source = root / 'mixed-source'
    mixed_source.mkdir()
    shutil.copyfile(root / 'config.json', mixed_source / 'config.json')
    for filename in COMPONENTS.values():
        shutil.copyfile(root / filename, mixed_source / filename)
    save_file({'blocks.0.linear.weight': source, 'blocks.1.linear.weight': source * 0.7},
              str(mixed_source / COMPONENTS['gaussian']))
    mixed_moments = root / 'mixed-moments.pt'
    torch.save({'gs.blocks.0.linear.weight': torch.diag(diagonal),
                'gs.blocks.1.linear.weight': torch.diag(diagonal)}, mixed_moments)
    mixed_moments_report = root / 'mixed-moments.json'
    mixed_moments_report.write_text(json.dumps({'status': 'complete', 'hashes_unchanged_after_run': True,
                                                'hessians_sha256': digest(mixed_moments),
                                                'inputs': {'gaussian_weights': digest(mixed_source / COMPONENTS['gaussian'])}}))
    mixed_q2, mixed_q4 = root / 'mixed-q2.npz', root / 'mixed-q4.npz'
    create_covariance(mixed_source / COMPONENTS['gaussian'], mixed_moments, mixed_moments_report, library, mixed_q2, 'Q2_K')
    create_covariance(mixed_source / COMPONENTS['gaussian'], mixed_moments, mixed_moments_report, library, mixed_q4, 'Q4_K')
    raw_latent, npy_latent = root / 'same.f32', root / 'same.npy'
    latent_values = np.arange(16, dtype=np.float32)
    latent_values.tofile(raw_latent)
    np.save(npy_latent, latent_values)
    mixed_args = SimpleNamespace(weights=mixed_source, sdk=sdk, q2_codes=mixed_q2, q4_codes=mixed_q4,
                                 train_latent=[raw_latent], eval_latent=[npy_latent], out=root / 'mixed-selection-bad.json',
                                 anchors=1, threads=1, max_groups=1, seed=42, min_available_gb=0, device='cpu')
    try:
        select_mixed_gaussian(mixed_args)
    except ValueError as exc:
        assert 'byte-disjoint' in str(exc)
    else:
        raise AssertionError('same decoded latent accepted in two file formats')
    assert not mixed_args.out.exists()
    mixed_selection = root / 'mixed-selection.json'
    mixed_selection.write_text(json.dumps({'status': 'complete', 'hashes_unchanged_after_run': True,
                                            'inputs_sha256': {'gaussian': digest(mixed_source / COMPONENTS['gaussian']),
                                                              'q2_codes': digest(mixed_q2), 'q4_codes': digest(mixed_q4)},
                                            'selected': [{'group': 'blocks.1'}]}))
    mixed_codes = root / 'mixed-codes.npz'
    mixed_result = create_mixed_gaussian(mixed_source / COMPONENTS['gaussian'], mixed_q2, mixed_q4,
                                         mixed_selection, mixed_codes, groups=1)
    assert mixed_result['q4_matrix_count'] == 1
    mixed_target = root / 'mixed-gaussian.gguf'
    export(mixed_source, mixed_target, 'RUNTIME_F16_GAUSSIAN_Q2Q4_COV', 'test-revision', gaussian_encoded=mixed_codes)
    model = IdolGGUF(mixed_target)
    assert model.tensors['gaussian']['blocks.0.linear.weight'].tensor_type == gguf.GGMLQuantizationType.Q2_K
    assert model.tensors['gaussian']['blocks.1.linear.weight'].tensor_type == gguf.GGMLQuantizationType.Q4_K
    assert model.reader.get_field('idol.quantization.encoded_selection_sha256').contents() == digest(mixed_selection)
    with np.load(mixed_codes, allow_pickle=False) as archive:
        mixed_values = {name: archive[name].copy() for name in archive.files}
    for name in ('blocks.0.linear.weight', 'blocks.1.linear.weight'):
        np.testing.assert_array_equal(model.tensors['gaussian'][name].data, mixed_values[name])
    del model
    metadata = json.loads(mixed_values['__metadata__'].tobytes())
    metadata['q4_names'] = []
    mixed_values['__metadata__'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8)
    bad_mixed_codes = root / 'bad-mixed-codes.npz'
    with bad_mixed_codes.open('xb') as stream:
        np.savez(stream, **mixed_values)
    bad_mixed = root / 'bad-mixed.gguf'
    try:
        export(mixed_source, bad_mixed, 'RUNTIME_F16_GAUSSIAN_Q2Q4_COV', 'test-revision', gaussian_encoded=bad_mixed_codes)
    except ValueError as exc:
        assert 'group allocation' in str(exc)
    else:
        raise AssertionError('inconsistent mixed Gaussian allocation accepted')
    assert not bad_mixed.exists() and not bad_mixed.with_name(bad_mixed.name + '.partial').exists()
    duplicate_plan = root / 'duplicate.npz'
    shutil.copyfile(plan, duplicate_plan)
    with zipfile.ZipFile(duplicate_plan, 'a') as archive:
        metadata_entry = archive.read('__metadata__.npy')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            archive.writestr('__metadata__.npy', metadata_entry)
    duplicate_target = root / 'duplicate.gguf'
    try:
        export(root, duplicate_target, 'RUNTIME_F16_GAUSSIAN_Q6K', 'test-revision', library, duplicate_plan)
    except ValueError as exc:
        assert 'duplicate' in str(exc)
    else:
        raise AssertionError('duplicate calibration entries accepted')
    assert not duplicate_target.exists()
    unbound_selection = root / 'unbound-selection.json'
    selection_data = json.loads(selection.read_text())
    selection_data['inputs_sha256']['moments'] = '0' * 64
    unbound_selection.write_text(json.dumps(selection_data))
    unbound_plan = root / 'unbound-selection.npz'
    try:
        create(root / COMPONENTS['gaussian'], moments, report, unbound_selection, 'Q6_K', unbound_plan)
    except ValueError as exc:
        assert 'selection report does not bind' in str(exc)
    else:
        raise AssertionError('unbound selection accepted')
    assert not unbound_plan.exists()
    report.write_text(json.dumps({'status': 'complete', 'hashes_unchanged_after_run': True,
                                  'hessians_sha256': '0' * 64,
                                  'inputs': {'gaussian_weights': digest(root / COMPONENTS['gaussian'])}}))
    bad_plan = root / 'wrong-moments.npz'
    try:
        create(root / COMPONENTS['gaussian'], moments, report, selection, 'Q4_K', bad_plan)
    except ValueError as exc:
        assert 'does not bind' in str(exc)
    else:
        raise AssertionError('unbound calibration moments accepted')
    assert not bad_plan.exists()

print('calibrated GGUF provenance and round trips passed')
