"""Export the four original Idol components to a single GGUF model file.

The image encoders referenced by Idol's configuration are separate licensed models and are not in the upstream Idol
weight repository. This file includes the denoiser, octree decoder, Gaussian decoder and background remover.
"""
import argparse
import ctypes
import hashlib
import json
import os
import tempfile
from pathlib import Path

import gguf
import numpy as np
from safetensors import safe_open

COMPONENTS = {"denoiser": "denoiser.safetensors", "octree": "decoder.safetensors",
              "gaussian": "decoder_gs.safetensors", "background": "birefnet.safetensors"}
SOURCE_REPO = "speridlabs/idol"
K_TIERS = ("Q2_K", "Q4_K", "Q6_K")
CALIBRATED_GAUSSIAN_TIERS = {'RUNTIME_F16_GAUSSIAN_Q2K': 'Q2_K',
                             'RUNTIME_F16_GAUSSIAN_Q4K': 'Q4_K',
                             'RUNTIME_F16_GAUSSIAN_Q6K': 'Q6_K'}
ENCODED_GAUSSIAN_TIERS = {'RUNTIME_F16_GAUSSIAN_Q2K_COV': 'Q2_K',
                          'RUNTIME_F16_GAUSSIAN_Q4K_COV': 'Q4_K',
                          'RUNTIME_F16_GAUSSIAN_Q6K_COV': 'Q6_K',
                          'RUNTIME_F16_GAUSSIAN_Q2Q4_COV': 'Q2_K+Q4_K'}


class GGMLQuantizer:
    """Use an explicitly supplied GGML library for formats gguf-py cannot encode."""
    def __init__(self, path):
        path = Path(path).resolve(strict=True)
        self.dll_directory = os.add_dll_directory(str(path.parent)) if os.name == 'nt' else None
        self.lib = ctypes.CDLL(str(path))
        self.lib.ggml_blck_size.argtypes = [ctypes.c_int]
        self.lib.ggml_blck_size.restype = ctypes.c_int64
        self.lib.ggml_type_size.argtypes = [ctypes.c_int]
        self.lib.ggml_type_size.restype = ctypes.c_size_t
        self.lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                                  ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t

    def quantize(self, value, quant_type, importance=None):
        block, size = gguf.GGML_QUANT_SIZES[quant_type]
        if (self.lib.ggml_blck_size(int(quant_type)) != block or
                self.lib.ggml_type_size(int(quant_type)) != size):
            raise ValueError('GGML library format mismatch')
        source = np.ascontiguousarray(value, dtype=np.float32)
        if source.ndim != 2 or source.shape[1] % block or not np.isfinite(source).all():
            raise ValueError('invalid matrix shape or values for GGML quantization')
        if importance is None:
            if self.lib.ggml_quantize_requires_imatrix(int(quant_type)):
                raise ValueError('GGML quantization requires an importance vector')
            importance_pointer = None
        else:
            importance = np.ascontiguousarray(importance, dtype=np.float32)
            if (importance.shape != (source.shape[1],) or not np.isfinite(importance).all()
                    or np.any(importance <= 0)):
                raise ValueError('invalid GGML importance vector')
            importance_pointer = importance.ctypes.data
        result = np.empty((source.shape[0], source.shape[1] // block * size), dtype=np.uint8)
        written = self.lib.ggml_quantize_chunk(int(quant_type), source.ctypes.data, result.ctypes.data, 0,
                                                source.shape[0], source.shape[1], importance_pointer)
        if written != result.nbytes:
            raise ValueError('GGML quantization wrote an unexpected number of bytes')
        return result


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _gaussian_group(name):
    parts = name.split('.')
    return '.'.join(parts[:2]) if parts[0] == 'blocks' else name


def load_gaussian_calibration(path, source, source_sha256, quant_type):
    """Validate a plan against all eligible source tensors before writing output."""
    with safe_open(str(source), framework='np') as weights:
        shapes = {name: tuple(weights.get_slice(name).get_shape()) for name in weights.keys()}
    eligible = sorted(name for name, shape in shapes.items()
                      if name.endswith('weight') and len(shape) == 2 and shape[-1] % 256 == 0)
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError('duplicate Gaussian calibration entries')
        if '__metadata__' not in archive.files:
            raise ValueError('Gaussian calibration has no metadata')
        raw = archive['__metadata__']
        if raw.dtype != np.uint8 or raw.ndim != 1 or raw.size > 1_000_000:
            raise ValueError('invalid Gaussian calibration metadata')
        metadata = json.loads(raw.tobytes().decode('utf-8'))
        if not isinstance(metadata, dict):
            raise ValueError('invalid Gaussian calibration metadata')
        groups = metadata.get('groups')
        selection_sha256 = metadata.get('selection_sha256')
        expected_groups = {_gaussian_group(name) for name in eligible}
        if (not eligible or metadata.get('version') != 1 or metadata.get('quant_type') != quant_type
                or metadata.get('gaussian_source_sha256') != source_sha256
                or not isinstance(selection_sha256, str) or len(selection_sha256) != 64
                or any(c not in '0123456789abcdef' for c in selection_sha256)
                or metadata.get('eligible') != eligible or not isinstance(groups, dict)
                or set(groups) != expected_groups or any(type(value) is not bool for value in groups.values())):
            raise ValueError('Gaussian calibration does not match the source model or quantization type')
        vector_names = {name for name in eligible if groups[_gaussian_group(name)]}
        if set(archive.files) != vector_names | {'__metadata__'}:
            raise ValueError('Gaussian calibration vectors do not match the selection')
        vectors = {}
        for name in vector_names:
            value = archive[name]
            if (value.dtype != np.float32 or value.shape != (shapes[name][-1],)
                    or not np.isfinite(value).all() or np.any(value <= 0)):
                raise ValueError('invalid Gaussian calibration vector: ' + name)
            vectors[name] = np.ascontiguousarray(value)
    return metadata, vectors


def load_gaussian_codes(path, source, source_sha256, quant_type):
    """Validate all pre-encoded matrices before creating an output file."""
    with safe_open(str(source), framework='np') as weights:
        shapes = {name: tuple(weights.get_slice(name).get_shape()) for name in weights.keys()}
    eligible = sorted(name for name, shape in shapes.items()
                      if name.endswith('weight') and len(shape) == 2 and shape[-1] % 256 == 0)
    if not eligible:
        raise ValueError('no eligible Gaussian matrices for encoded archive')
    kind = getattr(gguf.GGMLQuantizationType, quant_type)
    block, size = gguf.GGML_QUANT_SIZES[kind]
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(eligible) | {'__metadata__'}:
            raise ValueError('encoded Gaussian tensor names are missing, duplicated or unexpected')
        raw = archive['__metadata__']
        if raw.dtype != np.uint8 or raw.ndim != 1 or raw.size > 1_000_000:
            raise ValueError('invalid encoded Gaussian metadata')
        metadata = json.loads(raw.tobytes().decode('utf-8'))
        if (not isinstance(metadata, dict) or metadata.get('version') != 1
                or metadata.get('quant_type') != quant_type or metadata.get('source_sha256') != source_sha256
                or metadata.get('importance', 'none') not in ('none', 'covariance_diagonal')):
            raise ValueError('encoded Gaussian metadata does not match the source or quantization type')
        fields = ('moments_sha256', 'moments_report_sha256', 'ggml_sha256', 'tool_sha256')
        if 'quantizer_sha256' in metadata:
            fields += ('quantizer_sha256',)
        for field in fields:
            value = metadata.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid encoded Gaussian provenance: ' + field)
        damping = metadata.get('damping')
        if (type(damping) not in (int, float) or not np.isfinite(damping) or damping <= 0):
            raise ValueError('invalid encoded Gaussian damping')
        codes = {}
        for name in eligible:
            value = archive[name]
            shape = shapes[name]
            if (value.dtype != np.uint8 or value.shape != (shape[0], shape[1] // block * size)
                    or not np.isfinite(gguf.dequantize(value, kind)).all()):
                raise ValueError('invalid encoded Gaussian matrix: ' + name)
            codes[name] = np.ascontiguousarray(value)
    return metadata, codes


def load_denoiser_codes(path, source, source_sha256):
    """Validate a complete archive of encoded denoiser block matrices."""
    with safe_open(str(source), framework='np') as weights:
        shapes = {name: tuple(weights.get_slice(name).get_shape()) for name in weights.keys()}
    eligible = sorted(name for name, shape in shapes.items() if name.startswith('blocks.') and name.endswith('weight')
                      and len(shape) == 2 and all(dimension > 0 for dimension in shape) and shape[-1] % 256 == 0)
    if not eligible:
        raise ValueError('no eligible denoiser matrices for encoded archive')
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or '__metadata__' not in archive.files:
            raise ValueError('encoded denoiser tensor names are missing, duplicated or unexpected')
        raw = archive['__metadata__']
        if raw.dtype != np.uint8 or raw.ndim != 1 or raw.size > 1_000_000:
            raise ValueError('invalid encoded denoiser metadata')
        metadata = json.loads(raw.tobytes().decode('utf-8'))
        if (not isinstance(metadata, dict) or metadata.get('version') != 1 or metadata.get('component') != 'denoiser'
                or metadata.get('quant_type') not in ('Q2_K', 'Q4_K', 'Q6_K', 'Q8_0',
                                                       'Q2_K+F16', 'Q4_K+F16', 'Q6_K+F16', 'Q8_0+F16')
                or metadata.get('source_sha256') != source_sha256
                or metadata.get('importance', 'none') not in ('none', 'covariance_diagonal')):
            raise ValueError('encoded denoiser metadata does not match the source or quantization type')
        mixed = metadata['quant_type'].endswith('+F16')
        fields = ('moments_sha256', 'moments_report_sha256', 'ggml_sha256', 'tool_sha256',
                  'helper_sha256', 'reference_model_sha256')
        if 'quantizer_sha256' in metadata:
            fields += ('quantizer_sha256',)
        if mixed:
            fields += ('parent_codes_sha256', 'ranking_sha256', 'mix_tool_sha256')
        for field in fields:
            value = metadata.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid encoded denoiser provenance: ' + field)
        damping = metadata.get('damping')
        if type(damping) not in (int, float) or not np.isfinite(damping) or damping <= 0:
            raise ValueError('invalid encoded denoiser damping')
        if mixed:
            if any(len(name.split('.')) < 3 or not name.split('.')[1].isdigit() for name in eligible):
                raise ValueError('invalid denoiser block tensor names')
            available = {int(name.split('.')[1]) for name in eligible}
            preserved = metadata.get('preserved_blocks')
            if (not isinstance(preserved, list) or any(type(index) is not int for index in preserved)
                    or preserved != sorted(set(preserved)) or not set(preserved) <= available
                    or len(preserved) >= len(available)):
                raise ValueError('invalid preserved denoiser block allocation')
            encoded_names = [name for name in eligible if int(name.split('.')[1]) not in preserved]
        else:
            encoded_names = eligible
        if set(archive.files) != set(encoded_names) | {'__metadata__'}:
            raise ValueError('encoded denoiser tensor names are missing, duplicated or unexpected')
        kind = getattr(gguf.GGMLQuantizationType, metadata['quant_type'].removesuffix('+F16'))
        block, size = gguf.GGML_QUANT_SIZES[kind]
        codes = {}
        for name in encoded_names:
            value = archive[name]
            shape = shapes[name]
            if (value.dtype != np.uint8 or value.shape != (shape[0], shape[1] // block * size)
                    or not np.isfinite(gguf.dequantize(value, kind)).all()):
                raise ValueError('invalid encoded denoiser matrix: ' + name)
            codes[name] = np.ascontiguousarray(value)
    return metadata, codes


def load_mixed_gaussian_codes(path, source, source_sha256):
    """Validate per-matrix Q2/Q4 codes and the recorded group allocation."""
    with safe_open(str(source), framework='np') as weights:
        shapes = {name: tuple(weights.get_slice(name).get_shape()) for name in weights.keys()}
    eligible = sorted(name for name, shape in shapes.items()
                      if name.endswith('weight') and len(shape) == 2 and all(d > 0 for d in shape) and shape[-1] % 256 == 0)
    if not eligible:
        raise ValueError('no eligible Gaussian matrices for mixed archive')
    with np.load(path, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(eligible) | {'__metadata__'}:
            raise ValueError('mixed Gaussian tensor names are missing, duplicated or unexpected')
        raw = archive['__metadata__']
        if raw.dtype != np.uint8 or raw.ndim != 1 or raw.size > 1_000_000:
            raise ValueError('invalid mixed Gaussian metadata')
        metadata = json.loads(raw.tobytes().decode('utf-8'))
        if (not isinstance(metadata, dict) or metadata.get('version') != 1
                or metadata.get('quant_type') != 'Q2_K+Q4_K' or metadata.get('source_sha256') != source_sha256
                or metadata.get('q2_importance', 'none') not in ('none', 'covariance_diagonal')
                or metadata.get('q4_importance', 'none') not in ('none', 'covariance_diagonal')):
            raise ValueError('mixed Gaussian metadata does not match the source or quantization types')
        for field in ('q2_codes_sha256', 'q4_codes_sha256', 'selection_sha256', 'tool_sha256',
                      'q2_moments_sha256', 'q4_moments_sha256'):
            value = metadata.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('invalid mixed Gaussian provenance: ' + field)
        groups, q4_names = metadata.get('q4_groups'), metadata.get('q4_names')
        valid_groups = {_gaussian_group(name) for name in eligible}
        if (not isinstance(groups, list) or any(not isinstance(group, str) for group in groups)
                or len(set(groups)) != len(groups) or not set(groups) <= valid_groups
                or not isinstance(q4_names, list)
                or q4_names != sorted(name for name in eligible if _gaussian_group(name) in groups)):
            raise ValueError('invalid mixed Gaussian group allocation')
        codes, kinds = {}, {}
        for name in eligible:
            kind = gguf.GGMLQuantizationType.Q4_K if name in q4_names else gguf.GGMLQuantizationType.Q2_K
            block, size = gguf.GGML_QUANT_SIZES[kind]
            value, shape = archive[name], shapes[name]
            if (value.dtype != np.uint8 or value.shape != (shape[0], shape[1] // block * size)
                    or not np.isfinite(gguf.dequantize(value, kind)).all()):
                raise ValueError('invalid mixed Gaussian matrix: ' + name)
            codes[name], kinds[name] = np.ascontiguousarray(value), kind
    return metadata, codes, kinds


def encoding(component, name, value, tier, quantizer=None, gaussian_vectors=None, gaussian_codes=None,
             denoiser_codes=None, denoiser_kind=None, gaussian_kinds=None, octree_kind=None):
    """Optionally quantize octree matrices; preserve small tensors and convolution kernels."""
    if value.dtype.kind not in "fiub":
        raise ValueError(f"unsupported tensor dtype: {component}.{name}: {value.dtype}")
    if value.dtype.kind in "ub":
        if value.size and int(value.max()) > np.iinfo(np.int64).max:
            raise ValueError(f"unsigned integer outside GGUF signed range: {component}.{name}")
        return value.astype(np.int64), None
    if value.dtype.kind == "i":
        return value, None
    if not np.isfinite(value).all():
        raise ValueError(f"non-finite source tensor: {component}.{name}")
    if (component == 'octree' and octree_kind is not None and value.ndim == 2
            and value.shape[-1] % 32 == 0 and name.endswith('weight')):
        return gguf.quantize(value.astype(np.float32, copy=False), octree_kind), octree_kind
    if component == 'denoiser' and denoiser_codes is not None and name in denoiser_codes:
        return denoiser_codes[name], denoiser_kind
    if tier in ENCODED_GAUSSIAN_TIERS:
        if component == 'gaussian' and name in gaussian_codes:
            kind = gaussian_kinds[name] if gaussian_kinds is not None else getattr(gguf.GGMLQuantizationType, ENCODED_GAUSSIAN_TIERS[tier])
            return gaussian_codes[name], kind
        tier = 'RUNTIME_F16_Q8'
    if tier in CALIBRATED_GAUSSIAN_TIERS:
        if (component == 'gaussian' and name.endswith('weight') and value.ndim == 2
                and value.shape[-1] % 256 == 0):
            kind = getattr(gguf.GGMLQuantizationType, CALIBRATED_GAUSSIAN_TIERS[tier])
            return quantizer.quantize(value, kind, gaussian_vectors.get(name)), kind
        tier = 'RUNTIME_F16_Q8'
    if tier == 'RUNTIME_F16_Q8' and component in ('denoiser', 'background'):
        selected = name.startswith(('blocks.', 'noise_refiner.', 'context_refiner.'))
        if component == 'denoiser' and selected and value.ndim == 2 and value.shape[-1] % 32 == 0 and name.endswith('weight'):
            result = value.astype(np.float16)
            if not np.isfinite(result).all():
                raise ValueError(f"float16 overflow: {component}.{name}")
            return result, None
        return value, None
    if tier == 'Q8_MIXED' and component in ('denoiser', 'background'):
        parts = name.split('.')
        selected = (component == 'denoiser' and len(parts) >= 3 and parts[0] == 'blocks'
                    and parts[1].isdigit() and 12 <= int(parts[1]) < 24)
        if selected and value.ndim == 2 and value.shape[-1] % 32 == 0 and name.endswith('weight'):
            kind = gguf.GGMLQuantizationType.Q8_0
            return gguf.quantize(value.astype(np.float32, copy=False), kind), kind
        return value, None
    # Preserve the denoiser and background remover for the high-fidelity tiers.
    # The source background remover uses float16; retaining its original type
    # avoids both a preprocessing change and needless expansion to float32.
    if tier in ("Q8_SAFE", "Q4_SAFE") and component in ("denoiser", "background"):
        return value, None
    if component != "octree" and value.ndim == 2 and value.shape[-1] % 32 == 0 and name.endswith("weight"):
        if tier in K_TIERS:
            if component == 'denoiser' and value.shape[-1] % 256 == 0:
                quant_type = getattr(gguf.GGMLQuantizationType, tier)
                return quantizer.quantize(value, quant_type), quant_type
            if component != 'denoiser':
                quant_type = gguf.GGMLQuantizationType.Q8_0 if tier == 'Q6_K' else gguf.GGMLQuantizationType.Q4_0
                return gguf.quantize(value.astype(np.float32, copy=False), quant_type), quant_type
        if tier in ("Q8_0", "Q4_0", "Q8_SAFE", "Q4_SAFE", "Q8_MIXED", "RUNTIME_F16_Q8"):
            quant_name = {"Q8_SAFE": "Q8_0", "Q4_SAFE": "Q4_0", "Q8_MIXED": "Q8_0", "RUNTIME_F16_Q8": "Q8_0"}.get(tier, tier)
            quant_type = getattr(gguf.GGMLQuantizationType, quant_name)
            return gguf.quantize(value.astype(np.float32, copy=False), quant_type), quant_type
    if tier != "F32" and component == "background" and value.ndim >= 2:
        result = value.astype(np.float16)
        if not np.isfinite(result).all():
            raise ValueError(f"float16 overflow: {component}.{name}")
        return result, None
    return value.astype(np.float32, copy=False), None


def export(source, output, tier, revision, ggml_lib=None, gaussian_calibration=None, gaussian_encoded=None,
           denoiser_encoded=None, octree_quant=None):
    source, output = Path(source), Path(output)
    if tier not in ("F32", "Q8_0", "Q4_0", "Q8_SAFE", "Q4_SAFE", "Q8_MIXED", "RUNTIME_F16_Q8", *K_TIERS, *CALIBRATED_GAUSSIAN_TIERS, *ENCODED_GAUSSIAN_TIERS) or not revision:
        raise ValueError("unsupported tier or missing source revision")
    if octree_quant not in (None, 'Q4_0', 'Q8_0') or (octree_quant and tier == 'F32'):
        raise ValueError('unsupported octree quantization or incompatible F32 tier')
    octree_kind = getattr(gguf.GGMLQuantizationType, octree_quant) if octree_quant else None
    if tier in (*K_TIERS, *CALIBRATED_GAUSSIAN_TIERS) and ggml_lib is None:
        raise ValueError('K-block tiers require an explicitly supplied GGML library')
    if (tier in CALIBRATED_GAUSSIAN_TIERS) != (gaussian_calibration is not None):
        raise ValueError('a Gaussian calibration plan is required only for calibrated Gaussian tiers')
    if (tier in ENCODED_GAUSSIAN_TIERS) != (gaussian_encoded is not None):
        raise ValueError('encoded Gaussian matrices are required only for covariance-encoded tiers')
    if denoiser_encoded is not None and tier not in ('Q8_0', 'RUNTIME_F16_Q8', *CALIBRATED_GAUSSIAN_TIERS, *ENCODED_GAUSSIAN_TIERS):
        raise ValueError('encoded denoiser matrices require a compatible base tier')
    quantizer = GGMLQuantizer(ggml_lib) if tier in (*K_TIERS, *CALIBRATED_GAUSSIAN_TIERS) else None
    files = {"config.json": source / "config.json", **{name: source / file for name, file in COMPONENTS.items()}}
    initial = {name: digest(path) for name, path in files.items()}
    calibration_metadata = gaussian_vectors = encoded_metadata = gaussian_codes = gaussian_kinds = None
    denoiser_metadata = denoiser_codes = denoiser_kind = None
    if gaussian_calibration is not None:
        gaussian_calibration = Path(gaussian_calibration)
        initial['gaussian_calibration'] = digest(gaussian_calibration)
        calibration_metadata, gaussian_vectors = load_gaussian_calibration(
            gaussian_calibration, files['gaussian'], initial['gaussian'], CALIBRATED_GAUSSIAN_TIERS[tier])
    if gaussian_encoded is not None:
        gaussian_encoded = Path(gaussian_encoded)
        initial['gaussian_encoded'] = digest(gaussian_encoded)
        if ENCODED_GAUSSIAN_TIERS[tier] == 'Q2_K+Q4_K':
            encoded_metadata, gaussian_codes, gaussian_kinds = load_mixed_gaussian_codes(
                gaussian_encoded, files['gaussian'], initial['gaussian'])
        else:
            encoded_metadata, gaussian_codes = load_gaussian_codes(
                gaussian_encoded, files['gaussian'], initial['gaussian'], ENCODED_GAUSSIAN_TIERS[tier])
    if denoiser_encoded is not None:
        denoiser_encoded = Path(denoiser_encoded)
        initial['denoiser_encoded'] = digest(denoiser_encoded)
        denoiser_metadata, denoiser_codes = load_denoiser_codes(denoiser_encoded, files['denoiser'], initial['denoiser'])
        if tier == 'Q8_0' and denoiser_metadata['quant_type'] != 'Q8_0':
            raise ValueError('Q8_0 base tier requires a complete Q8_0 denoiser archive')
        denoiser_kind = getattr(gguf.GGMLQuantizationType, denoiser_metadata['quant_type'].removesuffix('+F16'))
    config = json.loads(files["config.json"].read_text(encoding="utf-8"))
    if not all(name in config for name in ("denoiser", "decoder", "decoder_gs", "image_encoders")):
        raise ValueError("not a complete Idol configuration")
    if output.exists():
        raise FileExistsError(output)
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    # Claim the name atomically: do not replace another run's work.
    with partial.open("xb"):
        pass
    old_tempdir = tempfile.tempdir
    writer = None
    try:
        tempfile.tempdir = str(output.parent.resolve())
        writer = gguf.GGUFWriter(partial, arch="idol", use_temp_file=True)
        writer.add_name("Idol")
        writer.add_description("Idol denoiser, octree and Gaussian decoders, and background remover")
        writer.add_string("idol.source.repository", SOURCE_REPO)
        writer.add_string("idol.source.revision", revision)
        writer.add_string("idol.config.json", json.dumps(config, sort_keys=True, separators=(",", ":")))
        stored_tier = tier if denoiser_metadata is None else tier + '_DENOISER_' + denoiser_metadata['quant_type'].replace('_', '').replace('+', '') + '_COV'
        writer.add_string("idol.quantization.tier", stored_tier)
        if calibration_metadata is not None:
            writer.add_string('idol.quantization.calibration_sha256', initial['gaussian_calibration'])
            writer.add_string('idol.quantization.calibration_selection_sha256', calibration_metadata['selection_sha256'])
        if encoded_metadata is not None:
            writer.add_string('idol.quantization.encoded_sha256', initial['gaussian_encoded'])
            if encoded_metadata['quant_type'] == 'Q2_K+Q4_K':
                writer.add_string('idol.quantization.encoded_selection_sha256', encoded_metadata['selection_sha256'])
            else:
                writer.add_string('idol.quantization.encoded_moments_sha256', encoded_metadata['moments_sha256'])
        if denoiser_metadata is not None:
            writer.add_string('idol.quantization.denoiser_encoded_sha256', initial['denoiser_encoded'])
            writer.add_string('idol.quantization.denoiser_moments_sha256', denoiser_metadata['moments_sha256'])
            if denoiser_metadata['quant_type'].endswith('+F16'):
                writer.add_string('idol.quantization.denoiser_preserved_blocks', json.dumps(denoiser_metadata['preserved_blocks']))
        policy = (f"eligible denoiser block and refiner matrices float16 as used by the pinned SDK; eligible Gaussian matrices {ENCODED_GAUSSIAN_TIERS[tier]} from covariance-compensated codes; other denoiser and background source precision; octree float32"
                  if tier in ENCODED_GAUSSIAN_TIERS else
                  f"eligible denoiser block and refiner matrices float16 as used by the pinned SDK; eligible Gaussian matrices {CALIBRATED_GAUSSIAN_TIERS[tier]} with optional per-channel importance; other denoiser and background source precision; octree float32"
                  if tier in CALIBRATED_GAUSSIAN_TIERS else
                  "eligible denoiser block and refiner matrices float16 as used by the pinned SDK; Gaussian matrices Q8_0; other denoiser and background source precision; octree float32"
                  if tier == 'RUNTIME_F16_Q8' else
                  "denoiser blocks 12-23 and Gaussian eligible matrices Q8_0; other denoiser and background source precision; octree float32"
                  if tier == 'Q8_MIXED' else
                  "eligible denoiser matrices use selected K-block type; other eligible matrices use Q4_0 or Q8_0; octree float32; remaining tensors retain float precision"
                  if tier in K_TIERS else
                  "denoiser and background retain source precision; octree float32; eligible Gaussian matrices quantized"
                  if tier in ("Q8_SAFE", "Q4_SAFE") else
                  "all floating-point values stored as float32" if tier == "F32" else
                  "octree float32; eligible matrices quantized; background kernels float16")
        if denoiser_metadata is not None:
            description = (f"eligible denoiser block matrices use {denoiser_metadata['quant_type']} with the listed blocks at original runtime precision; eligible refiner matrices float16 as used by the pinned SDK"
                           if denoiser_metadata['quant_type'].endswith('+F16') else
                           f"eligible denoiser block matrices {denoiser_metadata['quant_type']} from covariance-compensated codes; eligible refiner matrices float16 as used by the pinned SDK")
            policy = policy.replace('eligible denoiser block and refiner matrices float16 as used by the pinned SDK',
                                    description)
            if tier == 'Q8_0':
                policy += '; eligible denoiser block matrices use covariance-compensated Q8_0 codes'
        if octree_kind is not None:
            policy = policy.replace('octree float32', f'eligible octree matrices {octree_quant}; other octree tensors float32')
            writer.add_string('idol.quantization.octree', octree_quant)
        writer.add_string("idol.quantization.policy", policy)
        writer.add_string("idol.source.sha256.json", json.dumps(initial, sort_keys=True))
        writer.add_uint32("general.quantization_version", gguf.GGML_QUANT_VERSION)
        counts = {name: 0 for name in COMPONENTS}
        tensor_types = {}
        octree_quantized_count = 0
        for component, path in ((name, files[name]) for name in COMPONENTS):
            with safe_open(str(path), framework="np") as tensors:
                for name in sorted(tensors.keys()):
                    original = np.asarray(tensors.get_tensor(name))
                    encoded, raw_dtype = encoding(component, name, original, tier, quantizer, gaussian_vectors,
                                                  gaussian_codes, denoiser_codes, denoiser_kind, gaussian_kinds,
                                                  octree_kind)
                    writer.add_tensor(f"idol.{component}.{name}", encoded, raw_dtype=raw_dtype)
                    if component == 'octree' and octree_kind is not None and raw_dtype == octree_kind:
                        octree_quantized_count += 1
                    label = raw_dtype.name if raw_dtype is not None else str(encoded.dtype)
                    tensor_types[label] = tensor_types.get(label, 0) + 1
                    counts[component] += 1
            print(f"exported {component}: {counts[component]} tensors", flush=True)
        if octree_kind is not None and not octree_quantized_count:
            raise ValueError('no eligible octree matrices were quantized')
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        writer = None
        final_inputs = {name: digest(path) for name, path in files.items()}
        if gaussian_calibration is not None:
            final_inputs['gaussian_calibration'] = digest(gaussian_calibration)
        if gaussian_encoded is not None:
            final_inputs['gaussian_encoded'] = digest(gaussian_encoded)
        if denoiser_encoded is not None:
            final_inputs['denoiser_encoded'] = digest(denoiser_encoded)
        if initial != final_inputs:
            raise RuntimeError("source files changed during export")
        # Reader validation catches incomplete headers and confirms all four namespaces were written.
        reader = gguf.GGUFReader(partial)
        names = [tensor.name for tensor in reader.tensors]
        if len(names) != sum(counts.values()) or len(set(names)) != len(names):
            raise RuntimeError("GGUF tensor names are missing or duplicated")
        for component, count in counts.items():
            if count < 1 or sum(name.startswith(f"idol.{component}.") for name in names) != count:
                raise RuntimeError("GGUF component is missing: " + component)
        del reader
        # Publish without overwriting a concurrently created destination on any platform.
        os.link(partial, output)
        partial.unlink()
        return {"source_sha256": initial, "output_sha256": digest(output), "tier": stored_tier,
                "octree_quant": octree_quant, "octree_quantized_count": octree_quantized_count,
                "tensor_counts": counts, "stored_types": tensor_types,
                "source_revision": revision}
    except Exception:
        if writer is not None:
            writer.close()
        raise
    finally:
        tempfile.tempdir = old_tempdir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tier", choices=("F32", "Q8_0", "Q4_0", "Q8_SAFE", "Q4_SAFE", "Q8_MIXED", "RUNTIME_F16_Q8", *K_TIERS, *CALIBRATED_GAUSSIAN_TIERS, *ENCODED_GAUSSIAN_TIERS), required=True)
    parser.add_argument("--ggml-lib", type=Path, help="local GGML shared library required for K-block tiers")
    parser.add_argument('--gaussian-calibration', type=Path, help='plan produced by create_gaussian_calibration.py')
    parser.add_argument('--gaussian-encoded', type=Path, help='archive produced by calibrate_gaussian_covariance.py')
    parser.add_argument('--denoiser-encoded', type=Path, help='archive produced by calibrate_denoiser_covariance.py')
    parser.add_argument('--octree-quant', choices=('Q4_0', 'Q8_0'),
                        help='experimentally quantize eligible octree matrices; quality must be evaluated')
    parser.add_argument("--revision", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("report already exists")
    result = export(args.source, args.out, args.tier, args.revision, args.ggml_lib, args.gaussian_calibration,
                    args.gaussian_encoded, args.denoiser_encoded, args.octree_quant)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
