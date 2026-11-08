"""Native mesh extraction and bounded mesh comparisons for the local server."""
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_MESH_BYTES = 32 * 1024 * 1024
MAX_SPLAT_BYTES = 80 * 1024 * 1024


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def find_mesh_executable(explicit=None):
    if explicit:
        path = Path(explicit).resolve()
        if not path.is_file():
            raise ValueError('--mesh-executable must name an existing idol_mesh executable')
        return path
    found = shutil.which('idol_mesh')
    if found:
        return Path(found).resolve()
    root = Path(__file__).resolve().parents[1]
    for directory in ('', 'bin', 'build', 'build/Release'):
        for filename in ('idol_mesh', 'idol_mesh.exe'):
            path = root / directory / filename if directory else root / filename
            if path.is_file():
                return path
    return None


@dataclass(frozen=True)
class MeshOptions:
    resolution: int = 176
    sigma: float = 1.5
    iso: float = 0.2
    min_piece: float = 0.05
    lo: float = -0.55
    hi: float = 0.55

    def validate(self):
        if isinstance(self.resolution, bool) or int(self.resolution) != self.resolution or not 32 <= self.resolution <= 256:
            raise ValueError('Mesh resolution must be a whole number between 32 and 256')
        if any(not math.isfinite(x) for x in (self.sigma, self.iso, self.min_piece, self.lo, self.hi)):
            raise ValueError('Mesh settings must be finite')
        if not 0 < self.sigma <= 8 or not 0 < self.iso <= 1 or not 0 <= self.min_piece <= 1 or self.lo >= self.hi:
            raise ValueError('Invalid mesh extraction settings')


def load_mesh(path):
    import numpy as np
    import trimesh
    if Path(path).stat().st_size > MAX_MESH_BYTES:
        raise ValueError('Mesh exceeds 32 MiB')
    try:
        mesh = trimesh.load_mesh(str(path), file_type='obj', process=False, skip_materials=True)
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        raise ValueError('Invalid Wavefront OBJ triangle mesh') from exc
    if (not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces) or not len(mesh.vertices)
            or not np.isfinite(mesh.vertices).all()):
        raise ValueError('Mesh has no triangles; choose another extraction threshold or input')
    return mesh


def mesh_preview(source, output_dir):
    """Make a self-contained shaded preview; measurements use the original OBJ."""
    mesh = load_mesh(source)
    output = Path(output_dir) / (uuid.uuid4().hex + '.glb')
    with output.open('xb') as stream:
        stream.write(mesh.export(file_type='glb', include_normals=True))
    return output


def extract_mesh(source, output_dir, executable, options):
    options.validate()
    if not executable:
        raise ValueError('Mesh generation needs the native idol_mesh tool; build it or pass --mesh-executable')
    source, output_dir = Path(source), Path(output_dir)
    if source.stat().st_size > MAX_SPLAT_BYTES:
        raise ValueError('Gaussian point cloud exceeds 80 MiB')
    source_sha, executable_sha = digest(source), digest(executable)
    output = output_dir / (uuid.uuid4().hex + '.obj')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='idol-mesh-', dir=output_dir) as temporary:
        pending = Path(temporary) / 'mesh.obj'
        args = [str(executable), str(source), str(pending)]
        for key, value in asdict(options).items():
            args.extend(['--' + key.replace('_', '-'), str(value)])
        try:
            environment = os.environ.copy()
            environment['OMP_NUM_THREADS'] = '2'
            process = subprocess.run(args, capture_output=True, text=True, timeout=120,
                                     env=environment,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        except subprocess.TimeoutExpired as exc:
            raise ValueError('Mesh extraction exceeded the 120-second limit') from exc
        if process.returncode:
            raise ValueError('Native mesh extraction failed: ' + (process.stderr or process.stdout)[-2000:])
        mesh = load_mesh(pending)
        if digest(source) != source_sha or digest(executable) != executable_sha:
            raise RuntimeError('Mesh source or executable changed during extraction')
        metadata = {'source_name': source.name, 'source_ply_sha256': source_sha,
                    'mesh_sha256': digest(pending), 'extractor_sha256': executable_sha,
                    'settings': asdict(options), 'seconds': time.monotonic() - started,
                    'vertices': len(mesh.vertices), 'triangles': len(mesh.faces),
                    'scope': 'Opacity-weighted Gaussian centers, density blur, marching cubes, piece filtering. '
                             'Untextured geometry; Gaussian colors, scales, and rotations are not represented.'}
        try:
            with output.open('xb') as destination, pending.open('rb') as original:
                shutil.copyfileobj(original, destination)
            preview = output.with_suffix('.glb')
            with preview.open('xb') as stream:
                stream.write(mesh.export(file_type='glb', include_normals=True))
            metadata['preview_sha256'] = digest(preview)
            with output.with_suffix('.mesh.json').open('x', encoding='utf-8') as stream:
                json.dump(metadata, stream, indent=2, allow_nan=False)
        except Exception:
            output.unlink(missing_ok=True)
            output.with_suffix('.glb').unlink(missing_ok=True)
            raise
    return output


def compare_mesh_files(reference, candidate, output_dir, samples=20000, seed=23):
    from compare_meshes import compare
    if not 1000 <= samples <= 100000 or int(samples) != samples or not 0 <= seed < 2**32:
        raise ValueError('Comparison needs 1,000–100,000 surface samples and a seed from 0 to 2^32-1')
    hashes = {'reference': digest(reference), 'candidate': digest(candidate)}
    a, b = load_mesh(reference), load_mesh(candidate)
    report = {'scope': 'Sampled source points to closest target triangles in existing coordinates, without alignment. '
                      'Distances are in input model units. Primary F1 scores use 0.1%, 0.5% and 1% of the reference '
                      'bounding-box diagonal. Source-surface sampling still limits coverage; the legacy point-to-point '
                      'diagnostics include a reference resampling baseline. '
                      'This measures geometry agreement, not appearance or ground-truth reconstruction accuracy.',
              'input_sha256': hashes, 'measurements': compare(a, b, int(samples), int(seed))}
    if hashes != {'reference': digest(reference), 'candidate': digest(candidate)}:
        raise RuntimeError('Mesh inputs changed during comparison')
    for label, source in (('reference', reference), ('candidate', candidate)):
        metadata = Path(source).with_suffix('.mesh.json')
        if metadata.is_file():
            info = json.loads(metadata.read_text(encoding='utf-8'))
            if info['mesh_sha256'] != hashes[label]:
                raise ValueError('Mesh extraction metadata does not match the mesh bytes')
            report[label + '_extraction'] = info
    reference_info, candidate_info = report.get('reference_extraction'), report.get('candidate_extraction')
    report['extraction_settings_match'] = (reference_info['settings'] == candidate_info['settings']
                                          if reference_info and candidate_info else None)
    report['tool_sha256'] = {name: digest(Path(__file__).with_name(name))
                             for name in ('compare_meshes.py', 'idol_mesh_tools.py')}
    output = Path(output_dir) / (uuid.uuid4().hex + '.comparison.json')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report, output
