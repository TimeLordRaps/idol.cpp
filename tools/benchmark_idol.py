"""Run a frozen original-versus-quantized Idol fidelity protocol, one object at a time.

One resident model is loaded at a time. Receipts are resumable only with the same
protocol, runtime sources and model bytes. Timing observations include background
workstation activity and are not causal performance comparisons. No model is trained.
"""
import argparse
import contextlib
import gc
import gzip
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    pending = Path(str(path) + '.pending')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, required=True)
    parser.add_argument('--guard-root', type=Path, required=True)
    parser.add_argument('--mesh-executable', type=Path, required=True)
    parser.add_argument('--max-objects', type=int)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2',
                      OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sys.path[:0] = [str(args.sdk), str(args.guard_root)]
    from ops import guard
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    assert digest(args.model) == protocol['candidate_sha256']
    assert all(Path(case['id']).name == case['id'] and '..' not in case['id'] for case in protocol['cases'])
    for case in protocol['cases']:
        assert digest(args.protocol.parent / case['file']) == case['sha256']
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), Path(__file__).with_name('benchmark_render.py')]
    sources += [Path(__file__).with_name(n) for n in ('idol_session.py','idol_generate.py','idol_gguf.py',
                 'idol_packed.py','idol_triton.py','idol_mesh_tools.py','compare_meshes.py','render_splats_preview.py')]
    identity = {'protocol_sha256': digest(args.protocol), 'candidate_sha256': digest(args.model),
                'source_sha256': {p.name:digest(p) for p in sources},
                'original_component_sha256': {p.name:digest(p) for p in args.source.glob('*.safetensors')},
                'sdk_revision':subprocess.check_output(['git','-C',str(args.sdk),'rev-parse','HEAD'],text=True).strip(),
                'sdk_source_sha256':{str(p.relative_to(args.sdk)).replace('\\','/'):digest(p) for p in (args.sdk/'idol').rglob('*.py')},
                'source_config_sha256':digest(args.source/'config.json'),
                'extractor_sha256': digest(args.mesh_executable), 'render_size':256, 'render_extent':0.75}
    assert identity['sdk_revision']==protocol['sdk_revision']
    identity_path = args.output / 'identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, 'Benchmark identity changed; use a new output directory'
    else:
        if any(args.output.iterdir()):
            raise ValueError('A new benchmark output directory must be empty')
        write_json(identity_path, identity)
    cases = protocol['cases'][:args.max_objects] if args.max_objects else protocol['cases']
    with guard.guarded('idol-gso32-fidelity-benchmark', vram_gb=10, ram_gb=8.5, wait_s=300, category='external') as reservation:
        import diffusers.utils.import_utils as imports
        imports._xformers_available = False
        import numpy as np
        import psutil
        import torch
        from PIL import Image
        from safetensors.torch import load_file
        from idol.models import OctreeProbabilityFixedlenDecoder, GaussianFixedlenDecoder
        import idol_session
        import idol_generate as run
        from idol_mesh_tools import MeshOptions, extract_mesh, load_mesh
        from compare_meshes import compare as compare_meshes
        from render_splats_preview import read_splats, render as reference_render, metrics as render_metrics
        from benchmark_render import render, VIEWS
        torch.set_num_threads(2)
        process = psutil.Process()
        checkpoint_time = [0.0]
        def checkpoint():
            now = time.monotonic()
            if now - checkpoint_time[0] < 1:
                return
            checkpoint_time[0] = now
            if guard.foreign_claim() is not None or guard.claim_over('external') is not None:
                raise RuntimeError('YIELD: another job requested the graphics device; resume later')
            available = guard.ram_free_gb()
            if available is None or available < 3.0:
                raise RuntimeError(f'YIELD: protecting system memory; {available} GB available')
            if reservation.gov:
                reservation.gov.checkpoint()
        run.checkpoint = checkpoint
        # Do not park multi-gigabyte models in scarce host memory on a foreign claim.
        # The checkpoint above instead aborts this worker and its GPU allocations.
        run.hold_model = lambda module, name='': module
        original_container = idol_session.IdolGGUF
        assert original_container(args.model).config == json.loads((args.source/'config.json').read_text()), 'Original and quantized configurations differ'
        original_decoder_loader = run.load_gpu_decoders
        class OriginalWeights:
            def __init__(self, unused):
                self.config = original_container(args.model).config
            def load_module(self, component, module):
                names = {'denoiser':'denoiser','background':'birefnet','octree':'decoder','gaussian':'decoder_gs'}
                module.load_state_dict(load_file(str(args.source/(names[component]+'.safetensors'))),strict=True)
                return module
        def original_decoders(model, fused_kernels=False):
            octree = model.load_module('octree',OctreeProbabilityFixedlenDecoder(**model.config['decoder']['args']))
            gaussian = model.load_module('gaussian',GaussianFixedlenDecoder(**model.config['decoder_gs']['args']))
            return octree.eval().requires_grad_(False), gaussian.eval().requires_grad_(False)
        @contextlib.contextmanager
        def session(kind):
            checkpoint()
            validator = run.validate_runtime_half_storage
            if kind == 'original':
                idol_session.IdolGGUF = OriginalWeights
                run.load_gpu_decoders = original_decoders
                run.validate_runtime_half_storage = lambda *values: None
            started = time.monotonic()
            value = None
            try:
                value = idol_session.IdolSession(args.model, resident_gpu=True, batch_guidance=True,
                    cudnn_attention=True, fused_kernels=kind=='q8')
                print('MODEL_READY',kind,'initialization_seconds',round(time.monotonic()-started,3),flush=True)
                yield value
            finally:
                if value is not None:
                    value.close()
                idol_session.IdolGGUF = original_container
                run.load_gpu_decoders = original_decoder_loader
                run.validate_runtime_half_storage = validator
                gc.collect()
                torch.cuda.empty_cache()
        def check_sources():
            assert all(digest(p)==identity['source_sha256'][p.name] for p in sources), 'Runtime changed during benchmark'
            assert all(digest(args.sdk/name)==sha for name,sha in identity['sdk_source_sha256'].items()), 'SDK changed during benchmark'
        def render_all(cloud, directory, prefix):
            data = read_splats(cloud)
            frames = {}
            for name,axes,sign in VIEWS:
                rgb,alpha = render(data,axes,256,0.75,sign)
                frames[name] = (rgb,alpha)
                Image.fromarray(np.uint8(np.clip(rgb[::-1]*255+0.5,0,255))).save(directory/(prefix+'-'+name+'.png'))
            return frames
        def appearance(a,b):
            views = {}
            for name,_,_ in VIEWS:
                value=render_metrics(a[name],b[name])
                value['reference_foreground_pixels']=int(np.count_nonzero(a[name][1]>=1/255))
                value['candidate_foreground_pixels']=int(np.count_nonzero(b[name][1]>=1/255))
                value['both_views_empty']=value['reference_foreground_pixels']==value['candidate_foreground_pixels']==0
                views[name]=value
            valid=[v for v in views.values() if not v['both_views_empty']]
            return {'views':views,
                    'valid_view_count':len(valid),'empty_view_count':len(views)-len(valid),
                    'mean_foreground_color_error':float(np.mean([v['foreground_rgb_mean_absolute_difference'] for v in valid])) if valid else None,
                    'mean_silhouette_overlap':float(np.mean([v['alpha_threshold_0_5_iou'] for v in valid])) if valid else None,
                    'worst_view_foreground_color_error':float(max(v['foreground_rgb_mean_absolute_difference'] for v in valid)) if valid else None}
        parity_path = args.output / 'renderer-validation.json'
        if not parity_path.exists():
            sample = (args.output / 'renderer-fixture.ply')
            # Test both orientations and multiple projections on a previously produced real cloud.
            candidates = list(args.source.glob('assets/gallery/*.webp'))
            fixture = Path(os.environ.get('IDOL_RENDER_FIXTURE',''))
            if not fixture.is_file():
                raise ValueError('Set IDOL_RENDER_FIXTURE to a saved real Gaussian PLY for renderer parity')
            data = read_splats(fixture)
            data = tuple(value[:1024] for value in data)
            errors=[]
            for _,axes,sign in VIEWS:
                expected = reference_render(data,axes,32,0.75,sign)
                actual = render(data,axes,32,0.75,sign)
                for a,b in zip(expected,actual):
                    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-12)
                    errors.append(float(np.max(np.abs(a-b))))
            write_json(parity_path,{'fixture_sha256':digest(fixture),'maximum_absolute_difference':max(errors),
                                   'views':6,'gaussians':1024,'size':32,'fastmath':False})
        versions = {name:__import__(name).__version__ for name in ('numpy','scipy','trimesh','rtree','numba')}
        write_json(args.output/'environment.json',{'torch':torch.__version__,'cuda':torch.version.cuda,
             'gpu':torch.cuda.get_device_name(),'software':versions,
             'scope':'Fidelity run using the original and released runtime paths; one resident model at a time. '
                     'Timing and memory observations on a shared workstation; no controlled causal speed claim.'})
        options = MeshOptions(**protocol['mesh'])
        repeat_ids = {protocol['cases'][i]['id'] for i in (0,4,8,12)}
        for object_index,case in enumerate(cases):
            directory = args.output / case['id']
            directory.mkdir(exist_ok=True)
            target = directory/'result.json'
            if target.exists():
                print('RESUMING_SKIP',case['id'],flush=True)
                continue
            check_sources()
            checkpoint()
            print('OBJECT_START',object_index+1,len(protocol['cases']),case['id'],flush=True)
            item={'object':case,'protocol_sha256':identity['protocol_sha256'],'generations':{},'matched':{},
                  'original_seed_variation':{},'extraction_sensitivity':{},'same_seed_repeat':None}
            with tempfile.TemporaryDirectory(prefix='clouds-',dir=directory) as temporary:
                temporary=Path(temporary)
                for kind in ('original','q8'):
                    with session(kind) as model:
                        seeds = list(protocol['seeds']) + ([0] if kind=='original' and case['id'] in repeat_ids else [])
                        for index,seed in enumerate(seeds):
                            label=kind+'-'+str(seed)+('-repeat' if index>=len(protocol['seeds']) else '')
                            cloud=temporary/(label+'.ply')
                            torch.cuda.reset_peak_memory_stats()
                            before=process.memory_info().rss
                            started=time.monotonic()
                            try:
                                result=model.generate(args.protocol.parent/case['file'],cloud,mode=protocol['mode'],seed=seed,
                                                       steps=protocol['steps'],gaussians=protocol['gaussians'])
                            except Exception as exc:
                                if 'YIELD:' in str(exc) or 'out of memory' in str(exc).lower():
                                    raise
                                item['generations'][label]={'seed':seed,'failure':str(exc),'failure_class':'generation'}
                                for later_index,later in enumerate(seeds[index+1:],start=index+1):
                                    later_label=kind+'-'+str(later)+('-repeat' if later_index>=len(protocol['seeds']) else '')
                                    item['generations'][later_label]={'seed':later,'failure':'Session invalidated by previous inference failure'}
                                print('GENERATION_FAILED',case['id'],label,str(exc),flush=True)
                                break
                            torch.cuda.synchronize()
                            seconds=time.monotonic()-started
                            np.savez_compressed(directory/(label+'.latent.npz'),latent=result['latent'])
                            result['preprocessed_image'].save(directory/(label+'.prepared.png'))
                            item['generations'][label]={'seed':seed,'seconds':seconds,'ply_sha256':digest(cloud),
                                'camera':result['camera'],'peak_pytorch_allocated_bytes':torch.cuda.max_memory_allocated(),
                                'host_working_set_before':before,'host_working_set_after':process.memory_info().rss}
                            print('GENERATED',case['id'],label,round(seconds,2),'seconds',flush=True)
                            write_json(directory/'progress.json',item)
                # Release both pipelines before processor-based geometry and rendering analysis.
                frames,meshes={},{}
                for label in item['generations']:
                    if 'failure' in item['generations'][label]: continue
                    cloud=temporary/(label+'.ply')
                    frames[label]=render_all(cloud,directory,label)
                    try:
                        path=extract_mesh(cloud,temporary,args.mesh_executable,options)
                        meshes[label]=load_mesh(path)
                        info=json.loads(path.with_suffix('.mesh.json').read_text())
                        with path.open('rb') as stream,gzip.open(directory/(label+'.obj.gz'),'wb') as dest:
                            shutil.copyfileobj(stream,dest)
                        item['generations'][label]['mesh']=info
                    except ValueError as exc:
                        item['generations'][label]['mesh_failure']=str(exc)
                def comparison(left,right):
                    if 'failure' in item['generations'][left] or 'failure' in item['generations'][right]:
                        return {'generation_failure':'At least one inference failed; counted explicitly, not scored as agreement'}
                    a=np.load(directory/(left+'.latent.npz'))['latent'].astype(np.float64)
                    b=np.load(directory/(right+'.latent.npz'))['latent'].astype(np.float64)
                    result={'relative_latent_error':float(np.linalg.norm(a-b)/np.linalg.norm(a)),
                            'appearance':appearance(frames[left],frames[right])}
                    if left in meshes and right in meshes:
                        measured=compare_meshes(meshes[left],meshes[right],protocol['mesh_samples'],protocol['mesh_sample_seed'])
                        result['mesh']=measured
                    else:
                        result['mesh_failure']='At least one extraction failed; counted explicitly, not scored as agreement'
                    return result
                for seed in protocol['seeds']:
                    item['matched'][str(seed)]=comparison('original-'+str(seed),'q8-'+str(seed))
                for a,b in itertools.combinations(protocol['seeds'],2):
                    item['original_seed_variation'][str(a)+'-'+str(b)]=comparison('original-'+str(a),'original-'+str(b))
                if case['id'] in repeat_ids:
                    item['same_seed_repeat']=comparison('original-0','original-0-repeat')
                    item['same_seed_repeat']['identical_ply_bytes']=(item['generations']['original-0'].get('ply_sha256')==
                        item['generations']['original-0-repeat'].get('ply_sha256') if 'generation_failure' not in item['same_seed_repeat'] else None)
                if object_index % 4 == 0:
                    for name,settings in [('resolution128',MeshOptions(resolution=128)),
                                          ('resolution224',MeshOptions(resolution=224)),('no_component_filter',MeshOptions(min_piece=0))]:
                        try:
                            if any('failure' in item['generations'][kind+'-42'] for kind in ('original','q8')):
                                raise ValueError('Sensitivity measurement unavailable because inference failed')
                            a=extract_mesh(temporary/'original-42.ply',temporary,args.mesh_executable,settings)
                            b=extract_mesh(temporary/'q8-42.ply',temporary,args.mesh_executable,settings)
                            item['extraction_sensitivity'][name]=compare_meshes(load_mesh(a),load_mesh(b),20000,23)
                        except ValueError as exc:
                            item['extraction_sensitivity'][name]={'failure':str(exc)}
                # Keep every mesh, latent and six-view preview; Gaussian clouds are bounded scratch files.
                # Retain the six worst mesh-error cloud pairs for detailed inspection.
                worst_path=args.output/'worst-cases.json'
                worst=json.loads(worst_path.read_text()) if worst_path.exists() else []
                managed={row['case']+'-'+kind+'.ply.gz' for row in worst for kind in ('original','q8')}
                for seed,comparison_value in item['matched'].items():
                    score=comparison_value.get('mesh',{}).get('sampled_point_to_triangle',{}).get('symmetric_mean_over_reference_diagonal')
                    if score is None: continue
                    key=case['id']+'-seed'+seed
                    worst.append({'case':key,'object':case['id'],'seed':int(seed),'score':score})
                    worst=sorted(worst,key=lambda row:row['score'],reverse=True)[:6]
                    if any(row['case']==key for row in worst):
                        for kind in ('original','q8'):
                            with (temporary/(kind+'-'+seed+'.ply')).open('rb') as src,gzip.open(args.output/(key+'-'+kind+'.ply.gz'),'wb') as dst:
                                shutil.copyfileobj(src,dst)
                            managed.add(key+'-'+kind+'.ply.gz')
                keep={row['case']+'-'+kind+'.ply.gz' for row in worst for kind in ('original','q8')}
                for name in managed-keep:
                    path=args.output/name
                    assert path.resolve().parent==args.output.resolve()
                    path.unlink(missing_ok=True)
                write_json(worst_path,worst)
                check_sources()
                write_json(target,item)
                print('OBJECT_COMPLETE',object_index+1,len(protocol['cases']),case['id'],flush=True)
            gc.collect()
        write_json(args.output/'completion.json',{'completed_objects':sum((args.output/c['id']/'result.json').exists() for c in protocol['cases']),
            'expected_objects':len(protocol['cases']),'expected_matched_cases':len(protocol['cases'])*len(protocol['seeds']),
            'protocol_sha256':identity['protocol_sha256']})


if __name__=='__main__':
    main()
