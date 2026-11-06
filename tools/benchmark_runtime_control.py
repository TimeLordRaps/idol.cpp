"""Four-object exploratory controls for runtime, preprocessing and decoding differences.

This supplements the frozen cohort; it is not included in its headline averages.
The four objects are fixed by protocol position, not selected for their results.
"""
import argparse
import gc
import gzip
import hashlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('protocol','run','output','model','sdk','guard-root','mesh-executable'):
        p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    sys.path[:0]=[str(args.sdk),str(args.guard_root)]
    from ops import guard
    from idol_mesh_tools import MeshOptions,extract_mesh,load_mesh,digest
    protocol=json.loads(args.protocol.read_text())
    complete=json.loads((args.run/'completion.json').read_text())
    assert complete['completed_objects']==complete['expected_objects']==len(protocol['cases'])
    assert digest(args.model)==protocol['candidate_sha256']
    identity=json.loads((args.run/'identity.json').read_text())
    assert digest(args.protocol)==identity['protocol_sha256']
    for case in protocol['cases']:
        assert digest(args.protocol.parent/case['file'])==case['sha256']
    sources={Path(__file__):digest(Path(__file__)),args.mesh_executable:digest(args.mesh_executable)}
    for name,sha in identity['source_sha256'].items():
        assert digest(Path(__file__).with_name(name))==sha, 'Main benchmark source changed: '+name
        sources[Path(__file__).with_name(name)]=sha
    for name,sha in identity['sdk_source_sha256'].items():
        assert digest(args.sdk/name)==sha, 'SDK source changed: '+name
        sources[args.sdk/name]=sha
    assert digest(args.mesh_executable)==identity['extractor_sha256']
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'result.json').exists():
        raise ValueError('Control result already exists; choose a new output directory')
    selected=[protocol['cases'][i] for i in (0,4,8,12)]
    with guard.guarded('idol-q8-runtime-path-control',vram_gb=10,ram_gb=8.5,wait_s=300,category='external') as reservation:
        import diffusers.utils.import_utils as imports
        imports._xformers_available=False
        import numpy as np
        import torch
        from PIL import Image
        from idol_session import IdolSession
        import idol_generate as run
        from compare_meshes import compare
        from render_splats_preview import read_splats,metrics
        from benchmark_render import render,VIEWS
        from idol.models import OctreeProbabilityFixedlenDecoder
        torch.set_num_threads(2)
        last=[0.0]
        def checkpoint():
            if time.monotonic()-last[0]<1:return
            last[0]=time.monotonic()
            if guard.foreign_claim() is not None or guard.claim_over('external') is not None or (guard.ram_free_gb() or 0)<3:
                raise RuntimeError('YIELD: another job or system memory needs priority')
            if reservation.gov:reservation.gov.checkpoint()
        run.checkpoint=checkpoint
        run.hold_model=lambda value,name='':value
        result={'scope':__doc__,'model_sha256':digest(args.model),'protocol_sha256':digest(args.protocol),
                'main_run_identity_sha256':digest(args.run/'identity.json'),'seed':42,'cases':[],
                'intervention_scope':'Added during the main run, before these controls were executed. '
                    'Four objects fixed by protocol positions 0, 4, 8 and 12, not chosen by error. '
                    'Each intervention changes one pipeline boundary on the released artifact. '
                    'Results are conditional, nonadditive diagnostics and are excluded from the 96-case cohort.',
                'interventions':{
                    'default':'Same quantized weights, expanded default kernels instead of fused kernels.',
                    'fixed_prepared':'Fused quantized pipeline, replacing preprocessing output with the stored original-model prepared pixels; seed reset occurs after preprocessing.',
                    'fixed_latent':'Fused quantized octree and Gaussian decoders on the stored original latent with the same sampling seed; no denoising. Anchor identity is not assumed.'}}
        with tempfile.TemporaryDirectory(prefix='runtime-control-',dir=args.output) as temporary:
            temporary=Path(temporary); receipts={}
            for label,fused in [('fused',True),('default',False)]:
                with IdolSession(args.model,resident_gpu=True,batch_guidance=True,cudnn_attention=True,fused_kernels=fused) as session:
                    for index,case in enumerate(selected):
                        cloud=temporary/(str(index)+'-'+label+'.ply')
                        started=time.monotonic()
                        sample=session.generate(args.protocol.parent/case['file'],cloud,mode=protocol['mode'],seed=42,
                                                steps=protocol['steps'],gaussians=protocol['gaussians'])
                        receipts[(index,label)]={'latent':sample['latent'],'ply_sha256':digest(cloud),
                                                 'seconds':time.monotonic()-started,'camera':sample['camera']}
                        np.savez_compressed(args.output/(str(index)+'-'+label+'.latent.npz'),latent=sample['latent'])
                        sample['preprocessed_image'].save(args.output/(str(index)+'-'+label+'.prepared.png'))
                        print('CONTROL_GENERATED',case['id'],label,flush=True)
                        if fused:
                            original=args.run/case['id']
                            with Image.open(original/'original-42.prepared.png') as image:
                                prepared=image.copy()
                            preprocess=session.pipeline.preprocess
                            try:
                                session.pipeline.preprocess=lambda *a,**k:prepared.copy()
                                fixed=temporary/(str(index)+'-fixed_prepared.ply')
                                fixed_sample=session.generate(args.protocol.parent/case['file'],fixed,mode=protocol['mode'],
                                    seed=42,steps=protocol['steps'],gaussians=protocol['gaussians'])
                                assert np.array_equal(np.asarray(fixed_sample['preprocessed_image']),np.asarray(prepared))
                            finally:
                                session.pipeline.preprocess=preprocess
                            receipts[(index,'fixed_prepared')]={'latent':fixed_sample['latent'],'ply_sha256':digest(fixed),'camera':fixed_sample['camera']}
                            np.savez_compressed(args.output/(str(index)+'-fixed_prepared.latent.npz'),latent=fixed_sample['latent'])
                            print('CONTROL_GENERATED',case['id'],'fixed_prepared',flush=True)
                            latent=np.load(original/'original-42.latent.npz')['latent']
                            with torch.no_grad():
                                checkpoint()
                                cond=torch.from_numpy(np.ascontiguousarray(latent,dtype=np.float32)).to(session.device)
                                torch.manual_seed(42)
                                anchors=OctreeProbabilityFixedlenDecoder.sample(session.octree,cond,
                                    num_points=protocol['gaussians']//session.gaussian.rep_config['num_gaussians'],
                                    level=session.model.config['max_voxel_level'])
                                cloud_value=session.gaussian.to_representation(anchors,session.gaussian(anchors,cond))[0]
                                run.validate_gaussian_cloud(cloud_value,protocol['gaussians'])
                                fixed=temporary/(str(index)+'-fixed_latent.ply')
                                cloud_value.save_ply(str(fixed))
                                del cond,anchors,cloud_value
                            receipts[(index,'fixed_latent')]={'ply_sha256':digest(fixed)}
                            print('CONTROL_DECODED',case['id'],'fixed_latent',flush=True)
                gc.collect(); torch.cuda.empty_cache()
            for index,case in enumerate(selected):
                a,b=(receipts[(index,label)] for label in ('fused','default'))
                reference=json.loads((args.run/case['id']/'result.json').read_text())
                meshes={}; frames={}
                original=args.run/case['id']
                raw=gzip.open(original/'original-42.obj.gz','rb').read()
                assert hashlib.sha256(raw).hexdigest()==reference['generations']['original-42']['mesh']['mesh_sha256']
                import trimesh
                reference_mesh=trimesh.load_mesh(io.BytesIO(raw),file_type='obj',process=False,skip_materials=True)
                original_latent=np.load(original/'original-42.latent.npz')['latent'].astype(np.float64)
                for label in ('fused','default','fixed_prepared','fixed_latent'):
                    cloud=temporary/(str(index)+'-'+label+'.ply')
                    mesh=extract_mesh(cloud,temporary,args.mesh_executable,MeshOptions(**protocol['mesh']))
                    meshes[label]=load_mesh(mesh)
                    with mesh.open('rb') as src,gzip.open(args.output/(str(index)+'-'+label+'.obj.gz'),'wb') as dst:
                        dst.write(src.read())
                    if label in ('fused','default'):
                        data=read_splats(cloud)
                        frames[label]={name:render(data,axes,256,.75,sign) for name,axes,sign in VIEWS}
                case_result={'object':case['id'],'stratum':case['stratum'],
                    'fused_replay_matches_main_ply':a['ply_sha256']==reference['generations']['q8-42'].get('ply_sha256'),
                    'fused_ply_sha256':a['ply_sha256'],'default_ply_sha256':b['ply_sha256'],
                    'relative_latent_runtime_path_difference':float(np.linalg.norm(a['latent'].astype(np.float64)-b['latent'])/
                                                                 np.linalg.norm(a['latent'].astype(np.float64))),
                    'mesh_runtime_path_difference':compare(meshes['fused'],meshes['default'],20000,23),
                    'appearance_runtime_path_difference':{name:metrics(frames['fused'][name],frames['default'][name]) for name,_,_ in VIEWS},
                    'mesh_versus_original':{label:compare(reference_mesh,meshes[label],protocol['mesh_samples'],protocol['mesh_sample_seed'])
                        for label in meshes},
                    'relative_latent_error_versus_original':{label:float(np.linalg.norm(original_latent-receipts[(index,label)]['latent'])/np.linalg.norm(original_latent))
                        for label in ('fused','default','fixed_prepared')},
                    'prepared_original_png_sha256':digest(original/'original-42.prepared.png'),
                    'original_latent_archive_sha256':digest(original/'original-42.latent.npz'),
                    'cloud_sha256':{label:receipts[(index,label)]['ply_sha256'] for label in meshes},
                    'camera':{label:receipts[(index,label)]['camera'] for label in ('fused','default','fixed_prepared')}}
                result['cases'].append(case_result)
                print('CONTROL_ANALYZED',case['id'],flush=True)
        result['source_sha256']={name:digest(Path(__file__).with_name(name)) for name in
            ('benchmark_runtime_control.py','idol_session.py','idol_generate.py','idol_gguf.py','idol_packed.py','idol_triton.py')}
        assert all(digest(path)==sha for path,sha in sources.items()),'Control inputs changed during execution'
        assert digest(args.model)==protocol['candidate_sha256']
        (args.output/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('RUNTIME_PATH_CONTROL_COMPLETE',len(result['cases']),flush=True)


if __name__=='__main__':main()
