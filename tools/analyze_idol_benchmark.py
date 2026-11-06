"""Aggregate a frozen fidelity cohort without treating seeds as independent objects."""
import argparse
import hashlib
import gzip
import io
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image


METRICS = {
 'mesh_mean_error_percent': ('Mean mesh distance / reference diagonal (%)',False),
 'mesh_p95_error_percent': ('Directional 95th-percentile mesh distance / diagonal (%)',False),
 'mesh_sampled_max_percent': ('Maximum sampled mesh distance / diagonal (%)',False),
 'mesh_f1_0_1_percent': ('Mesh F1 within 0.1% of reference diagonal (%)',True),
 'mesh_f1_0_5_percent': ('Mesh F1 within 0.5% of reference diagonal (%)',True),
 'foreground_color_error_percent': ('Mean foreground color absolute error (%)',False),
 'silhouette_overlap_percent': ('Mean silhouette overlap (%)',True),
 'relative_latent_error_percent': ('Relative latent Euclidean error (%)',False),
 'prepared_image_error_percent': ('Preprocessed input channel absolute error (%)',False),
 'prepared_image_changed_pixels_percent': ('Preprocessed input pixels with any channel change (%)',False),
 'camera_yaw_error_degrees': ('Predicted camera yaw difference (degrees)',False),
 'camera_pitch_error_degrees': ('Predicted camera pitch difference (degrees)',False),
 'camera_field_of_view_error_degrees': ('Predicted vertical field-of-view difference (degrees)',False),
 'camera_radius_error_percent': ('Predicted camera radius relative difference (%)',False),
}
EXPLORATORY_METRICS = {name for name in METRICS if name.startswith(('prepared_', 'camera_'))}


def scalars(value):
    result = {}
    if 'relative_latent_error' in value:
        result['relative_latent_error_percent'] = 100*value['relative_latent_error']
    appearance=value.get('appearance',{})
    if appearance.get('mean_foreground_color_error') is not None:
        result['foreground_color_error_percent']=100*appearance['mean_foreground_color_error']
        result['silhouette_overlap_percent']=100*appearance['mean_silhouette_overlap']
    mesh=value.get('mesh',{})
    if 'sampled_point_to_triangle' in mesh:
        m=mesh['sampled_point_to_triangle']; diagonal=mesh['reference_bounding_box_diagonal']
        result.update(mesh_mean_error_percent=100*m['symmetric_mean_distance']/diagonal,
                      mesh_p95_error_percent=100*m['symmetric_p95_distance']/diagonal,
                      mesh_sampled_max_percent=100*m['maximum_sampled_distance']/diagonal,
                      mesh_f1_0_1_percent=100*m['f_scores_by_reference_diagonal_fraction']['0.001']['f1'],
                      mesh_f1_0_5_percent=100*m['f_scores_by_reference_diagonal_fraction']['0.005']['f1'])
    return result


def distribution(values):
    if not values: return None
    v=np.asarray(values,dtype=np.float64)
    return {'count':len(v),'mean':float(v.mean()),'median':float(np.median(v)),
            'p05':float(np.percentile(v,5)),'p95':float(np.percentile(v,95)),
            'min':float(v.min()),'max':float(v.max())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--require-complete',action='store_true')
    args=p.parse_args()
    protocol=json.loads(args.protocol.read_text())
    identity=json.loads((args.run/'identity.json').read_text())
    assert hashlib.sha256(args.protocol.read_bytes()).hexdigest()==identity['protocol_sha256']
    for case in protocol['cases']:
        image=args.protocol.parent/case['file']
        if not image.is_file() or hashlib.sha256(image.read_bytes()).hexdigest()!=case['sha256']:
            raise ValueError('Protocol input is missing or changed: '+case['id'])
    data=[]
    for case in protocol['cases']:
        path=args.run/case['id']/'result.json'
        if path.exists():
            item=json.loads(path.read_text()); assert item['object']==case and item['protocol_sha256']==identity['protocol_sha256']
            assert set(item['matched'])=={str(seed) for seed in protocol['seeds']}
            assert set(item['original_seed_variation'])=={str(a)+'-'+str(b) for a,b in itertools.combinations(protocol['seeds'],2)}
            data.append(item)
    expected=len(protocol['cases'])*len(protocol['seeds'])
    count=sum(len(d['matched']) for d in data)
    complete=len(data)==len(protocol['cases']) and count==expected
    if args.require_complete and not complete:
        raise RuntimeError(f'Incomplete benchmark: {len(data)}/{len(protocol["cases"])} objects, {count}/{expected} matched cases')
    if args.require_complete:
        completion=json.loads((args.run/'completion.json').read_text())
        if (completion.get('protocol_sha256')!=identity['protocol_sha256'] or
                completion.get('completed_objects')!=len(protocol['cases']) or
                completion.get('expected_objects')!=len(protocol['cases']) or
                completion.get('expected_matched_cases')!=expected):
            raise RuntimeError('Worker completion receipt does not confirm the complete protocol')
    args.out.mkdir(parents=True,exist_ok=True)
    records=[]; failures=[]; repeats=[]; sensitivity=[]; topology=[]; timing=[]
    generation_counts={'successful':0,'failed_attempts':0,'not_attempted_after_session_failure':0}
    for item in data:
        row={'id':item['object']['id'],'name':item['object']['name'],'stratum':item['object']['stratum']}
        prepared={}
        def diagnostics(left,right):
            if any('failure' in item['generations'][label] for label in (left,right)):
                return {}
            for label in (left,right):
                if label not in prepared:
                    file=args.run/row['id']/(label+'.prepared.png')
                    with Image.open(file) as image:
                        pixels=np.asarray(image.convert('RGB'))
                    prepared[label]=(pixels,hashlib.sha256(file.read_bytes()).hexdigest())
            a,sha_a=prepared[left]; b,sha_b=prepared[right]
            assert a.shape==b.shape
            delta=np.abs(a.astype(np.int16)-b.astype(np.int16))
            ca,cb=(item['generations'][label]['camera'] for label in (left,right))
            yaw=abs((ca['yaw']-cb['yaw']+np.pi)%(2*np.pi)-np.pi)
            return {'prepared_image_error_percent':float(delta.mean()*100/255),
                'prepared_image_changed_pixels_percent':float(np.any(delta!=0,axis=-1).mean()*100),
                'prepared_images_pixel_identical':bool(not delta.any()),
                'prepared_reference_png_sha256':sha_a,'prepared_candidate_png_sha256':sha_b,
                'camera_yaw_error_degrees':float(np.degrees(yaw)),
                'camera_pitch_error_degrees':float(np.degrees(abs(ca['pitch']-cb['pitch']))),
                'camera_field_of_view_error_degrees':abs(ca['fov']-cb['fov']),
                'camera_radius_error_percent':100*abs(ca['radius']-cb['radius'])/abs(ca['radius'])}
        for arm,key in [('quantization','matched'),('original_seed_variation','original_seed_variation')]:
            row[arm]=[]
            for seed,value in item[key].items():
                values=scalars(value)
                left,right=('original-'+seed,'q8-'+seed) if arm=='quantization' else tuple('original-'+s for s in seed.split('-'))
                values.update(diagnostics(left,right))
                row[arm].append({'seed_or_pair':seed,**values})
                if 'generation_failure' in value or 'mesh_failure' in value:
                    failures.append({'object':row['id'],'arm':arm,'seed_or_pair':seed,
                                     'generation_failure':value.get('generation_failure'),'mesh_failure':value.get('mesh_failure')})
                if value.get('appearance',{}).get('valid_view_count')==0:
                    failures.append({'object':row['id'],'arm':arm,'seed_or_pair':seed,
                                     'appearance_diagnostic_failure':'No nonempty orthographic views'})
        for label,generation in item['generations'].items():
            if 'failure' in generation:
                generation_counts['failed_attempts' if generation.get('failure_class')=='generation' else 'not_attempted_after_session_failure']+=1
                failures.append({'object':row['id'],'arm':label,'generation_failure':generation['failure']})
            else:
                generation_counts['successful']+=1
                if not label.endswith('repeat'):
                    timing.append({'object':row['id'],'model':label.split('-')[0],
                                   'request':'first' if generation['seed']==protocol['seeds'][0] else 'subsequent',
                                   'seconds':generation['seconds'],
                                   'peak_pytorch_allocated_bytes':generation['peak_pytorch_allocated_bytes'],
                                   'host_working_set_before':generation['host_working_set_before'],
                                   'host_working_set_after':generation['host_working_set_after']})
                archive=args.run/row['id']/(label+'.obj.gz')
                if archive.exists():
                    import trimesh
                    raw=gzip.open(archive,'rb').read()
                    assert hashlib.sha256(raw).hexdigest()==generation['mesh']['mesh_sha256']
                    mesh=trimesh.load_mesh(io.BytesIO(raw),file_type='obj',process=False,skip_materials=True)
                    topology.append({'object':row['id'],'label':label,'vertices':len(mesh.vertices),
                        'faces':len(mesh.faces),'components':int(mesh.body_count),'euler_characteristic':int(mesh.euler_number),
                        'watertight':bool(mesh.is_watertight),'winding_consistent':bool(mesh.is_winding_consistent),
                        'surface_area':float(mesh.area),
                        'absolute_oriented_triangle_volume':abs(float(mesh.volume)) if mesh.is_watertight and mesh.is_winding_consistent else None})
        if item.get('same_seed_repeat') is not None:
            repeats.append({'object':row['id'],'identical_ply_bytes':item['same_seed_repeat'].get('identical_ply_bytes'),
                            **scalars(item['same_seed_repeat']),**diagnostics('original-0','original-0-repeat')})
        if item['extraction_sensitivity']:
            values={}
            standard=item['matched'].get('42',{}).get('mesh')
            if standard: values['default']=scalars({'mesh':standard})
            for setting,value in item['extraction_sensitivity'].items():
                values[setting]=scalars({'mesh':value}) if 'sampled_point_to_triangle' in value else {'failure':value}
            sensitivity.append({'object':row['id'],'stratum':row['stratum'],'measurements':values})
        records.append(row)
    summary={}
    rng=np.random.default_rng(20261105)
    for metric,(description,higher_better) in METRICS.items():
        values={arm:[sample[metric] for r in records for sample in r[arm] if metric in sample]
                for arm in ('quantization','original_seed_variation')}
        object_rows=[]
        for row in records:
            arms={arm:[sample[metric] for sample in row[arm] if metric in sample]
                  for arm in ('quantization','original_seed_variation')}
            if all(len(v)==3 for v in arms.values()):
                object_rows.append({'id':row['id'],'stratum':row['stratum'],
                                    **{arm:float(np.mean(v)) for arm,v in arms.items()}})
        boot=[]; boot_delta=[]
        strata=sorted({r['stratum'] for r in object_rows})
        if object_rows:
            for _ in range(2000):
                sampled=[]
                for stratum in strata:
                    group=[r for r in object_rows if r['stratum']==stratum]
                    sampled.extend(group[i] for i in rng.integers(0,len(group),len(group)))
                boot.append(np.mean([r['quantization'] for r in sampled]))
                boot_delta.append(np.mean([r['quantization']-r['original_seed_variation'] for r in sampled]))
        summary[metric]={'description':description,'higher_is_better':higher_better,
                         'quantization':distribution(values['quantization']),
                         'original_seed_variation':distribution(values['original_seed_variation']),
                         'complete_objects_for_resampling':len(object_rows),
                         'cohort_mean_resampling_interval_95':np.percentile(boot,[2.5,97.5]).tolist() if boot else None,
                         'paired_cohort_mean_minus_seed_baseline_interval_95':np.percentile(boot_delta,[2.5,97.5]).tolist() if boot_delta else None,
                         'objects_quantization_worse_than_seed_baseline':sum(
                             r['quantization']<r['original_seed_variation'] if higher_better else
                             r['quantization']>r['original_seed_variation'] for r in object_rows),
                         'object_means':object_rows}
    strata_summary={}
    for name in sorted({r['stratum'] for r in records}):
        group=[r for r in records if r['stratum']==name]
        strata_summary[name]={metric:distribution([x[metric] for r in group for x in r['quantization'] if metric in x]) for metric in METRICS}
    worst=sorted(({'object':r['id'],'stratum':r['stratum'],**x} for r in records for x in r['quantization']
                  if 'mesh_mean_error_percent' in x),key=lambda v:v['mesh_mean_error_percent'],reverse=True)[:12]
    from scipy.stats import spearmanr
    correlations={}
    means={key:{r['id']:r['quantization'] for r in value['object_means']} for key,value in summary.items()}
    for other in ('relative_latent_error_percent','foreground_color_error_percent'):
        ids=sorted(set(means['mesh_mean_error_percent'])&set(means[other]))
        a=[means['mesh_mean_error_percent'][key] for key in ids]; b=[means[other][key] for key in ids]
        rho=float(spearmanr(a,b).statistic) if len(ids)>2 and np.ptp(a)>0 and np.ptp(b)>0 else None
        correlations[other]={'objects':len(ids),'spearman_rank_correlation_with_mean_mesh_error':rho,
                             'scope':'Exploratory association across complete object means; no causal attribution.'}
    topology_index={(r['object'],r['label']):r for r in topology}
    topology_changes=[]
    for item in data:
        for seed in protocol['seeds']:
            key=item['object']['id']; left=topology_index.get((key,'original-'+str(seed))); right=topology_index.get((key,'q8-'+str(seed)))
            if left and right:
                topology_changes.append({'object':key,'seed':seed,
                  'component_count_changed':left['components']!=right['components'],
                  'euler_characteristic_changed':left['euler_characteristic']!=right['euler_characteristic'],
                  'reference_components':left['components'],'candidate_components':right['components'],
                  'reference_euler':left['euler_characteristic'],'candidate_euler':right['euler_characteristic'],
                  'reference_watertight':left['watertight'],'candidate_watertight':right['watertight'],
                  'surface_area_change_percent':100*(right['surface_area']/left['surface_area']-1)})
    timing_summary={model:{kind:{'seconds':distribution([r['seconds'] for r in timing if r['model']==model and r['request']==kind]),
                      'pytorch_peak_allocated_bytes':distribution([r['peak_pytorch_allocated_bytes'] for r in timing if r['model']==model and r['request']==kind])}
                      for kind in ('first','subsequent')} for model in ('original','q8')}
    result={'status':'COMPLETE_MEASUREMENTS' if complete else 'PARTIAL',
        'objects_completed':len(data),'objects_expected':len(protocol['cases']), 'matched_cases':count,'matched_cases_expected':expected,
        'protocol_sha256':identity['protocol_sha256'],
        'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'A household-object geometry challenge cohort specified before inference, not a random population sample. '
                'Three seeds are clustered within each object. Original-versus-original seed variation changes the whole '
                'stochastic pipeline; it is context, not an error allowance or a subtraction from quantization error. '
                'The intervals are empirical stability intervals from 2,000 stratified object resamples, not population '
                'coverage guarantees. Ground-truth 3D reconstruction accuracy is not established. Approximate orthographic '
                'rendering is separate from upstream rendering and target-image accuracy. No quality threshold was chosen after seeing results.',
        'metrics':summary,'strata':strata_summary,'worst_cases':worst,'failures':failures,'generation_counts':generation_counts,
        'measurement_definitions':{
            'geometry':'Area-weighted surface samples to closest triangles; unsquared Euclidean distances in model units, with normalized percentages relative to the original mesh bounding-box diagonal.',
            'color':'Absolute differences of red, green, blue channel values in [0,1], converted from stored Idol coefficients; no additional gamma transformation. Foreground is the union of pixels with alpha >= 1/255 in either render. Percentage means 100 times mean absolute channel difference, averaged equally across valid views.',
            'silhouette':'Intersection divided by union of rendered pixels with alpha >= 0.5; both-empty views are counted and excluded from the appearance mean.',
            'render_resolution':'256 pixels square, fixed extent -0.75 to +0.75 model units on each image axis, six orthographic directions. Fixed framing can clip geometry; out-of-frame coverage was not measured for all clouds.',
            'prepared_image':'Exploratory analysis added during the run. Stored preprocessed images are compared in red, green, blue channels. Mean absolute channel difference is divided by 255; changed pixels have any unequal channel. Percentages are on [0,100].',
            'camera':'Exploratory analysis added during the run. Shortest circular yaw difference and absolute pitch difference are converted from radians to degrees. Vertical field of view is already in degrees; radius difference is divided by the original radius in model-coordinate units.',
            'peak_signal_to_noise_ratio':'Raw view reports use decibels and maximum channel value 1. A null value at exactly zero squared error represents an infinite ratio, not a failed measurement.'},
        'timing_observations':timing_summary,
        'timing_scope':'Generation calls after model loading, including preprocessing, sampling, decoding and PLY writing. '
                       'Shared workstation, original model loaded before Q8 for each object, no randomized order or controlled '
                       'isolation. First calls can include kernel initialization. These observations do not establish causal speed gains. '
                       'PyTorch allocated bytes exclude the graphics context, unused reserved allocations and other processes. '
                       'Host working-set readings are endpoints, not measured peaks.',
        'exploratory_correlations':correlations,'exploratory_topology':topology,'exploratory_topology_changes':topology_changes,
        'topology_scope':'Exploratory diagnostics of the extracted meshes, not ground-truth topology. Euler characteristic '
                         'is vertices minus edges plus faces; it is not interpreted as a hole count. Triangle volume '
                         'is reported only for watertight consistently wound meshes; self-intersections were not checked. '
                         'Area and volume use squared and cubed model-coordinate units.',
        'same_seed_repeats':repeats,'extraction_sensitivity':sensitivity,'per_object':records}
    result['exploratory_metric_names']=sorted(EXPLORATORY_METRICS)
    (args.out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# Idol Q8 fidelity benchmark', '',f"Status: **{result['status']}**. Objects: {len(data)}/{len(protocol['cases'])}; matched cases: {count}/{expected}.",
           '',result['scope'],'','| Measure | Valid matched cases | Mean | Median | 95th percentile | Range | Original seed-baseline mean |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for metric,value in summary.items():
        if metric in EXPLORATORY_METRICS: continue
        q,b=value['quantization'],value['original_seed_variation']
        if q:
            baseline=f"{b['mean']:.5g}" if b else 'UNKNOWN'
            lines.append(f"| {value['description']} | {q['count']} | {q['mean']:.5g} | {q['median']:.5g} | {q['p95']:.5g} | {q['min']:.5g}–{q['max']:.5g} | {baseline} |")
    lines.extend(['','## Strata','', '| Geometry/use stratum | Matched cases | Mean mesh error (%) | Tight F1 mean (%) | Foreground color error (%) |',
                  '|---|---:|---:|---:|---:|'])
    for name,value in strata_summary.items():
        g,f,c=(value.get(key) for key in ('mesh_mean_error_percent','mesh_f1_0_1_percent','foreground_color_error_percent'))
        if g:
            lines.append(f"| {name} | {g['count']} | {g['mean']:.5g} | {f['mean']:.5g} | {c['mean']:.5g} |")
    lines.extend(['','## Empirical stability intervals','',
                  'Each object is one resampling unit, with seeds retained together. Objects are resampled within their declared strata. These intervals describe stability of this selected cohort; they are not population coverage guarantees.',
                  '', '| Measure | 95% resampling interval for cohort mean | Paired mean difference from original seed baseline |',
                  '|---|---:|---:|'])
    for metric,value in summary.items():
        if metric in EXPLORATORY_METRICS: continue
        interval=value['cohort_mean_resampling_interval_95']; delta=value['paired_cohort_mean_minus_seed_baseline_interval_95']
        if interval:
            lines.append(f"| {value['description']} | {interval[0]:.5g}–{interval[1]:.5g} | {delta[0]:.5g}–{delta[1]:.5g} |")
    lines.extend(['','## Exploratory preprocessing and camera diagnostics','',
                  'These analyses were added during the run; they are not part of the analysis specified before inference.',
                  '', '| Diagnostic | Valid cases | Mean | Median | 95th percentile | Maximum |', '|---|---:|---:|---:|---:|---:|'])
    for metric in METRICS:
        if metric not in EXPLORATORY_METRICS: continue
        value=summary[metric]; q=value['quantization']
        if q: lines.append(f"| {value['description']} | {q['count']} | {q['mean']:.5g} | {q['median']:.5g} | {q['p95']:.5g} | {q['max']:.5g} |")
    lines.extend(['','## Failures', '',f'{len(failures)} recorded failure entries; generation and derived comparison failures may refer to the same failed inference.',
                  json.dumps(generation_counts),'','## Same-seed repeatability','',json.dumps(repeats,indent=2),'','## Worst matched cases','',
                  '| Object | Seed | Mean mesh error (%) | Tight-threshold mesh F1 (%) | Foreground color error (%) |',
                  '|---|---:|---:|---:|---:|'])
    for row in worst:
        lines.append(f"| {row['object']} | {row['seed_or_pair']} | {row['mesh_mean_error_percent']:.5g} | {row['mesh_f1_0_1_percent']:.5g} | {row.get('foreground_color_error_percent',float('nan')):.5g} |")
    (args.out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'objects':len(data),'matched':count,'failure_entries':len(failures)}))


if __name__=='__main__':main()
