"""Check cohort completeness, input binding and metric units without model inference."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT=Path(__file__).resolve().parents[2]/'tools/analyze_idol_benchmark.py'


class CohortAnalysis(unittest.TestCase):
    def fixture(self,root):
        run=root/'run'; run.mkdir()
        image=root/'input.png'; Image.new('RGB',(2,2),'white').save(image)
        case={'id':'fixture','name':'Synthetic unit-test fixture','stratum':'synthetic',
              'file':image.name,'sha256':hashlib.sha256(image.read_bytes()).hexdigest()}
        protocol=root/'protocol.json'
        protocol.write_text(json.dumps({'cases':[case],'seeds':[0,42,2026]}))
        sha=hashlib.sha256(protocol.read_bytes()).hexdigest()
        (run/'identity.json').write_text(json.dumps({'protocol_sha256':sha}))
        (run/'completion.json').write_text(json.dumps({'protocol_sha256':sha,'completed_objects':1,
            'expected_objects':1,'expected_matched_cases':3}))
        directory=run/'fixture'; directory.mkdir()
        generations={}
        for arm in ('original','q8'):
            for seed in (0,42,2026):
                label=f'{arm}-{seed}'
                generations[label]={'seed':seed,'seconds':1,'peak_pytorch_allocated_bytes':100,
                    'host_working_set_before':200,'host_working_set_after':300,
                    'camera':{'yaw':0,'pitch':0,'fov':50,'radius':2}}
                Image.new('RGB',(2,2),'white').save(directory/(label+'.prepared.png'))
        comparison={'relative_latent_error':.05,
            'appearance':{'mean_foreground_color_error':.02,'mean_silhouette_overlap':.9,'valid_view_count':6},
            'mesh':{'reference_bounding_box_diagonal':2,'sampled_point_to_triangle':{
                'symmetric_mean_distance':.02,'symmetric_p95_distance':.04,'maximum_sampled_distance':.08,
                'f_scores_by_reference_diagonal_fraction':{'0.001':{'f1':.8},'0.005':{'f1':.99}}}}}
        receipt={'object':case,'protocol_sha256':sha,'generations':generations,
            'matched':{str(seed):comparison for seed in (0,42,2026)},
            'original_seed_variation':{pair:comparison for pair in ('0-42','0-2026','42-2026')},
            'same_seed_repeat':None,'extraction_sensitivity':{}}
        target=directory/'result.json'; target.write_text(json.dumps(receipt))
        return protocol,run,target,image

    def invoke(self,root,protocol,run):
        return subprocess.run([sys.executable,str(SCRIPT),'--protocol',str(protocol),'--run',str(run),
            '--out',str(root/'analysis'),'--require-complete'],capture_output=True,text=True)

    def test_metric_units_and_object_resampling(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); protocol,run,_,_=self.fixture(root)
            result=self.invoke(root,protocol,run)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/'analysis/summary.json').read_text())
            self.assertEqual(report['matched_cases'],3)
            self.assertEqual(report['generation_counts']['successful'],6)
            for key,expected in [('mesh_mean_error_percent',1),('relative_latent_error_percent',5),
                                 ('foreground_color_error_percent',2),('mesh_f1_0_1_percent',80)]:
                metric=report['metrics'][key]
                self.assertAlmostEqual(metric['quantization']['mean'],expected)
                self.assertEqual(metric['complete_objects_for_resampling'],1)
                for endpoint in metric['cohort_mean_resampling_interval_95']:
                    self.assertAlmostEqual(endpoint,expected)

    def test_incomplete_cohort_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); protocol,run,target,_=self.fixture(root); target.unlink()
            result=self.invoke(root,protocol,run)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Incomplete benchmark',result.stderr)

    def test_changed_input_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); protocol,run,_,image=self.fixture(root)
            image.write_bytes(b'changed')
            result=self.invoke(root,protocol,run)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Protocol input is missing or changed',result.stderr)

    def test_stale_worker_completion_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); protocol,run,_,_=self.fixture(root)
            path=run/'completion.json'; value=json.loads(path.read_text()); value['completed_objects']=0
            path.write_text(json.dumps(value))
            result=self.invoke(root,protocol,run)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Worker completion receipt',result.stderr)


if __name__=='__main__':unittest.main()
