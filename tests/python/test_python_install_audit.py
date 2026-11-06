import base64,hashlib,importlib.metadata,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from audit_python_install import audit


class InstalledFileAudit(unittest.TestCase):
    def test_empty_environment_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            self.assertEqual(audit([],name)['status'],'FAIL')

    def fixture(self,root):
        info=root/'fixture-1.0.dist-info';info.mkdir()
        (info/'METADATA').write_text('Name: fixture\nVersion: 1.0\n')
        body=b'expected installed bytes\n';(root/'module.py').write_bytes(body)
        (root/'empty.py').write_bytes(b'')
        entries=[]
        for name in ('module.py','empty.py'):
            data=(root/name).read_bytes()
            sha=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            entries.append(f'{name},sha256={sha},{len(data)}')
        (info/'RECORD').write_text('\n'.join(entries)+'\n')
        return info

    def test_valid_and_legitimately_empty_files(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.fixture(root)
            report=audit(importlib.metadata.distributions(path=[root]),root)
            self.assertEqual(report['status'],'PASS');self.assertEqual(report['hashed_files'],2)

    def test_truncated_package_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.fixture(root);(root/'module.py').write_bytes(b'')
            report=audit(importlib.metadata.distributions(path=[root]),root)
            self.assertEqual(report['status'],'FAIL')
            self.assertEqual(report['failures'][0]['file'],'module.py')

    def test_missing_record_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);info=self.fixture(root);(info/'RECORD').unlink()
            report=audit(importlib.metadata.distributions(path=[root]),root)
            self.assertEqual(report['status'],'FAIL')


if __name__=='__main__':unittest.main()
