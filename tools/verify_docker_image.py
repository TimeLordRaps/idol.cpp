"""Audit an exported Docker image using a checker mounted from this checkout."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    info=json.loads(subprocess.check_output(['docker','image','inspect',args.image],text=True))[0]
    checker=Path(__file__).with_name('audit_python_install.py').resolve()
    checksum=hashlib.sha256(checker.read_bytes()).hexdigest()
    command=['docker','run','--rm','--network','none','--read-only','--cpus','2','--memory','8g',
        '--tmpfs','/tmp:rw,nosuid,nodev,size=256m',
        '--mount',f'type=bind,source={checker},target=/audit.py,readonly',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/opt/idol-cpp/tools',
        '-e','HF_HUB_OFFLINE=1','--entrypoint','/opt/venv/bin/python',info['Id'],
        '/audit.py','--runtime-imports']
    process=subprocess.run(command,capture_output=True,text=True,timeout=600)
    try:result=json.loads(process.stdout)
    except json.JSONDecodeError:
        raise RuntimeError('Image checker did not return its JSON receipt: '+process.stderr[-1500:])
    assert hashlib.sha256(checker.read_bytes()).hexdigest()==checksum,'Checker changed during execution'
    result.update(image_id=info['Id'],checker_sha256=checksum,
        checker_scope='Read-only checker supplied from the host checkout; an empty checker in the image cannot silently pass.')
    if args.report:
        with args.report.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({key:value for key,value in result.items() if key not in ('failures','scope')},indent=2))
    if (process.returncode or result.get('status')!='PASS' or not result.get('hashed_files') or
            result.get('runtime_imports')!='PASS'):
        print(json.dumps(result.get('failures',[])[:20],indent=2))
        raise SystemExit(1)


if __name__=='__main__':main()
