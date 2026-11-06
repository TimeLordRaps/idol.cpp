"""Check installed wheel files against their installation records, then import the runtime.

This detects missing/truncated installed files. It assumes trusted package records;
it does not authenticate upstream publishers or cover files outside those records.
"""
import argparse
import base64
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path


def audit(distributions,prefix):
    prefix=Path(prefix).resolve();checked=0;unchecked=0;failures=[];packages=0
    for distribution in distributions:
        packages+=1
        name=distribution.metadata.get('Name') or 'unknown distribution'
        if not distribution.metadata.get('Name') or not distribution.version:
            failures.append({'package':name,'failure':'Missing package identity metadata'})
        files=distribution.files
        if not files:
            failures.append({'package':name,'failure':'Missing or empty installation record'})
            continue
        for entry in files:
            path=Path(distribution.locate_file(entry))
            if not path.resolve().is_relative_to(prefix):
                failures.append({'package':name,'file':str(entry),'failure':'Record path escapes the environment'})
                continue
            if not path.is_file():
                failures.append({'package':name,'file':str(entry),'failure':'Installed file missing'})
                continue
            if entry.hash is None:
                unchecked+=1;continue
            try:
                value=hashlib.new(entry.hash.mode)
                with path.open('rb') as source:
                    for block in iter(lambda:source.read(4*1024*1024),b''):value.update(block)
                actual=base64.urlsafe_b64encode(value.digest()).rstrip(b'=').decode()
                if actual!=entry.hash.value or (entry.size is not None and path.stat().st_size!=entry.size):
                    failures.append({'package':name,'file':str(entry),'failure':'Installed bytes differ from record'})
                checked+=1
            except (OSError,ValueError) as error:
                failures.append({'package':name,'file':str(entry),'failure':str(error)})
    if not packages:failures.append({'failure':'No installed distributions discovered'})
    return {'status':'FAIL' if failures else 'PASS','packages':packages,'hashed_files':checked,
            'unhashed_record_entries':unchecked,'failures':failures,
            'scope':__doc__}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--runtime-imports',action='store_true')
    args=parser.parse_args()
    result=audit(importlib.metadata.distributions(),sys.prefix)
    if result['status']=='PASS' and args.runtime_imports:
        try:
            import torch
            from idol_server import Runtime,create_app
            from idol_session import IdolSession
            from idol.models import OctreeProbabilityFixedlenDecoder,GaussianFixedlenDecoder
            assert all(callable(value) for value in (Runtime,create_app,IdolSession,
                OctreeProbabilityFixedlenDecoder,GaussianFixedlenDecoder))
            result['runtime_imports']='PASS';result['torch']=torch.__version__
        except Exception as error:
            result['status']='FAIL';result['runtime_imports']=str(error)
    if args.report:
        with args.report.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({key:value for key,value in result.items() if key!='scope'}))
    if result['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
