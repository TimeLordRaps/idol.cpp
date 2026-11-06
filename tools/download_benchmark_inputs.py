"""Download the exact public images named by a benchmark protocol and verify every digest."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('protocol',type=Path)
    p.add_argument('--cache-dir',type=Path)
    args=p.parse_args()
    protocol=json.loads(args.protocol.read_text(encoding='utf-8'))
    root=args.protocol.resolve().parent
    for item in protocol['cases']:
        target=(root/item['file']).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Input path escapes the protocol directory')
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('Existing input differs: '+item['id'])
            continue
        source=Path(hf_hub_download(protocol['dataset'],item['source_file'],repo_type='dataset',
                    revision=protocol['dataset_revision'],cache_dir=str(args.cache_dir) if args.cache_dir else None))
        if hashlib.sha256(source.read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError('Downloaded input differs: '+item['id'])
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as output,source.open('rb') as stream:
            shutil.copyfileobj(stream,output)
        print(item['id'])
    print('All protocol inputs verified')


if __name__=='__main__':main()
