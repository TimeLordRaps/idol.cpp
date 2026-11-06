"""Send a real image to an Idol server and check its returned point cloud."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import requests
from plyfile import PlyData


def exercise(base, image, *, steps=50, gaussians=262144, seed=42, mode='real', timeout=900):
    parsed=urlsplit(base)
    if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError('Server URL must be an HTTP or HTTPS base URL')
    base=base.rstrip('/')
    health=requests.get(base+'/health',timeout=10);health.raise_for_status()
    data=Path(image).read_bytes()
    image_sha256=hashlib.sha256(data).hexdigest()
    started=time.monotonic()
    response=requests.post(base+'/v1/reconstructions',
        files={'image':(Path(image).name,data,'application/octet-stream')},
        data={'mode':mode,'steps':str(steps),'gaussians':str(gaussians),'seed':str(seed)},timeout=timeout)
    elapsed=time.monotonic()-started
    response.raise_for_status()
    metadata_url=response.headers.get('X-Idol-Metadata')
    if not metadata_url or not metadata_url.startswith('/v1/reconstructions/'):
        raise ValueError('Server did not provide a valid reconstruction metadata path')
    metadata_response=requests.get(base+metadata_url,timeout=10);metadata_response.raise_for_status()
    metadata=metadata_response.json()
    if metadata.get('input')!='uploaded_image' or metadata.get('image_sha256')!=image_sha256:
        raise ValueError('Reconstruction metadata does not match the uploaded image')
    cloud=PlyData.read(io.BytesIO(response.content))['vertex'].data
    if len(cloud)!=gaussians:raise ValueError('Returned point count differs from the requested count')
    required={'x','y','z','opacity',*(f'f_dc_{i}' for i in range(3)),
              *(f'scale_{i}' for i in range(3)),*(f'rot_{i}' for i in range(4))}
    if not required.issubset(cloud.dtype.names) or any(
            not np.issubdtype(cloud[name].dtype,np.floating) for name in required):
        raise ValueError('Returned point cloud lacks floating-point Gaussian properties')
    if any(not np.isfinite(cloud[name]).all() for name in cloud.dtype.names
           if np.issubdtype(cloud[name].dtype,np.floating)):
        raise ValueError('Returned point cloud contains non-finite values')
    loaded=requests.get(base+'/health',timeout=10);loaded.raise_for_status()
    if not loaded.json().get('model_loaded'):raise ValueError('Server does not report a loaded model')
    report={'status':'PASS','scope':'One real uploaded-image reconstruction; not a quality or minimum-memory test.',
        'input_sha256':image_sha256,'point_cloud_sha256':hashlib.sha256(response.content).hexdigest(),
        'point_count':len(cloud),'all_float_properties_finite':True,'request_seconds':elapsed,
        'settings':{'mode':mode,'steps':steps,'gaussians':gaussians,'seed':seed},'metadata':metadata}
    return report,response.content


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8127')
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New directory for the point cloud and receipt')
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--gaussians',type=int,default=262144)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--mode',choices=('real','synthetic'),default='real')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report,cloud=exercise(args.url,args.image,steps=args.steps,gaussians=args.gaussians,seed=args.seed,mode=args.mode)
    (args.output/'result.ply').write_bytes(cloud)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
