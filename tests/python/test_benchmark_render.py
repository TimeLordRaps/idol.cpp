"""Check accelerated diagnostic compositing against the existing implementation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import numpy as np
from benchmark_render import render,VIEWS
from render_splats_preview import render as reference

rng=np.random.default_rng(20261105)
positions=rng.uniform(-1,1,(128,3))
positions[:4]=[[0,0,0],[0.75,0.75,0],[0,0,0],[-0.75,-0.75,0]]
matrix=rng.normal(size=(128,3,3))
covariance=matrix@np.swapaxes(matrix,1,2)*0.001+np.eye(3)[None]*0.0001
opacity=rng.uniform(0,1,128); opacity[:2]=[0,1]
data=(positions,opacity,rng.uniform(0,1,(128,3)),covariance)
for _,axes,sign in VIEWS:
    for size in (16,33,64):
        a=reference(data,axes,size,0.75,sign)
        b=render(data,axes,size,0.75,sign)
        for left,right in zip(a,b):
            np.testing.assert_allclose(left,right,rtol=1e-10,atol=1e-12)
print('Accelerated renderer agrees on 18 synthetic camera/size cases, including clipping and tied depths')
