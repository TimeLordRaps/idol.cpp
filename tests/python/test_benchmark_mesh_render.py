"""Check depth interpolation, occlusion and degenerate handling for display previews."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
import numpy as np
from benchmark_mesh_render import render_mesh

class Mesh:pass
mesh=Mesh()
mesh.vertices=np.array([[-.9,-.9,0],[.9,-.9,0],[0,.9,0],[-.9,-.9,.5],[.9,-.9,.5],[0,.9,.5]])
mesh.faces=np.array([[0,1,2],[3,4,5],[0,0,0]])
for faces in (mesh.faces,mesh.faces[::-1]):
    mesh.faces=faces
    image,depth=render_mesh(mesh,size=32,extent=1)
    assert depth[16,16]==.5 and np.all(image[16,16]<1)
    _,back=render_mesh(mesh,size=32,extent=1,sign=-1)
    assert back[16,16]==0
    assert np.isneginf(depth[0,0]) and np.all(image[0,0]==1)
mesh.vertices[:,2]=mesh.vertices[:,0]
_,depth=render_mesh(mesh,size=32,extent=1)
assert abs(depth[16,16]-1/32)<1e-12
print('Mesh display previews pass front/back occlusion, face-order, plane-depth and degenerate-triangle checks')
