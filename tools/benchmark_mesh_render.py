"""Opaque orthographic mesh previews for inspection; no metric uses these pixels."""
import numpy as np
from numba import njit


@njit(cache=True)
def _rasterize(vertices,faces,axes,size,extent,sign):
    image=np.ones((size,size,3),dtype=np.float64)
    depth=np.full((size,size),-np.inf)
    scale=size/(2*extent)
    for face in faces:
        tri=vertices[face]
        x=tri[:,axes[0]]*scale+size/2
        y=tri[:,axes[1]]*scale+size/2
        z=tri[:,axes[2]]*sign
        denominator=(y[1]-y[2])*(x[0]-x[2])+(x[2]-x[1])*(y[0]-y[2])
        if abs(denominator)<1e-12:continue
        normal=np.cross(tri[1]-tri[0],tri[2]-tri[0]); length=np.sqrt(np.sum(normal*normal))
        if length==0:continue
        normal/=length
        shade=.25+.65*abs((normal[axes[0]]*.3+normal[axes[1]]*.4+normal[axes[2]]*sign*.866025403784))
        for row in range(max(0,int(np.floor(y.min()))),min(size,int(np.ceil(y.max())))):
            for col in range(max(0,int(np.floor(x.min()))),min(size,int(np.ceil(x.max())))):
                px,py=col+.5,row+.5
                a=((y[1]-y[2])*(px-x[2])+(x[2]-x[1])*(py-y[2]))/denominator
                b=((y[2]-y[0])*(px-x[2])+(x[0]-x[2])*(py-y[2]))/denominator
                c=1-a-b
                if min(a,b,c)<-1e-10:continue
                distance=a*z[0]+b*z[1]+c*z[2]
                if distance>depth[row,col]:
                    depth[row,col]=distance
                    image[row,col,:]=shade
    return image[::-1].copy(),depth[::-1].copy()


def render_mesh(mesh,axes=(0,1,2),size=256,extent=.75,sign=1):
    vertices=np.asarray(mesh.vertices,dtype=np.float64)
    faces=np.asarray(mesh.faces,dtype=np.int64)
    if (vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all() or
            faces.ndim!=2 or faces.shape[1]!=3 or not len(faces) or faces.min()<0 or
            faces.max()>=len(vertices) or set(axes)!={0,1,2} or size<1 or
            not np.isfinite(extent) or extent<=0 or sign not in (-1,1)):
        raise ValueError('Invalid mesh preview arguments')
    return _rasterize(vertices,faces,np.asarray(axes),size,extent,sign)
