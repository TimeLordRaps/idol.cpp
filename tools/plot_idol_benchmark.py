"""Create bounded scientific figures and worst-case contact sheets from benchmark receipts."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.summary.read_text()); protocol=json.loads(args.protocol.read_text())
    if data['status']!='COMPLETE_MEASUREMENTS':
        raise ValueError('Figures require complete measurements, including explicit failures')
    args.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(16,5.2))
    for axis,key,title in [(axes[0],'mesh_mean_error_percent','Geometry agreement'),
                           (axes[1],'foreground_color_error_percent','Approximate-view appearance')]:
        for arm,color,label in [('quantization','#147d92','Matched-seed Q8 versus original'),
                                ('original_seed_variation','#b86c3a','Original versus original, different seeds')]:
            values=sorted(row[key] for item in data['per_object'] for row in item[arm] if key in row)
            if values: axis.step(values,np.arange(1,len(values)+1)/len(values),where='post',color=color,label=label)
        axis.set_xscale('symlog',linthresh=0.001)
        axis.set_xlabel('Mean mesh distance / reference diagonal (%)' if key.startswith('mesh') else 'Foreground channel absolute error (%)')
        axis.set_ylabel('Fraction of valid comparisons')
        axis.set_title(title); axis.grid(alpha=.2); axis.legend(fontsize=8,loc='lower right')
    names=list(data['strata'])
    quant=[data['strata'][name]['mesh_f1_0_1_percent']['mean'] if data['strata'][name]['mesh_f1_0_1_percent'] else np.nan for name in names]
    baseline=[]
    for name in names:
        v=[row['mesh_f1_0_1_percent'] for item in data['per_object'] if item['stratum']==name
           for row in item['original_seed_variation'] if 'mesh_f1_0_1_percent' in row]
        baseline.append(np.mean(v) if v else np.nan)
    y=np.arange(len(names))
    axes[2].barh(y-.18,quant,.35,label='Matched-seed Q8',color='#147d92')
    axes[2].barh(y+.18,baseline,.35,label='Original seed variation',color='#b86c3a')
    axes[2].set_yticks(y,[n.replace('_',' ') for n in names],fontsize=8)
    axes[2].set_xlabel('Mesh F1: harmonic mean of precision/recall\nwithin 0.1% diagonal (%)')
    axes[2].set_xlim(0,100); axes[2].set_title('Geometry/use strata'); axes[2].legend(fontsize=8)
    fig.suptitle(f"Idol mixed-precision eight-bit fidelity: {data['objects_completed']} objects, {data['matched_cases']} matched-seed cases")
    fig.text(.5,.015,'Q8 denotes the released artifact with eligible matrices in eight-bit blocks. Selected household-object challenge cohort; agreement with original Idol, not ground-truth accuracy.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,.95))
    for ext in ('png','svg'): fig.savefig(args.out/('overview.'+ext),dpi=180)
    plt.close(fig)

    objects=[]
    for row in data['per_object']:
        q=[x['mesh_mean_error_percent'] for x in row['quantization'] if 'mesh_mean_error_percent' in x]
        b=[x['mesh_mean_error_percent'] for x in row['original_seed_variation'] if 'mesh_mean_error_percent' in x]
        if q: objects.append((row,float(np.mean(q)),q,b))
    objects.sort(key=lambda row:row[1],reverse=True)
    fig,axis=plt.subplots(figsize=(12,max(6,len(objects)*.32)))
    for i,(row,mean,q,b) in enumerate(objects):
        axis.plot([min(q),max(q)],[i,i],color='#147d92',linewidth=2)
        axis.scatter([mean],[i],color='#147d92',s=22,zorder=3)
        if b: axis.scatter([np.mean(b)],[i],color='#b86c3a',marker='x',s=26,zorder=3)
    axis.set_yticks(range(len(objects)),[r[0]['name'][:43] for r in objects],fontsize=8)
    axis.invert_yaxis(); axis.set_xscale('symlog',linthresh=.001)
    axis.set_xlabel('Mean mesh distance / reference diagonal (%)')
    axis.set_title('Per-object geometry: Q8 mean and range over valid seeds; crosses show original-seed baseline mean')
    axis.grid(axis='x',alpha=.2); fig.tight_layout()
    for ext in ('png','svg'): fig.savefig(args.out/('object-geometry.'+ext),dpi=170)
    plt.close(fig)

    inputs={row['id']:row for row in protocol['cases']}
    selected=objects[:6]
    width,height=10*256,len(selected)*310+60
    canvas=Image.new('RGB',(width,height),'white'); draw=ImageDraw.Draw(canvas)
    from matplotlib import font_manager
    font=ImageFont.truetype(font_manager.findfont('DejaVu Sans'),14)
    draw.text((10,5),'Six valid-mesh objects with largest mean error; each at its worst geometry seed. Fixed views of model-coordinate x-y, x-z, y-z planes.',fill='black',font=font)
    draw.text((10,26),'Each view: original | Q8 | absolute channel difference ×5 (display only). The full report scores all six views.',fill='black',font=font)
    for index,(row,mean,q,b) in enumerate(selected):
        seed=max((x for x in row['quantization'] if 'mesh_mean_error_percent' in x),key=lambda x:x['mesh_mean_error_percent'])['seed_or_pair']
        top=60+index*310
        shown=next(x['mesh_mean_error_percent'] for x in row['quantization'] if x['seed_or_pair']==seed)
        draw.text((8,top),f"{row['name'][:55]} — object mean {mean:.4g}%; shown seed {seed}: {shown:.4g}% mesh error",fill='black',font=font)
        image=Image.open(args.protocol.parent/inputs[row['id']]['file']).convert('RGB'); image.thumbnail((250,250))
        canvas.paste(image,(3,top+35))
        for view_index,view in enumerate(('xy-front','xz-front','yz-front')):
            ref=Image.open(args.run/row['id']/('original-'+seed+'-'+view+'.png')).convert('RGB')
            quant=Image.open(args.run/row['id']/('q8-'+seed+'-'+view+'.png')).convert('RGB')
            error=Image.fromarray(np.uint8(np.clip(np.abs(np.asarray(ref,dtype=float)-np.asarray(quant,dtype=float))*5,0,255)))
            for column,image in enumerate((ref,quant,error)):
                canvas.paste(image,(256+(view_index*3+column)*256,top+35))
    canvas.save(args.out/'worst-case-views.png')
    import trimesh
    from benchmark_mesh_render import render_mesh
    mesh_canvas=Image.new('RGB',(7*256,len(selected)*310+65),'white')
    draw=ImageDraw.Draw(mesh_canvas)
    draw.text((10,5),'Same six worst-mean objects and seeds: input | x-y original / Q8 | x-z original / Q8 | y-z original / Q8',fill='black',font=font)
    draw.text((10,27),'Opaque shaded meshes, fixed orthographic extent ±0.75 model units. Display only; metrics use original triangle coordinates.',fill='black',font=font)
    for index,(row,mean,q,b) in enumerate(selected):
        case=max((x for x in row['quantization'] if 'mesh_mean_error_percent' in x),key=lambda x:x['mesh_mean_error_percent'])
        seed=case['seed_or_pair']; top=65+index*310
        draw.text((8,top),f"{row['name'][:65]} — seed {seed}; mean mesh error {case['mesh_mean_error_percent']:.4g}% of diagonal",fill='black',font=font)
        with Image.open(args.protocol.parent/inputs[row['id']]['file']) as original:
            thumbnail=original.convert('RGB'); thumbnail.thumbnail((250,250)); mesh_canvas.paste(thumbnail,(3,top+35))
        receipt=json.loads((args.run/row['id']/'result.json').read_text())
        for column,label in enumerate(('original-'+seed,'q8-'+seed)):
            raw=gzip.open(args.run/row['id']/(label+'.obj.gz'),'rb').read()
            assert hashlib.sha256(raw).hexdigest()==receipt['generations'][label]['mesh']['mesh_sha256']
            mesh=trimesh.load_mesh(io.BytesIO(raw),file_type='obj',process=False,skip_materials=True)
            for view_index,axes in enumerate(((0,1,2),(0,2,1),(1,2,0))):
                pixels,_=render_mesh(mesh,axes)
                preview=Image.fromarray(np.uint8(np.clip(pixels*255+.5,0,255)))
                mesh_canvas.paste(preview,(256+(view_index*2+column)*256,top+35))
    mesh_canvas.save(args.out/'worst-case-meshes.png')
    print('Saved overview, object-level geometry, worst-case appearance and shaded mesh contact sheets')


if __name__=='__main__':main()
