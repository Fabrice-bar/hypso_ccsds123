#!/usr/bin/env python3
"""Recreate the Green Book Figure 4-21 experiment with HYPSO captures.

Train on a 16-row strip immediately above each test strip IN THE SAME CAPTURE.
Default pilot: 5 images x 2 locations x 3 test heights x 2 methods = 60 tests,
plus 10 reusable training compressions. Full native width and bands by default.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import numpy as np
import psutil
import _study_support as core
import sweep_workers as sw

VERSION = "1.0.0-fig421"
GIB = 1024**3
ALIASES = {"lines":"y","line":"y","frames":"y","frame":"y","rows":"y","y":"y",
           "samples":"x","sample":"x","columns":"x","cols":"x","x":"x",
           "bands":"z","band":"z","wavelength":"z","wavelengths":"z","z":"z"}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024**2), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.asarray(a,dtype="<i8").tobytes()).hexdigest()


def csv_table(path, rows, first=()):
    fields = list(first)
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    temp = path.with_suffix(path.suffix+".tmp")
    with temp.open("w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=fields)
        w.writeheader();w.writerows(rows)
    os.replace(temp,path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir",type=Path,default=Path("data"))
    group=p.add_mutually_exclusive_group()
    group.add_argument("--images",nargs="+",choices=list(core.SCENES),default=list(core.SCENES))
    group.add_argument("--files",nargs="+",type=Path,help="Alternative NetCDF captures; each is trained separately.")
    p.add_argument("--heights",type=int,nargs="+",default=[16,64,256])
    p.add_argument("--train-height",type=int,default=16)
    group=p.add_mutually_exclusive_group()
    group.add_argument("--full-width",action="store_true",help="All native columns (default).")
    group.add_argument("--width",type=int,help="Smaller shared training/test width for a quick check.")
    group=p.add_mutually_exclusive_group()
    group.add_argument("--locations",type=int,default=2,help="Locations per image, spread from top to bottom.")
    group.add_argument("--train-rows",type=int,nargs="+",help="Explicit zero-based training start rows.")
    p.add_argument("--col",type=int,help="First column with --width; default centre.")
    p.add_argument("--p",type=int,default=3)
    p.add_argument("--local-sum",choices=list(core.LOCAL_SUM_OPTIONS),default="wide-neighbor")
    p.add_argument("--prediction-mode",choices=["full","reduced"],default="full")
    p.add_argument("--omega",type=int,default=19)
    p.add_argument("--q",type=int,help="Default floor(Omega/2); changing this is a different experiment.")
    p.add_argument("--nu-max",type=int,default=3)
    p.add_argument("--t-inc",type=int,default=64)
    p.add_argument("--default-vmin",type=int,default=-1)
    p.add_argument("--custom-vmin",type=int,default=3)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--max-memory-gb",type=float,default=90,help="Conservative per-job estimate limit, GiB.")
    p.add_argument("--total-memory-gb",type=float,default=90,help="Concurrent worker estimate budget, GiB.")
    p.add_argument("--variable",default="products/dn")
    p.add_argument("--axis-order",choices=["yxz","yzx","xyz","xzy","zxy","zyx"])
    p.add_argument("--repo",type=Path)
    p.add_argument("--output",type=Path,default=Path("results/fig421"))
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--plot-only",action="store_true")
    a=p.parse_args(argv)
    a.heights=sorted(set(a.heights));a.q=a.omega//2 if a.q is None else a.q
    a.images=list(dict.fromkeys(a.images))
    if a.train_rows is not None:
        a.train_rows=sorted(set(a.train_rows));a.locations=len(a.train_rows)
    if min(a.heights)<1 or a.train_height<1 or a.locations<1:
        p.error("Heights and number of locations must be positive.")
    if not (-6<=min(a.default_vmin,a.custom_vmin)<=max(a.default_vmin,a.custom_vmin)<=a.nu_max<=9):
        p.error("Require -6 <= both nu_min values <= nu_max <= 9.")
    if not (16<=a.t_inc<=2048 and a.t_inc & (a.t_inc-1)==0):
        p.error("t-inc must be a power of two from 16 to 2048.")
    if not (1<=a.p<=15 and 4<=a.omega<=19 and 3<=a.q<=a.omega+3):
        p.error("Require P=1..15, Omega=4..19, Q=3..Omega+3.")
    if a.width is not None and a.width<2:
        p.error("Width must be at least two.")
    if any(x is not None and x<0 for x in [a.col]+(a.train_rows or [])):
        p.error("Row/column origins cannot be negative.")
    if a.width is None and a.col not in (None,0):
        p.error("Full-width strips start at column zero; --col requires --width.")
    if a.workers<1 or not all(math.isfinite(x) and x>0 for x in [a.max_memory_gb,a.total_memory_gb]):
        p.error("Workers and memory budgets must be positive.")
    return a


def find_repo(explicit):
    for root in ([explicit] if explicit else [Path.cwd(),*Path(__file__).resolve().parents]):
        if (root/"ccsds123_i2_hlm/header.py").is_file():
            return root.resolve()
    raise ValueError("Run from the verification-model root or supply --repo PATH.")


def estimate(shape,p):
    return math.prod(shape)*(320+16*(p+3))/GIB+.5


def inspect_input(path,args):
    with core.open_netcdf(path) as ds:
        v=ds[args.variable]
        order=args.axis_order or "".join(ALIASES.get(d.lower(),"?") for d in v.dimensions)
        if v.ndim!=3 or np.dtype(v.dtype).kind not in "iu" or len(order)!=3 or set(order)!=set("yxz"):
            raise ValueError(f"Expected integer 3-D DN in {path}; dimensions={v.dimensions}. Specify --axis-order if needed.")
        if float(getattr(v,"scale_factor",1))!=1 or float(getattr(v,"add_offset",0))!=0:
            raise ValueError("Use original unscaled integer DN.")
        shape=[int(v.shape[order.index(k)]) for k in "yxz"]
        if min(shape)<1 or max(shape)>65535:
            raise ValueError(f"Unsupported dimensions: {shape}.")
        attrs={k:str(ds.getncattr(k)) for k in ("sat_id","instrument","processing_level","timestamp_acquired") if k in ds.ncattrs()}
    st=path.stat()
    return dict(file=str(path),shape_yxz=shape,axis_order=order,file_bytes=st.st_size,
                file_mtime_ns=st.st_mtime_ns,file_sha256=sha_file(path),attributes=attrs)


def build_study(a,repo):
    paths=([(p.stem,p.resolve()) for p in a.files] if a.files else
           [(s,core.find_capture(a.input_dir.resolve(),core.SCENES[s])) for s in a.images])
    if len({s for s,_ in paths})!=len(paths):
        raise ValueError("Input names collide.")
    inputs={s:inspect_input(p,a) for s,p in paths}
    if len({d['file_sha256'] for d in inputs.values()})!=len(inputs):
        raise ValueError("Duplicate input captures detected.")
    jobs=[]
    for i,(name,d) in enumerate(inputs.items()):
        y,x,z=d['shape_yxz'];width=a.width or x
        col=a.col if a.col is not None else (x-width)//2
        last=y-a.train_height-max(a.heights)
        starts=a.train_rows or ([last//2] if a.locations==1 else np.rint(np.linspace(0,last,a.locations)).astype(int).tolist())
        if min(starts+[col])<0 or max(starts)>last or col+width>x or len(set(starts))!=len(starts):
            raise ValueError(f"Training/test regions do not fit {name} ({y}x{x}). Reduce heights/locations or set valid --train-rows.")
        if min([a.train_height]+a.heights)*width<3:
            raise ValueError("Each strip must contain at least three spatial samples.")
        d.update(label=name.split('_')[0].title(),width=width,col=col,training_starts=starts,
                 locations_overlap=any(b<c+a.train_height+max(a.heights) for c,b in zip(starts,starts[1:])))
        for loc,start in enumerate(starts):
            key=f"i{i:02d}_l{loc:02d}"
            base=dict(image=name,location=loc,training_id=key+'_training',columns=width,bands=z,crop_col=col)
            jobs.append(dict(base,job_id=key+'_training',stage='training',initialization='default',
                             nu_min=a.default_vmin,rows=a.train_height,crop_row=start,estimated_ram_gib=estimate((a.train_height,width,z),a.p)))
            for h in a.heights:
                for method,nu in [('default',a.default_vmin),('custom',a.custom_vmin)]:
                    jobs.append(dict(base,job_id=f'{key}_h{h}_{method}',stage='test',initialization=method,
                                     nu_min=nu,rows=h,crop_row=start+a.train_height,estimated_ram_gib=estimate((h,width,z),a.p)))
    profile={k:getattr(a,k) for k in ('p','local_sum','prediction_mode','omega','q','nu_max','t_inc','variable')}
    profile.update(a=0,a_values=[0],phi=0,phi_values=[0],psi=0,D=16,R=64,encoder='sample-adaptive',
                   Umax=18,gamma_star=6,gamma_0=1,K=3,weight_exponent_offsets='ALL_ZERO')
    own=Path(__file__).resolve().parent
    source=list((repo/'ccsds123_i2_hlm').rglob('*.py'))+list((repo/'ccsds123_i2_hlm').rglob('*.txt'))
    study=dict(tool='fig421_weight_init',version=VERSION,profile=profile,inputs=inputs,jobs=jobs,
               heights=a.heights,train_height=a.train_height,locations=a.locations,full_width=a.width is None,
               default_vmin=a.default_vmin,custom_vmin=a.custom_vmin,
               design='Adjacent training/test strips within the same capture. Heights are nested prefixes at each location; report mean percentage reduction across locations.',
               weight_transfer='Q signed MSBs of post-final-training adaptive weights. All other predictor and coder state reset.',
               model_sha256=core.file_digest(source,repo),bundle_sha256=core.file_digest(list(own.glob('*.py')),own),
               python=platform.python_version(),numpy=np.__version__)
    return dict(study_id=hashlib.sha256(json.dumps(study,sort_keys=True).encode()).hexdigest(),study=study)


def active_components(bands,p,full):
    n=p+(3 if full else 0)
    return np.arange(n)[None,:]<(np.minimum(np.arange(bands),p)+(3 if full else 0))[:,None]


def quantize(weights,omega,q):
    """Blue Book 4.6.3.3: signed Q MSBs, prescribed 0 followed by 1s suffix."""
    shift=omega+3-q
    table=np.asarray(weights,dtype=np.int64)>>shift
    suffix=(1<<(shift-1))-1 if shift else 0
    initial=(table<<shift)+suffix
    if np.any(table<-(1<<(q-1))) or np.any(table>=(1<<(q-1))):
        raise ValueError("Weights exceed signed Q-bit range.")
    return table,initial


def post_sample_weights(pred,y,x):
    """The reference updates lazily: export state AFTER the final sample."""
    if np.any(pred.weight_exponent_offset!=0):
        raise ValueError("This experiment requires zero weight exponent offsets.")
    rho=int(pred.weight_update_scaling_exponent[y*pred.header.x_size+x])
    sign=np.where(pred.double_resolution_prediction_error[y,x]>=0,1,-1)
    n=sign[:,None]*pred.local_difference_vector[y,x]
    scaled=n>>rho if rho>=0 else n<<(-rho)
    return np.clip(pred.weight_vector[y,x]+(scaled+1)//2,pred.weight_min,pred.weight_max)


def worker(spec_path):
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    s=json.loads(Path(spec_path).read_text(encoding='utf-8'))
    job,src=s['job'],s['source']
    a=SimpleNamespace(**s['profile'],nu_min=job['nu_min'],low_memory=True,max_memory_gb=s['max_memory_gb'],
                      full_image=False,crop=(job['rows'],job['columns']),row=job['crop_row'],col=job['crop_col'],axis_order=src['axis_order'])
    model=sw.load_model(Path(s['repo']),'vmax');hd=model[0]
    st=Path(src['file']).stat()
    if (st.st_size,st.st_mtime_ns)!=(src['file_bytes'],src['file_mtime_ns']):
        raise ValueError("Input changed after planning.")
    cube,info=core.read_cube(Path(src['file']),a)
    h=core.make_header(cube,a,0,a.nu_max,hd)
    h.weight_component_resolution=a.omega-4
    mask=active_components(h.z_size,a.p,a.prediction_mode=='full')
    table=initial=None
    if job['initialization']=='custom':
        trained=s['table']
        if (trained['image'],trained['location'],trained['training_end_row'])!=(job['image'],job['location'],job['crop_row']):
            raise ValueError("Training must immediately precede this test strip in the same image.")
        table,initial=quantize(trained['final_weights'],a.omega,a.q)
        table[~mask]=0
        if array_sha(table)!=trained['lambda_sha256']:
            raise ValueError("Weight-table checksum mismatch.")
        h.weight_init_method=hd.WeightInitMethod.CUSTOM
        h.weight_init_table_flag=hd.WeightInitTableFlag.INCLUDED
        h.weight_init_resolution=a.q
        h.weight_init_table=table.copy()
    h.check_legal_config()
    hb,optional=h.get_header_bitstreams()
    if len(optional):
        raise ValueError("Unexpected external metadata.")
    hp=Path(s['result']).with_suffix('.header.bin')
    with hp.open('wb') as f: hb.tofile(f)
    decoded=hd.Header();decoded.set_config_from_file(str(hp))
    if table is not None and (decoded.weight_init_resolution!=a.q or not np.array_equal(decoded.weight_init_table[mask],table[mask])):
        raise RuntimeError("Weight table failed header serialization round-trip.")
    captured={}
    class ObservedPredictor(model[2].Predictor):
        def run_predictor(self):
            super().run_predictor()
            if initial is not None and not np.array_equal(self.weight_vector[0,1][mask],initial[mask]):
                raise RuntimeError("Model did not apply the requested initial weights.")
            py,px=divmod(self.header.y_size*self.header.x_size-2,self.header.x_size)
            if not np.array_equal(post_sample_weights(self,py,px)[mask],self.weight_vector[-1,-1][mask]):
                raise RuntimeError("Weight export arithmetic differs from the reference update.")
            if job['stage']=='training':
                w=post_sample_weights(self,self.header.y_size-1,self.header.x_size-1)
                w[~mask]=0
                captured.update(final_weights=w.tolist(),final_weights_sha256=array_sha(w))
    observed=(model[0],model[1],SimpleNamespace(Predictor=ObservedPredictor),model[3])
    row=core.encode_one(cube,h,a,0,a.nu_max,0,observed)
    if row['header_bits']!=len(hb):
        raise RuntimeError("Header changed during compression.")
    table_bits=int(mask.sum())*a.q if table is not None else 0
    if table is not None:
        base=copy.deepcopy(h)
        base.weight_init_method=hd.WeightInitMethod.DEFAULT
        base.weight_init_table_flag=hd.WeightInitTableFlag.NOT_INCLUDED
        base.weight_init_resolution=0;base.weight_init_table=None
        bb,_=base.get_header_bitstreams()
        if len(hb)-len(bb)!=8*((table_bits+7)//8):
            raise RuntimeError("Custom weight-table overhead is not correctly counted.")
    row.update(job_id=job['job_id'],image=job['image'],location=job['location'],stage=job['stage'],
               initialization=job['initialization'],training_id=job['training_id'],omega=a.omega,q=a.q if table is not None else 0,
               crop_row=job['crop_row'],crop_col=job['crop_col'],crop_sha256=info['crop_sha256'],
               weight_table_payload_bits=table_bits,weight_table_padded_bits=8*((table_bits+7)//8),
               worker_pid=os.getpid(),predictor_workers=1,initialization_verified=1,header_roundtrip_verified=1,
               export_update_verified=1,header_sha256=sha_file(hp))
    core.write_json(Path(s['result']),dict(study_id=s['study_id'],row=row,crop=info,weights=captured,header=core.header_snapshot(h)))


def records(out,meta):
    result=[]
    for job in meta['study']['jobs']:
        p=out/'jobs'/(job['job_id']+'.json')
        if p.exists():
            rec=json.loads(p.read_text(encoding='utf-8'))
            if rec['study_id']!=meta['study_id'] or rec['row']['job_id']!=job['job_id']:
                raise ValueError(f"Saved result identity mismatch: {p}.")
            result.append(rec)
    return result


def make_tables(out,meta):
    p=meta['study']['profile'];tables={}
    training=[r for r in records(out,meta) if r['row']['stage']=='training']
    if len(training)!=sum(j['stage']=='training' for j in meta['study']['jobs']):
        raise ValueError("Training incomplete; repeat the same command to resume.")
    for rec in training:
        row=rec['row'];w=np.array(rec['weights']['final_weights'],dtype=np.int64)
        if array_sha(w)!=rec['weights']['final_weights_sha256']:
            raise ValueError("Saved training weights checksum mismatch.")
        table,initial=quantize(w,p['omega'],p['q'])
        mask=active_components(row['bands'],p['p'],p['prediction_mode']=='full');table[~mask]=0
        result=dict(image=row['image'],location=row['location'],training_end_row=row['crop_row']+row['rows'],
                    training_crop_sha256=row['crop_sha256'],final_weights=w.tolist(),lambda_sha256=array_sha(table),
                    initial_weights=initial.tolist(),lambda_values=table.tolist(),q=p['q'],omega=p['omega'])
        tables[row['job_id']]=result
        core.write_json(out/'weights'/(row['job_id']+'.json'),result)
    return tables


def comparisons(rows,study):
    lookup={(r['image'],r['location'],r['rows'],r['initialization']):r for r in rows if r['stage']=='test'}
    paired=[];averages=[]
    for image in study['inputs']:
        for height in study['heights']:
            group=[]
            for loc in range(study['locations']):
                d=lookup.get((image,loc,height,'default'));c=lookup.get((image,loc,height,'custom'))
                if d is None or c is None: continue
                for k in ('crop_sha256','sample_count','columns','bands','a','nu_max','t_inc','p','local_sum','prediction_mode','omega','encoder'):
                    if d[k]!=c[k]: raise RuntimeError(f"Comparison mismatch: {image}, {height}, {k}.")
                item=dict(image=image,location=loc,height=height,default_bps=d['bits_per_sample'],custom_bps=c['bits_per_sample'],
                          rate_reduction_percent=100*(d['stream_bits']-c['stream_bits'])/d['stream_bits'],
                          default_bits=d['stream_bits'],custom_bits=c['stream_bits'],custom_weight_table_bits=c['weight_table_padded_bits'])
                group.append(item);paired.append(item)
            if group:
                v=[r['rate_reduction_percent'] for r in group]
                db=float(np.mean([r['default_bps'] for r in group]));cb=float(np.mean([r['custom_bps'] for r in group]))
                averages.append(dict(image=image,height=height,locations_completed=len(group),locations_planned=study['locations'],
                    mean_rate_reduction_percent=float(np.mean(v)),min_reduction_percent=min(v),max_reduction_percent=max(v),
                    default_mean_bps=db,custom_mean_bps=cb,ratio_of_mean_rates_reduction_percent=100*(db-cb)/db))
    return paired,averages


def save_metrics(out,meta):
    rows=[r['row'] for r in records(out,meta)]
    prefix=('image','location','rows','initialization','nu_min','bits_per_sample')
    csv_table(out/'results.csv',[r for r in rows if r['stage']=='test'],prefix)
    csv_table(out/'training_results.csv',[r for r in rows if r['stage']=='training'],prefix)
    pairs,means=comparisons(rows,meta['study'])
    csv_table(out/'comparisons.csv',pairs,('image','location','height','rate_reduction_percent'))
    csv_table(out/'fig421_summary.csv',means,('image','height','mean_rate_reduction_percent','locations_completed'))
    return rows,means


def plot_results(out,meta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows,means=save_metrics(out,meta)
    if not means: return
    s=meta['study'];p=s['profile'];images=list(s['inputs'])
    planned=sum(j['stage']=='test' for j in s['jobs']);done=sum(r['stage']=='test' for r in rows)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
    subtitle=(f"Lossless | P={p['p']} | {p['local_sum']}, {p['prediction_mode']} | sample-adaptive\n"
              f"Omega={p['omega']}, Q={p['q']}, nu_max={p['nu_max']}, t_inc={p['t_inc']} | training: {s['train_height']} rows")
    def configure(ax):
        ax.set_xscale('log',base=2);ax.set_xticks(s['heights']);ax.set_xticklabels([str(x) for x in s['heights']])
        ax.set_xlabel('Test-image height (rows)');ax.grid(alpha=.22)
    def save(fig,name,title,handles,labels):
        fig.suptitle(title+'\n'+subtitle,fontsize=11,y=.985)
        fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.072),ncol=min(5,len(labels)),fontsize=10)
        fig.text(.5,.015,f"Mean over {s['locations']} locations per image. Adjacent training/test strips; "
                 f"{'full native width' if s['full_width'] else 'selected width'} and all native bands.\n"
                 "Rate includes custom weight table, header, payload and padding. Hollow points: incomplete location average.",
                 ha='center',va='bottom',fontsize=8)
        fig.tight_layout(rect=(0,.18,1,1));fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,7));handles=[];labels=[]
    for i,image in enumerate(images):
        d={r['height']:r for r in means if r['image']==image}
        line,=ax.plot(s['heights'],[d.get(h,{}).get('mean_rate_reduction_percent',np.nan) for h in s['heights']],
                      marker='o',color=colors[i%len(colors)],label=s['inputs'][image]['label'])
        partial=[h for h in s['heights'] if h in d and d[h]['locations_completed']<s['locations']]
        if partial: ax.scatter(partial,[d[h]['mean_rate_reduction_percent'] for h in partial],facecolors='white',edgecolors=line.get_color(),zorder=4)
        handles.append(line);labels.append(line.get_label())
    configure(ax);ax.axhline(0,color='0.35',linewidth=.8)
    ax.set_ylabel('Average compressed data rate reduction (%)')
    ax.set_title(f"Custom initialization (nu_min={s['custom_vmin']}) versus default (nu_min={s['default_vmin']})\nLarger values are better",fontsize=10)
    save(fig,'fig421_rate_reduction',f'HYPSO custom weight initialization | Figure 4-21 experiment | {done}/{planned} tests',handles,labels)
    nc=min(3,len(images));nr=math.ceil(len(images)/nc)
    fig,axes=plt.subplots(nr,nc,figsize=(5.2*nc,3.5*nr+1.8),squeeze=False);handles=[];labels=[]
    for i,image in enumerate(images):
        ax=axes.flat[i];d={r['height']:r for r in means if r['image']==image}
        for method,key,color in [('Default','default_mean_bps',colors[0]),('Custom','custom_mean_bps',colors[1])]:
            line,=ax.plot(s['heights'],[d.get(h,{}).get(key,np.nan) for h in s['heights']],marker='o',color=color,label=method)
            if i==0: handles.append(line);labels.append(method)
        info=s['inputs'][image]
        ax.set_title(f"{info['label']} | width {info['width']}, {info['shape_yxz'][2]} bands",fontsize=10)
        configure(ax);ax.set_ylabel('Mean compressed bits/sample')
    for ax in list(axes.flat)[len(images):]: ax.set_visible(False)
    save(fig,'rates_by_height',f'HYPSO mean rates | {done}/{planned} tests',handles,labels)


def cpu_allowance(n):
    try: count=len(os.sched_getaffinity(0))
    except AttributeError: count=os.cpu_count() or 1
    slurm=os.environ.get('SLURM_CPUS_PER_TASK','')
    if slurm.isdecimal(): count=min(count,max(1,int(slurm)))
    return min(n,count)


def stop_process(proc):
    if proc.poll() is not None: return
    try:
        if os.name=='posix': os.killpg(proc.pid,signal.SIGTERM)
        else: proc.terminate()
        proc.wait(timeout=3)
    except ProcessLookupError: pass
    except subprocess.TimeoutExpired:
        try:
            if os.name=='posix': os.killpg(proc.pid,signal.SIGKILL)
            else: proc.kill()
        except ProcessLookupError: pass
        proc.wait()


def run_phase(stage,a,meta,repo,out,tables=None):
    jobs=[j for j in meta['study']['jobs'] if j['stage']==stage]
    complete={r['row']['job_id'] for r in records(out,meta)}
    pending=[j for j in jobs if j['job_id'] not in complete]
    active={};failures=[];peak=0;limit=cpu_allowance(a.workers);waiting=False
    try:
        while pending or active:
            while pending and len(active)<limit:
                j=pending[0];need=j['estimated_ram_gib']
                reserved=sum(max(v['job']['estimated_ram_gib'],v['rss']/GIB) for v in active.values())
                remaining=sum(max(0,v['job']['estimated_ram_gib']-v['rss']/GIB) for v in active.values())
                if need+reserved+.35>a.total_memory_gb or need+remaining+.35>sw._available_memory():
                    if not active: raise ValueError(f"Insufficient available RAM/budget for estimate {need:.2f} GiB. Completed jobs are saved.")
                    if not waiting: print(f"RAM budget: {len(active)} active workers; remaining jobs queued.",flush=True);waiting=True
                    break
                pending.pop(0);stem=out/'jobs'/j['job_id'];result=stem.with_suffix('.worker.json');result.unlink(missing_ok=True)
                spec=dict(study_id=meta['study_id'],job=j,source=meta['study']['inputs'][j['image']],profile=meta['study']['profile'],
                          max_memory_gb=a.max_memory_gb,repo=str(repo),result=str(result))
                if j['initialization']=='custom': spec['table']=tables[j['training_id']]
                sp=stem.with_suffix('.spec.json');core.write_json(sp,spec)
                log=stem.with_suffix('.log').open('w',encoding='utf-8');env=dict(os.environ)
                for key in sw._NATIVE_THREADS: env[key]='1'
                try:
                    proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--_worker',str(sp)],cwd=repo,
                                          stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=os.name=='posix')
                except BaseException: log.close();raise
                active[j['job_id']]=dict(proc=proc,job=j,log=log,result=result,rss=0)
                peak=max(peak,len(active));waiting=False
                print(f"START {j['job_id']}: {j['image']} {j['rows']}x{j['columns']}x{j['bands']}, active={len(active)}/{limit}",flush=True)
            for key,v in list(active.items()):
                try: v['rss']=psutil.Process(v['proc'].pid).memory_info().rss
                except (psutil.NoSuchProcess,psutil.AccessDenied): v['rss']=0
                code=v['proc'].poll()
                if code is None: continue
                v['log'].close();del active[key];waiting=False
                if code or not v['result'].exists():
                    failures.append(dict(job_id=key,reason=f'Exit {code}; inspect jobs/{key}.log'))
                    print(f'FAILED {key}; inspect jobs/{key}.log',flush=True);continue
                rec=json.loads(v['result'].read_text(encoding='utf-8'))
                if rec['study_id']!=meta['study_id']: raise RuntimeError('Worker returned mismatched study identity.')
                core.write_json(out/'jobs'/(key+'.json'),rec);v['result'].unlink();complete.add(key)
                save_metrics(out,meta)
                count=sum(j['job_id'] in complete for j in jobs)
                print(f"DONE {stage} {count}/{len(jobs)}: {key}, {rec['row']['bits_per_sample']:.6f} bits/sample, max error={rec['row']['max_error']}",flush=True)
            if active: time.sleep(.1)
    finally:
        for v in active.values(): stop_process(v['proc']);v['log'].close()
        csv_table(out/(stage+'_failures.csv'),failures,('job_id','reason'))
        core.write_json(out/(stage+'_execution.json'),dict(workers_requested=a.workers,peak_concurrent_workers=peak,
                        execution='one complete configuration per worker; original serial predictor'))
        save_metrics(out,meta)
    return not failures


def main():
    a=parse_args();out=a.output.resolve()
    if a.plot_only:
        if not (out/'metadata.json').is_file(): raise ValueError('Missing metadata.json in output directory.')
        with sw.output_lock(out):
            meta=json.loads((out/'metadata.json').read_text(encoding='utf-8'))
            if meta['study']['tool']!='fig421_weight_init': raise ValueError('Not a Figure 4-21 study.')
            plot_results(out,meta)
        return 0
    if not __debug__: raise ValueError('Do not use python -O; reference assertions must remain enabled.')
    repo=find_repo(a.repo);sw.load_model(repo,'vmax');meta=build_study(a,repo);jobs=meta['study']['jobs']
    for j in jobs:
        if j['estimated_ram_gib']>a.max_memory_gb or j['estimated_ram_gib']+.35>a.total_memory_gb:
            raise ValueError(f"{j['job_id']} estimate {j['estimated_ram_gib']:.2f} GiB exceeds the memory budget.")
    n=sum(j['stage']=='test' for j in jobs)
    print(f"{len(meta['study']['inputs'])} images x {len(a.heights)} heights x {a.locations} locations x 2 methods = {n} TEST compressions; plus {len(jobs)-n} TRAINING runs.")
    print(f"Lossless; sample-adaptive; P={a.p}, {a.local_sum}/{a.prediction_mode}, Omega={a.omega}, Q={a.q}, nu_max={a.nu_max}, t_inc={a.t_inc}.")
    print(f"Default/training nu_min={a.default_vmin}; custom nu_min={a.custom_vmin}. One config per worker; requested {a.workers}, CPU allowance {cpu_allowance(a.workers)}.")
    for name,d in meta['study']['inputs'].items():
        for loc,tr in enumerate(d['training_starts']):
            print(f"{name} location {loc}: native {d['shape_yxz']}; train rows [{tr}:{tr+a.train_height}), test starts {tr+a.train_height}; columns [{d['col']}:{d['col']+d['width']}).")
        if d['locations_overlap']: print(f'Note: different {name} locations overlap; these are descriptive, not independent replicates.')
    if a.dry_run: print('Dry run: no compression or output writes.');return 0
    with sw.output_lock(out):
        path=out/'metadata.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=meta:
            raise ValueError('Inputs, grid, profile or code changed. Use a new output directory.')
        if not path.exists() and ((out/'jobs').exists() or (out/'results.csv').exists()):
            raise ValueError('Results exist without matching metadata. Use a new output directory.')
        (out/'jobs').mkdir(exist_ok=True);(out/'weights').mkdir(exist_ok=True)
        core.write_json(path,meta);csv_table(out/'plan.csv',jobs)
        try:
            if not run_phase('training',a,meta,repo,out): return 1
            tables=make_tables(out,meta)
            if not run_phase('test',a,meta,repo,out,tables): return 1
        finally: plot_results(out,meta)
        print(f'Finished. Results: {out}')
        print('Error checks use encoder-side reconstruction, not an independent decoder.')
    return 0


if __name__=='__main__':
    try:
        if len(sys.argv)==3 and sys.argv[1]=='--_worker': worker(sys.argv[2])
        else: raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Completed jobs saved; repeat the same command to resume.');raise SystemExit(130)
    except Exception as exc:
        if '--_worker' in sys.argv: traceback.print_exc()
        else: print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(1)
