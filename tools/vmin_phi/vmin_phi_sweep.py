#!/usr/bin/env python3
r"""Plot compressed bits/sample versus nu_min, with one curve per damping phi.

Defaults: five HYPSO captures, identical 256x256 centre crops within each scene;
nu_min=-6..2 inclusive, phi=0..5, A=0/3/15. Fixed P=3, reduced,
narrow-neighbour, nu_max=4, t_inc=128, Theta=3, Omega=19, R=64, hybrid.

Example (activated environment, from the verification repository root):
  python tools/vmin_phi/vmin_phi_sweep.py --input-dir data --workers 12

--workers N parallelizes prediction INSIDE one configuration. Configurations
run sequentially; hybrid encoding remains serial. Low-memory storage is on.
Runtime/memory are recorded in CSV but never plotted. Completed points resume.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import psutil
import _damping_core as core
import _sweep_engine as engine
import sweep_workers as sw
import wavefront_predictor as wavefront

VERSION = "1.0.0-vmin-phi"


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir",type=Path,default=Path("data"))
    select=parser.add_mutually_exclusive_group()
    select.add_argument("--images",nargs="+",choices=list(core.SCENES),help="Default: all five study images.")
    select.add_argument("--files",type=Path,nargs="+",help="Explicit NetCDF paths instead of the five-scene list.")
    parser.add_argument("--crop",type=int,nargs=2,metavar=("HEIGHT","WIDTH"),default=[256,256])
    parser.add_argument("--anchor",choices=("center","top-left"),default="center")
    parser.add_argument("--vmin-min",type=int,help="Inclusive range start; default -6.")
    parser.add_argument("--vmin-max",type=int,help="Inclusive range end; default min(2, nu_max).")
    parser.add_argument("--vmin-values",type=int,nargs="+",help="Explicit values instead of a range, e.g. -6 -3 0 2.")
    parser.add_argument("--phi-values",type=int,nargs="+",default=list(range(6)),help="Damping integers 0..7; default 0 1 2 3 4 5.")
    parser.add_argument("--a-values",type=int,nargs="+",default=[0,3,15],help="Absolute-error limits in DN; default 0 3 15.")
    parser.add_argument("--nu-max","--vmax",dest="nu_max",type=int,default=4,help="Fixed for every curve; default 4.")
    parser.add_argument("--t-inc",type=int,default=128,help="Fixed literal interval, a power of two from 16 to 2048.")
    parser.add_argument("--p",type=int,default=3)
    parser.add_argument("--prediction-mode",choices=("reduced","full"),default="reduced")
    parser.add_argument("--local-sum",choices=list(core.LOCAL_SUM_OPTIONS),default="narrow-neighbor")
    parser.add_argument("--psi",type=int,default=7,help="Offset for A>0; always zero for A=0.")
    parser.add_argument("--variable",default="products/dn")
    parser.add_argument("--axis-order",choices=("yxz","yzx","xyz","xzy","zxy","zyx"))
    storage=parser.add_mutually_exclusive_group()
    storage.add_argument("--low-memory",dest="low_memory",action="store_true",help="Default: omit hybrid text traces.")
    storage.add_argument("--standard-memory",dest="low_memory",action="store_false")
    parser.set_defaults(low_memory=True)
    parser.add_argument("--workers",type=int,default=1,help="Predictor processes per configuration, e.g. 12; one configuration at a time.")
    parser.add_argument("--rows-per-task",type=int,default=0,help="0: automatic dependency-aware row-block size.")
    parser.add_argument("--work-dir",type=Path,help="Local disk for temporary shared arrays; default OUTPUT/_work.")
    parser.add_argument("--max-memory-gb",type=float,default=90,help="Estimated job ceiling in GiB, not an OS limit.")
    parser.add_argument("--total-memory-gb",type=float,default=90,help="Planning memory ceiling in GiB.")
    parser.add_argument("--memory-poll-ms",type=int,default=50)
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--output",type=Path,default=Path("results/vmin_phi_256"))
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--plot-only",action="store_true",help="Replot saved results without loading images or running the codec.")
    args=parser.parse_args(argv)
    if not -6<=args.nu_max<=9:parser.error("--nu-max must be -6..9.")
    if args.vmin_values is not None:
        if args.vmin_min is not None or args.vmin_max is not None:
            parser.error("Use --vmin-values OR --vmin-min/--vmin-max, not both.")
        args.vmin_values=sorted(set(args.vmin_values))
    else:
        first=-6 if args.vmin_min is None else args.vmin_min
        last=min(2,args.nu_max) if args.vmin_max is None else args.vmin_max
        if first>last:parser.error("--vmin-min must be <= --vmin-max.")
        args.vmin_values=list(range(first,last+1))
    if any(not -6<=v<=args.nu_max for v in args.vmin_values):
        parser.error("Every nu_min must satisfy -6 <= nu_min <= fixed nu_max.")
    args.phi_values=sorted(set(args.phi_values));args.a_values=sorted(set(args.a_values))
    if any(not 0<=p<=7 for p in args.phi_values):parser.error("With Theta=3, phi must be 0..7.")
    if any(not 0<=a<=32767 for a in args.a_values):parser.error("A must be 0..32767 DN.")
    if any(not 2<=n<=65535 for n in args.crop):parser.error("Crop dimensions must be 2..65535.")
    if not 0<=args.p<=15 or not 0<=args.psi<=7:parser.error("Require P=0..15 and psi=0..7.")
    if args.t_inc<16 or args.t_inc>2048 or args.t_inc&(args.t_inc-1):
        parser.error("t_inc must be a power of two from 16 to 2048.")
    if args.workers<1 or args.rows_per_task<0:parser.error("Workers must be positive; rows-per-task nonnegative.")
    if not 10<=args.memory_poll_ms<=1000:parser.error("memory-poll-ms must be 10..1000.")
    if any(not math.isfinite(v) or v<=0 for v in (args.max_memory_gb,args.total_memory_gb)):
        parser.error("Memory budgets must be positive and finite.")
    args.images=list(dict.fromkeys(args.images or list(core.SCENES)))
    return args


def build_jobs(args,inputs):
    jobs=[]
    y,x=args.crop
    for scene,info in inputs.items():
        sy,sx,z=info["source_shape_yxz"]
        row,col=(0,0) if args.anchor=="top-left" else ((sy-y)//2,(sx-x)//2)
        issue="" if y<=sy and x<=sx else f"Requested {y}x{x} does not fit source {sy}x{sx}."
        for A,phi,nmin in itertools.product(args.a_values,args.phi_values,args.vmin_values):
            identity=[scene,y,x,row,col,A,phi,nmin,args.nu_max,args.t_inc]
            job_id=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
            jobs.append(dict(job_id=job_id,image=scene,size=f"{y}x{x}",repeat=1,is_full_image=0,
                             rows=y,columns=x,bands=z,crop_row=row,crop_col=col,t_inc=args.t_inc,
                             nu_min=nmin,phi=phi,a=A,
                             estimated_ram_gib=engine.estimate_gib((y,x,z),args.p,args.low_memory),
                             dimension_issue=issue))
    return jobs


def plot_results(records,metadata,out):
    if not records:return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    study=metadata["study"];profile=study["profile"]
    images=list(study["inputs"]);avals=study["a_values"];phis=study["phi_values"];vmins=study["vmin_values"]
    rows=[r["row"] for r in records]
    values={(r["image"],int(r["a"]),int(r["phi"]),int(r["nu_min"])):float(r["bits_per_sample"]) for r in rows}
    colors=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#5B4BCE","#8C564B","#666666"]
    markers=["o","s","D","^","v","P","X",">"]
    setting=(f"P={profile['p']} | {profile['local_sum']}, {profile['prediction_mode']} | "
             rf"$\nu_{{max}}={profile['nu_max']}$, $t_{{inc}}={profile['t_inc']}$, $\Theta=3$")

    def draw(selected,filename):
        nrows=len(selected);ncols=len(avals)
        fig,axes=plt.subplots(nrows,ncols,squeeze=False,figsize=(5.1*ncols,3.15*nrows+1.9))
        for i,scene in enumerate(selected):
            info=study["inputs"][scene];bands=info["source_shape_yxz"][2]
            for j,A in enumerate(avals):
                ax=axes[i,j];present=0
                for phi in phis:
                    yy=[values.get((scene,A,phi,v),np.nan) for v in vmins]
                    present+=int(np.count_nonzero(np.isfinite(yy)))
                    ax.plot(vmins,yy,color=colors[phi],marker=markers[phi],markersize=4,
                            linewidth=1.4,label=rf"$\phi={phi}$")
                expected=len(phis)*len(vmins)
                partial=f" | {present}/{expected} points" if present!=expected else ""
                ax.set_title(f"{info['label']} | A = {A} DN{partial}\n"
                             f"{study['crop'][0]} x {study['crop'][1]} pixels, {bands} bands",fontsize=10.5)
                ax.set_xlabel(r"Initial adaptation parameter, $\nu_{min}$",fontsize=10)
                ax.set_ylabel("Compressed bits/sample",fontsize=10)
                ax.set_xticks(vmins);ax.set_xlim(min(vmins)-.3,max(vmins)+.3)
                ax.ticklabel_format(axis="y",useOffset=False)
                ax.grid(True,alpha=.25);ax.set_axisbelow(True)
                if not present:ax.text(.5,.5,"No completed points",ha="center",va="center",transform=ax.transAxes)
        done=sum(r["image"] in selected for r in rows)
        total=len(selected)*len(avals)*len(phis)*len(vmins)
        fig.suptitle(f"HYPSO initial adaptation / damping study | hybrid encoder | {done}/{total}\n"+setting,fontsize=13)
        handles=[Line2D([],[],color=colors[p],marker=markers[p],linewidth=1.4,label=rf"$\phi={p}$") for p in phis]
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.035),ncol=min(8,len(phis)),frameon=False)
        fig.text(.5,.009,"Rate includes header, entropy payload/tail and word padding; external initialization excluded. Lower is better.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.10 if nrows==1 else .065,1,.84 if nrows==1 else .94),h_pad=2.3,w_pad=2.0)
        for ext in ("png","pdf"):fig.savefig(out/(filename+"."+ext),dpi=160,bbox_inches="tight")
        plt.close(fig)

    draw(images,"all_images_vmin_phi")
    for scene in images:draw([scene],scene+"_vmin_phi")
    # Descriptive minima of this grid/crop only, not a general HYPSO recommendation.
    best=[]
    for scene in images:
        for A in avals:
            group=[r for r in rows if r["image"]==scene and int(r["a"])==A]
            if group:
                r=min(group,key=lambda x:(float(x["bits_per_sample"]),int(x["phi"]),int(x["nu_min"])))
                best.append({k:r[k] for k in ("image","a","nu_min","phi","nu_max","t_inc","bits_per_sample","max_error")} |
                            {"completed_points":len(group),"expected_points":len(phis)*len(vmins)})
    engine.table(out/"best_per_image_a.csv",["image","a","nu_min","phi","nu_max","t_inc","bits_per_sample","max_error","completed_points","expected_points"],best)


def make_study(args,repo,inputs,jobs):
    own=Path(__file__).resolve().parent
    profile={k:getattr(args,k) for k in ("p","nu_max","t_inc","psi","local_sum","prediction_mode")}
    profile.update(D=16,Omega=19,R=64,Theta=3,encoder="HYBRID",psi_lossless=0)
    model_files=list((repo/"ccsds123_i2_hlm").rglob("*.py"))+list((repo/"ccsds123_i2_hlm").rglob("*.txt"))
    study=dict(tool="vmin_phi",version=VERSION,profile=profile,inputs=inputs,jobs=jobs,
               crop=args.crop,anchor=args.anchor,vmin_values=args.vmin_values,phi_values=args.phi_values,
               a_values=args.a_values,low_memory=args.low_memory,python=platform.python_version(),
               numpy=np.__version__,psutil=psutil.__version__,platform=platform.platform(),
               model_sha256=core.file_digest(model_files,repo),
               bundle_sha256=core.file_digest(list(own.glob("*.py")),own),
               reference="CCSDS 120.2-G-2, Figure 4-10 style; HYPSO crops and fixed nu_max/t_inc selected for this study.",
               metric="Header + entropy payload/tail + word padding per scalar sample. External initialization excluded; inclusive rate also saved.",
               execution="Sequential configurations, parallel predictor within each crop; hybrid encoding stays serial.",
               checks="Encoder-side reconstruction bounds and actual applied adaptation schedule; no independent decoder.",
               memory="OS peak_rss is crop coordinator only; process-tree RSS double-counts shared pages; sampled Linux PSS apportions them.")
    return {"study_id":hashlib.sha256(json.dumps(study,sort_keys=True).encode()).hexdigest(),
            "study":study,"model_path":str(repo)}


def main():
    args=parse_args();out=args.output.resolve()
    if args.plot_only:
        if not out.is_dir():raise ValueError("Output folder does not exist.")
        with sw.output_lock(out):
            metadata=json.loads((out/"metadata.json").read_text(encoding="utf-8"))
            if metadata.get("study",{}).get("tool")!="vmin_phi":raise ValueError("Not a vmin/phi study output folder.")
            records=engine.regenerate(out,metadata);plot_results(records,metadata,out)
        print(f"Plots: {out}");return 0
    if not __debug__:raise ValueError("Do not run python -O; model assertions must remain enabled.")
    repo=engine.find_repo(args.repo);sw.load_model(repo,"vmax")
    if args.workers>1:wavefront.validate_reference()
    if args.low_memory:
        from hybrid_no_traces import validate_reference
        validate_reference()
    paths=([(p.stem,p.resolve()) for p in args.files] if args.files else
           [(s,core.find_capture(args.input_dir.resolve(),core.SCENES[s])) for s in args.images])
    inputs={}
    for scene,path in paths:
        if scene in inputs:raise ValueError("Input names collide; use distinct filenames.")
        if not scene or any(ch in scene for ch in '/\\:*?"<>|'):raise ValueError("Use input names without special path characters.")
        info=engine.inspect_input(path,args)
        info["label"]=scene.title() if scene in core.SCENES else scene.replace('_',' ')
        inputs[scene]=info
    jobs=build_jobs(args,inputs);metadata=make_study(args,repo,inputs,jobs)
    limit=engine.allowed_predictor_workers(args.workers)
    print(f"{len(inputs)} images x {len(args.a_values)} A values x {len(args.phi_values)} phi values x "
          f"{len(args.vmin_values)} nu_min values = {len(jobs)} configurations.")
    print(f"nu_min={args.vmin_values}; phi={args.phi_values}; A={args.a_values}; "
          f"fixed nu_max={args.nu_max}, t_inc={args.t_inc}.")
    print(f"One configuration at a time; predictor workers requested {args.workers}, CPU allowance {limit}.")
    for scene,info in inputs.items():
        job=next(j for j in jobs if j["image"]==scene)
        _,_,pool=wavefront.task_geometry(job["rows"],job["bands"],limit,args.rows_per_task)
        need=job["estimated_ram_gib"]+(0.1*pool if limit>1 else 0)
        disk=wavefront.workspace_bytes((job["rows"],job["columns"],job["bands"]),args.p,args.prediction_mode=="full")/engine.GIB if limit>1 else 0
        reason=job["dimension_issue"]
        if not reason and need>args.max_memory_gb:reason="exceeds per-job RAM limit"
        if not reason and need+.35>engine.execution_budget(args):reason="exceeds current available/total RAM budget"
        print(f"{info['label']}: source {info['source_shape_yxz']}, crop {job['rows']}x{job['columns']} at "
              f"({job['crop_row']},{job['crop_col']}), estimated RAM {need:.2f} GiB, temporary files {disk:.2f} GiB"+
              (f"; SKIP NOW: {reason}" if reason else ""))
    if args.dry_run:
        print("Dry run: no compression or output writes.");return 0
    with sw.output_lock(out):
        meta_path=out/"metadata.json"
        if meta_path.exists():
            old=json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("study_id")!=metadata["study_id"] or old.get("study")!=metadata["study"]:
                raise ValueError("Inputs, grid, fixed settings, storage mode or code changed. Use a new output folder.")
        elif (out/"results.csv").exists() or (out/"jobs").exists():
            raise ValueError("Results exist without matching metadata. Use a new output folder.")
        (out/"jobs").mkdir(exist_ok=True)
        core.write_json(meta_path,metadata)
        engine.table(out/"plan.csv",list(jobs[0]),jobs)
        return engine.run_jobs(args,metadata,repo,out,plot_results)


if __name__=="__main__":
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Completed configurations are saved; repeat the same command to resume.")
        raise SystemExit(130)
    except (ValueError,OSError,ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr);raise SystemExit(1)
