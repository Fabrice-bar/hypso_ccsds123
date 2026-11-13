#!/usr/bin/env python3
r"""HYPSO Figure 4-12-style sample-representative offset study.

Plot bits/sample against psi=0..7, with phi=0..5 curves and image/A panels.
Defaults: all five scenes, 128x128 centre crops, A=3/15 only, t_inc=512,
nu_min=-1 and nu_max=7-phi. P=3, reduced, narrow-neighbour, hybrid;
D=16, R=64, Omega=19 and Theta=3. All native raw DN bands are retained.

One worker runs one COMPLETE sequential predictor+encoder configuration.
--workers 12 allows up to 12 configurations concurrently, subject to CPU/RAM.
Use --crop 256 256 for the larger study and a different output folder.
Lossless A=0 is intentionally excluded; A>15 is rejected.
Runtime/memory are recorded, not plotted. Completed configurations resume.
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

VERSION = "1.0.0-offset-phi-config-workers"


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir",type=Path,default=Path("data"))
    select=parser.add_mutually_exclusive_group()
    select.add_argument("--images",nargs="+",choices=list(core.SCENES),help="Default: all five study scenes.")
    select.add_argument("--files",type=Path,nargs="+",help="Explicit NetCDF files instead of scene names.")
    parser.add_argument("--crop",type=int,nargs=2,default=[128,128],metavar=("HEIGHT","WIDTH"))
    parser.add_argument("--anchor",choices=("center","top-left"),default="center")
    parser.add_argument("--psi-values",type=int,nargs="+",default=list(range(8)),help="Offset values; default all integers 0..7.")
    parser.add_argument("--phi-values",type=int,nargs="+",default=list(range(6)),help="Damping curves; default 0..5. Use 3 4 for a focused run.")
    parser.add_argument("--a-values",type=int,nargs="+",default=[3,15],help="Near-lossless absolute limits in DN: 1..15; default 3 15.")
    parser.add_argument("--t-inc",type=int,default=512,help="FIXED interval for the whole study; power of two 16..2048. Default 512.")
    parser.add_argument("--nu-min",type=int,default=-1)
    parser.add_argument("--p",type=int,default=3)
    parser.add_argument("--prediction-mode",choices=("reduced","full"),default="reduced")
    parser.add_argument("--local-sum",choices=list(core.LOCAL_SUM_OPTIONS),default="narrow-neighbor")
    parser.add_argument("--variable",default="products/dn")
    parser.add_argument("--axis-order",choices=("yxz","yzx","xyz","xzy","zxy","zyx"))
    storage=parser.add_mutually_exclusive_group()
    storage.add_argument("--low-memory",dest="low_memory",action="store_true",help="Default: omit hybrid text traces.")
    storage.add_argument("--standard-memory",dest="low_memory",action="store_false")
    parser.set_defaults(low_memory=True)
    parser.add_argument("--workers",type=int,default=2,help="Concurrent COMPLETE configurations; default 2. Each worker has one native thread.")
    parser.add_argument("--max-memory-gb",type=float,default=90,help="Estimated per-job ceiling, GiB.")
    parser.add_argument("--total-memory-gb",type=float,default=90,help="Estimated active-job budget, GiB; RAM availability may reduce concurrency.")
    parser.add_argument("--memory-poll-ms",type=int,default=50)
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--output",type=Path,help="Default results/offset_phi_HEIGHTxWIDTH_tincN.")
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--plot-only",action="store_true",help="Replot saved results without encoding.")
    args=parser.parse_args(argv)
    for key in ("psi_values","phi_values","a_values"):
        setattr(args,key,sorted(set(getattr(args,key))))
    if any(not 0<=p<=7 for p in args.psi_values):parser.error("With Theta=3, psi must be 0..7.")
    if any(not 0<=p<=7 for p in args.phi_values):parser.error("With Theta=3, phi must be 0..7.")
    if any(not 1<=a<=15 for a in args.a_values):
        parser.error("Use A=1..15 DN. Lossless A=0 has no offset effect and is omitted; A>15 is not allowed.")
    if not 16<=args.t_inc<=2048 or args.t_inc&(args.t_inc-1):
        parser.error("t_inc must be a power of two from 16 to 2048.")
    if not -6<=args.nu_min<=min(7-p for p in args.phi_values):
        parser.error("Require -6 <= nu_min <= 7-phi for EVERY selected phi.")
    if any(not 2<=n<=65535 for n in args.crop):parser.error("Crop dimensions must be 2..65535.")
    if not 0<=args.p<=15:parser.error("Require P=0..15.")
    if args.workers<1:parser.error("Workers must be positive.")
    if not 10<=args.memory_poll_ms<=1000:parser.error("memory-poll-ms must be 10..1000.")
    if any(not math.isfinite(v) or v<=0 for v in (args.max_memory_gb,args.total_memory_gb)):
        parser.error("Memory budgets must be positive and finite.")
    if args.output is None:args.output=Path(f"results/offset_phi_{args.crop[0]}x{args.crop[1]}_tinc{args.t_inc}")
    args.images=list(dict.fromkeys(args.images or list(core.SCENES)))
    return args


def build_jobs(args,inputs):
    jobs=[];y,x=args.crop
    for scene,info in inputs.items():
        sy,sx,z=info["source_shape_yxz"]
        row,col=(0,0) if args.anchor=="top-left" else ((sy-y)//2,(sx-x)//2)
        issue="" if y<=sy and x<=sx else f"Requested {y}x{x} does not fit source {sy}x{sx}."
        for A,phi,psi in itertools.product(args.a_values,args.phi_values,args.psi_values):
            nmax=7-phi
            identity=[scene,y,x,row,col,A,phi,psi,args.nu_min,nmax,args.t_inc]
            job_id=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
            segments=engine.schedule_segments(y,x,args.nu_min,args.t_inc,nmax)
            jobs.append(dict(job_id=job_id,image=scene,size=f"{y}x{x}",rows=y,columns=x,bands=z,
                             crop_row=row,crop_col=col,t_inc=args.t_inc,nu_min=args.nu_min,nu_max=nmax,
                             phi=phi,psi=psi,a=A,
                             planned_nu_max_reached=int(any(s["nu"]==nmax for s in segments)),
                             estimated_ram_gib=engine.estimate_gib((y,x,z),args.p,args.low_memory),
                             dimension_issue=issue))
    return jobs


def write_summaries(rows,study,out):
    """Crop/grid minima, and improvements relative to psi=7 at the SAME phi."""
    images=list(study["inputs"]);avals=study["a_values"];phis=study["phi_values"]
    baseline={(r["image"],int(r["a"]),int(r["phi"])):r for r in rows if int(r["psi"])==7}
    keys=["image","a","phi","psi","nu_min","nu_max","t_inc","bits_per_sample","max_error","rmse_dn","nu_max_reached"]
    fields=keys+["baseline_psi7_bits_per_sample","gain_vs_psi7_bits_per_sample","gain_vs_psi7_percent",
                 "completed_points","expected_points","complete","tied_best_phi_psi"]
    def summarize(group,expected):
        if not group:return None
        # Rate is exactly stream_bits/sample_count. Prefer the existing psi=7
        # baseline only when there is an exact rate tie, never a near tie.
        minimum=min(float(r["bits_per_sample"]) for r in group)
        ties=[r for r in group if float(r["bits_per_sample"])==minimum]
        winner=min(ties,key=lambda r:(abs(7-int(r["psi"])),int(r["phi"])))
        ref=baseline.get((winner["image"],int(winner["a"]),int(winner["phi"])))
        result={k:winner[k] for k in keys}
        b=float(ref["bits_per_sample"]) if ref else None
        gain=b-minimum if b is not None else None
        result.update(baseline_psi7_bits_per_sample=b if b is not None else "",
                      gain_vs_psi7_bits_per_sample=gain if gain is not None else "",
                      gain_vs_psi7_percent=100*gain/b if b else "",
                      completed_points=len(group),expected_points=expected,complete=int(len(group)==expected),
                      tied_best_phi_psi=";".join(f"{r['phi']}:{r['psi']}" for r in ties))
        return result
    per_phi=[];joint=[]
    for scene,A in itertools.product(images,avals):
        group=[r for r in rows if r["image"]==scene and int(r["a"])==A]
        entry=summarize(group,len(phis)*len(study["psi_values"]))
        if entry:joint.append(entry)
        for phi in phis:
            entry=summarize([r for r in group if int(r["phi"])==phi],len(study["psi_values"]))
            if entry:per_phi.append(entry)
    engine.table(out/"best_psi_per_image_a_phi.csv",fields,per_phi)
    engine.table(out/"best_phi_psi_per_image_a.csv",fields,joint)


def plot_results(records,metadata,out):
    if not records:return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    study=metadata["study"];profile=study["profile"]
    images=list(study["inputs"]);avals=study["a_values"];phis=study["phi_values"]
    xx=study["psi_values"];rows=[r["row"] for r in records]
    by_point={(r["image"],int(r["a"]),int(r["phi"]),int(r["psi"])):r for r in rows}
    colors=["#0072B2","#D55E00","#009E73","#CC79A7","#E69F00","#5B4BCE","#8C564B","#666666"]
    markers=["o","s","D","^","v","P","X",">"]
    setting=(f"P={profile['p']} | {profile['local_sum']}, {profile['prediction_mode']} | "
             rf"$\nu_{{min}}={profile['nu_min']}$, $\nu_{{max}}=7-\phi$, "
             rf"$t_{{inc}}={profile['t_inc']}$, $\Theta=3$")
    def draw(selected,filename):
        nrows=len(selected);ncols=len(avals)
        width=max(8.2,5.8*ncols);height=3.2*nrows+2.0
        fig,axes=plt.subplots(nrows,ncols,squeeze=False,figsize=(width,height))
        for i,scene in enumerate(selected):
            info=study["inputs"][scene]
            for j,A in enumerate(avals):
                ax=axes[i,j];present=0
                for phi in phis:
                    pts=[by_point.get((scene,A,phi,psi)) for psi in xx]
                    yy=[float(r["bits_per_sample"]) if r else np.nan for r in pts]
                    ax.plot(xx,yy,color=colors[phi],linewidth=1.4)
                    for x,y,r in zip(xx,yy,pts):
                        if r is None:continue
                        present+=1
                        ax.plot(x,y,linestyle="none",marker=markers[phi],markersize=4.5,
                                markeredgecolor=colors[phi],markeredgewidth=1.1,
                                markerfacecolor=colors[phi] if int(r["nu_max_reached"]) else "white")
                expected=len(phis)*len(xx)
                partial=f" | {present}/{expected}" if present!=expected else ""
                ax.set_title(f"{info['label']} | A = {A} DN{partial}\n"
                             f"{study['crop'][0]} x {study['crop'][1]} pixels, {info['source_shape_yxz'][2]} bands",fontsize=10.5)
                ax.set_xlabel(r"Sample-representative offset, $\psi$",fontsize=10)
                ax.set_ylabel("Compressed bits/sample",fontsize=10)
                ax.set_xticks(xx);ax.set_xlim(min(xx)-.25,max(xx)+.25)
                ax.ticklabel_format(axis="y",useOffset=False)
                ax.grid(True,alpha=.25);ax.set_axisbelow(True)
                if not present:ax.text(.5,.5,"No completed points",ha="center",va="center",transform=ax.transAxes)
        done=sum(r["image"] in selected for r in rows)
        total=len(selected)*len(avals)*len(phis)*len(xx)
        fig.suptitle(f"HYPSO offset / damping study | hybrid encoder | {done}/{total}\n"+setting,fontsize=12)
        handles=[Line2D([],[],color=colors[p],marker=markers[p],linewidth=1.4,
                        label=rf"$\phi={p}$, $\nu_{{max}}={7-p}$") for p in phis]
        columns=min(len(phis),3 if ncols<3 else 6)
        fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.065 if nrows==1 else .045),
                   ncol=columns,frameon=False,fontsize=10)
        fig.text(.5,.012,"Hollow markers: final adaptation parameter not reached within this crop.\n"
                 "Rate includes header, entropy payload/tail and word padding; external initialization excluded.",ha="center",fontsize=8)
        fig.tight_layout(rect=(0,.25 if nrows==1 else .10,1,.86 if nrows==1 else .94),h_pad=2.3,w_pad=2.0)
        for ext in ("png","pdf"):fig.savefig(out/(filename+"."+ext),dpi=170,bbox_inches="tight")
        plt.close(fig)
    draw(images,"all_images_offset_phi")
    for scene in images:draw([scene],scene+"_offset_phi")
    write_summaries(rows,study,out)


def make_study(args,repo,inputs,jobs):
    own=Path(__file__).resolve().parent
    profile={k:getattr(args,k) for k in ("p","nu_min","t_inc","local_sum","prediction_mode")}
    profile.update(D=16,Omega=19,R=64,Theta=3,encoder="HYBRID",nu_max_rule="7-phi")
    model_files=list((repo/"ccsds123_i2_hlm").rglob("*.py"))+list((repo/"ccsds123_i2_hlm").rglob("*.txt"))
    study=dict(tool="offset_phi",version=VERSION,profile=profile,inputs=inputs,jobs=jobs,
               crop=args.crop,anchor=args.anchor,psi_values=args.psi_values,phi_values=args.phi_values,
               a_values=args.a_values,a_upper_limit=15,low_memory=args.low_memory,python=platform.python_version(),
               numpy=np.__version__,psutil=psutil.__version__,platform=platform.platform(),
               model_sha256=core.file_digest(model_files,repo),bundle_sha256=core.file_digest(list(own.glob("*.py")),own),
               reference="CCSDS 120.2-G-2 Figure 4-12 style; HYPSO data, no lossless cases, A<=15 DN.",
               choice_note="Default t_inc=512 is a common compromise inferred visually from the supplied 128x128 t_inc/phi plots, not a numerically established global optimum. All phi curves retained to check offset/damping interaction.",
               metric="Header + entropy payload/tail + word padding per scalar sample; external initialization excluded, inclusive rate also saved.",
               execution="Independent complete configurations in separate processes, one native thread each; no parallelization within a configuration.",
               checks="Encoder-side reconstruction bounds and actual applied adaptation schedule; no independent decoder.",
               adaptation="nu_max_reached means the programmed exponent reached its cap, not learned-weight convergence.")
    return {"study_id":hashlib.sha256(json.dumps(study,sort_keys=True).encode()).hexdigest(),
            "study":study,"model_path":str(repo)}


def main():
    args=parse_args();out=args.output.resolve()
    if args.plot_only:
        if not out.is_dir():raise ValueError("Output folder does not exist.")
        with sw.output_lock(out):
            metadata=json.loads((out/"metadata.json").read_text(encoding="utf-8"))
            if metadata.get("study",{}).get("tool")!="offset_phi":raise ValueError("Not an offset/phi study output folder.")
            records=engine.regenerate(out,metadata);plot_results(records,metadata,out)
        print(f"Plots: {out}");return 0
    if not __debug__:raise ValueError("Do not use python -O; the model uses assertions.")
    repo=engine.find_repo(args.repo);sw.load_model(repo,"vmax")
    if args.low_memory:
        from hybrid_no_traces import validate_reference
        validate_reference()
    paths=([(p.stem,p.resolve()) for p in args.files] if args.files else
           [(s,core.find_capture(args.input_dir.resolve(),core.SCENES[s])) for s in args.images])
    inputs={}
    for scene,path in paths:
        if scene in inputs:raise ValueError("Input names collide; use distinct filenames.")
        if not scene or any(ch in scene for ch in '/\\:*?"<>|'):raise ValueError("Use filenames without special path characters.")
        info=engine.inspect_input(path,args)
        info["label"]=scene.title() if scene in core.SCENES else scene.replace('_',' ')
        inputs[scene]=info
    jobs=build_jobs(args,inputs);metadata=make_study(args,repo,inputs,jobs)
    limit=engine.allowed_workers(args.workers)
    print(f"{len(inputs)} images x {len(args.a_values)} A values x {len(args.phi_values)} phi values x "
          f"{len(args.psi_values)} offsets = {len(jobs)} configurations.")
    print(f"A={args.a_values}; phi={args.phi_values}; psi={args.psi_values}; FIXED t_inc={args.t_inc}; "
          f"nu_min={args.nu_min}; nu_max=7-phi.")
    print(f"One complete configuration per process; requested workers {args.workers}, CPU allowance {limit}.")
    for scene,info in inputs.items():
        group=[j for j in jobs if j["image"]==scene];job=group[0];need=job["estimated_ram_gib"]
        reason=job["dimension_issue"]
        if not reason and need>args.max_memory_gb:reason="exceeds per-job RAM ceiling"
        budget=engine.execution_budget(args)
        if not reason and need+.35>budget:reason="exceeds current available/total RAM budget"
        fits=max(0,min(limit,int(max(0,budget-.35)//need)))
        print(f"{info['label']}: source {info['source_shape_yxz']}, crop {job['size']} at "
              f"({job['crop_row']},{job['crop_col']}), estimated {need:.2f} GiB/config; about {fits} jobs fit now"+
              (f"; SKIP NOW: {reason}" if reason else ""))
        unfinished=sorted({(j['phi'],j['nu_max']) for j in group if not j['planned_nu_max_reached']})
        if unfinished:print("  Hollow markers will flag unreached caps: "+", ".join(
                            f"phi={p}, nu_max={v}" for p,v in unfinished))
    if args.dry_run:
        print("Dry run: no compression or output writes.");return 0
    with sw.output_lock(out):
        meta_path=out/"metadata.json"
        if meta_path.exists():
            old=json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("study_id")!=metadata["study_id"] or old.get("study")!=metadata["study"]:
                raise ValueError("Inputs, crop, grid, fixed settings, storage or code changed. Use a new output folder.")
        elif (out/"results.csv").exists() or (out/"jobs").exists():
            raise ValueError("Results exist without matching metadata. Use a new output folder.")
        (out/"jobs").mkdir(exist_ok=True)
        core.write_json(meta_path,metadata);engine.table(out/"plan.csv",list(jobs[0]),jobs)
        return engine.run_jobs(args,metadata,repo,out,plot_results)


if __name__=="__main__":
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Completed configurations are saved; repeat the same command to resume.")
        raise SystemExit(130)
    except (ValueError,OSError,ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr);raise SystemExit(1)
