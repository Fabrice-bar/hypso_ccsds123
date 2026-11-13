#!/usr/bin/env python3
r"""HYPSO Figure 4-13-style offset/damping/noise study.

Heatmaps show 100*(compressed_size/best_tested_size - 1) at each noise level.
Default: five scenes, centre 128x128, A=3/15, full phi/psi grid 0..7 (Theta=3),
nu_min=-1, FIXED nu_max=4, t_inc=512, P=3, reduced/narrow-neighbour/hybrid.
First spectral band uses phi=psi=0, as in Figure 4-13.

Default added Gaussian sigma/A levels: 0, 0.5, 1, 2, 5, 10.
Alternatively pass --sigmas with literal DN standard deviations.
Each worker runs one complete configuration, with deterministic paired noise.
--sigmas 0 runs just the original-image joint grid; --theta 4 refines to 16x16.
Runtime/memory are recorded, not plotted. Repeat the command to resume.
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
import _noise_study as ns
import sweep_workers as sw
from _fig413_plots import plot_results

VERSION="1.0.0-fig413"


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir",type=Path,default=Path("data"))
    choose=p.add_mutually_exclusive_group()
    choose.add_argument("--images",nargs="+",choices=list(core.SCENES))
    choose.add_argument("--files",type=Path,nargs="+")
    p.add_argument("--crop",type=int,nargs=2,default=[128,128],metavar=("HEIGHT","WIDTH"))
    p.add_argument("--anchor",choices=("center","top-left"),default="center")
    p.add_argument("--a-values",type=int,nargs="+",default=[3,15])
    p.add_argument("--theta",type=int,choices=(3,4),default=3,help="3: full 8x8 grid; 4: full 16x16 grid.")
    p.add_argument("--phi-values",type=int,nargs="+",help="Default: every legal damping value for Theta.")
    p.add_argument("--psi-values",type=int,nargs="+",help="Default: every legal offset value for Theta.")
    noise=p.add_mutually_exclusive_group()
    noise.add_argument("--noise-ratios",type=float,nargs="+",help="sigma/A values; default 0 .5 1 2 5 10.")
    noise.add_argument("--sigmas",type=float,nargs="+",help="Literal added-noise sigma values in DN, e.g. 0 25 50 100 200 400.")
    p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--repeats",type=int,default=1,help="Independent noise realizations per nonzero sigma; default 1.")
    p.add_argument("--first-band",choices=("zero","uniform"),default="zero",help="zero matches the figure; uniform applies the pair to every band.")
    p.add_argument("--nu-min",type=int,default=-1)
    p.add_argument("--nu-max",type=int,default=4,help="Fixed over the WHOLE grid; default 4.")
    p.add_argument("--t-inc",type=int,default=512)
    p.add_argument("--p",type=int,default=3)
    p.add_argument("--prediction-mode",choices=("reduced","full"),default="reduced")
    p.add_argument("--local-sum",choices=list(core.LOCAL_SUM_OPTIONS),default="narrow-neighbor")
    p.add_argument("--variable",default="products/dn")
    p.add_argument("--axis-order",choices=("yxz","yzx","xyz","xzy","zxy","zyx"))
    storage=p.add_mutually_exclusive_group()
    storage.add_argument("--low-memory",dest="low_memory",action="store_true")
    storage.add_argument("--standard-memory",dest="low_memory",action="store_false")
    p.set_defaults(low_memory=True)
    p.add_argument("--workers",type=int,default=2)
    p.add_argument("--max-memory-gb",type=float,default=90)
    p.add_argument("--total-memory-gb",type=float,default=90)
    p.add_argument("--memory-poll-ms",type=int,default=50)
    p.add_argument("--repo",type=Path)
    p.add_argument("--output",type=Path)
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--plot-only",action="store_true")
    args=p.parse_args(argv)
    limit=2**args.theta
    args.phi_values=sorted(set(args.phi_values if args.phi_values is not None else range(limit)))
    args.psi_values=sorted(set(args.psi_values if args.psi_values is not None else range(limit)))
    args.a_values=sorted(set(args.a_values))
    if any(not 1<=a<=15 for a in args.a_values):p.error("A must be 1..15 DN; no lossless or A>15 cases.")
    if any(not 0<=v<limit for v in args.phi_values+args.psi_values):p.error(f"Theta={args.theta} requires phi and psi in 0..{limit-1}.")
    if args.sigmas is None:
        values=args.noise_ratios if args.noise_ratios is not None else [0,.5,1,2,5,10]
        args.noise_mode="ratio"
    else:values=args.sigmas;args.noise_mode="dn"
    if any(not math.isfinite(v) or v<0 for v in values):p.error("Noise levels must be finite and nonnegative.")
    args.noise_values=sorted(set(0.0 if v==0 else float(v) for v in values))
    if any((v*a if args.noise_mode=="ratio" else v)>65535 for v in args.noise_values for a in args.a_values):
        p.error("Added-noise sigma must not exceed 65535 DN.")
    if args.seed<0 or args.seed>=2**64:p.error("Seed must be in 0..2^64-1.")
    if args.repeats<1:p.error("Repeats must be positive.")
    if not -6<=args.nu_min<=args.nu_max<=9:p.error("Require -6 <= nu_min <= nu_max <= 9.")
    if not 16<=args.t_inc<=2048 or args.t_inc&(args.t_inc-1):p.error("t_inc must be a power of two from 16 through 2048.")
    if any(not 2<=n<=65535 for n in args.crop):p.error("Crop dimensions must be 2..65535.")
    if not 0<=args.p<=15:p.error("P must be 0..15.")
    if args.workers<1:p.error("Workers must be positive.")
    if not 10<=args.memory_poll_ms<=1000:p.error("memory-poll-ms must be 10..1000.")
    if any(not math.isfinite(v) or v<=0 for v in (args.max_memory_gb,args.total_memory_gb)):
        p.error("Memory budgets must be positive and finite.")
    args.images=list(dict.fromkeys(args.images or list(core.SCENES)))
    if args.output is None:args.output=Path(f"results/fig413_{args.crop[0]}x{args.crop[1]}_theta{args.theta}")
    return args


def build_jobs(args,inputs):
    jobs=[];y,x=args.crop
    for scene,info in inputs.items():
        sy,sx,z=info["source_shape_yxz"]
        row,col=(0,0) if args.anchor=="top-left" else ((sy-y)//2,(sx-x)//2)
        issue="" if y<=sy and x<=sx else f"Requested {y}x{x} does not fit source {sy}x{sx}."
        if z<2 and args.first_band=="zero":issue="At least two bands are needed when the first band is fixed to zero."
        for A,level in itertools.product(args.a_values,args.noise_values):
            sigma=float(level*A if args.noise_mode=="ratio" else level)
            # sigma=0 contains no random component: encode it once per pair.
            reps=1 if sigma==0 else args.repeats
            for rep,phi,psi in itertools.product(range(reps),args.phi_values,args.psi_values):
                seed=ns.noise_seed(info["file_sha256"],y,x,row,col,args.seed,rep)
                identity=[scene,y,x,row,col,A,phi,psi,args.theta,args.first_band,args.nu_min,args.nu_max,
                          args.t_inc,level,sigma,rep,seed]
                jid=hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
                segs=engine.schedule_segments(y,x,args.nu_min,args.t_inc,args.nu_max)
                jobs.append(dict(job_id=jid,image=scene,size=f"{y}x{x}",rows=y,columns=x,bands=z,
                                 crop_row=row,crop_col=col,t_inc=args.t_inc,nu_min=args.nu_min,nu_max=args.nu_max,
                                 a=A,phi=phi,psi=psi,noise_level=level,sigma_dn=sigma,sigma_over_a=sigma/A,
                                 noise_rep=rep,noise_seed=seed,
                                 planned_nu_max_reached=int(any(s["nu"]==args.nu_max for s in segs)),
                                 estimated_ram_gib=engine.estimate_gib((y,x,z),args.p,args.low_memory),
                                 dimension_issue=issue))
    return jobs


def make_study(args,repo,inputs,jobs):
    own=Path(__file__).resolve().parent
    profile={k:getattr(args,k) for k in ("p","nu_min","nu_max","t_inc","local_sum","prediction_mode","theta","first_band")}
    profile.update(D=16,R=64,Omega=19,encoder="HYBRID",nu_max_rule="fixed")
    model=list((repo/"ccsds123_i2_hlm").rglob("*.py"))+list((repo/"ccsds123_i2_hlm").rglob("*.txt"))
    study=dict(tool="fig413",version=VERSION,profile=profile,inputs=inputs,jobs=jobs,
               crop=args.crop,anchor=args.anchor,a_values=args.a_values,a_upper_limit=15,
               phi_values=args.phi_values,psi_values=args.psi_values,
               noise_mode=args.noise_mode,noise_values=args.noise_values,repeats=args.repeats,seed=args.seed,
               noise_method="PCG64 standard_normal; common field for all pairs, A and sigma at each scene/crop/replicate; rint then clip to 0..65535. Sigma=0 uses the original crop exactly.",
               reference="CCSDS 120.2-G-2 Figure 4-13 style; HYPSO raw crops retain existing instrument noise. No AVIRIS-NG preprocessing or noise removal is claimed.",
               book_differences="Book uses processed high-SNR AVIRIS-NG, P=2, A=40, Theta=4; HYPSO defaults P=3, A=3/15, Theta=3, fixed nu_max=4, t_inc=512. First-band representative parameters zero by default.",
               metric="Expansion=100*(mean compressed rate/minimum mean compressed rate in the same image/A/sigma grid - 1). Mean is over paired noise realizations. Incomplete cells excluded; incomplete panels provisional.",
               stream_definition="Header (including representative tables) + entropy payload/tail + word padding per scalar sample; external initialization excluded, inclusive rate saved.",
               checks="Encoder-side reconstruction bound against the NOISY compressor input, not the original crop; no independent decoder.",
               low_memory=args.low_memory,python=platform.python_version(),numpy=np.__version__,psutil=psutil.__version__,
               platform=platform.platform(),model_sha256=core.file_digest(model,repo),
               bundle_sha256=core.file_digest(list(own.glob("*.py")),own))
    return {"study_id":hashlib.sha256(json.dumps(study,sort_keys=True).encode()).hexdigest(),
            "study":study,"model_path":str(repo)}


def main():
    args=parse_args();out=args.output.resolve()
    if args.plot_only:
        if not out.is_dir():raise ValueError("Output folder does not exist.")
        with sw.output_lock(out):
            metadata=json.loads((out/"metadata.json").read_text(encoding="utf-8"))
            if metadata.get("study",{}).get("tool")!="fig413":raise ValueError("Not a Figure 4-13 study folder.")
            plot_results(engine.regenerate(out,metadata),metadata,out)
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
        if scene in inputs:raise ValueError("Input names collide.")
        if not scene or any(ch in scene for ch in '/\\:*?"<>|'):raise ValueError("Use filenames without special path characters.")
        info=engine.inspect_input(path,args)
        info["label"]=scene.title() if scene in core.SCENES else scene.replace('_',' ')
        inputs[scene]=info
    jobs=build_jobs(args,inputs);metadata=make_study(args,repo,inputs,jobs)
    print(f"{len(jobs)} configurations; {len(inputs)} images, A={args.a_values}, "
          f"{len(args.phi_values)}x{len(args.psi_values)} phi/psi grid, {len(args.noise_values)} noise levels.")
    print(f"Theta={args.theta}, P={args.p}, nu_min={args.nu_min}, FIXED nu_max={args.nu_max}, t_inc={args.t_inc}, first band={args.first_band}.")
    for A in args.a_values:
        sigmas=[v*A if args.noise_mode=="ratio" else v for v in args.noise_values]
        print(f"A={A}: added sigma (DN) = {sigmas}; repeats={args.repeats} (sigma=0 runs once).")
    print(f"One COMPLETE configuration per worker; requested {args.workers}, CPU allowance {engine.allowed_workers(args.workers)}.")
    for scene,info in inputs.items():
        job=next(j for j in jobs if j["image"]==scene);need=job["estimated_ram_gib"]
        fits=min(engine.allowed_workers(args.workers),int(max(0,engine.execution_budget(args)-.35)//need))
        print(f"{info['label']}: {job['size']} at ({job['crop_row']},{job['crop_col']}), "
              f"{need:.2f} GiB estimated/job, about {fits} fit now; "+
              (job['dimension_issue'] or f"nu_max reached={bool(job['planned_nu_max_reached'])}"))
    if args.dry_run:print("Dry run: no compression or output writes.");return 0
    with sw.output_lock(out):
        path=out/"metadata.json"
        if path.exists():
            old=json.loads(path.read_text(encoding="utf-8"))
            if old.get("study_id")!=metadata["study_id"] or old.get("study")!=metadata["study"]:
                raise ValueError("Inputs, crop, grid, noise, settings, storage or code changed. Use a new output folder.")
        elif (out/"results.csv").exists() or (out/"jobs").exists():
            raise ValueError("Existing results have no matching metadata. Use a new output folder.")
        (out/"jobs").mkdir(exist_ok=True)
        core.write_json(path,metadata);engine.table(out/"plan.csv",list(jobs[0]),jobs)
        return engine.run_jobs(args,metadata,repo,out,plot_results)


if __name__=="__main__":
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Completed configurations are saved; repeat the same command to resume.")
        raise SystemExit(130)
    except (ValueError,OSError,ModuleNotFoundError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr);raise SystemExit(1)
